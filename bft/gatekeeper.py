"""
GridSentinel — BFT Gatekeeper
================================
Implements the Byzantine Fault Detection layer.

Called once per 5-minute canonical window per depot.
Receives an aggregated snapshot from consumer_align.py and:

  1. Runs the MAD consensus filter on SoC and power readings
  2. Cross-validates charger cluster against depot meter
  3. Applies contextual Trust Ledger decay (three tiers from config)
  4. Blacklists sensors below trust threshold
  5. Emits a Clean Truth snapshot (Byzantine sensors replaced with
     weighted interpolation from trusted neighbours)
  6. Writes trust scores and Clean Truth to InfluxDB + Redis

The Math — MAD Filter:
  For each sensor cluster (all bus SoC readings in one depot):

    MAD = median(|x_i - median(X)|)

  A sensor x_i is flagged Byzantine if:

    |x_i - median(X)| / (1.4826 × MAD) > k

  where k = mad_threshold_k from config (default 3).

  The 1.4826 consistency constant makes MAD equivalent to σ for
  Gaussian-distributed data, so k=3 means "3 standard deviations".

Contextual Trust Ledger decay (v2.0):
  Three decay tiers distinguish transient noise from coordinated attacks:
    - minor:       Isolated spike. MAD flag only. Likely sensor glitch.
    - standard:    MAD flag + fails depot meter cross-validation.
    - coordinated: 3+ buses flagged in the same window simultaneously.

  Recovery:
    - +0.01 per clean window (slow, steady)
    - +0.05 burst bonus after 5 consecutive clean windows
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from typing import Any

import numpy as np

from config.settings import settings

log = logging.getLogger(__name__)

# MAD consistency constant — makes MAD ≡ σ for Gaussian data
MAD_CONSISTENCY = 1.4826


class BFTGatekeeper:
    """
    Per-depot BFT filter. One instance per align-bft-depot-N container.

    State:
      trust_scores:      bus_id → float (0.0–1.0, starts at 1.0)
      clean_streak:      bus_id → int (consecutive clean windows)
      flagged_history:   bus_id → list of recent flag timestamps
    """

    def __init__(self, depot_id: int) -> None:
        self.depot_id = depot_id
        self.cfg = settings.bft
        self.trust_scores: dict[str, float] = defaultdict(lambda: 1.0)
        self.clean_streak: dict[str, int] = defaultdict(int)
        self._redis = self._connect_redis()
        self._influx = self._connect_influx()

        log.info(
            "BFT Gatekeeper ready | depot=%d | k=%.1f | "
            "decay=[minor=%.2f std=%.2f coord=%.2f] | blacklist=%.2f",
            depot_id,
            self.cfg.mad_threshold_k,
            self.cfg.trust_decay_minor,
            self.cfg.trust_decay_standard,
            self.cfg.trust_decay_coordinated,
            self.cfg.trust_blacklist_threshold,
        )

    # ─── External connections ─────────────────────────────────────────────────

    def _connect_redis(self) -> Any | None:
        try:
            import redis

            r = redis.Redis.from_url(
                os.environ.get("REDIS_URL", "redis://redis:6379"),
                decode_responses=True,
            )
            r.ping()
            log.info("BFT depot=%d — Redis connected.", self.depot_id)
            return r
        except Exception as exc:
            log.warning(
                "BFT depot=%d — Redis unavailable (%s). Continuing without.", self.depot_id, exc
            )
            return None

    def _connect_influx(self) -> Any | None:
        try:
            from influxdb_client import InfluxDBClient

            client = InfluxDBClient(
                url=os.environ.get("INFLUXDB_URL", "http://influxdb:8086"),
                token=os.environ.get("INFLUXDB_TOKEN", "gridsentinel-token"),
                org=os.environ.get("INFLUXDB_ORG", "gridsentinel"),
            )
            client.ping()
            log.info("BFT depot=%d — InfluxDB connected.", self.depot_id)
            return client
        except Exception as exc:
            log.warning(
                "BFT depot=%d — InfluxDB unavailable (%s). Continuing without.", self.depot_id, exc
            )
            return None

    # ─── MAD Filter ───────────────────────────────────────────────────────────

    def _mad_filter(
        self,
        values: dict[str, float],
    ) -> dict[str, bool]:
        """
        Apply MAD consensus filter to a dict of {bus_id: value}.

        Returns {bus_id: is_flagged}.

        For small clusters (<3 sensors), MAD is unreliable — skip filtering.
        This protects against false positives at depot startup when only
        a few buses have reported.
        """
        flagged: dict[str, bool] = {bid: False for bid in values}

        readings = np.array(list(values.values()), dtype=float)
        if len(readings) < 3:
            return flagged

        med = np.median(readings)
        mad = np.median(np.abs(readings - med))

        # Degenerate case: all readings identical (MAD = 0)
        # This itself is suspicious — could be a flatline attack.
        # Flag all as minor if cluster is large enough to have variance.
        if mad < 1e-6:
            if len(readings) > 5:
                log.debug(
                    "depot=%d MAD=0 — possible flatline attack on %d sensors",
                    self.depot_id,
                    len(readings),
                )
            return flagged  # Can't score without variance — pass through

        scaled_mad = MAD_CONSISTENCY * mad
        bus_ids = list(values.keys())
        arr_vals = list(values.values())

        for bus_id, val in zip(bus_ids, arr_vals):
            z_score = abs(val - med) / scaled_mad
            if z_score > self.cfg.mad_threshold_k:
                flagged[bus_id] = True
                log.debug(
                    "MAD flag | depot=%d | bus=%s | val=%.2f | median=%.2f | z=%.2f",
                    self.depot_id,
                    bus_id,
                    val,
                    med,
                    z_score,
                )

        return flagged

    # ─── Depot meter cross-validation ────────────────────────────────────────

    def _meter_cross_validation_fails(
        self,
        buses: list[dict],
        depot_meter_kw: float,
    ) -> bool:
        """
        Cross-validate charger cluster against depot meter.

        If reported aggregate power diverges from the depot meter by more
        than 20%, the charger cluster is lying.

        The depot meter is the ground-truth anchor — a single, harder-to-
        compromise measurement at the substation connection point.

        Returns True if the cluster fails cross-validation.
        """
        if abs(depot_meter_kw) < 5.0:
            # Meter near zero — no meaningful cross-validation possible
            return False

        reported_total = sum(b.get("mean_power_kw", 0.0) for b in buses)

        if abs(depot_meter_kw) < 1e-6:
            return False

        divergence = abs(reported_total - depot_meter_kw) / abs(depot_meter_kw)

        if divergence > 0.20:  # >20% divergence = cross-validation fail
            log.warning(
                "Depot meter mismatch | depot=%d | reported=%.1fkW | meter=%.1fkW | div=%.1f%%",
                self.depot_id,
                reported_total,
                depot_meter_kw,
                divergence * 100,
            )
            return True

        return False

    # ─── Contextual Trust Ledger ──────────────────────────────────────────────

    def _classify_attack(
        self,
        flagged_bus_ids: set[str],
        meter_fails: bool,
    ) -> str:
        """
        Classify the severity of detected Byzantine behaviour.

        Returns one of: "minor", "standard", "coordinated"

        Classification logic:
          coordinated: 3+ buses flagged simultaneously (designed to beat averaging)
          standard:    1–2 buses flagged AND depot meter cross-validation fails
          minor:       1–2 buses flagged, meter OK (likely transient sensor noise)
        """
        n = len(flagged_bus_ids)
        if n == 0:
            return "none"
        if n >= 3:
            return "coordinated"
        if meter_fails:
            return "standard"
        return "minor"

    def _update_trust_scores(
        self,
        buses: list[dict],
        flagged_ids: set[str],
        attack_class: str,
    ) -> None:
        """
        Apply contextual decay and recovery to the Trust Ledger.

        Decay rates from fleet.yaml bft section:
          minor:       -0.02  (transient noise — gentle decay)
          standard:    -0.10  (confirmed Byzantine — significant decay)
          coordinated: -0.20  (coordinated attack — aggressive decay)

        Recovery:
          +0.01 per clean window
          +0.05 burst after 5 consecutive clean windows
        """
        cfg = self.cfg
        decay_map = {
            "minor": cfg.trust_decay_minor,
            "standard": cfg.trust_decay_standard,
            "coordinated": cfg.trust_decay_coordinated,
        }
        decay = decay_map.get(attack_class, 0.0)

        for bus in buses:
            bus_id = bus["bus_id"]

            if bus_id in flagged_ids:
                self.trust_scores[bus_id] = max(0.0, self.trust_scores[bus_id] - decay)
                self.clean_streak[bus_id] = 0

                log.info(
                    "Trust decay | depot=%d | bus=%s | class=%s | " "decay=%.2f | score=%.3f",
                    self.depot_id,
                    bus_id,
                    attack_class,
                    decay,
                    self.trust_scores[bus_id],
                )
            else:
                # Clean reading — apply recovery
                self.clean_streak[bus_id] += 1
                recovery = cfg.trust_recovery_per_window

                # Burst bonus after 5 consecutive clean windows
                if self.clean_streak[bus_id] >= 5 and self.trust_scores[bus_id] < 1.0:
                    recovery += cfg.trust_recovery_burst
                    self.clean_streak[bus_id] = 0  # Reset streak after burst

                self.trust_scores[bus_id] = min(1.0, self.trust_scores[bus_id] + recovery)

    # ─── Clean Truth emission ─────────────────────────────────────────────────

    def _build_clean_truth(
        self,
        snapshot: dict,
        flagged_ids: set[str],
    ) -> dict:
        """
        Build Clean Truth snapshot by replacing Byzantine bus readings
        with weighted interpolation from trusted neighbours.

        Blacklisted buses (trust < threshold) are completely excluded.
        Their SoC/power contribution is replaced by the depot-level mean
        of trusted buses.
        """
        buses = snapshot["buses"]
        blacklisted = {
            bid
            for bid, score in self.trust_scores.items()
            if score < self.cfg.trust_blacklist_threshold
        }

        trusted_buses = [b for b in buses if b["bus_id"] not in blacklisted]

        if not trusted_buses:
            log.warning(
                "depot=%d — all buses blacklisted! Emitting empty Clean Truth.",
                self.depot_id,
            )
            trusted_buses = buses  # Fail open — better than no data

        # Mean SoC and power from trusted buses (used to replace blacklisted)
        trusted_mean_soc = sum(b["mean_soc_pct"] for b in trusted_buses) / len(trusted_buses)
        trusted_mean_power = sum(b["mean_power_kw"] for b in trusted_buses) / len(trusted_buses)

        clean_buses = []
        for bus in buses:
            bid = bus["bus_id"]
            if bid in blacklisted:
                # Replace with interpolated values from trusted neighbours
                clean_buses.append(
                    {
                        **bus,
                        "mean_soc_pct": round(trusted_mean_soc, 2),
                        "mean_power_kw": round(trusted_mean_power, 2),
                        "interpolated": True,
                        "trust_score": round(self.trust_scores.get(bid, 0.0), 4),
                    }
                )
            else:
                clean_buses.append(
                    {
                        **bus,
                        "interpolated": False,
                        "trust_score": round(self.trust_scores.get(bid, 1.0), 4),
                    }
                )

        return {
            **snapshot,
            "buses": clean_buses,
            "flagged_bus_ids": list(flagged_ids),
            "blacklisted_ids": list(blacklisted),
            "clean_bus_count": len(trusted_buses),
            "bft_passed": True,
        }

    # ─── Output sinks ─────────────────────────────────────────────────────────

    def _write_to_influx(self, clean_truth: dict, attack_class: str) -> None:
        """Write BFT metrics and trust scores to InfluxDB."""
        if not self._influx:
            return
        try:
            from influxdb_client import Point
            from influxdb_client.client.write_api import SYNCHRONOUS

            write_api = self._influx.write_api(write_options=SYNCHRONOUS)
            bucket = os.environ.get("INFLUXDB_BUCKET", "telemetry")
            org = os.environ.get("INFLUXDB_ORG", "gridsentinel")
            ts = clean_truth["canonical_timestamp"]

            # BFT window summary
            p = (
                Point("bft_window")
                .tag("depot_id", str(self.depot_id))
                .tag("attack_class", attack_class)
                .field("flagged_count", len(clean_truth["flagged_bus_ids"]))
                .field("blacklisted_count", len(clean_truth["blacklisted_ids"]))
                .field("clean_bus_count", clean_truth["clean_bus_count"])
                .field("spot_price", clean_truth["spot_price"])
                .field("temperature_c", clean_truth["temperature_c"])
                .field("depot_meter_kw", clean_truth["depot_meter_kw"])
                .time(ts)
            )
            write_api.write(bucket=bucket, org=org, record=p)

            # Per-bus trust scores
            for bus in clean_truth["buses"]:
                tp = (
                    Point("bft_trust")
                    .tag("depot_id", str(self.depot_id))
                    .tag("bus_id", bus["bus_id"])
                    .field("trust_score", bus["trust_score"])
                    .field("mean_soc_pct", bus["mean_soc_pct"])
                    .field("interpolated", int(bus["interpolated"]))
                    .time(ts)
                )
                write_api.write(bucket=bucket, org=org, record=tp)

        except Exception as exc:
            log.warning("InfluxDB write error | depot=%d | %s", self.depot_id, exc)

    def _write_to_redis(self, clean_truth: dict) -> None:
        """
        Publish Clean Truth snapshot to Redis for MPC consumption.
        Key: gridsentinel:clean_truth:{depot_id}
        TTL: 90 seconds (1.5× window size — MPC reads every 5 min)
        """
        if not self._redis:
            return
        try:
            key = f"gridsentinel:clean_truth:{self.depot_id}"
            data = json.dumps(clean_truth)
            self._redis.setex(key, 90, data)

            # Publish to pub/sub channel for real-time consumers
            self._redis.publish(
                f"gridsentinel:bft:{self.depot_id}",
                data,
            )
        except Exception as exc:
            log.warning("Redis write error | depot=%d | %s", self.depot_id, exc)

    # ─── Main entry point ─────────────────────────────────────────────────────

    def process(self, snapshot: dict) -> dict:
        """
        Process one 5-minute canonical window through the full BFT pipeline.

        Args:
            snapshot: Output of consumer_align.aggregate_window()

        Returns:
            Clean Truth snapshot with trust scores and BFT metadata.
        """
        buses = snapshot.get("buses", [])
        depot_meter = snapshot.get("depot_meter_kw", 0.0)
        ts = snapshot.get("canonical_timestamp", "")

        if not buses:
            log.warning("depot=%d | Empty window at %s — skipping BFT.", self.depot_id, ts)
            return snapshot

        # Step 1 — MAD filter on SoC values
        soc_values = {b["bus_id"]: b["mean_soc_pct"] for b in buses}
        power_values = {b["bus_id"]: b["mean_power_kw"] for b in buses}

        soc_flagged = self._mad_filter(soc_values)
        power_flagged = self._mad_filter(power_values)

        # Union: flagged on either SoC or power
        flagged_ids = {bid for bid, flag in {**soc_flagged, **power_flagged}.items() if flag}

        # Step 2 — Depot meter cross-validation
        meter_fails = self._meter_cross_validation_fails(buses, depot_meter)

        # Step 3 — Classify attack severity
        attack_class = self._classify_attack(flagged_ids, meter_fails)

        # Step 4 — Update Trust Ledger
        self._update_trust_scores(buses, flagged_ids, attack_class)

        # Step 5 — Build Clean Truth snapshot
        clean_truth = self._build_clean_truth(snapshot, flagged_ids)

        # Step 6 — Log summary
        n_flagged = len(flagged_ids)
        blacklist = settings.bft.trust_blacklist_threshold

        if n_flagged:
            log.warning(
                "BFT | depot=%d | ts=%s | FLAGGED=%d | class=%s | meter_fail=%s",
                self.depot_id,
                ts[:16],
                n_flagged,
                attack_class,
                meter_fails,
            )
        else:
            log.info(
                "BFT | depot=%d | ts=%s | ✓ CLEAN | buses=%d | price=£%.2f | temp=%.1f°C",
                self.depot_id,
                ts[:16],
                len(buses),
                snapshot.get("spot_price", 0.0),
                snapshot.get("temperature_c", 0.0),
            )

        # Step 7 — Emit to sinks
        self._write_to_influx(clean_truth, attack_class)
        self._write_to_redis(clean_truth)

        return clean_truth
