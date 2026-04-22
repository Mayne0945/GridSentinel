"""
GridSentinel — BFT Gatekeeper
================================
Implements the Byzantine Fault Detection layer.

Called once per 5-minute canonical window per depot.
Receives an aggregated snapshot from consumer_align.py and:

  1. Checks price_metadata.confidence — tags MPC mode before any filtering
  2. Runs the MAD consensus filter on SoC and power readings
  3. Cross-validates charger cluster against depot meter
  4. Applies contextual Trust Ledger decay (three tiers from config)
  5. Blacklists sensors below trust threshold
  6. Emits a Clean Truth snapshot (Byzantine sensors replaced with
     weighted interpolation from trusted neighbours)
  7. Writes trust scores and Clean Truth to InfluxDB + Redis

The Math — MAD Filter:
  For each sensor cluster (all bus SoC readings in one depot):

    MAD = median(|x_i - median(X)|)

  A sensor x_i is flagged Byzantine if:

    |x_i - median(X)| / (1.4826 × MAD) > k

  where k = mad_threshold_k from config (default 3).

  The 1.4826 consistency constant makes MAD equivalent to sigma for
  Gaussian-distributed data, so k=3 means "3 standard deviations".

Contextual Trust Ledger decay (v2.0):
  Three decay tiers distinguish transient noise from coordinated attacks:
    - minor:       Isolated spike. MAD flag only. Likely sensor glitch.
    - standard:    MAD flag + fails depot meter cross-validation.
    - coordinated: 3+ buses flagged in the same window simultaneously.

  Recovery:
    - +0.01 per clean window (slow, steady)
    - +0.05 burst bonus after 5 consecutive clean windows

MPC Mode tagging (v2.1):
  The Clean Truth snapshot now carries an mpc_mode field:
    - "normal":   price_confidence >= 0.5 — full arbitrage allowed
    - "safety":   price_confidence == 0.0 — departure-critical charging only
  The MPC reads this field and restricts its objective accordingly.
"""

from __future__ import annotations

# Path fix — allows both `python bft/gatekeeper.py` and `python -m bft.gatekeeper`
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import json
import logging
import os
from collections import defaultdict
from typing import Any

import numpy as np

from config.settings import settings

log = logging.getLogger(__name__)

# MAD consistency constant — makes MAD equivalent to sigma for Gaussian data
MAD_CONSISTENCY = 1.4826


class BFTGatekeeper:
    """
    Per-depot BFT filter. One instance per align-bft-depot-N container.

    State:
      trust_scores:  bus_id -> float (0.0-1.0, starts at 1.0)
      clean_streak:  bus_id -> int (consecutive clean windows)
    """

    def __init__(self, depot_id: int) -> None:
        self.depot_id = depot_id
        self.cfg = settings.bft
        self.market_cfg = settings.market
        self.trust_scores: dict[str, float] = defaultdict(lambda: 1.0)
        self.clean_streak: dict[str, int] = defaultdict(int)
        self._redis = self._connect_redis()
        self._influx = self._connect_influx()

        log.info(
            "BFT Gatekeeper ready | depot=%d | k=%.1f | "
            "decay=[minor=%.2f std=%.2f coord=%.2f] | blacklist=%.2f | "
            "emergency_price=%.4f EUR/kWh | max_stale=%ds",
            depot_id,
            self.cfg.mad_threshold_k,
            self.cfg.trust_decay_minor,
            self.cfg.trust_decay_standard,
            self.cfg.trust_decay_coordinated,
            self.cfg.trust_blacklist_threshold,
            self.market_cfg.emergency_base_price,
            self.market_cfg.max_stale_window_s,
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
        url = os.environ.get("INFLUXDB_URL", "https://influxddb:8086")
        if url == "disabled":
            return None
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

    # ─── Price confidence check ───────────────────────────────────────────────

    def _resolve_mpc_mode(self, snapshot: dict) -> str:
        """
        Determine MPC operating mode from price_metadata.confidence.

        Returns:
          "normal"  — confidence >= 0.5 — live or fresh LKG price
                      Full arbitrage optimisation allowed.
          "safety"  — confidence == 0.0 — stale or missing price data
                      MPC must restrict to departure-critical charging only.
                      Emergency price is already active in the snapshot.

        This is the firewall between a data outage and a grid event.
        The BFT layer is the last component that sees price_metadata
        before the clean truth reaches the MPC — so it is the right
        place to enforce this gate.
        """
        confidence = snapshot.get("price_metadata", {}).get("confidence", 1.0)
        source = snapshot.get("price_metadata", {}).get("source", "live")

        if confidence == 0.0:
            log.warning(
                "BFT | depot=%d | price_confidence=0.0 (source=%s) | "
                "MPC mode -> SAFETY (departure-critical charging only)",
                self.depot_id,
                source,
            )
            return "safety"

        if confidence < 1.0:
            log.info(
                "BFT | depot=%d | price_confidence=%.1f (source=%s) | "
                "MPC mode -> NORMAL (LKG price — arbitrage allowed with caution)",
                self.depot_id,
                confidence,
                source,
            )

        return "normal"

    # ─── MAD Filter ───────────────────────────────────────────────────────────

    def _mad_filter(self, values: dict[str, float]) -> dict[str, bool]:
        """
        Apply MAD consensus filter to {bus_id: value}.
        Returns {bus_id: is_flagged}.
        Skips clusters smaller than 3 sensors — MAD is unreliable on tiny sets.
        """
        flagged: dict[str, bool] = {bid: False for bid in values}
        readings = np.array(list(values.values()), dtype=float)

        if len(readings) < 3:
            return flagged

        med = np.median(readings)
        mad = np.median(np.abs(readings - med))

        if mad < 1e-6:
            if len(readings) > 5:
                log.debug(
                    "depot=%d MAD=0 — possible flatline attack on %d sensors",
                    self.depot_id,
                    len(readings),
                )
            return flagged

        scaled_mad = MAD_CONSISTENCY * mad
        for bus_id, val in zip(values.keys(), values.values()):
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
        >20% divergence = cluster is lying.
        The depot meter is the ground-truth anchor.
        """
        if abs(depot_meter_kw) < 5.0:
            return False

        reported_total = sum(b.get("mean_power_kw", 0.0) for b in buses)
        if abs(depot_meter_kw) < 1e-6:
            return False

        divergence = abs(reported_total - depot_meter_kw) / abs(depot_meter_kw)
        if divergence > 0.20:
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

    def _classify_attack(self, flagged_bus_ids: set[str], meter_fails: bool) -> str:
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
                    "Trust decay | depot=%d | bus=%s | class=%s | decay=%.2f | score=%.3f",
                    self.depot_id,
                    bus_id,
                    attack_class,
                    decay,
                    self.trust_scores[bus_id],
                )
            else:
                self.clean_streak[bus_id] += 1
                recovery = cfg.trust_recovery_per_window
                if self.clean_streak[bus_id] >= 5 and self.trust_scores[bus_id] < 1.0:
                    recovery += cfg.trust_recovery_burst
                    self.clean_streak[bus_id] = 0
                self.trust_scores[bus_id] = min(1.0, self.trust_scores[bus_id] + recovery)

    # ─── Clean Truth emission ─────────────────────────────────────────────────

    def _build_clean_truth(
        self,
        snapshot: dict,
        flagged_ids: set[str],
        mpc_mode: str,
    ) -> dict:
        """
        Build Clean Truth snapshot.
        Byzantine bus readings replaced with weighted interpolation from trusted neighbours.
        Blacklisted buses completely excluded and interpolated.
        mpc_mode injected here — the MPC reads this field directly.
        """
        buses = snapshot["buses"]
        blacklisted = {
            bid
            for bid, score in self.trust_scores.items()
            if score < self.cfg.trust_blacklist_threshold
        }

        trusted_buses = [b for b in buses if b["bus_id"] not in blacklisted]
        if not trusted_buses:
            log.warning("depot=%d — all buses blacklisted! Failing open.", self.depot_id)
            trusted_buses = buses

        trusted_mean_soc = sum(b["mean_soc_pct"] for b in trusted_buses) / len(trusted_buses)
        trusted_mean_power = sum(b["mean_power_kw"] for b in trusted_buses) / len(trusted_buses)

        clean_buses = []
        for bus in buses:
            bid = bus["bus_id"]
            if bid in blacklisted:
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
            "mpc_mode": mpc_mode,  # "normal" | "safety"
        }

    # ─── Output sinks ─────────────────────────────────────────────────────────

    def _write_to_influx(self, clean_truth: dict, attack_class: str) -> None:
        if not self._influx:
            return
        try:
            from influxdb_client import Point
            from influxdb_client.client.write_api import SYNCHRONOUS

            write_api = self._influx.write_api(write_options=SYNCHRONOUS)
            bucket = os.environ.get("INFLUXDB_BUCKET", "telemetry")
            org = os.environ.get("INFLUXDB_ORG", "gridsentinel")
            ts = clean_truth["canonical_timestamp"]

            p = (
                Point("bft_window")
                .tag("depot_id", str(self.depot_id))
                .tag("attack_class", attack_class)
                .tag("mpc_mode", clean_truth.get("mpc_mode", "normal"))
                .field("flagged_count", len(clean_truth.get("flagged_bus_ids", [])))
                .field("blacklisted_count", len(clean_truth.get("blacklisted_ids", [])))
                .field("clean_bus_count", clean_truth.get("clean_bus_count", 0))
                .field("spot_price", clean_truth.get("spot_price", 0.0))
                .field(
                    "price_confidence", clean_truth.get("price_metadata", {}).get("confidence", 1.0)
                )
                .field("temperature_c", clean_truth.get("temperature_c", 21.0))
                .field("depot_meter_kw", clean_truth.get("depot_meter_kw", 0.0))
                .time(ts)
            )
            write_api.write(bucket=bucket, org=org, record=p)

            for bus in clean_truth["buses"]:
                tp = (
                    Point("bft_trust")
                    .tag("depot_id", str(self.depot_id))
                    .tag("bus_id", bus["bus_id"])
                    .field("trust_score", bus["trust_score"])
                    .field("mean_soc_pct", bus["mean_soc_pct"])
                    .field("interpolated", int(bus.get("interpolated", False)))
                    .time(ts)
                )
                write_api.write(bucket=bucket, org=org, record=tp)

        except Exception as exc:
            log.warning("InfluxDB write error | depot=%d | %s", self.depot_id, exc)

    def _write_to_redis(self, clean_truth: dict) -> None:
        """
        Publish Clean Truth to Redis for MPC consumption.
        Key: gridsentinel:clean_truth:{depot_id}
        TTL: 90s (1.5x window — MPC reads every 5 min)
        """
        if not self._redis:
            return
        try:
            key = f"gridsentinel:clean_truth:{self.depot_id}"
            data = json.dumps(clean_truth)
            self._redis.setex(key, 90, data)
            self._redis.publish(f"gridsentinel:bft:{self.depot_id}", data)
        except Exception as exc:
            log.warning("Redis write error | depot=%d | %s", self.depot_id, exc)

    # ─── Main entry point ─────────────────────────────────────────────────────

    def process(self, snapshot: dict) -> dict:
        """
        Process one 5-minute canonical window through the full BFT pipeline.

        Args:
            snapshot: Output of consumer_align.aggregate_window()

        Returns:
            Clean Truth snapshot with trust scores, BFT metadata, and mpc_mode.
        """
        buses = snapshot.get("buses", [])
        depot_meter = snapshot.get("depot_meter_kw", 0.0)
        ts = snapshot.get("canonical_timestamp", "")

        if not buses:
            log.warning("depot=%d | Empty window at %s — skipping BFT.", self.depot_id, ts)
            return snapshot

        # Step 1 — Resolve MPC mode from price confidence BEFORE filtering
        # This runs first so the mode is set regardless of sensor state
        mpc_mode = self._resolve_mpc_mode(snapshot)

        # Step 2 — MAD filter on SoC and power
        soc_values = {b["bus_id"]: b["mean_soc_pct"] for b in buses}
        power_values = {b["bus_id"]: b["mean_power_kw"] for b in buses}
        soc_flagged = self._mad_filter(soc_values)
        power_flagged = self._mad_filter(power_values)
        flagged_ids = {bid for bid, flag in {**soc_flagged, **power_flagged}.items() if flag}

        # Step 3 — Depot meter cross-validation
        meter_fails = self._meter_cross_validation_fails(buses, depot_meter)

        # Step 4 — Classify attack severity
        attack_class = self._classify_attack(flagged_ids, meter_fails)

        # Step 5 — Update Trust Ledger
        self._update_trust_scores(buses, flagged_ids, attack_class)

        # Step 6 — Build Clean Truth with mpc_mode injected
        clean_truth = self._build_clean_truth(snapshot, flagged_ids, mpc_mode)

        # Step 7 — Log summary
        if flagged_ids:
            log.warning(
                "BFT | depot=%d | ts=%s | FLAGGED=%d | class=%s | " "meter_fail=%s | mpc_mode=%s",
                self.depot_id,
                ts[:16],
                len(flagged_ids),
                attack_class,
                meter_fails,
                mpc_mode,
            )
        else:
            log.info(
                "BFT | depot=%d | ts=%s | CLEAN | buses=%d | "
                "price=%.2f EUR/MWh | confidence=%.1f | mpc_mode=%s",
                self.depot_id,
                ts[:16],
                len(buses),
                snapshot.get("spot_price", 0.0),
                snapshot.get("price_metadata", {}).get("confidence", 1.0),
                mpc_mode,
            )

        # Step 8 — Emit to sinks
        self._write_to_influx(clean_truth, attack_class)
        self._write_to_redis(clean_truth)

        return clean_truth


if __name__ == "__main__":
    import argparse
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="GridSentinel BFT Gatekeeper")
    parser.add_argument(
        "--snapshot",
        default="data/aligned/latest_snapshot.json",
        help="Path to aligned snapshot JSON",
    )
    parser.add_argument("--depot-id", type=int, default=1)
    parser.add_argument("--watch", action="store_true", help="Poll continuously every 5 seconds")
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    gatekeeper = BFTGatekeeper(depot_id=args.depot_id)

    def run_once() -> None:
        if not snapshot_path.exists():
            log.warning("Snapshot not found at %s — generating synthetic snapshot", snapshot_path)
            # Build a synthetic snapshot from latest dispatch so dashboard has data
            dispatch_path = Path("data/dispatch/latest_dispatch.json")
            if not dispatch_path.exists():
                log.error("No dispatch found either — run mpc/dispatch.py first")
                return

            dispatch = json.loads(dispatch_path.read_text())
            commands = dispatch.get("commands", [])

            buses = [
                {
                    "bus_id": cmd["bus_id"],
                    "mean_soc_pct": float(np.random.uniform(30, 90)),
                    "mean_power_kw": cmd.get("power_kw", 0.0),
                    "status": cmd.get("action", "hold"),
                }
                for cmd in commands
            ]

            snapshot = {
                "canonical_timestamp": dispatch.get("timestamp", ""),
                "spot_price": 50.0,
                "price_metadata": {"confidence": 1.0},
                "depot_meter_kw": sum(b["mean_power_kw"] for b in buses),
                "temperature_c": 18.0,
                "solar_irradiance_wm2": 250.0,
                "buses": buses,
            }
        else:
            snapshot = json.loads(snapshot_path.read_text())

        clean_truth = gatekeeper.process(snapshot)

        # Write trust ledger for dashboard
        trust_path = Path("data/bft/trust_ledger.json")
        trust_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = {
            "timestamp": clean_truth.get("canonical_timestamp", ""),
            "buses": dict(gatekeeper.trust_scores),
            "flagged": list(clean_truth.get("flagged_ids", [])),
        }
        trust_path.write_text(json.dumps(ledger, indent=2))
        log.info("Trust ledger written → %s | flagged=%d", trust_path, len(ledger["flagged"]))

    if args.watch:
        log.info("BFT Gatekeeper watching — polling every 5s")
        while True:
            run_once()
            time.sleep(5)
    else:
        run_once()
