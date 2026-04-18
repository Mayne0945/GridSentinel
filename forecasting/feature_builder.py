"""
forecasting/feature_builder.py
--------------------------------
Transforms the Clean Truth stream into a numerical feature matrix for XGBoost.

Input:  Redis pub/sub channel  gridsentinel:bft:{depot_id}
        Clean Truth JSON published by bft.gatekeeper every 5 minutes.

Output: Redis key             gridsentinel:features:{depot_id}  (90s TTL)
        Redis pub/sub channel gridsentinel:features:{depot_id}
        FeatureVector dataclass (for testing and in-process use)

Feature set (all strictly numerical — no categoricals, no nulls):
    Temporal (cyclical encoding):
        sin_hour, cos_hour       — 24h cycle
        sin_dow, cos_dow         — 7-day cycle
        sin_month, cos_month     — 12-month cycle

    Price (from ENTSO-E via Clean Truth):
        spot_price_eur_mwh       — current window price
        price_lag_1              — t-1 window (5 min ago)
        price_lag_3              — t-3 (15 min ago)
        price_lag_6              — t-6 (30 min ago)
        price_lag_12             — t-12 (1 hour ago)
        price_lag_288            — t-288 (24 hours ago)
        price_rolling_mean_6h    — 72-window rolling mean
        price_rolling_std_6h     — 72-window rolling std
        price_rolling_mean_24h   — 288-window rolling mean
        price_rolling_std_24h    — 288-window rolling std

    Weather (from Open-Meteo via Clean Truth):
        temperature_c            — current ambient temperature
        solar_irradiance_wm2     — current solar irradiance
        wind_speed_kmh           — current wind speed

    Fleet (aggregated from Clean Truth bus states):
        fleet_mean_soc_pct       — mean SoC across clean buses in depot
        fleet_min_soc_pct        — min SoC (tightest constraint)
        fleet_available_kw       — estimated dispatchable discharge capacity
        clean_bus_count          — number of buses passing BFT this window
        byzantine_bus_count      — number flagged (attack signal)

    Grid proxy:
        is_peak_hour             — 1 if hour in {7,8,9,17,18,19,20}, else 0
        is_weekend               — 1 if Saturday or Sunday

Rolling windows are maintained in-memory using a deque — O(1) append,
O(N) mean/std where N <= 288. No database reads required per window.

Design decision: one FeatureBuilder instance per depot, running in a
dedicated process (one per align-bft-depotN container). At 5 depots
this is 5 independent processes — no shared state, no coordination.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import redis

from config.settings import get_config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FeatureVector:
    """
    One 5-minute feature snapshot for a single depot.
    All fields are float — XGBoost requires no non-numeric input.
    """

    # Identity (not fed to model — used for labelling and joining)
    canonical_timestamp: str
    depot_id: int
    feature_version: str = "1.0"

    # --- Temporal ---
    sin_hour: float = 0.0
    cos_hour: float = 0.0
    sin_dow: float = 0.0  # day of week
    cos_dow: float = 0.0
    sin_month: float = 0.0
    cos_month: float = 0.0

    # --- Price ---
    spot_price_eur_mwh: float = 0.0
    price_lag_1: float = 0.0  # 5 min ago
    price_lag_3: float = 0.0  # 15 min ago
    price_lag_6: float = 0.0  # 30 min ago
    price_lag_12: float = 0.0  # 1 hour ago
    price_lag_288: float = 0.0  # 24 hours ago
    price_rolling_mean_6h: float = 0.0
    price_rolling_std_6h: float = 0.0
    price_rolling_mean_24h: float = 0.0
    price_rolling_std_24h: float = 0.0

    # --- Weather ---
    temperature_c: float = 15.0
    solar_irradiance_wm2: float = 0.0
    wind_speed_kmh: float = 0.0

    # --- Fleet ---
    fleet_mean_soc_pct: float = 50.0
    fleet_min_soc_pct: float = 20.0
    fleet_available_kw: float = 0.0
    clean_bus_count: int = 0
    byzantine_bus_count: int = 0

    # --- Grid proxy ---
    is_peak_hour: float = 0.0
    is_weekend: float = 0.0

    def to_model_input(self) -> dict[str, float]:
        """
        Return only the numerical features the model trains on.
        Strips identity fields (timestamp, depot_id, version).
        """
        excluded = {"canonical_timestamp", "depot_id", "feature_version"}
        return {k: float(v) for k, v in asdict(self).items() if k not in excluded}


# ---------------------------------------------------------------------------
# Rolling window — O(1) append, no Pandas dependency at inference time
# ---------------------------------------------------------------------------


class RollingWindow:
    """
    Fixed-size deque with mean and std computed on demand.

    Why not Pandas? The feature builder runs in the ingestion container,
    not in the forecasting container. Keeping it NumPy-free makes the
    ingestion layer lightweight. Pandas is only needed in train.py.
    """

    def __init__(self, maxlen: int) -> None:
        self._q: deque[float] = deque(maxlen=maxlen)

    def append(self, value: float) -> None:
        self._q.append(value)

    def mean(self) -> float:
        if not self._q:
            return 0.0
        return sum(self._q) / len(self._q)

    def std(self) -> float:
        if len(self._q) < 2:
            return 0.0
        m = self.mean()
        variance = sum((x - m) ** 2 for x in self._q) / (len(self._q) - 1)
        return math.sqrt(variance)

    def __len__(self) -> int:
        return len(self._q)

    def get(self, lag: int) -> float:
        """
        Return the value `lag` steps back from the most recent.
        lag=1 → most recent, lag=2 → one before that, etc.
        Returns 0.0 if window not yet full enough.
        """
        if lag > len(self._q):
            return 0.0
        return self._q[-lag]


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------


class FeatureBuilder:
    """
    Subscribes to one depot's Clean Truth channel and emits feature vectors.

    Lifecycle:
        builder = FeatureBuilder(depot_id=0)
        builder.run()   # blocking — subscribe loop
    """

    # 5-min windows per period
    WINDOWS_PER_6H = 72
    WINDOWS_PER_24H = 288

    # Peak hours for GB electricity market (BST-aligned)
    PEAK_HOURS = {7, 8, 9, 17, 18, 19, 20}

    def __init__(self, depot_id: int) -> None:
        self.depot_id = depot_id
        self.cfg = get_config()

        self._redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        self._r: redis.Redis | None = None

        # Rolling price windows
        self._price_6h = RollingWindow(maxlen=self.WINDOWS_PER_6H)
        self._price_24h = RollingWindow(maxlen=self.WINDOWS_PER_24H)

        # Full 24h price history for lag-288 lookup (kept separate for clarity)
        self._price_history = RollingWindow(maxlen=self.WINDOWS_PER_24H + 1)

        self._windows_processed = 0

    def _connect(self) -> None:
        self._r = redis.from_url(self._redis_url, decode_responses=True)
        self._r.ping()
        log.info("Depot %d: Redis connected at %s", self.depot_id, self._redis_url)

    # ------------------------------------------------------------------
    # Temporal encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_time(ts: datetime) -> dict[str, float]:
        """
        Cyclical encoding of hour, day-of-week, and month.

        Why cyclical? A linear encoding treats hour=23 and hour=0 as
        maximally distant (23 units apart). sin/cos encoding makes them
        adjacent (both near the same point on the circle). XGBoost splits
        on feature values — cyclical encoding lets it learn midnight
        patterns correctly without needing to see both sides of the day.

        Encoding:
            sin(2π * value / period), cos(2π * value / period)

        hour  ∈ [0, 23]  → period = 24
        dow   ∈ [0, 6]   → period = 7   (0=Monday per isoweekday)
        month ∈ [1, 12]  → period = 12
        """
        hour_rad = 2 * math.pi * ts.hour / 24
        dow_rad = 2 * math.pi * ts.weekday() / 7
        month_rad = 2 * math.pi * (ts.month - 1) / 12

        return {
            "sin_hour": math.sin(hour_rad),
            "cos_hour": math.cos(hour_rad),
            "sin_dow": math.sin(dow_rad),
            "cos_dow": math.cos(dow_rad),
            "sin_month": math.sin(month_rad),
            "cos_month": math.cos(month_rad),
        }

    # ------------------------------------------------------------------
    # Fleet feature extraction
    # ------------------------------------------------------------------

    def _extract_fleet_features(
        self,
        buses: list[dict[str, Any]],
    ) -> dict[str, float]:
        """
        Aggregate per-bus Clean Truth states into depot-level fleet features.

        Only buses that passed BFT (is_byzantine=False) contribute to the
        fleet aggregates. Byzantine buses are counted separately as a signal.
        """
        clean = [b for b in buses if not b.get("is_byzantine", False)]
        flagged = [b for b in buses if b.get("is_byzantine", False)]

        if not clean:
            log.warning(
                "Depot %d: no clean buses in window — using safe defaults",
                self.depot_id,
            )
            return {
                "fleet_mean_soc_pct": 50.0,
                "fleet_min_soc_pct": 20.0,
                "fleet_available_kw": 0.0,
                "clean_bus_count": 0,
                "byzantine_bus_count": len(flagged),
            }

        socs = [b.get("soc_pct", 50.0) for b in clean]
        mean_soc = sum(socs) / len(socs)
        min_soc = min(socs)

        # Dispatchable discharge capacity:
        # Buses with SoC > 20% can discharge down to 20% floor
        # Available kWh = (soc - 0.20) * capacity per bus
        # Convert to kW assuming 1-hour dispatch window
        battery_kwh = self.cfg.fleet.battery_capacity_kwh
        max_dis_kw = self.cfg.fleet.max_charge_rate_kw

        available_kw = sum(
            min(
                (b.get("soc_pct", 0) / 100 - 0.20) * battery_kwh,
                max_dis_kw,
            )
            for b in clean
            if b.get("soc_pct", 0) > 20
        )

        return {
            "fleet_mean_soc_pct": round(mean_soc, 2),
            "fleet_min_soc_pct": round(min_soc, 2),
            "fleet_available_kw": round(max(available_kw, 0.0), 1),
            "clean_bus_count": len(clean),
            "byzantine_bus_count": len(flagged),
        }

    # ------------------------------------------------------------------
    # Price feature extraction
    # ------------------------------------------------------------------

    def _update_price_windows(self, price: float) -> dict[str, float]:
        """
        Append new price to rolling windows, return lag and rolling features.
        Call this AFTER reading current lags, BEFORE updating for next window.
        """
        # Read lags from history BEFORE appending current price
        lag_1 = self._price_history.get(1)  # previous window
        lag_3 = self._price_history.get(3)
        lag_6 = self._price_history.get(6)
        lag_12 = self._price_history.get(12)
        lag_288 = self._price_history.get(288)  # 24h ago (0.0 if < 24h data)

        # Now append current price
        self._price_history.append(price)
        self._price_6h.append(price)
        self._price_24h.append(price)

        return {
            "price_lag_1": lag_1,
            "price_lag_3": lag_3,
            "price_lag_6": lag_6,
            "price_lag_12": lag_12,
            "price_lag_288": lag_288,
            "price_rolling_mean_6h": round(self._price_6h.mean(), 4),
            "price_rolling_std_6h": round(self._price_6h.std(), 4),
            "price_rolling_mean_24h": round(self._price_24h.mean(), 4),
            "price_rolling_std_24h": round(self._price_24h.std(), 4),
        }

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process(self, clean_truth: dict[str, Any]) -> FeatureVector:
        """
        Transform one Clean Truth window into a FeatureVector.

        Args:
            clean_truth: JSON payload from bft.gatekeeper via Redis pub/sub.

        Returns:
            FeatureVector with all features populated.
        """
        ts_str = clean_truth.get("canonical_timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str).replace(tzinfo=UTC)
        except (ValueError, TypeError):
            ts = datetime.now(tz=UTC)
            log.warning("Depot %d: could not parse timestamp '%s'", self.depot_id, ts_str)

        # Temporal features
        time_feats = self._encode_time(ts)

        # Grid-proxy features
        is_peak = float(ts.hour in self.PEAK_HOURS)
        is_weekend = float(ts.weekday() >= 5)

        # Price features
        price = float(clean_truth.get("spot_price_eur_mwh", 0.0))
        price_feats = self._update_price_windows(price)

        # Weather features
        weather = clean_truth.get("weather", {})
        temp = float(weather.get("temperature_c", 15.0))
        solar = float(weather.get("solar_irradiance_wm2", 0.0))
        wind = float(weather.get("wind_speed_kmh", 0.0))

        # Fleet features
        buses = clean_truth.get("buses", [])
        fleet_feats = self._extract_fleet_features(buses)

        vector = FeatureVector(
            canonical_timestamp=ts_str,
            depot_id=self.depot_id,
            # temporal
            sin_hour=time_feats["sin_hour"],
            cos_hour=time_feats["cos_hour"],
            sin_dow=time_feats["sin_dow"],
            cos_dow=time_feats["cos_dow"],
            sin_month=time_feats["sin_month"],
            cos_month=time_feats["cos_month"],
            # price
            spot_price_eur_mwh=price,
            price_lag_1=price_feats["price_lag_1"],
            price_lag_3=price_feats["price_lag_3"],
            price_lag_6=price_feats["price_lag_6"],
            price_lag_12=price_feats["price_lag_12"],
            price_lag_288=price_feats["price_lag_288"],
            price_rolling_mean_6h=price_feats["price_rolling_mean_6h"],
            price_rolling_std_6h=price_feats["price_rolling_std_6h"],
            price_rolling_mean_24h=price_feats["price_rolling_mean_24h"],
            price_rolling_std_24h=price_feats["price_rolling_std_24h"],
            # weather
            temperature_c=temp,
            solar_irradiance_wm2=solar,
            wind_speed_kmh=wind,
            # fleet
            fleet_mean_soc_pct=fleet_feats["fleet_mean_soc_pct"],
            fleet_min_soc_pct=fleet_feats["fleet_min_soc_pct"],
            fleet_available_kw=fleet_feats["fleet_available_kw"],
            clean_bus_count=fleet_feats["clean_bus_count"],
            byzantine_bus_count=fleet_feats["byzantine_bus_count"],
            # grid proxy
            is_peak_hour=is_peak,
            is_weekend=is_weekend,
        )

        self._windows_processed += 1
        log.debug(
            "Depot %d | window=%s | price=%.2f | soc=%.1f%% | clean=%d | byz=%d",
            self.depot_id,
            ts_str[:16],
            price,
            fleet_feats["fleet_mean_soc_pct"],
            fleet_feats["clean_bus_count"],
            fleet_feats["byzantine_bus_count"],
        )

        return vector

    def _emit(self, vector: FeatureVector) -> None:
        """Write feature vector to Redis — TTL key + pub/sub publish."""
        if self._r is None:
            return

        payload = json.dumps(asdict(vector))
        key = f"gridsentinel:features:{self.depot_id}"

        self._r.setex(key, 90, payload)
        self._r.publish(key, payload)

        log.info(
            "Depot %d | features emitted | window=%s | "
            "price=%.2f EUR/MWh | temp=%.1fC | soc_mean=%.1f%% | "
            "clean=%d | byz=%d | windows_total=%d",
            self.depot_id,
            vector.canonical_timestamp[:16],
            vector.spot_price_eur_mwh,
            vector.temperature_c,
            vector.fleet_mean_soc_pct,
            vector.clean_bus_count,
            vector.byzantine_bus_count,
            self._windows_processed,
        )

    def run(self) -> None:
        """
        Blocking subscribe loop.
        Subscribes to gridsentinel:bft:{depot_id}, processes each window,
        emits features to gridsentinel:features:{depot_id}.
        """
        self._connect()

        in_channel = f"gridsentinel:bft:{self.depot_id}"
        out_channel = f"gridsentinel:features:{self.depot_id}"

        pubsub = self._r.pubsub()
        pubsub.subscribe(in_channel)

        log.info(
            "Depot %d | FeatureBuilder subscribed | in=%s out=%s",
            self.depot_id,
            in_channel,
            out_channel,
        )

        for message in pubsub.listen():
            if message["type"] != "message":
                continue

            try:
                clean_truth = json.loads(message["data"])
            except json.JSONDecodeError as e:
                log.warning("Depot %d: malformed Clean Truth JSON — %s", self.depot_id, e)
                continue

            try:
                vector = self.process(clean_truth)
                self._emit(vector)
            except Exception as e:
                log.error(
                    "Depot %d: feature extraction failed — %s",
                    self.depot_id,
                    e,
                    exc_info=True,
                )
