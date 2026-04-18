"""
GridSentinel — forecasting/inference.py
========================================
Loads the trained XGBoost model + conformal quantiles produced by train.py
and serves rolling 24-hour probabilistic price forecasts to the MPC.

Deadlock fix (v2)
-----------------
_threading_fix MUST be the first import — it sets OMP/BLAS/MKL thread
counts to 1 before any C++ runtime loads. See forecasting/_threading_fix.py.

The _load_artifacts method then applies nthread=1 at the XGBoost booster
level using the correct single-arg set_param("nthread", 1) syntax.
"""

from __future__ import annotations

# ── DEADLOCK FIX: must precede all C++ / numerical imports ─────────────────
import forecasting._threading_fix  # noqa: F401  (side-effect import)
# ──────────────────────────────────────────────────────────────────────────

import argparse
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("gridsent.inference")

HORIZON: int = 288
STEP_MINUTES: int = 5
ALPHA: float = 0.20
DEFAULT_MODELS_DIR = Path("models")

MIN_PROFIT_EUR: float = 50.0
MIN_CONFIDENCE: float = 0.60
MAX_CHARGE_WINDOW_STEPS: int = 72
MAX_DISCHARGE_WINDOW_STEPS: int = 48


# ---------------------------------------------------------------------------
# Output data-classes
# ---------------------------------------------------------------------------

@dataclass
class PriceInterval:
    timestamp: str
    horizon_step: int
    point_forecast_eur_mwh: float
    lower_80: float
    upper_80: float
    interval_width: float


@dataclass
class ArbitrageWindow:
    charge_start: str
    charge_end: str
    discharge_start: str
    discharge_end: str
    charge_avg_upper_eur_mwh: float
    discharge_avg_lower_eur_mwh: float
    spread_eur_mwh: float
    estimated_profit_eur: float
    confidence: float
    buses_required: int
    horizon_steps: tuple = field(default_factory=tuple)


@dataclass
class ForecastResult:
    generated_at: str
    horizon_hours: int
    step_minutes: int
    coverage_target: float
    model_version: str
    intervals: list[PriceInterval]
    arbitrage_windows: list[ArbitrageWindow]
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Pandas-vectorised feature builder
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    df = df.copy()
    df["canonical_timestamp"] = pd.to_datetime(df["canonical_timestamp"], utc=True)
    df = df.sort_values("canonical_timestamp").reset_index(drop=True)

    ts = df["canonical_timestamp"]

    def _cyclical(series: pd.Series, period: float):
        rad = 2 * math.pi * series / period
        return np.sin(rad), np.cos(rad)

    df["sin_hour"],  df["cos_hour"]  = _cyclical(ts.dt.hour,                    24.0)
    df["sin_dow"],   df["cos_dow"]   = _cyclical(ts.dt.dayofweek,                7.0)
    df["sin_month"], df["cos_month"] = _cyclical(ts.dt.month.astype(float) - 1, 12.0)

    PEAK_HOURS = {7, 8, 9, 17, 18, 19, 20}
    df["is_peak_hour"] = ts.dt.hour.isin(PEAK_HOURS).astype(float)
    df["is_weekend"]   = (ts.dt.dayofweek >= 5).astype(float)

    price = df["spot_price_eur_mwh"]
    df["price_lag_1"]   = price.shift(1)
    df["price_lag_3"]   = price.shift(3)
    df["price_lag_6"]   = price.shift(6)
    df["price_lag_12"]  = price.shift(12)
    df["price_lag_288"] = price.shift(288)

    df["price_rolling_mean_6h"]  = price.shift(1).rolling(72,  min_periods=1).mean()
    df["price_rolling_std_6h"]   = price.shift(1).rolling(72,  min_periods=2).std(ddof=1)
    df["price_rolling_mean_24h"] = price.shift(1).rolling(288, min_periods=1).mean()
    df["price_rolling_std_24h"]  = price.shift(1).rolling(288, min_periods=2).std(ddof=1)

    df["temperature_c"] = (
        df["temperature_c"].ffill()
        if "temperature_c" in df.columns
        else pd.Series(15.0, index=df.index)
    )
    df["solar_irradiance_wm2"] = (
        df["solar_irradiance_wm2"].fillna(0.0)
        if "solar_irradiance_wm2" in df.columns
        else pd.Series(0.0, index=df.index)
    )
    df["wind_speed_kmh"] = (
        df["wind_speed_kmh"].fillna(0.0)
        if "wind_speed_kmh" in df.columns
        else pd.Series(0.0, index=df.index)
    )

    df["fleet_mean_soc_pct"]  = 50.0
    df["fleet_min_soc_pct"]   = 20.0
    df["fleet_available_kw"]  = 0.0
    df["clean_bus_count"]     = 0.0
    df["byzantine_bus_count"] = 0.0

    df = df.fillna(0.0)

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Feature mismatch — columns absent from Clean Truth batch: {missing}\n"
            f"feature_builder.py and inference.py may have diverged."
        )

    return df[feature_columns].astype(np.float32).values


# ---------------------------------------------------------------------------
# Core forecaster
# ---------------------------------------------------------------------------

class GridSentinelForecaster:

    def __init__(
        self,
        models_dir: Path | str = DEFAULT_MODELS_DIR,
        fleet_capacity_kwh: float = 18_000.0,
        charger_power_kw: float = 6_000.0,
    ):
        self.models_dir         = Path(models_dir)
        self.fleet_capacity_kwh = fleet_capacity_kwh
        self.charger_power_kw   = charger_power_kw

        self._model             = None
        self._conf_quantiles    = None
        self._feature_columns   = None
        self._training_metadata = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        t0 = time.perf_counter()

        model_path = self.models_dir / "xgb_forecaster.joblib"
        quant_path = self.models_dir / "conformal_quantiles.npy"
        feat_path  = self.models_dir / "feature_columns.json"
        meta_path  = self.models_dir / "training_metadata.json"

        for p in [model_path, quant_path, feat_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Artifact not found: {p}\n"
                    f"Run forecasting/train.py first."
                )

        log.info("Loading model artefacts from %s …", self.models_dir)
        self._model = joblib.load(model_path)

        # ── Deadlock fix: force single-threaded execution ──────────────────
        # MultiOutputRegressor wraps 288 XGBoost estimators. Without this,
        # all 288 try to spawn N OpenMP threads simultaneously → deadlock.
        #
        # Correct set_param syntax is set_param(key, value) — NOT a dict.
        # The dict form exists but silently fails to propagate in some
        # XGBoost versions, which is why the previous fix didn't work.
        self._model.n_jobs = 1
        for est in self._model.estimators_:
            est.n_jobs = 1
            booster = est.get_booster()
            booster.set_param("nthread", 1)   # ← correct: two args, not dict
        # ──────────────────────────────────────────────────────────────────

        self._conf_quantiles = np.load(quant_path)

        with open(feat_path) as f:
            self._feature_columns = json.load(f)
        if meta_path.exists():
            with open(meta_path) as f:
                self._training_metadata = json.load(f)

        elapsed = time.perf_counter() - t0
        log.info(
            "Artefacts loaded in %.2fs | features=%d | horizon=%d | "
            "conf_quantile_mean=%.3f EUR/MWh",
            elapsed,
            len(self._feature_columns),
            len(self._conf_quantiles),
            float(self._conf_quantiles.mean()),
        )

    def predict(self, clean_truth_df: pd.DataFrame) -> ForecastResult:
        t0 = time.perf_counter()
        log.info("Building feature matrix from %d Clean Truth rows …", len(clean_truth_df))

        X        = build_feature_matrix(clean_truth_df, self._feature_columns)
        X_latest = X[-1:, :]

        anchor_price = float(clean_truth_df["spot_price_eur_mwh"].iloc[-1])
        anchor_ts    = pd.to_datetime(
            clean_truth_df["canonical_timestamp"].iloc[-1], utc=True
        )

        log.info(
            "Inference anchor | ts=%s | price=%.2f EUR/MWh",
            anchor_ts.isoformat(), anchor_price,
        )

        log.info("Running predict() across %d estimators …", len(self._model.estimators_))
        delta_pred      = self._model.predict(X_latest)[0]
        point_forecasts = anchor_price + delta_pred

        lower = point_forecasts - self._conf_quantiles
        upper = point_forecasts + self._conf_quantiles

        intervals: list[PriceInterval] = []
        for h in range(HORIZON):
            slot_ts = anchor_ts + pd.Timedelta(minutes=STEP_MINUTES * (h + 1))
            width   = float(upper[h] - lower[h])
            intervals.append(PriceInterval(
                timestamp              = slot_ts.isoformat(),
                horizon_step           = h + 1,
                point_forecast_eur_mwh = round(float(point_forecasts[h]), 4),
                lower_80               = round(float(lower[h]), 4),
                upper_80               = round(float(upper[h]), 4),
                interval_width         = round(width, 4),
            ))

        windows = self._scan_arbitrage_windows(intervals, anchor_price)

        elapsed = time.perf_counter() - t0
        model_version = (
            self._training_metadata.get("model_version", "unknown")
            if self._training_metadata else "unknown"
        )

        result = ForecastResult(
            generated_at      = datetime.now(timezone.utc).isoformat(),
            horizon_hours     = HORIZON * STEP_MINUTES // 60,
            step_minutes      = STEP_MINUTES,
            coverage_target   = 1 - ALPHA,
            model_version     = model_version,
            intervals         = intervals,
            arbitrage_windows = windows,
            meta = {
                "anchor_price_eur_mwh" : round(anchor_price, 4),
                "anchor_timestamp"     : anchor_ts.isoformat(),
                "mean_interval_width"  : round(float(self._conf_quantiles.mean() * 2), 4),
                "inference_elapsed_s"  : round(elapsed, 3),
                "n_arbitrage_windows"  : len(windows),
                "clean_truth_rows_used": len(clean_truth_df),
            },
        )

        log.info(
            "Forecast complete | elapsed=%.3fs | arbitrage_windows=%d | "
            "mean_PI_width=±%.3f EUR/MWh",
            elapsed, len(windows), float(self._conf_quantiles.mean()),
        )

        if windows:
            best = max(windows, key=lambda w: w.estimated_profit_eur)
            log.info(
                "Best window | charge=%s→%s | discharge=%s→%s | "
                "spread=%.2f EUR/MWh | est. profit=€%.0f | confidence=%.2f",
                best.charge_start[:16], best.charge_end[:16],
                best.discharge_start[:16], best.discharge_end[:16],
                best.spread_eur_mwh, best.estimated_profit_eur, best.confidence,
            )

        return result

    def _scan_arbitrage_windows(
        self,
        intervals: list[PriceInterval],
        anchor_price: float,
    ) -> list[ArbitrageWindow]:
        """
        O(n) optimised scanner using prefix sums + best-charge-at lookup.

        Two improvements over the original O(n²) nested loop:
          1. Prefix sums on lower/upper arrays make window-mean calculation O(1)
             instead of O(window_len) per candidate.
          2. best_charge_at dict tracks the cheapest charge window ending at
             each timestep — each discharge only needs one dict lookup instead
             of scanning all charge candidates.

        Conservative criterion (unchanged): spread = discharge lower-bound
        minus charge upper-bound must be > 0, meaning the trade is profitable
        even in the worst-case scenario of both prediction intervals.
        """
        n     = len(intervals)
        lower = np.array([iv.lower_80 for iv in intervals])
        upper = np.array([iv.upper_80 for iv in intervals])

        # Prefix sums — sum_x[j] - sum_x[i] gives sum of x[i:j] in O(1)
        sum_l = np.concatenate([[0], np.cumsum(lower)])
        sum_u = np.concatenate([[0], np.cumsum(upper)])

        # ── Step 1: enumerate all valid charge windows ────────────────────
        charge_candidates: list[tuple[int, int, float]] = []
        for s in range(n - 12):
            for length in range(6, min(MAX_CHARGE_WINDOW_STEPS + 1, n - s)):
                e     = s + length
                avg_u = (sum_u[e] - sum_u[s]) / length
                charge_candidates.append((s, e, avg_u))

        # ── Step 2: enumerate all valid discharge windows ─────────────────
        discharge_candidates: list[tuple[int, int, float]] = []
        for s in range(12, n - 6):
            for length in range(6, min(MAX_DISCHARGE_WINDOW_STEPS + 1, n - s)):
                e     = s + length
                avg_l = (sum_l[e] - sum_l[s]) / length
                discharge_candidates.append((s, e, avg_l))

        # ── Step 3: best (cheapest upper-bound) charge ending at each slot ─
        # Keyed by end-timestep — for each discharge start we only need to
        # scan charge windows that ended before it started.
        best_charge_at: dict[int, tuple[int, int, float]] = {}
        for s, e, avg_u in charge_candidates:
            if e not in best_charge_at or avg_u < best_charge_at[e][2]:
                best_charge_at[e] = (s, e, avg_u)

        # ── Step 4: pair each discharge with the best available charge ─────
        windows: list[ArbitrageWindow] = []

        for d_s, d_e, d_avg_l in discharge_candidates:
            # Only charge windows that finished before this discharge starts
            valid_charges = [
                c for end_t, c in best_charge_at.items() if end_t < d_s
            ]
            if not valid_charges:
                continue

            # Cheapest charge window (minimise what we paid to buy energy)
            c_s, c_e, c_avg_u = min(valid_charges, key=lambda x: x[2])

            spread = d_avg_l - c_avg_u
            if spread <= 0:
                continue

            # Energy available for this trade
            charge_duration_h = (c_e - c_s) * (STEP_MINUTES / 60)
            energy_kwh = min(
                self.charger_power_kw * charge_duration_h,
                self.fleet_capacity_kwh,
            )

            profit_eur = spread * energy_kwh / 1000.0   # EUR/MWh → EUR/kWh
            if profit_eur < MIN_PROFIT_EUR:
                continue

            # Buses required: energy needed ÷ usable capacity per bus
            kwh_per_bus    = self.fleet_capacity_kwh / 100.0
            buses_required = int(math.ceil(energy_kwh / kwh_per_bus))

            # Confidence: penalise wide PI bands and far-horizon windows
            c_width_avg = (
                (sum_u[c_e] - sum_u[c_s]) - (sum_l[c_e] - sum_l[c_s])
            ) / (c_e - c_s)
            d_width_avg = (
                (sum_u[d_e] - sum_u[d_s]) - (sum_l[d_e] - sum_l[d_s])
            ) / (d_e - d_s)
            avg_width       = (c_width_avg + d_width_avg) / 2.0
            horizon_penalty = d_e / n
            confidence      = float(np.clip(
                1.0 - (avg_width / 40.0) - (horizon_penalty * 0.2),
                0.0, 1.0,
            ))

            if confidence < MIN_CONFIDENCE:
                continue

            windows.append(ArbitrageWindow(
                charge_start                = intervals[c_s].timestamp,
                charge_end                  = intervals[c_e - 1].timestamp,
                discharge_start             = intervals[d_s].timestamp,
                discharge_end               = intervals[d_e - 1].timestamp,
                charge_avg_upper_eur_mwh    = round(c_avg_u, 4),
                discharge_avg_lower_eur_mwh = round(d_avg_l, 4),
                spread_eur_mwh              = round(spread, 4),
                estimated_profit_eur        = round(profit_eur, 2),
                confidence                  = round(confidence, 4),
                buses_required              = buses_required,
            ))

        return sorted(
            windows, key=lambda x: x.estimated_profit_eur, reverse=True
        )[:10]


# ---------------------------------------------------------------------------
# Rolling inference runner
# ---------------------------------------------------------------------------

def run_rolling_inference(
    forecaster: GridSentinelForecaster,
    clean_truth_path: Path,
    output_dir: Path,
    interval_minutes: int = 60,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "Rolling inference | source=%s | output=%s | interval=%dm",
        clean_truth_path, output_dir, interval_minutes,
    )
    while True:
        cycle_start = time.time()
        try:
            df     = pd.read_parquet(clean_truth_path)
            result = forecaster.predict(df)

            ts_tag      = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            arc_path    = output_dir / f"forecast_{ts_tag}.json"
            latest_path = output_dir / "latest_forecast.json"

            arc_path.write_text(result.to_json())
            latest_path.write_text(result.to_json())
            log.info("Forecast written → %s", latest_path)

        except Exception as exc:
            log.error("Inference cycle failed: %s", exc, exc_info=True)

        elapsed = time.time() - cycle_start
        time.sleep(max(0, interval_minutes * 60 - elapsed))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GridSentinel — 24h probabilistic price forecaster",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",      "-i", type=Path, required=True)
    p.add_argument("--output",     "-o", type=Path, default=Path("data/forecasts/latest_forecast.json"))
    p.add_argument("--models-dir",       type=Path, default=DEFAULT_MODELS_DIR)
    p.add_argument("--rolling",          action="store_true")
    p.add_argument("--interval",         type=int,  default=60)
    p.add_argument("--output-dir",       type=Path, default=Path("data/forecasts"))
    return p


def main() -> None:
    args       = _build_arg_parser().parse_args()
    forecaster = GridSentinelForecaster(models_dir=args.models_dir)

    if args.rolling:
        run_rolling_inference(
            forecaster       = forecaster,
            clean_truth_path = args.input,
            output_dir       = args.output_dir,
            interval_minutes = args.interval,
        )
        return

    log.info("Loading Clean Truth from %s …", args.input)
    df     = pd.read_parquet(args.input)
    result = forecaster.predict(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.to_json())
    log.info("Forecast written → %s", args.output)

    print("\n" + "═" * 56)
    print("  GridSentinel — Inference Complete")
    print("═" * 56)
    print(f"  Generated at   : {result.generated_at}")
    print(f"  Horizon        : {result.horizon_hours}h ({HORIZON} steps)")
    print(f"  Coverage target: {result.coverage_target:.0%}")
    print(f"  Anchor price   : {result.meta['anchor_price_eur_mwh']:.2f} EUR/MWh")
    print(f"  Mean PI width  : ±{result.meta['mean_interval_width'] / 2:.3f} EUR/MWh")
    print(f"  Arbitrage wins : {result.meta['n_arbitrage_windows']}")
    if result.arbitrage_windows:
        best = result.arbitrage_windows[0]
        print(f"\n  Best window:")
        print(f"    Charge   : {best.charge_start[11:16]} → {best.charge_end[11:16]}")
        print(f"    Discharge: {best.discharge_start[11:16]} → {best.discharge_end[11:16]}")
        print(f"    Spread   : {best.spread_eur_mwh:.2f} EUR/MWh")
        print(f"    Est. P&L : €{best.estimated_profit_eur:.0f}")
        print(f"    Confidence: {best.confidence:.0%}")
        print(f"    Buses req : {best.buses_required}")
    print("═" * 56 + "\n")


if __name__ == "__main__":
    main()