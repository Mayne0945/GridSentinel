"""
forecasting/train.py
---------------------
Trains the GridSentinel 24-hour price delta forecaster.

Architecture:
    MultiOutputRegressor(XGBRegressor(...))
    — one XGBoost tree ensemble per forecast horizon h ∈ {1, 2, ..., 288}
    — each tree sees the same 26-feature input vector (price history, weather,
      temporal encoding, zero-filled fleet features)
    — target: Δprice[h] = spot_price_{t+h} − spot_price_t  (EUR/MWh delta)

Uncertainty quantification:
    Split conformal prediction (manual implementation).
    After final model training, a held-out calibration window (last CAL_DAYS
    days of training data) is used to compute the 80th-percentile absolute
    residual per horizon. At inference, the interval is:
        [ŷ_h − q80_h,  ŷ_h + q80_h]
    This gives valid marginal 80% coverage under exchangeability without
    requiring MAPIE's multi-output API (which has had breaking changes across
    versions).

Pipeline:
    1.  load_data()          — read Parquet files written by data_fetch.py
    2.  build_features()     — vectorized offline feature matrix (N × 26)
    3.  build_targets()      — delta matrix (M × 288), M = N − 288
    4.  walk_forward_splits()— time-ordered CV split generator (no leakage)
    5.  optuna_objective()   — Optuna trial → mean horizon MAE on last fold
    6.  train_final()        — refit on full train set with best params
    7.  calibrate_conformal()— compute q80 quantiles on calibration window
    8.  save_artifacts()     — joblib model + npy quantiles + json metadata

Usage:
    python forecasting/train.py \\
        --price-parquet  data/raw/entso_e_prices.parquet \\
        --weather-parquet data/raw/weather_london.parquet \\
        --train-end  2024-10-31 \\
        --output-dir models \\
        --n-trials   30

Outputs:
    models/xgb_forecaster.joblib      — fitted MultiOutputRegressor
    models/conformal_quantiles.npy    — shape (288,) float64 array
    models/feature_columns.json       — ordered list of 26 feature names
    models/training_metadata.json     — run stats, val MAE, coverage
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

log = logging.getLogger("forecasting.train")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stdout,
)

# Silence Optuna's noisy per-trial logs — we print our own summary
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Forecast horizon: 24h at 5-min resolution
HORIZON = 288

# Feature columns — must match FeatureVector.to_model_input() field order exactly.
# Identity fields (canonical_timestamp, depot_id, feature_version) are excluded.
# Fleet features (indices 19–23) are zero-filled at train time — the XGBoost
# trees will assign them zero information gain, which is the correct behaviour:
# GB grid spot price does not depend on 500 buses' SoC in a single depot.
FEATURE_COLUMNS: list[str] = [
    # Temporal (6)
    "sin_hour",
    "cos_hour",
    "sin_dow",
    "cos_dow",
    "sin_month",
    "cos_month",
    # Price (10)
    "spot_price_eur_mwh",
    "price_lag_1",
    "price_lag_3",
    "price_lag_6",
    "price_lag_12",
    "price_lag_288",
    "price_rolling_mean_6h",
    "price_rolling_std_6h",
    "price_rolling_mean_24h",
    "price_rolling_std_24h",
    # Weather (3)
    "temperature_c",
    "solar_irradiance_wm2",
    "wind_speed_kmh",
    # Fleet — zero-filled at train time (5)
    "fleet_mean_soc_pct",
    "fleet_min_soc_pct",
    "fleet_available_kw",
    "clean_bus_count",
    "byzantine_bus_count",
    # Grid proxy (2)
    "is_peak_hour",
    "is_weekend",
]

assert len(FEATURE_COLUMNS) == 26, f"Expected 26 features, got {len(FEATURE_COLUMNS)}"

# Fleet feature indices — must be zero-filled in training matrix
FLEET_FEATURE_DEFAULTS = {
    "fleet_mean_soc_pct": 50.0,  # neutral midpoint — model ignores this
    "fleet_min_soc_pct": 20.0,
    "fleet_available_kw": 0.0,
    "clean_bus_count": 0.0,  # treated as float for XGBoost
    "byzantine_bus_count": 0.0,
}

# Walk-forward CV config
MIN_TRAIN_DAYS = 30  # Minimum training window size (rows = × 288)
VAL_DAYS = 1  # Validation window per fold (rows = × 288)
STEP_DAYS = 7  # Slide validation window by this many days
N_CV_FOLDS = 3  # Use only the last N folds for speed

# Conformal calibration
CAL_DAYS = 30  # Calibration window (tail of training set)
COVERAGE_80 = 0.80  # Target coverage level


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(
    price_parquet: Path,
    weather_parquet: Path,
) -> pd.DataFrame:
    """
    Load price and weather Parquets written by data_fetch.py.
    Returns a single DataFrame at 5-minute UTC resolution with columns:
        spot_price_eur_mwh, temperature_c, solar_irradiance_wm2, wind_speed_kmh

    Validates alignment before returning.
    """
    log.info("Loading price data from  %s", price_parquet)
    df_price = pd.read_parquet(price_parquet)

    log.info("Loading weather data from %s", weather_parquet)
    df_weather = pd.read_parquet(weather_parquet)

    # Both should have a UTC DatetimeIndex from data_fetch.py
    if not isinstance(df_price.index, pd.DatetimeIndex):
        raise ValueError("Price Parquet index is not a DatetimeIndex")
    if not isinstance(df_weather.index, pd.DatetimeIndex):
        raise ValueError("Weather Parquet index is not a DatetimeIndex")

    # Ensure UTC-aware
    if df_price.index.tz is None:
        df_price.index = df_price.index.tz_localize("UTC")
    if df_weather.index.tz is None:
        df_weather.index = df_weather.index.tz_localize("UTC")

    # Align on intersection — defensive, should be exact match if data_fetch ran correctly
    idx = df_price.index.intersection(df_weather.index)
    if len(idx) < len(df_price):
        log.warning(
            "Index intersection dropped %d rows (price has more rows than weather). "
            "Check data_fetch.py output alignment.",
            len(df_price) - len(idx),
        )

    df = pd.concat([df_price.loc[idx], df_weather.loc[idx]], axis=1)

    expected_cols = {
        "spot_price_eur_mwh",
        "temperature_c",
        "solar_irradiance_wm2",
        "wind_speed_kmh",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Raw data missing columns: {missing}")

    log.info(
        "Loaded %d rows | %s → %s | price %.2f–%.2f EUR/MWh",
        len(df),
        df.index[0].date(),
        df.index[-1].date(),
        df["spot_price_eur_mwh"].min(),
        df["spot_price_eur_mwh"].max(),
    )

    return df


# ---------------------------------------------------------------------------
# Feature matrix construction  (offline batch — mirrors feature_builder.py)
# ---------------------------------------------------------------------------


def _cyclical(values: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    """sin/cos cyclical encoding with the given period."""
    rad = 2 * math.pi * values / period
    return np.sin(rad), np.cos(rad)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the offline feature matrix from raw price + weather data.
    Mirrors the logic in FeatureBuilder.process() but vectorised with pandas.

    Key decisions:
        - NaN cells from lag/rolling at the start of the series are filled
          with 0.0 to match FeatureBuilder's deque returning 0.0 when empty.
        - Fleet features are zero-filled (see module docstring).
        - All outputs are float64.

    Returns a DataFrame with columns = FEATURE_COLUMNS, same index as df.
    """
    log.info("Building feature matrix for %d rows...", len(df))
    idx = df.index
    price = df["spot_price_eur_mwh"]

    feat = pd.DataFrame(index=idx)

    # ── Temporal encoding ────────────────────────────────────────────────────
    feat["sin_hour"], feat["cos_hour"] = _cyclical(idx.hour, 24.0)
    feat["sin_dow"], feat["cos_dow"] = _cyclical(idx.dayofweek, 7.0)
    feat["sin_month"], feat["cos_month"] = _cyclical(idx.month.astype(float) - 1, 12.0)

    # ── Price features ───────────────────────────────────────────────────────
    feat["spot_price_eur_mwh"] = price.values

    # Lags — shift by N 5-min steps
    for lag in [1, 3, 6, 12, 288]:
        feat[f"price_lag_{lag}"] = price.shift(lag).fillna(0.0).values

    # Rolling stats — window in number of 5-min steps
    # ddof=1 (sample std) matches RollingWindow.std() in feature_builder.py
    for window, label in [(72, "6h"), (288, "24h")]:
        roll = price.rolling(window, min_periods=1)
        feat[f"price_rolling_mean_{label}"] = roll.mean().values
        feat[f"price_rolling_std_{label}"] = roll.std(ddof=1).fillna(0.0).values

    # ── Weather features ─────────────────────────────────────────────────────
    feat["temperature_c"] = df["temperature_c"].values
    feat["solar_irradiance_wm2"] = df["solar_irradiance_wm2"].values
    feat["wind_speed_kmh"] = df["wind_speed_kmh"].values

    # ── Fleet features (zero-filled at training time) ────────────────────────
    for col, default_val in FLEET_FEATURE_DEFAULTS.items():
        feat[col] = default_val

    # ── Grid proxy ───────────────────────────────────────────────────────────
    peak_hours = {7, 8, 9, 17, 18, 19, 20}
    feat["is_peak_hour"] = idx.hour.isin(peak_hours).astype(float)
    feat["is_weekend"] = (idx.dayofweek >= 5).astype(float)

    # Enforce column order — must match FEATURE_COLUMNS exactly
    feat = feat[FEATURE_COLUMNS].astype(np.float32)

    nan_count = feat.isna().sum().sum()
    if nan_count > 0:
        log.warning("%d NaN values in feature matrix — filling with 0.0", nan_count)
        feat = feat.fillna(0.0)

    log.info("Feature matrix: %s — dtype %s", feat.shape, feat.dtypes.iloc[0])
    return feat


# ---------------------------------------------------------------------------
# Target matrix construction
# ---------------------------------------------------------------------------


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the Δprice target matrix.

    target[t, h] = spot_price_{t+h} − spot_price_t    h ∈ {1, ..., HORIZON}

    Why deltas (not absolute prices)?
        1. Stationarity: raw prices drift with seasonal baselines and fuel
           costs. Deltas around the current observation are far more
           stationary — the model generalises across seasons without retraining.
        2. Arbitrage framing: the MPC objective needs the spread between
           charge and discharge windows, not absolute prices. Predicting
           deltas gives the arbitrage detector exactly what it needs.

    The last HORIZON rows have no valid targets and are dropped.
    Returns a DataFrame with HORIZON columns (h=1..288) and
    len(df) - HORIZON rows.
    """
    log.info("Building target matrix (HORIZON=%d)...", HORIZON)
    price = df["spot_price_eur_mwh"]

    # Vectorised: shift prices forward and subtract current price
    # price.shift(-h) moves future values to current row
    cols = {}
    for h in range(1, HORIZON + 1):
        cols[f"delta_h{h:03d}"] = price.shift(-h).values - price.values

    targets = pd.DataFrame(cols, index=df.index)

    # Drop the last HORIZON rows — they reference future data we don't have
    targets = targets.iloc[:-HORIZON]

    nan_count = targets.isna().sum().sum()
    if nan_count > 0:
        raise ValueError(
            f"{nan_count} NaN values in target matrix after dropping last "
            f"{HORIZON} rows. This should not happen — check data alignment."
        )

    log.info(
        "Target matrix: %s | delta range: [%.2f, %.2f] EUR/MWh",
        targets.shape,
        targets.values.min(),
        targets.values.max(),
    )
    return targets


# ---------------------------------------------------------------------------
# Walk-forward cross-validation splits
# ---------------------------------------------------------------------------


class CVSplit(NamedTuple):
    fold: int
    train_idx: np.ndarray  # integer positions
    val_idx: np.ndarray  # integer positions


def walk_forward_splits(
    n_rows: int,
    min_train_rows: int,
    val_rows: int,
    step_rows: int,
    horizon: int,
    n_folds: int,
) -> Generator[CVSplit, None, None]:
    """
    Generate walk-forward CV splits with a mandatory horizon gap.

    Gap explanation:
        The target for row t is price[t+HORIZON]. If we train on rows 0..T
        and validate starting at T+1, the training targets for rows near T
        "see" prices up to T+HORIZON. To prevent any information leak, the
        validation window must start at T + horizon + 1.

    Yields the last n_folds splits only (earlier splits are less predictive
    of current model quality and slow down Optuna with redundant evaluations).
    """
    # Determine all valid fold start positions
    splits: list[CVSplit] = []
    fold = 0

    val_start = min_train_rows + horizon  # First valid val start position
    while val_start + val_rows <= n_rows:
        train_end = val_start - horizon
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_start + val_rows)
        splits.append(CVSplit(fold=fold, train_idx=train_idx, val_idx=val_idx))
        fold += 1
        val_start += step_rows

    if not splits:
        raise ValueError(
            f"No valid CV splits found. "
            f"Need at least {min_train_rows + horizon + val_rows} rows, "
            f"got {n_rows}. Reduce min_train_days or val_days."
        )

    # Yield only the last n_folds — most relevant for current regime
    for split in splits[-n_folds:]:
        yield split


# ---------------------------------------------------------------------------
# Model training helpers
# ---------------------------------------------------------------------------


def _make_model(params: dict) -> MultiOutputRegressor:
    """
    Construct MultiOutputRegressor(XGBRegressor) from hyperparameter dict.

    n_jobs=-1 on MultiOutputRegressor parallelises the 288 independent fits
    across all available CPU cores. Each XGBRegressor itself uses n_jobs=1
    to avoid nested parallelism contention.
    """
    xgb = XGBRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_weight=params["min_child_weight"],
        objective="reg:squarederror",
        tree_method="hist",  # Fast histogram-based algorithm
        n_jobs=1,  # 1 per estimator — parallelism at wrapper level
        random_state=42,
        verbosity=0,
    )
    return MultiOutputRegressor(xgb, n_jobs=-1)


def _mean_horizon_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Mean absolute error averaged across all HORIZON outputs.
    Shape: y_true, y_pred both (N, HORIZON).
    """
    return float(np.mean(np.abs(y_true - y_pred)))


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def make_optuna_objective(
    X: np.ndarray,
    Y: np.ndarray,
    min_train_rows: int,
    val_rows: int,
    step_rows: int,
    horizon: int,
    n_cv_folds: int,
) -> callable:
    """
    Factory — returns a closure that Optuna calls per trial.

    The objective runs walk-forward CV with n_cv_folds and returns mean
    horizon MAE across all folds. Lower is better.
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

        fold_maes: list[float] = []

        for split in walk_forward_splits(
            n_rows=len(X),
            min_train_rows=min_train_rows,
            val_rows=val_rows,
            step_rows=step_rows,
            horizon=horizon,
            n_folds=n_cv_folds,
        ):
            X_train = X[split.train_idx]
            Y_train = Y[split.train_idx]
            X_val = X[split.val_idx]
            Y_val = Y[split.val_idx]

            model = _make_model(params)
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_val)

            fold_maes.append(_mean_horizon_mae(Y_val, Y_pred))

        return float(np.mean(fold_maes))

    return objective


# ---------------------------------------------------------------------------
# Conformal calibration
# ---------------------------------------------------------------------------


def calibrate_conformal(
    model: MultiOutputRegressor,
    X_cal: np.ndarray,
    Y_cal: np.ndarray,
    coverage: float = COVERAGE_80,
) -> np.ndarray:
    """
    Split conformal prediction calibration.

    For each horizon h, compute the `coverage`-quantile of absolute residuals
    on the calibration set. At inference, the prediction interval is:
        [ŷ_h − q_h,  ŷ_h + q_h]

    This guarantees marginal coverage at the `coverage` level under
    exchangeability (see Angelopoulos & Bates 2021, §2).

    Args:
        model:    Fitted MultiOutputRegressor
        X_cal:    Calibration features  (N_cal × 26)
        Y_cal:    Calibration targets   (N_cal × 288)
        coverage: Target coverage level (0.80)

    Returns:
        quantiles: shape (HORIZON,) — one q per horizon h
    """
    log.info(
        "Calibrating conformal prediction on %d rows (target coverage %.0f%%)...",
        len(X_cal),
        coverage * 100,
    )

    Y_pred = model.predict(X_cal)  # (N_cal, HORIZON)
    residuals = np.abs(Y_cal - Y_pred)  # (N_cal, HORIZON)

    # Per-horizon quantile — axis=0 takes quantile across calibration samples
    quantiles = np.quantile(residuals, coverage, axis=0)  # (HORIZON,)

    log.info(
        "Conformal quantiles | mean: %.3f EUR/MWh | "
        "h=1: %.3f | h=144 (12h): %.3f | h=288 (24h): %.3f",
        quantiles.mean(),
        quantiles[0],
        quantiles[143],
        quantiles[287],
    )

    return quantiles


def evaluate_conformal_coverage(
    model: MultiOutputRegressor,
    quantiles: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
) -> dict[str, float]:
    """
    Measure empirical interval coverage and mean interval width on a
    held-out validation set. Used for reporting in training metadata.

    Returns dict with 'empirical_coverage' and 'mean_interval_width_eur'.
    """
    Y_pred = model.predict(X_val)  # (N_val, HORIZON)
    lower = Y_pred - quantiles[np.newaxis, :]
    upper = Y_pred + quantiles[np.newaxis, :]

    inside = ((Y_val >= lower) & (Y_val <= upper)).astype(float)
    empirical_coverage = float(inside.mean())

    mean_width = float(2 * quantiles.mean())

    log.info(
        "Conformal coverage check | empirical: %.3f (target: %.2f) | "
        "mean interval width: ±%.3f EUR/MWh",
        empirical_coverage,
        COVERAGE_80,
        quantiles.mean(),
    )

    return {
        "empirical_coverage": empirical_coverage,
        "target_coverage": COVERAGE_80,
        "mean_interval_width_eur": mean_width,
    }


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------


def save_artifacts(
    model: MultiOutputRegressor,
    quantiles: np.ndarray,
    metadata: dict,
    output_dir: Path,
) -> None:
    """
    Persist all model artifacts required for inference.

    Saved files:
        xgb_forecaster.joblib     — fitted MultiOutputRegressor (288 trees)
        conformal_quantiles.npy   — shape (288,) float64 array
        feature_columns.json      — ordered feature list for inference contract
        training_metadata.json    — run stats for reproducibility
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "xgb_forecaster.joblib"
    quantiles_path = output_dir / "conformal_quantiles.npy"
    features_path = output_dir / "feature_columns.json"
    metadata_path = output_dir / "training_metadata.json"

    joblib.dump(model, model_path, compress=3)
    log.info("Saved model → %s", model_path)

    np.save(quantiles_path, quantiles)
    log.info("Saved conformal quantiles → %s", quantiles_path)

    with features_path.open("w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)
    log.info("Saved feature columns → %s", features_path)

    metadata["saved_at"] = datetime.now(tz=UTC).isoformat()
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Saved training metadata → %s", metadata_path)


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------


def main(
    price_parquet: Path,
    weather_parquet: Path,
    train_end: datetime | None,
    output_dir: Path,
    n_trials: int,
) -> None:
    """
    End-to-end training pipeline.

    Data split:
        All rows up to train_end (or the full dataset if None) are used.
        Within that window:
            Walk-forward CV folds → hyperparameter search
            Final model trains on ALL rows minus the calibration window
            Calibration window → conformal quantile estimation (last CAL_DAYS)
    """
    t0 = time.time()

    # ── 1. Load ─────────────────────────────────────────────────────────────
    df = load_data(price_parquet, weather_parquet)

    if train_end is not None:
        train_end_utc = train_end.replace(tzinfo=UTC)
        df = df[df.index < train_end_utc]
        log.info("Truncated to train_end %s → %d rows", train_end_utc.date(), len(df))

    # ── 2. Feature matrix ────────────────────────────────────────────────────
    X_df = build_features(df)

    # ── 3. Target matrix ─────────────────────────────────────────────────────
    # build_targets returns len(df) - HORIZON rows — align X accordingly
    Y_df = build_targets(df)
    X_df = X_df.iloc[: len(Y_df)]  # trim last HORIZON rows (no valid targets)

    assert len(X_df) == len(Y_df), f"Feature/target row mismatch: X={len(X_df)}, Y={len(Y_df)}"

    X = X_df.values.astype(np.float32)  # (N, 26)
    Y = Y_df.values.astype(np.float32)  # (N, 288)

    log.info(
        "Training matrix: X=%s  Y=%s  | dtype X=%s Y=%s",
        X.shape,
        Y.shape,
        X.dtype,
        Y.dtype,
    )

    # Determine walk-forward split parameters in rows
    min_train_rows = MIN_TRAIN_DAYS * HORIZON  # 90 days × 288 = 25920
    val_rows = VAL_DAYS * HORIZON  # 14 days × 288 = 4032
    step_rows = STEP_DAYS * HORIZON  # 7 days  × 288 = 2016
    cal_rows = CAL_DAYS * HORIZON  # 30 days × 288 = 8640

    min_required = min_train_rows + HORIZON + val_rows
    if len(X) < min_required:
        raise ValueError(
            f"Dataset has only {len(X)} rows but needs ≥ {min_required} for "
            f"walk-forward CV. Fetch more history with data_fetch.py "
            f"(current: MIN_TRAIN_DAYS={MIN_TRAIN_DAYS}, VAL_DAYS={VAL_DAYS})."
        )

    # ── 4. Hyperparameter search ─────────────────────────────────────────────
    log.info("Starting Optuna search: %d trials, %d CV folds each...", n_trials, N_CV_FOLDS)

    study = optuna.create_study(
        direction="minimize",
        study_name="gridsentinel_xgb",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    objective = make_optuna_objective(
        X=X,
        Y=Y,
        min_train_rows=min_train_rows,
        val_rows=val_rows,
        step_rows=step_rows,
        horizon=HORIZON,
        n_cv_folds=N_CV_FOLDS,
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_cv_mae = study.best_value

    log.info(
        "Optuna complete | best mean-horizon MAE: %.4f EUR/MWh | best params: %s",
        best_cv_mae,
        best_params,
    )

    # ── 5. Final model — train on everything except calibration tail ─────────
    log.info(
        "Training final model on %d rows (holding out last %d rows for calibration)...",
        len(X) - cal_rows,
        cal_rows,
    )

    X_final_train = X[:-cal_rows]
    Y_final_train = Y[:-cal_rows]
    X_cal = X[-cal_rows:]
    Y_cal = Y[-cal_rows:]

    final_model = _make_model(best_params)
    final_model.fit(X_final_train, Y_final_train)

    log.info("Final model fitted.")

    # ── 6. Conformal calibration ─────────────────────────────────────────────
    quantiles = calibrate_conformal(final_model, X_cal, Y_cal, coverage=COVERAGE_80)

    # ── 7. Coverage report on calibration set (diagnostic only) ─────────────
    coverage_stats = evaluate_conformal_coverage(final_model, quantiles, X_cal, Y_cal)

    # ── 8. Final MAE on calibration window (point forecast quality) ──────────
    Y_cal_pred = final_model.predict(X_cal)
    final_mae = _mean_horizon_mae(Y_cal, Y_cal_pred)

    # Per-horizon MAE — useful for spotting degradation at long horizons
    per_horizon_mae = np.mean(np.abs(Y_cal - Y_cal_pred), axis=0)  # (HORIZON,)
    mae_h1 = float(per_horizon_mae[0])
    mae_h12h = float(per_horizon_mae[143])
    mae_h24h = float(per_horizon_mae[287])

    elapsed = time.time() - t0

    log.info(
        "\n"
        "══════════════════════════════════════════════════════\n"
        "  GridSentinel — Training Complete\n"
        "══════════════════════════════════════════════════════\n"
        "  Rows trained : %d  |  Rows calibrated : %d\n"
        "  Best CV MAE  : %.4f EUR/MWh\n"
        "  Final MAE    : %.4f EUR/MWh (mean across 288 horizons)\n"
        "    h=1   (5m)  : %.4f EUR/MWh\n"
        "    h=144 (12h) : %.4f EUR/MWh\n"
        "    h=288 (24h) : %.4f EUR/MWh\n"
        "  Conformal coverage : %.3f  (target %.2f)\n"
        "  Elapsed      : %.1fs\n"
        "══════════════════════════════════════════════════════",
        len(X_final_train),
        len(X_cal),
        best_cv_mae,
        final_mae,
        mae_h1,
        mae_h12h,
        mae_h24h,
        coverage_stats["empirical_coverage"],
        COVERAGE_80,
        elapsed,
    )

    # ── 9. Save ──────────────────────────────────────────────────────────────
    metadata = {
        "horizon": HORIZON,
        "feature_count": len(FEATURE_COLUMNS),
        "train_rows": len(X_final_train),
        "cal_rows": len(X_cal),
        "train_start": str(df.index[0].date()),
        "train_end_cutoff": str(df.index[len(X_final_train) - 1].date()),
        "best_params": best_params,
        "best_cv_mae": round(best_cv_mae, 4),
        "final_mae": round(final_mae, 4),
        "mae_h1_5min": round(mae_h1, 4),
        "mae_h144_12h": round(mae_h12h, 4),
        "mae_h288_24h": round(mae_h24h, 4),
        "conformal": coverage_stats,
        "elapsed_s": round(elapsed, 1),
        "n_optuna_trials": n_trials,
        "n_cv_folds": N_CV_FOLDS,
    }

    save_artifacts(final_model, quantiles, metadata, output_dir)

    log.info(
        "All artifacts written to %s\n"
        "Next step: forecasting/inference.py loads xgb_forecaster.joblib "
        "+ conformal_quantiles.npy and serves predictions to the MPC.",
        output_dir,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train GridSentinel 24h price delta forecaster.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--price-parquet",
        default="data/raw/entso_e_prices.parquet",
        help="Path to Parquet file written by data_fetch.py",
    )
    p.add_argument(
        "--weather-parquet",
        default="data/raw/weather_london.parquet",
        help="Path to Parquet file written by data_fetch.py",
    )
    p.add_argument(
        "--train-end",
        default=None,
        help="Exclude data on or after this date (YYYY-MM-DD). "
        "Defaults to full dataset. Set to prevent future leakage in backtest.",
    )
    p.add_argument(
        "--output-dir",
        default="models",
        help="Directory to write model artifacts",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of Optuna hyperparameter search trials (default 30)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    train_end_dt: datetime | None = None
    if args.train_end:
        train_end_dt = datetime.fromisoformat(args.train_end).replace(tzinfo=UTC)

    main(
        price_parquet=Path(args.price_parquet),
        weather_parquet=Path(args.weather_parquet),
        train_end=train_end_dt,
        output_dir=Path(args.output_dir),
        n_trials=args.n_trials,
    )
