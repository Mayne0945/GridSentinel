"""
tests/unit/test_forecasting_pipeline.py
-----------------------------------------
Unit tests for forecasting/data_fetch.py and forecasting/train.py.

Tests are fully offline — no network calls, no Parquet files required.
All external I/O is replaced with pytest fixtures and monkeypatching.

Coverage targets:
    data_fetch.py  — XML parser, monthly chunker, alignment, NaN handling
    train.py       — feature builder, target builder, walk-forward splits,
                     conformal calibration, coverage evaluation
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Imports under test ────────────────────────────────────────────────────────
from forecasting.data_fetch import (
    _month_ranges,
    _parse_entso_e_xml,
)
from forecasting.train import (
    FEATURE_COLUMNS,
    FLEET_FEATURE_DEFAULTS,
    HORIZON,
    build_features,
    build_targets,
    calibrate_conformal,
    evaluate_conformal_coverage,
    make_optuna_objective,
    walk_forward_splits,
    _make_model,
    _mean_horizon_mae,
    CVSplit,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _make_raw_df(n_rows: int = 500, base_price: float = 60.0) -> pd.DataFrame:
    """
    Create a minimal raw DataFrame that load_data() would return.
    n_rows at 5-minute intervals starting 2023-01-01 UTC.
    """
    idx = pd.date_range("2023-01-01", periods=n_rows, freq="5min", tz="UTC")
    rng = np.random.default_rng(42)
    prices = base_price + rng.normal(0, 5, n_rows).cumsum() * 0.05
    return pd.DataFrame(
        {
            "spot_price_eur_mwh":    prices,
            "temperature_c":         15.0 + rng.normal(0, 3, n_rows),
            "solar_irradiance_wm2":  np.clip(rng.normal(200, 100, n_rows), 0, None),
            "wind_speed_kmh":        10.0 + rng.uniform(0, 20, n_rows),
        },
        index=idx,
    )


# ===========================================================================
# data_fetch — month range chunker
# ===========================================================================

class TestMonthRanges:
    def test_single_month_returns_one_chunk(self):
        start = _utc("2023-03-01")
        end   = _utc("2023-04-01")
        chunks = list(_month_ranges(start, end))
        assert len(chunks) == 1
        s, e = chunks[0]
        assert s.year == 2023 and s.month == 3
        assert e == end

    def test_two_months_returns_two_chunks(self):
        start = _utc("2023-01-01")
        end   = _utc("2023-03-01")
        chunks = list(_month_ranges(start, end))
        assert len(chunks) == 2

    def test_twelve_months_returns_twelve_chunks(self):
        start = _utc("2023-01-01")
        end   = _utc("2024-01-01")
        chunks = list(_month_ranges(start, end))
        assert len(chunks) == 12

    def test_chunks_are_contiguous(self):
        start = _utc("2023-01-01")
        end   = _utc("2023-06-01")
        chunks = list(_month_ranges(start, end))
        for i in range(len(chunks) - 1):
            assert chunks[i][1] == chunks[i + 1][0], (
                "Chunk boundary gap between chunk %d and %d" % (i, i + 1)
            )

    def test_end_is_respected(self):
        start = _utc("2023-06-15")
        end   = _utc("2023-07-20")
        chunks = list(_month_ranges(start, end))
        assert chunks[-1][1] == end

    def test_same_start_end_returns_no_chunks(self):
        ts = _utc("2023-01-01")
        chunks = list(_month_ranges(ts, ts))
        assert chunks == []


# ===========================================================================
# data_fetch — ENTSO-E XML parser
# ===========================================================================

MINIMAL_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">
  <TimeSeries>
    <Period>
      <timeInterval>
        <start>2023-06-01T00:00Z</start>
        <end>2023-06-02T00:00Z</end>
      </timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><price.amount>55.23</price.amount></Point>
      <Point><position>2</position><price.amount>52.10</price.amount></Point>
      <Point><position>3</position><price.amount>48.75</price.amount></Point>
    </Period>
  </TimeSeries>
</Publication_MarketDocument>"""


class TestEntsoEXmlParser:
    def test_parses_three_hourly_prices(self):
        series = _parse_entso_e_xml(
            MINIMAL_XML,
            _utc("2023-06-01"),
            _utc("2023-06-02"),
        )
        assert len(series) == 3

    def test_first_price_correct(self):
        series = _parse_entso_e_xml(MINIMAL_XML, _utc("2023-06-01"), _utc("2023-06-02"))
        assert abs(series.iloc[0] - 55.23) < 1e-6

    def test_second_price_offset_by_one_hour(self):
        series = _parse_entso_e_xml(MINIMAL_XML, _utc("2023-06-01"), _utc("2023-06-02"))
        assert series.index[1].hour == 1

    def test_index_is_utc_aware(self):
        series = _parse_entso_e_xml(MINIMAL_XML, _utc("2023-06-01"), _utc("2023-06-02"))
        assert series.index.tz is not None

    def test_empty_xml_returns_empty_series(self):
        empty_xml = (
            '<Publication_MarketDocument '
            'xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">'
            '</Publication_MarketDocument>'
        )
        series = _parse_entso_e_xml(empty_xml, _utc("2023-06-01"), _utc("2023-06-02"))
        assert series.empty

    def test_no_duplicate_timestamps(self):
        series = _parse_entso_e_xml(MINIMAL_XML, _utc("2023-06-01"), _utc("2023-06-02"))
        assert not series.index.duplicated().any()


# ===========================================================================
# train — feature builder
# ===========================================================================

class TestBuildFeatures:
    def test_output_shape(self):
        df = _make_raw_df(n_rows=300)
        X = build_features(df)
        assert X.shape == (300, 26)

    def test_column_names_match_feature_columns(self):
        df = _make_raw_df(n_rows=100)
        X = build_features(df)
        assert list(X.columns) == FEATURE_COLUMNS

    def test_no_nan_values(self):
        df = _make_raw_df(n_rows=400)
        X = build_features(df)
        assert not X.isna().any().any()

    def test_fleet_features_are_zero_filled(self):
        df = _make_raw_df(n_rows=100)
        X = build_features(df)
        for col, default_val in FLEET_FEATURE_DEFAULTS.items():
            assert (X[col] == default_val).all(), (
                f"Fleet feature '{col}' should be {default_val}, got {X[col].unique()}"
            )

    def test_temporal_sin_cos_magnitude_is_one(self):
        df = _make_raw_df(n_rows=50)
        X = build_features(df)
        for prefix in ["hour", "dow", "month"]:
            mag = np.sqrt(X[f"sin_{prefix}"] ** 2 + X[f"cos_{prefix}"] ** 2)
            assert np.allclose(mag, 1.0, atol=1e-5), (
                f"sin/cos magnitude for {prefix} is not 1.0"
            )

    def test_is_weekend_correct_on_known_date(self):
        # 2023-01-07 is a Saturday
        idx = pd.date_range("2023-01-07", periods=12, freq="5min", tz="UTC")
        df = pd.DataFrame(
            {
                "spot_price_eur_mwh": 60.0,
                "temperature_c": 15.0,
                "solar_irradiance_wm2": 0.0,
                "wind_speed_kmh": 10.0,
            },
            index=idx,
        )
        X = build_features(df)
        assert (X["is_weekend"] == 1.0).all()

    def test_is_weekend_zero_on_monday(self):
        # 2023-01-09 is a Monday
        idx = pd.date_range("2023-01-09", periods=12, freq="5min", tz="UTC")
        df = pd.DataFrame(
            {
                "spot_price_eur_mwh": 60.0,
                "temperature_c": 15.0,
                "solar_irradiance_wm2": 0.0,
                "wind_speed_kmh": 10.0,
            },
            index=idx,
        )
        X = build_features(df)
        assert (X["is_weekend"] == 0.0).all()

    def test_is_peak_hour_flagged_at_17h(self):
        idx = pd.date_range("2023-01-09 17:00", periods=12, freq="5min", tz="UTC")
        df = pd.DataFrame(
            {
                "spot_price_eur_mwh": 60.0,
                "temperature_c": 15.0,
                "solar_irradiance_wm2": 0.0,
                "wind_speed_kmh": 10.0,
            },
            index=idx,
        )
        X = build_features(df)
        assert (X["is_peak_hour"] == 1.0).all()

    def test_price_lag_1_is_previous_row(self):
        idx = pd.date_range("2023-01-01", periods=50, freq="5min", tz="UTC")
        prices = np.arange(50, dtype=float)
        df = pd.DataFrame(
            {
                "spot_price_eur_mwh":   prices,
                "temperature_c":         15.0,
                "solar_irradiance_wm2":  0.0,
                "wind_speed_kmh":        10.0,
            },
            index=idx,
        )
        X = build_features(df)
        # Row 5: price=5, lag_1 should be 4
        assert X["price_lag_1"].iloc[5] == 4.0

    def test_price_lag_at_start_is_zero(self):
        df = _make_raw_df(n_rows=50)
        X = build_features(df)
        # Row 0 has no history — lag_1 should be 0.0
        assert X["price_lag_1"].iloc[0] == 0.0

    def test_dtype_is_float32(self):
        df = _make_raw_df(n_rows=50)
        X = build_features(df)
        assert X.dtypes.iloc[0] == np.float32


# ===========================================================================
# train — target builder
# ===========================================================================

class TestBuildTargets:
    def test_output_rows_is_n_minus_horizon(self):
        df = _make_raw_df(n_rows=HORIZON + 100)
        Y = build_targets(df)
        assert len(Y) == 100

    def test_output_columns_is_horizon(self):
        df = _make_raw_df(n_rows=HORIZON + 50)
        Y = build_targets(df)
        assert Y.shape[1] == HORIZON

    def test_no_nan_values(self):
        df = _make_raw_df(n_rows=HORIZON + 50)
        Y = build_targets(df)
        assert not Y.isna().any().any()

    def test_delta_is_zero_for_constant_price(self):
        idx = pd.date_range("2023-01-01", periods=HORIZON + 50, freq="5min", tz="UTC")
        df = pd.DataFrame(
            {
                "spot_price_eur_mwh":   50.0,
                "temperature_c":         15.0,
                "solar_irradiance_wm2":  0.0,
                "wind_speed_kmh":        10.0,
            },
            index=idx,
        )
        Y = build_targets(df)
        assert np.allclose(Y.values, 0.0, atol=1e-5)

    def test_h1_delta_is_one_step_price_diff(self):
        n = HORIZON + 10
        prices = np.arange(n, dtype=float)
        idx = pd.date_range("2023-01-01", periods=n, freq="5min", tz="UTC")
        df = pd.DataFrame(
            {
                "spot_price_eur_mwh":   prices,
                "temperature_c":         15.0,
                "solar_irradiance_wm2":  0.0,
                "wind_speed_kmh":        10.0,
            },
            index=idx,
        )
        Y = build_targets(df)
        # For linearly increasing prices, all h-1 deltas should equal 1.0
        assert np.allclose(Y["delta_h001"].values, 1.0, atol=1e-5)

    def test_column_names_are_correct(self):
        df = _make_raw_df(n_rows=HORIZON + 10)
        Y = build_targets(df)
        assert Y.columns[0]   == "delta_h001"
        assert Y.columns[287] == "delta_h288"


# ===========================================================================
# train — walk-forward splits
# ===========================================================================

class TestWalkForwardSplits:
    def _get_splits(self, n_rows=10000):
        return list(
            walk_forward_splits(
                n_rows         = n_rows,
                min_train_rows = 2000,
                val_rows       = 500,
                step_rows      = 500,
                horizon        = HORIZON,
                n_folds        = 3,
            )
        )

    def test_returns_up_to_n_folds(self):
        splits = self._get_splits()
        assert len(splits) <= 3

    def test_train_indices_precede_val_indices(self):
        for split in self._get_splits():
            assert split.train_idx[-1] < split.val_idx[0]

    def test_horizon_gap_is_respected(self):
        """Val start must be at least HORIZON rows after train end."""
        for split in self._get_splits():
            gap = split.val_idx[0] - split.train_idx[-1]
            assert gap >= HORIZON, (
                f"Gap {gap} < HORIZON {HORIZON} — data leakage risk"
            )

    def test_val_indices_do_not_overlap_train(self):
        for split in self._get_splits():
            overlap = set(split.train_idx) & set(split.val_idx)
            assert len(overlap) == 0

    def test_splits_are_ordered_by_fold(self):
        splits = self._get_splits()
        for i in range(1, len(splits)):
            assert splits[i].val_idx[0] > splits[i - 1].val_idx[0]

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError, match="No valid CV splits"):
            list(
                walk_forward_splits(
                    n_rows         = 100,
                    min_train_rows = 5000,   # impossible
                    val_rows       = 500,
                    step_rows      = 500,
                    horizon        = HORIZON,
                    n_folds        = 3,
                )
            )

    def test_train_indices_are_0_indexed(self):
        splits = self._get_splits()
        assert splits[0].train_idx[0] == 0


# ===========================================================================
# train — mean horizon MAE
# ===========================================================================

class TestMeanHorizonMAE:
    def test_zero_for_perfect_prediction(self):
        Y = np.ones((100, 288))
        assert _mean_horizon_mae(Y, Y) == 0.0

    def test_correct_for_constant_error(self):
        Y_true = np.zeros((10, 5))
        Y_pred = np.ones((10, 5))
        assert abs(_mean_horizon_mae(Y_true, Y_pred) - 1.0) < 1e-6

    def test_symmetric(self):
        rng = np.random.default_rng(0)
        A = rng.normal(0, 1, (50, 288)).astype(np.float32)
        B = rng.normal(0, 1, (50, 288)).astype(np.float32)
        assert abs(_mean_horizon_mae(A, B) - _mean_horizon_mae(B, A)) < 1e-5


# ===========================================================================
# train — conformal calibration
# ===========================================================================

class TestConformalCalibration:
    def _make_perfect_model(self, n_cal: int = 200) -> tuple:
        """
        Returns a mock model that always predicts the true values,
        plus matching X_cal and Y_cal arrays.
        """
        rng = np.random.default_rng(7)
        X_cal = rng.normal(0, 1, (n_cal, 26)).astype(np.float32)
        Y_cal = rng.normal(0, 10, (n_cal, HORIZON)).astype(np.float32)
        model = MagicMock()
        model.predict.return_value = Y_cal.copy()
        return model, X_cal, Y_cal

    def _make_noisy_model(self, n_cal: int = 200, noise_std: float = 5.0) -> tuple:
        rng = np.random.default_rng(13)
        X_cal = rng.normal(0, 1, (n_cal, 26)).astype(np.float32)
        Y_cal = rng.normal(0, 10, (n_cal, HORIZON)).astype(np.float32)
        Y_pred = Y_cal + rng.normal(0, noise_std, (n_cal, HORIZON)).astype(np.float32)
        model = MagicMock()
        model.predict.return_value = Y_pred
        return model, X_cal, Y_cal

    def test_output_shape_is_horizon(self):
        model, X_cal, Y_cal = self._make_perfect_model()
        q = calibrate_conformal(model, X_cal, Y_cal, coverage=0.80)
        assert q.shape == (HORIZON,)

    def test_perfect_model_has_near_zero_quantiles(self):
        model, X_cal, Y_cal = self._make_perfect_model()
        q = calibrate_conformal(model, X_cal, Y_cal, coverage=0.80)
        assert q.max() < 1e-4, f"Perfect model should have ~zero residuals, got max={q.max():.4f}"

    def test_noisy_model_has_positive_quantiles(self):
        model, X_cal, Y_cal = self._make_noisy_model(noise_std=5.0)
        q = calibrate_conformal(model, X_cal, Y_cal, coverage=0.80)
        assert q.min() > 0.0

    def test_wider_coverage_gives_larger_quantiles(self):
        model, X_cal, Y_cal = self._make_noisy_model(noise_std=5.0)
        q80 = calibrate_conformal(model, X_cal, Y_cal, coverage=0.80)
        q95 = calibrate_conformal(model, X_cal, Y_cal, coverage=0.95)
        assert q95.mean() > q80.mean(), "95% coverage should give wider intervals than 80%"

    def test_quantiles_are_non_negative(self):
        model, X_cal, Y_cal = self._make_noisy_model()
        q = calibrate_conformal(model, X_cal, Y_cal, coverage=0.80)
        assert (q >= 0).all()


# ===========================================================================
# train — coverage evaluation
# ===========================================================================

class TestEvaluateCoverage:
    def test_coverage_near_target_with_correct_quantiles(self):
        """
        If quantiles are calibrated from the same distribution as val data,
        empirical coverage should be close to the target (±5%).
        """
        rng = np.random.default_rng(99)
        n_val  = 500
        n_cal  = 500
        noise  = 3.0

        Y_true_cal  = rng.normal(0, 10, (n_cal, HORIZON)).astype(np.float32)
        Y_pred_cal  = Y_true_cal + rng.normal(0, noise, (n_cal, HORIZON)).astype(np.float32)
        residuals   = np.abs(Y_true_cal - Y_pred_cal)
        quantiles   = np.quantile(residuals, 0.80, axis=0)

        Y_true_val  = rng.normal(0, 10, (n_val, HORIZON)).astype(np.float32)
        Y_pred_val  = Y_true_val + rng.normal(0, noise, (n_val, HORIZON)).astype(np.float32)

        model = MagicMock()
        model.predict.return_value = Y_pred_val

        X_val = rng.normal(0, 1, (n_val, 26)).astype(np.float32)
        Y_val = Y_true_val

        stats = evaluate_conformal_coverage(model, quantiles, X_val, Y_val)

        assert abs(stats["empirical_coverage"] - 0.80) < 0.06, (
            f"Empirical coverage {stats['empirical_coverage']:.3f} "
            f"is more than 6% away from target 0.80"
        )

    def test_coverage_dict_has_required_keys(self):
        rng = np.random.default_rng(1)
        model = MagicMock()
        model.predict.return_value = rng.normal(0, 1, (50, HORIZON)).astype(np.float32)
        q = np.ones(HORIZON)
        X = rng.normal(0, 1, (50, 26)).astype(np.float32)
        Y = rng.normal(0, 1, (50, HORIZON)).astype(np.float32)
        stats = evaluate_conformal_coverage(model, q, X, Y)
        for key in ("empirical_coverage", "target_coverage", "mean_interval_width_eur"):
            assert key in stats


# ===========================================================================
# train — make_model sanity checks
# ===========================================================================

class TestMakeModel:
    BASE_PARAMS = {
        "n_estimators":     50,      # Small for test speed
        "max_depth":        3,
        "learning_rate":    0.1,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
    }

    def test_model_fits_and_predicts(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 26)).astype(np.float32)
        Y = rng.normal(0, 5, (100, 10)).astype(np.float32)   # 10 horizons for speed
        params = dict(self.BASE_PARAMS, n_estimators=5)       # 5 trees per horizon

        # Patch HORIZON to 10 for this test to avoid training 288 trees
        from sklearn.multioutput import MultiOutputRegressor
        from xgboost import XGBRegressor

        model = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=5, max_depth=3, learning_rate=0.1,
                objective="reg:squarederror", tree_method="hist",
                n_jobs=1, random_state=42, verbosity=0,
            ),
            n_jobs=1,
        )
        model.fit(X, Y)
        preds = model.predict(X)
        assert preds.shape == (100, 10)

    def test_make_model_returns_multioutput_regressor(self):
        from sklearn.multioutput import MultiOutputRegressor
        model = _make_model(self.BASE_PARAMS)
        assert isinstance(model, MultiOutputRegressor)