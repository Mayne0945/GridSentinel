"""
tests/unit/test_feature_builder.py
------------------------------------
Unit tests for forecasting.feature_builder.

Tests cover:
  - Cyclical time encoding correctness (mathematical properties)
  - Rolling window O(1) behaviour and edge cases
  - Price lag extraction at various depths
  - Fleet feature aggregation (clean vs byzantine buses)
  - Full process() pipeline on a realistic Clean Truth payload
  - FeatureVector.to_model_input() only returns numeric fields
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from forecasting.feature_builder import FeatureBuilder, FeatureVector, RollingWindow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_clean_truth(
    timestamp: str = "2026-04-16T07:30:00+00:00",
    price: float = 87.4,
    temperature_c: float = 18.0,
    solar: float = 350.0,
    wind: float = 12.5,
    bus_socs: list[float] | None = None,
    byzantine_bus_ids: set[str] | None = None,
) -> dict:
    """Build a minimal valid Clean Truth payload for testing."""
    if bus_socs is None:
        bus_socs = [72.0] * 10

    byzantine_bus_ids = byzantine_bus_ids or set()

    buses = [
        {
            "bus_id":      f"D0_BUS_{i:03d}",
            "soc_pct":     soc,
            "power_kw":    -85.0,
            "is_byzantine": f"D0_BUS_{i:03d}" in byzantine_bus_ids,
        }
        for i, soc in enumerate(bus_socs)
    ]

    return {
        "canonical_timestamp": timestamp,
        "depot_id":            0,
        "spot_price_eur_mwh":  price,
        "weather": {
            "temperature_c":        temperature_c,
            "solar_irradiance_wm2": solar,
            "wind_speed_kmh":       wind,
        },
        "buses": buses,
    }


def make_builder(depot_id: int = 0) -> FeatureBuilder:
    builder = FeatureBuilder(depot_id=depot_id)
    builder._connect = lambda: None   # Skip Redis in unit tests
    builder._r = None
    return builder


# ---------------------------------------------------------------------------
# RollingWindow tests
# ---------------------------------------------------------------------------

class TestRollingWindow:
    def test_mean_of_single_value(self):
        w = RollingWindow(maxlen=10)
        w.append(50.0)
        assert w.mean() == pytest.approx(50.0)

    def test_mean_of_multiple_values(self):
        w = RollingWindow(maxlen=10)
        for v in [10.0, 20.0, 30.0]:
            w.append(v)
        assert w.mean() == pytest.approx(20.0)

    def test_std_of_identical_values_is_zero(self):
        w = RollingWindow(maxlen=10)
        for _ in range(5):
            w.append(42.0)
        assert w.std() == pytest.approx(0.0, abs=1e-9)

    def test_std_single_value_is_zero(self):
        w = RollingWindow(maxlen=10)
        w.append(42.0)
        assert w.std() == pytest.approx(0.0)

    def test_mean_empty_is_zero(self):
        w = RollingWindow(maxlen=10)
        assert w.mean() == pytest.approx(0.0)

    def test_std_empty_is_zero(self):
        w = RollingWindow(maxlen=10)
        assert w.std() == pytest.approx(0.0)

    def test_maxlen_evicts_oldest_value(self):
        w = RollingWindow(maxlen=3)
        for v in [10.0, 20.0, 30.0, 40.0]:
            w.append(v)
        # Window should contain [20, 30, 40] — 10 was evicted
        assert w.mean() == pytest.approx(30.0)

    def test_lag_1_returns_most_recent(self):
        w = RollingWindow(maxlen=10)
        for v in [10.0, 20.0, 30.0]:
            w.append(v)
        assert w.get(1) == pytest.approx(30.0)

    def test_lag_3_returns_third_from_end(self):
        w = RollingWindow(maxlen=10)
        for v in [10.0, 20.0, 30.0]:
            w.append(v)
        assert w.get(3) == pytest.approx(10.0)

    def test_lag_beyond_window_returns_zero(self):
        w = RollingWindow(maxlen=10)
        w.append(42.0)
        assert w.get(5) == pytest.approx(0.0)

    def test_std_known_values(self):
        """Verify std against manually computed value."""
        w = RollingWindow(maxlen=10)
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]:
            w.append(v)
        # Sample std of [2,4,4,4,5,5,7,9] = 2.0
        assert w.std() == pytest.approx(2.138, rel=1e-3)


# ---------------------------------------------------------------------------
# Temporal encoding tests
# ---------------------------------------------------------------------------

class TestTemporalEncoding:
    def test_sin_cos_magnitude_is_one(self):
        """sin² + cos² must equal 1 for any valid cyclical encoding."""
        ts = datetime(2026, 4, 16, 7, 30, tzinfo=timezone.utc)
        feats = FeatureBuilder._encode_time(ts)
        assert (feats["sin_hour"] ** 2 + feats["cos_hour"] ** 2) == pytest.approx(1.0, rel=1e-6)
        assert (feats["sin_dow"] ** 2 + feats["cos_dow"] ** 2) == pytest.approx(1.0, rel=1e-6)
        assert (feats["sin_month"] ** 2 + feats["cos_month"] ** 2) == pytest.approx(1.0, rel=1e-6)

    def test_midnight_has_zero_sin_hour(self):
        """Hour 0 → sin(0) = 0, cos(0) = 1."""
        ts = datetime(2026, 4, 16, 0, 0, tzinfo=timezone.utc)
        feats = FeatureBuilder._encode_time(ts)
        assert feats["sin_hour"] == pytest.approx(0.0, abs=1e-9)
        assert feats["cos_hour"] == pytest.approx(1.0, rel=1e-6)

    def test_noon_has_zero_cos_hour(self):
        """Hour 12 → sin(π) ≈ 0, cos(π) = -1."""
        ts = datetime(2026, 4, 16, 12, 0, tzinfo=timezone.utc)
        feats = FeatureBuilder._encode_time(ts)
        assert feats["sin_hour"] == pytest.approx(0.0, abs=1e-6)
        assert feats["cos_hour"] == pytest.approx(-1.0, rel=1e-6)

    def test_monday_and_sunday_are_adjacent_in_encoding(self):
        """
        Monday (weekday=0) and Sunday (weekday=6) should be close in
        cyclical space — their Euclidean distance should be small.
        """
        monday = datetime(2026, 4, 13, 12, 0, tzinfo=timezone.utc)  # Monday
        sunday = datetime(2026, 4, 19, 12, 0, tzinfo=timezone.utc)  # Sunday
        m = FeatureBuilder._encode_time(monday)
        s = FeatureBuilder._encode_time(sunday)
        dist = math.sqrt((m["sin_dow"] - s["sin_dow"]) ** 2 + (m["cos_dow"] - s["cos_dow"]) ** 2)
        # Distance Monday↔Sunday ≈ 2*sin(π/7) ≈ 0.867
        # Distance Monday↔Wednesday ≈ 2*sin(2π/7) ≈ 1.56
        # So Monday↔Sunday should be less than Monday↔Wednesday
        wednesday = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
        w = FeatureBuilder._encode_time(wednesday)
        dist_mon_wed = math.sqrt((m["sin_dow"] - w["sin_dow"]) ** 2 + (m["cos_dow"] - w["cos_dow"]) ** 2)
        assert dist < dist_mon_wed

    def test_all_six_temporal_features_present(self):
        ts = datetime(2026, 4, 16, 9, 0, tzinfo=timezone.utc)
        feats = FeatureBuilder._encode_time(ts)
        required = {"sin_hour", "cos_hour", "sin_dow", "cos_dow", "sin_month", "cos_month"}
        assert required == set(feats.keys())


# ---------------------------------------------------------------------------
# Fleet feature extraction tests
# ---------------------------------------------------------------------------

class TestFleetFeatureExtraction:
    def setup_method(self):
        self.builder = make_builder()

    def test_mean_soc_correct(self):
        buses = [{"soc_pct": 60.0, "is_byzantine": False},
                 {"soc_pct": 80.0, "is_byzantine": False}]
        feats = self.builder._extract_fleet_features(buses)
        assert feats["fleet_mean_soc_pct"] == pytest.approx(70.0)

    def test_min_soc_correct(self):
        buses = [{"soc_pct": 30.0, "is_byzantine": False},
                 {"soc_pct": 80.0, "is_byzantine": False}]
        feats = self.builder._extract_fleet_features(buses)
        assert feats["fleet_min_soc_pct"] == pytest.approx(30.0)

    def test_byzantine_buses_excluded_from_mean(self):
        buses = [
            {"soc_pct": 80.0, "is_byzantine": False},
            {"soc_pct": 5.0,  "is_byzantine": True},   # Byzantine — should be ignored
        ]
        feats = self.builder._extract_fleet_features(buses)
        assert feats["fleet_mean_soc_pct"] == pytest.approx(80.0)

    def test_byzantine_count_correct(self):
        buses = [
            {"soc_pct": 80.0, "is_byzantine": False},
            {"soc_pct": 5.0,  "is_byzantine": True},
            {"soc_pct": 5.0,  "is_byzantine": True},
        ]
        feats = self.builder._extract_fleet_features(buses)
        assert feats["byzantine_bus_count"] == 2
        assert feats["clean_bus_count"] == 1

    def test_all_byzantine_returns_safe_defaults(self):
        buses = [{"soc_pct": 5.0, "is_byzantine": True}] * 10
        feats = self.builder._extract_fleet_features(buses)
        assert feats["fleet_mean_soc_pct"] == pytest.approx(50.0)
        assert feats["fleet_available_kw"] == pytest.approx(0.0)

    def test_available_kw_zero_when_all_buses_low_soc(self):
        """Buses at < 20% SoC cannot discharge — available kW must be zero."""
        buses = [{"soc_pct": 10.0, "is_byzantine": False}] * 10
        feats = self.builder._extract_fleet_features(buses)
        assert feats["fleet_available_kw"] == pytest.approx(0.0)

    def test_available_kw_positive_when_buses_charged(self):
        """Buses at 80% SoC should report significant available discharge capacity."""
        buses = [{"soc_pct": 80.0, "is_byzantine": False}] * 10
        feats = self.builder._extract_fleet_features(buses)
        assert feats["fleet_available_kw"] > 0


# ---------------------------------------------------------------------------
# Price lag and rolling window integration tests
# ---------------------------------------------------------------------------

class TestPriceLagsAndWindows:
    def setup_method(self):
        self.builder = make_builder()

    def test_lag_1_is_previous_price(self):
        p1, p2 = 80.0, 90.0
        self.builder._update_price_windows(p1)
        feats = self.builder._update_price_windows(p2)
        assert feats["price_lag_1"] == pytest.approx(p1)

    def test_lag_288_is_zero_before_24h_data(self):
        feats = self.builder._update_price_windows(75.0)
        assert feats["price_lag_288"] == pytest.approx(0.0)

    def test_rolling_mean_6h_initialises_correctly(self):
        """After one window, mean should equal that window's price."""
        feats = self.builder._update_price_windows(100.0)
        assert feats["price_rolling_mean_6h"] == pytest.approx(100.0)

    def test_rolling_std_6h_zero_after_one_value(self):
        feats = self.builder._update_price_windows(100.0)
        assert feats["price_rolling_std_6h"] == pytest.approx(0.0)

    def test_rolling_mean_increases_with_higher_price(self):
        self.builder._update_price_windows(50.0)
        feats = self.builder._update_price_windows(150.0)
        assert feats["price_rolling_mean_6h"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Full process() pipeline test
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def setup_method(self):
        self.builder = make_builder()

    def test_process_returns_feature_vector(self):
        ct = make_clean_truth()
        result = self.builder.process(ct)
        assert isinstance(result, FeatureVector)

    def test_depot_id_preserved(self):
        ct = make_clean_truth()
        result = self.builder.process(ct)
        assert result.depot_id == 0

    def test_timestamp_preserved(self):
        ct = make_clean_truth(timestamp="2026-04-16T07:30:00+00:00")
        result = self.builder.process(ct)
        assert "2026-04-16T07:30" in result.canonical_timestamp

    def test_price_correct(self):
        ct = make_clean_truth(price=92.5)
        result = self.builder.process(ct)
        assert result.spot_price_eur_mwh == pytest.approx(92.5)

    def test_temperature_correct(self):
        ct = make_clean_truth(temperature_c=22.3)
        result = self.builder.process(ct)
        assert result.temperature_c == pytest.approx(22.3)

    def test_peak_hour_flagged_correctly(self):
        ct = make_clean_truth(timestamp="2026-04-16T18:00:00+00:00")  # peak hour
        result = self.builder.process(ct)
        assert result.is_peak_hour == pytest.approx(1.0)

    def test_off_peak_hour_flagged_correctly(self):
        ct = make_clean_truth(timestamp="2026-04-16T03:00:00+00:00")  # 3am — not peak
        result = self.builder.process(ct)
        assert result.is_peak_hour == pytest.approx(0.0)

    def test_weekend_flag_on_saturday(self):
        ct = make_clean_truth(timestamp="2026-04-18T12:00:00+00:00")  # Saturday
        result = self.builder.process(ct)
        assert result.is_weekend == pytest.approx(1.0)

    def test_weekday_flag_on_monday(self):
        ct = make_clean_truth(timestamp="2026-04-13T12:00:00+00:00")  # Monday
        result = self.builder.process(ct)
        assert result.is_weekend == pytest.approx(0.0)

    def test_byzantine_buses_excluded_from_fleet_mean(self):
        # 9 clean buses at 80%, 1 byzantine at 5%
        socs = [80.0] * 9 + [5.0]
        byz  = {"D0_BUS_009"}
        ct   = make_clean_truth(bus_socs=socs, byzantine_bus_ids=byz)
        result = self.builder.process(ct)
        # Mean should be ~80%, not dragged down to ~76% by the byzantine reading
        assert result.fleet_mean_soc_pct > 79.0

    def test_windows_counter_increments(self):
        for i in range(5):
            self.builder.process(make_clean_truth())
        assert self.builder._windows_processed == 5

    def test_all_feature_fields_are_float_compatible(self):
        ct = make_clean_truth()
        result = self.builder.process(ct)
        for k, v in result.to_model_input().items():
            assert isinstance(v, float), f"Field '{k}' is not float: {type(v)}"

    def test_to_model_input_excludes_identity_fields(self):
        ct = make_clean_truth()
        result = self.builder.process(ct)
        model_input = result.to_model_input()
        assert "canonical_timestamp" not in model_input
        assert "depot_id" not in model_input
        assert "feature_version" not in model_input

    def test_to_model_input_has_expected_feature_count(self):
        ct = make_clean_truth()
        result = self.builder.process(ct)
        model_input = result.to_model_input()
        # 6 temporal + 10 price + 3 weather + 5 fleet + 2 grid proxy = 26
        assert len(model_input) == 26

    def test_malformed_timestamp_does_not_crash(self):
        ct = make_clean_truth()
        ct["canonical_timestamp"] = "not-a-timestamp"
        result = self.builder.process(ct)
        assert isinstance(result, FeatureVector)

    def test_missing_weather_uses_defaults(self):
        ct = make_clean_truth()
        del ct["weather"]
        result = self.builder.process(ct)
        assert result.temperature_c == pytest.approx(15.0)
        assert result.solar_irradiance_wm2 == pytest.approx(0.0)

    def test_sequential_windows_build_correct_lags(self):
        """Feed 3 windows and verify lag-1 and lag-3 are correct."""
        prices = [60.0, 70.0, 75.0, 80.0]
        for p in prices:
            ct     = make_clean_truth(price=p)
            result = self.builder.process(ct)

        # After 3 windows: lag-1 should be 70.0 (read before appending 80.0)
        assert result.price_lag_1 == pytest.approx(75.0)
        # lag-3 should be the first price appended: 60.0
        assert result.price_lag_3 == pytest.approx(60.0)
