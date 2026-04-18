"""
forecasting/mock_data_fetch.py
------------------------------
TACTICAL BYPASS SCRIPT.
Generates 1 year of mathematically realistic mock ENTSO-E and Open-Meteo data
to unblock pipeline development while waiting for API token approval.

Usage:
    python forecasting/mock_data_fetch.py
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("mock_data")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", stream=sys.stdout)

def run_mock_generation():
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp("2023-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-01", tz="UTC")
    idx = pd.date_range(start, end, freq="5min", inclusive="left")
    n = len(idx)

    log.info(f"Injecting {n} rows of synthetic grid telemetry...")

    # Time axes for cyclical math
    hour_of_day = idx.hour + idx.minute / 60.0
    day_of_year = idx.dayofyear

    # ---------------------------------------------------------
    # 1. Price: Base 60 + Daily double peak + Weekend drop + Noise
    # ---------------------------------------------------------
    base_price = 60.0
    # Peak 1 (morning), Peak 2 (evening)
    daily_cycle = 20 * np.sin(np.pi * (hour_of_day - 6) / 12) + 15 * np.sin(np.pi * (hour_of_day - 14) / 6)
    weekend_penalty = np.where(idx.dayofweek >= 5, -15.0, 0.0)
    price_noise = np.random.normal(0, 3.0, n)
    
    prices = base_price + daily_cycle + weekend_penalty + price_noise
    prices = np.clip(prices, 10.0, 300.0) # Keep within realistic bounds

    df_prices = pd.DataFrame({"spot_price_eur_mwh": prices}, index=idx)
    price_path = output_dir / "entso_e_prices.parquet"
    df_prices.to_parquet(price_path, compression="snappy")
    log.info(f"Mock prices sealed → {price_path}")

    # ---------------------------------------------------------
    # 2. Weather: Temp (Annual + Daily), Solar (Daylight), Wind
    # ---------------------------------------------------------
    # Temp: Annual cycle (baseline 12C, swing 10C) + Daily cycle (swing 5C)
    annual_temp = 12.0 + 10.0 * np.sin(2 * np.pi * (day_of_year - 100) / 365.0)
    daily_temp = 5.0 * np.sin(2 * np.pi * (hour_of_day - 8) / 24.0)
    temp = annual_temp + daily_temp + np.random.normal(0, 1.0, n)

    # Solar: Bell curve between 6 AM and 6 PM, 0 otherwise
    solar = 600.0 * np.sin(np.pi * (hour_of_day - 6) / 12.0)
    solar = np.where((hour_of_day < 6) | (hour_of_day > 18), 0.0, solar)
    solar = solar + np.random.normal(0, 20.0, n)
    solar = np.clip(solar, 0.0, None)

    # Wind: Random walk baseline + noise
    wind = 15.0 + np.random.normal(0, 4.0, n)
    wind = np.clip(wind, 0.0, None)

    df_weather = pd.DataFrame({
        "temperature_c": temp,
        "solar_irradiance_wm2": solar,
        "wind_speed_kmh": wind
    }, index=idx)

    weather_path = output_dir / "weather_london.parquet"
    df_weather.to_parquet(weather_path, compression="snappy")
    log.info(f"Mock weather sealed → {weather_path}")

    log.info("Synthetic grid active. You are unblocked.")

if __name__ == "__main__":
    run_mock_generation()