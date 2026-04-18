"""
GridSentinel — Open-Meteo Weather Producer
============================================
Publishes weather data to Kinesis every 15 minutes.

Uses the Open-Meteo free API — no API key required.
Fetches for the Johannesburg depot location (primary) since the fleet
simulator uses JHB coordinates and the battery physics uses JHB temperature.

Real data from: https://api.open-meteo.com
  - Solar irradiance (direct_radiation W/m²)
  - Temperature 2m (°C)
  - Wind speed 10m (km/h)
  - Cloud cover (%) — used as proxy for solar forecast confidence

Synthetic fallback: if the API is unreachable, uses the same diurnal
temperature model as bus.py so the data stays internally consistent.

Record schema emitted to Kinesis:
  {
    "source":               "open_meteo",
    "event_timestamp":      "2026-04-15T14:00:00+00:00",
    "ingestion_timestamp":  "2026-04-15T14:00:02+00:00",
    "location":             "johannesburg",
    "latitude":             -26.2041,
    "longitude":            28.0473,
    "temperature_c":        28.4,
    "solar_irradiance_wm2": 612.0,
    "wind_speed_kmh":       14.2,
    "cloud_cover_pct":      15.0,
    "is_forecast":          false
  }
"""

from __future__ import annotations

import logging
import math
import random
import sys
import time
from datetime import UTC, datetime

import requests

from ingestion.kinesis_client import KinesisClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-32s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("ingestion.producer_weather")

# Johannesburg depot coordinates (Depot 0 — Park Station)
JHB_LAT = -26.2041
JHB_LON = 28.0473
POLL_INTERVAL_S = 15 * 60  # 15 minutes
PARTITION_KEY = "shared"

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


# ─── Synthetic weather (JHB diurnal model) ───────────────────────────────────


def synthetic_weather(dt: datetime) -> dict:
    """
    Johannesburg diurnal weather model — consistent with bus.py temperature model.

    JHB climate (Highveld, summer):
      Temperature: 12–30°C, sinusoidal, peak ~14:00 local (UTC+2)
      Solar:       0 overnight, peak ~900 W/m² at solar noon (12:00 local)
      Wind:        8–25 km/h, afternoon gusts typical on Highveld
      Cloud:       Low in dry season (Apr–Sep), higher in summer (Nov–Feb)
    """
    hour_utc = dt.hour + dt.minute / 60.0
    hour_local = (hour_utc + 2.0) % 24.0  # SAST = UTC+2

    # Temperature: 21°C mean, ±9°C amplitude, peak at 14:00 local
    temperature_c = 21.0 + 9.0 * math.sin(math.pi * (hour_local - 6.0) / 12.0)
    temperature_c = round(temperature_c + random.gauss(0, 0.8), 1)

    # Solar irradiance: 0 at night, cosine bell centred on solar noon (12:00 local)
    if 6.0 <= hour_local <= 18.0:
        solar_angle = math.pi * (hour_local - 6.0) / 12.0
        peak_irradiance = 900.0  # W/m² at solar noon, JHB clear sky
        solar_irradiance = peak_irradiance * math.sin(solar_angle)
        cloud_factor = random.uniform(0.75, 1.00)  # JHB mostly clear
        solar_irradiance = round(solar_irradiance * cloud_factor, 1)
        cloud_cover_pct = round((1.0 - cloud_factor) * 100.0, 1)
    else:
        solar_irradiance = 0.0
        cloud_cover_pct = round(random.uniform(5, 30), 1)

    # Wind: afternoon gusts typical, calmer at night
    if 12.0 <= hour_local <= 18.0:
        wind_speed_kmh = round(random.uniform(14.0, 28.0), 1)
    else:
        wind_speed_kmh = round(random.uniform(5.0, 14.0), 1)

    return {
        "temperature_c": max(5.0, temperature_c),
        "solar_irradiance_wm2": max(0.0, solar_irradiance),
        "wind_speed_kmh": wind_speed_kmh,
        "cloud_cover_pct": cloud_cover_pct,
    }


# ─── Real Open-Meteo API fetch ────────────────────────────────────────────────


def fetch_real_weather(dt: datetime) -> dict | None:
    """
    Fetch current weather from Open-Meteo (free, no API key).

    Returns None on any error — caller falls back to synthetic.
    """
    params = {
        "latitude": JHB_LAT,
        "longitude": JHB_LON,
        "current": [
            "temperature_2m",
            "direct_radiation",
            "wind_speed_10m",
            "cloud_cover",
        ],
        "timezone": "auto",
        "forecast_days": 1,
    }
    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        current = data.get("current", {})

        return {
            "temperature_c": round(current.get("temperature_2m", 20.0), 1),
            "solar_irradiance_wm2": round(current.get("direct_radiation", 0.0), 1),
            "wind_speed_kmh": round(current.get("wind_speed_10m", 10.0), 1),
            "cloud_cover_pct": round(current.get("cloud_cover", 20.0), 1),
        }

    except Exception as exc:
        log.warning("Open-Meteo API error — falling back to synthetic: %s", exc)
        return None


# ─── Record builder ───────────────────────────────────────────────────────────


def build_record(weather: dict, dt: datetime, mode: str) -> dict:
    return {
        "source": "open_meteo",
        "mode": mode,
        "event_timestamp": dt.isoformat(),
        "ingestion_timestamp": datetime.now(UTC).isoformat(),
        "location": "johannesburg",
        "latitude": JHB_LAT,
        "longitude": JHB_LON,
        "temperature_c": weather["temperature_c"],
        "solar_irradiance_wm2": weather["solar_irradiance_wm2"],
        "wind_speed_kmh": weather["wind_speed_kmh"],
        "cloud_cover_pct": weather["cloud_cover_pct"],
        "is_forecast": False,
    }


# ─── Main loop ────────────────────────────────────────────────────────────────


def main() -> None:
    log.info(
        "Weather Producer starting | source=Open-Meteo | interval=%dm | lat=%.4f lon=%.4f",
        POLL_INTERVAL_S // 60,
        JHB_LAT,
        JHB_LON,
    )

    client = KinesisClient()
    client.wait_for_stream()

    while True:
        now = datetime.now(UTC)
        weather = fetch_real_weather(now)
        mode = "real"

        if weather:
            log.info(
                "Open-Meteo | %.1f°C | solar=%.0f W/m² | wind=%.1f km/h | cloud=%.0f%%",
                weather["temperature_c"],
                weather["solar_irradiance_wm2"],
                weather["wind_speed_kmh"],
                weather["cloud_cover_pct"],
            )
        else:
            weather = synthetic_weather(now)
            mode = "synthetic"
            log.info(
                "Synthetic weather | %.1f°C | solar=%.0f W/m² | wind=%.1f km/h",
                weather["temperature_c"],
                weather["solar_irradiance_wm2"],
                weather["wind_speed_kmh"],
            )

        record = build_record(weather, now, mode)
        client.put_record(record, partition_key=PARTITION_KEY)

        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    main()
