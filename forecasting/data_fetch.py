"""
forecasting/data_fetch.py
--------------------------
One-time ETL script. Pulls historical GB day-ahead spot prices from ENTSO-E
and weather observations from Open-Meteo, aligns both to a 5-minute UTC index,
and writes immutable Parquet files to data/raw/.

Run once before any training. Never run from train.py.

Usage:
    python forecasting/data_fetch.py \\
        --start 2023-01-01 \\
        --end   2024-12-31 \\
        --output-dir data/raw \\
        --force

Environment variables required:
    ENTSO_E_TOKEN   — API security token from https://transparency.entsoe.eu/

Outputs:
    data/raw/entso_e_prices.parquet
        columns: timestamp (UTC, 5-min), spot_price_eur_mwh

    data/raw/weather_london.parquet
        columns: timestamp (UTC, 5-min), temperature_c,
                 solar_irradiance_wm2, wind_speed_kmh

Design decisions:
    - ENTSO-E returns hourly day-ahead prices (document type A44).
      These are forward-filled to 5-minute resolution to match the
      canonical stream cadence.

    - Open-Meteo archive API returns hourly data.
      Temperature and wind are linearly interpolated to 5-min.
      Solar irradiance (shortwave_radiation) is linearly interpolated.

    - Both sources are aligned to a single 5-minute UTC DatetimeIndex
      spanning [start, end). No row is dropped — gaps are forward-filled
      with a warning so the training pipeline can see them.

    - The script is idempotent: if both Parquet files already exist and
      --force is not passed, it exits immediately. This makes it safe to
      re-run in CI without re-pulling.

    - Requests are chunked into monthly windows to stay within ENTSO-E
      API limits (one month max per request).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger("data_fetch")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stdout,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# GB bidding zone EIC code (Great Britain)
ENTSO_E_AREA = "10YGB----------A"
ENTSO_E_BASE = "https://web-api.tp.entsoe.eu/api"
ENTSO_E_DOCTYPE = "A44"  # Day-ahead prices

# London coords for Open-Meteo
WEATHER_LAT = 51.5074
WEATHER_LON = -0.1278

OPEN_METEO_BASE = "https://archive-api.open-meteo.com/v1/archive"

# 5-minute frequency string for pandas
FREQ_5MIN = "5min"

# ENTSO-E XML namespace
NS = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"}

# Retry config — ENTSO-E occasionally 429s on burst requests
MAX_RETRIES = 5
RETRY_BACKOFF = 2.0  # seconds, doubles each retry


# ---------------------------------------------------------------------------
# ENTSO-E helpers
# ---------------------------------------------------------------------------


def _month_ranges(start: datetime, end: datetime) -> Iterator[tuple[datetime, datetime]]:
    """
    Yield (chunk_start, chunk_end) monthly windows covering [start, end).
    ENTSO-E enforces a maximum query window of one month per request.
    """
    cursor = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while cursor < end:
        next_month = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
        chunk_end = min(next_month, end)
        yield cursor, chunk_end
        cursor = next_month


def _fetch_entso_e_month(
    token: str,
    area: str,
    period_start: datetime,
    period_end: datetime,
) -> pd.Series:
    """
    Fetch one month of day-ahead spot prices from ENTSO-E.
    Returns a pd.Series indexed by UTC datetime, values in EUR/MWh.
    ENTSO-E timestamps are period-start UTC.
    """
    params = {
        "securityToken": token,
        "documentType": ENTSO_E_DOCTYPE,
        "in_Domain": area,
        "out_Domain": area,
        "periodStart": period_start.strftime("%Y%m%d%H%M"),
        "periodEnd": period_end.strftime("%Y%m%d%H%M"),
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(ENTSO_E_BASE, params=params, timeout=30)
            if resp.status_code == 429:
                wait = RETRY_BACKOFF * (2 ** (attempt - 1))
                log.warning("ENTSO-E rate limit hit — sleeping %.1fs (attempt %d)", wait, attempt)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"ENTSO-E fetch failed after {MAX_RETRIES} attempts: {e}") from e
            time.sleep(RETRY_BACKOFF * attempt)
    else:
        raise RuntimeError("ENTSO-E fetch exhausted all retries")

    return _parse_entso_e_xml(resp.text, period_start, period_end)


def _parse_entso_e_xml(xml_text: str, period_start: datetime, period_end: datetime) -> pd.Series:
    """
    Parse ENTSO-E Publication_MarketDocument XML into a pd.Series.

    ENTSO-E returns hourly point observations. Each <TimeSeries> contains a
    <Period> with a <timeInterval> and <resolution>, then <Point> children
    with <position> (1-indexed) and <price.amount>.

    Resolution may be PT60M (hourly) or PT30M (GB intraday). We normalise
    everything to hourly by taking the first point in each hour window.
    """
    root = ET.fromstring(xml_text)
    records: dict[datetime, float] = {}

    for ts_el in root.findall(".//ns:TimeSeries", NS):
        for period_el in ts_el.findall("ns:Period", NS):
            start_el = period_el.find("ns:timeInterval/ns:start", NS)
            res_el = period_el.find("ns:resolution", NS)
            if start_el is None or res_el is None:
                continue

            # Parse interval start — format: 2023-01-01T00:00Z
            try:
                p_start = datetime.fromisoformat(start_el.text.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                log.warning("Could not parse period start: %s", start_el.text)
                continue

            # Resolution in minutes
            res_text = res_el.text  # e.g. "PT60M" or "PT30M"
            try:
                res_min = int(res_text.replace("PT", "").replace("M", ""))
            except ValueError:
                res_min = 60

            for point_el in period_el.findall("ns:Point", NS):
                pos_el = point_el.find("ns:position", NS)
                price_el = point_el.find("ns:price.amount", NS)
                if pos_el is None or price_el is None:
                    continue
                try:
                    position = int(pos_el.text)
                    price = float(price_el.text)
                except (ValueError, TypeError):
                    continue

                # Point timestamp = period_start + (position-1) * resolution
                offset = timedelta(minutes=res_min * (position - 1))
                ts = p_start + offset
                # Normalise to UTC-aware
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)

                # Keep first point in each resolution window (handles duplicates)
                if ts not in records:
                    records[ts] = price

    if not records:
        log.warning(
            "ENTSO-E returned no price points for %s – %s", period_start.date(), period_end.date()
        )
        return pd.Series(dtype=float)

    series = pd.Series(records, name="spot_price_eur_mwh").sort_index()
    series.index = pd.DatetimeIndex(series.index, tz="UTC")
    return series


def fetch_entso_e_prices(
    token: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """
    Fetch all months of GB day-ahead prices in [start, end).
    Returns DataFrame with columns: [spot_price_eur_mwh]
    at 5-minute UTC resolution (forward-filled from hourly source).
    """
    log.info("Fetching ENTSO-E GB prices %s → %s", start.date(), end.date())
    chunks: list[pd.Series] = []

    for chunk_start, chunk_end in _month_ranges(start, end):
        log.info("  ENTSO-E chunk %s → %s", chunk_start.date(), chunk_end.date())
        series = _fetch_entso_e_month(token, ENTSO_E_AREA, chunk_start, chunk_end)
        if not series.empty:
            chunks.append(series)
        time.sleep(0.3)  # polite rate limiting

    if not chunks:
        raise RuntimeError("ENTSO-E returned no data for the requested range")

    hourly = pd.concat(chunks).sort_index()

    # Remove duplicates that can appear at month boundaries
    hourly = hourly[~hourly.index.duplicated(keep="first")]

    # Reindex to 5-minute UTC canonical grid
    idx_5min = pd.date_range(start=start, end=end, freq=FREQ_5MIN, tz="UTC", inclusive="left")
    df = hourly.reindex(idx_5min)

    # Forward-fill hourly value into each 5-min slot within that hour
    gaps_before = df.isna().sum()
    df = df.ffill()
    gaps_after = df.isna().sum()

    if gaps_after > 0:
        log.warning(
            "ENTSO-E: %d 5-min slots still NaN after forward-fill "
            "(leading gap at start of range — filling with series mean)",
            gaps_after,
        )
        df = df.fillna(df.mean())

    log.info(
        "ENTSO-E: %d hourly records → %d 5-min rows (%d gaps filled via ffill)",
        len(hourly),
        len(df),
        int(gaps_before),
    )

    return df.rename("spot_price_eur_mwh").to_frame()


# ---------------------------------------------------------------------------
# Open-Meteo helpers
# ---------------------------------------------------------------------------


def fetch_open_meteo_weather(
    start: datetime,
    end: datetime,
    lat: float = WEATHER_LAT,
    lon: float = WEATHER_LON,
) -> pd.DataFrame:
    """
    Fetch historical hourly weather from Open-Meteo archive API.

    Variables pulled:
        temperature_2m        → temperature_c
        shortwave_radiation   → solar_irradiance_wm2
        windspeed_10m         → wind_speed_kmh

    Returns DataFrame at 5-minute UTC resolution, linearly interpolated
    from the hourly source.
    """
    log.info("Fetching Open-Meteo weather %s → %s", start.date(), end.date())

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": (end - timedelta(days=1)).strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,shortwave_radiation,windspeed_10m",
        "timezone": "UTC",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(OPEN_METEO_BASE, params=params, timeout=60)
            if resp.status_code == 429:
                wait = RETRY_BACKOFF * (2 ** (attempt - 1))
                log.warning("Open-Meteo rate limit — sleeping %.1fs", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Open-Meteo fetch failed: {e}") from e
            time.sleep(RETRY_BACKOFF * attempt)

    data = resp.json()

    if "hourly" not in data:
        raise RuntimeError(
            f"Open-Meteo response missing 'hourly' key: {data.get('reason', 'unknown')}"
        )

    hourly = data["hourly"]
    timestamps = pd.to_datetime(hourly["time"], utc=True)

    df_hourly = pd.DataFrame(
        {
            "temperature_c": hourly["temperature_2m"],
            "solar_irradiance_wm2": hourly["shortwave_radiation"],
            "wind_speed_kmh": hourly["windspeed_10m"],
        },
        index=timestamps,
    )

    # Clip solar to non-negative (API occasionally returns tiny negatives at night)
    df_hourly["solar_irradiance_wm2"] = df_hourly["solar_irradiance_wm2"].clip(lower=0.0)

    # Reindex to 5-minute canonical grid and interpolate
    idx_5min = pd.date_range(start=start, end=end, freq=FREQ_5MIN, tz="UTC", inclusive="left")
    df_5min = df_hourly.reindex(df_hourly.index.union(idx_5min)).sort_index()
    df_5min = df_5min.interpolate(method="time")
    df_5min = df_5min.reindex(idx_5min)

    remaining_gaps = df_5min.isna().sum().sum()
    if remaining_gaps > 0:
        log.warning(
            "Open-Meteo: %d NaN cells after interpolation — forward-filling", remaining_gaps
        )
        df_5min = df_5min.ffill().bfill()

    log.info(
        "Open-Meteo: %d hourly rows → %d 5-min rows",
        len(df_hourly),
        len(df_5min),
    )

    return df_5min


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    start: datetime,
    end: datetime,
    output_dir: Path,
    force: bool = False,
) -> None:
    """
    Pull ENTSO-E + Open-Meteo, align, write Parquet.
    Skips if both files exist and force=False.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    price_path = output_dir / "entso_e_prices.parquet"
    weather_path = output_dir / "weather_london.parquet"

    if price_path.exists() and weather_path.exists() and not force:
        log.info(
            "Both Parquet files already exist. Pass --force to re-pull.\n" "  %s\n  %s",
            price_path,
            weather_path,
        )
        return

    # --- ENTSO-E ---
    token = os.environ.get("ENTSO_E_TOKEN", "")
    if not token:
        raise OSError(
            "ENTSO_E_TOKEN environment variable is not set. "
            "Register at https://transparency.entsoe.eu/ to get a free token."
        )

    df_prices = fetch_entso_e_prices(token, start, end)
    df_prices.to_parquet(price_path, index=True, compression="snappy")
    log.info("Saved → %s  (%d rows)", price_path, len(df_prices))

    # --- Open-Meteo ---
    df_weather = fetch_open_meteo_weather(start, end)
    df_weather.to_parquet(weather_path, index=True, compression="snappy")
    log.info("Saved → %s  (%d rows)", weather_path, len(df_weather))

    # --- Alignment sanity check ---
    assert len(df_prices) == len(df_weather), (
        f"Price rows ({len(df_prices)}) ≠ weather rows ({len(df_weather)}). "
        "Index mismatch — investigate before training."
    )
    assert df_prices.index.equals(df_weather.index), (
        "Price and weather timestamps do not align. "
        "Check timezone handling in both fetch functions."
    )

    log.info(
        "✓ Data fetch complete.\n"
        "  Range  : %s → %s\n"
        "  5-min rows : %d\n"
        "  Price  : %.2f – %.2f EUR/MWh\n"
        "  Temp   : %.1f – %.1f °C",
        start.date(),
        end.date(),
        len(df_prices),
        df_prices["spot_price_eur_mwh"].min(),
        df_prices["spot_price_eur_mwh"].max(),
        df_weather["temperature_c"].min(),
        df_weather["temperature_c"].max(),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pull ENTSO-E prices + Open-Meteo weather into data/raw/ Parquet files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD (exclusive)")
    p.add_argument("--output-dir", default="data/raw", help="Output directory for Parquet files")
    p.add_argument("--force", action="store_true", help="Re-pull even if files already exist")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end_dt = datetime.fromisoformat(args.end).replace(tzinfo=UTC)

    if start_dt >= end_dt:
        log.error("--start must be before --end")
        sys.exit(1)

    run(
        start=start_dt,
        end=end_dt,
        output_dir=Path(args.output_dir),
        force=args.force,
    )
