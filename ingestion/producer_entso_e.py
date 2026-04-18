"""
GridSentinel — ENTSO-E Price Producer
========================================
Publishes day-ahead electricity spot prices to Kinesis every 60 seconds.

Real data mode:
  Set ENTSOE_API_TOKEN in environment (free at transparency.entsoe.eu).
  Fetches GB (10YGB----------A) day-ahead prices from the ENTSO-E
  Transparency Platform REST API.

Synthetic fallback (no token):
  Generates a realistic diurnal price curve so the rest of the pipeline
  keeps running during development without an API key.

Record schema emitted to Kinesis:
  {
    "source":             "entso_e",
    "event_timestamp":    "2026-04-15T14:00:00+00:00",
    "ingestion_timestamp":"2026-04-15T14:00:05+00:00",
    "price_area":         "GB",
    "currency":           "GBP",
    "unit":               "MWh",
    "spot_price":         87.43,
    "resolution":         "PT60M"
  }
"""

from __future__ import annotations

import logging
import math
import os
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
log = logging.getLogger("ingestion.producer_entso_e")

# ENTSO-E Transparency Platform
ENTSOE_BASE_URL = "https://web-api.tp.entsoe.eu/api"
GB_PRICE_AREA = "10YGB----------A"
POLL_INTERVAL_S = 60  # Emit current price once per minute
PARTITION_KEY = "shared"


# ─── Synthetic price curve ────────────────────────────────────────────────────


def synthetic_price(dt: datetime) -> float:
    """
    Realistic GB electricity price curve.

    Based on typical GB day-ahead price patterns:
    - Overnight trough: ~£40/MWh (00:00–05:00)
    - Morning peak:     ~£95/MWh (07:00–09:00)
    - Midday dip:       ~£65/MWh (12:00–14:00, solar generation)
    - Evening peak:     ~£130/MWh (17:00–19:00, demand peak)
    - Late evening:     ~£60/MWh (21:00–23:00)

    Uses overlapping sinusoids to avoid a simplistic single-peak shape.
    Adds ±8% stochastic noise to simulate real price volatility.
    """
    h = dt.hour + dt.minute / 60.0

    # Base overnight level
    base = 55.0

    # Morning peak (centred 08:00)
    morn = 40.0 * math.exp(-0.5 * ((h - 8.0) / 1.5) ** 2)

    # Midday solar dip (centred 13:00) — solar suppresses wholesale prices
    solar = -20.0 * math.exp(-0.5 * ((h - 13.0) / 2.0) ** 2)

    # Evening peak (centred 18:00) — UK demand peak
    eve = 75.0 * math.exp(-0.5 * ((h - 18.0) / 1.5) ** 2)

    price = base + morn + solar + eve
    noise = random.gauss(0, price * 0.08)
    return round(max(5.0, price + noise), 2)


# ─── Real ENTSO-E API fetch ───────────────────────────────────────────────────


def fetch_real_price(token: str, dt: datetime) -> float | None:
    """
    Fetch the current hour's day-ahead price from ENTSO-E.

    Returns None on any error — caller falls back to synthetic.
    """
    # Day-ahead prices are published for tomorrow — fetch today's published prices
    date_str = dt.strftime("%Y%m%d")
    params = {
        "securityToken": token,
        "documentType": "A44",  # Price document
        "in_Domain": GB_PRICE_AREA,
        "out_Domain": GB_PRICE_AREA,
        "periodStart": f"{date_str}0000",
        "periodEnd": f"{date_str}2300",
    }
    try:
        resp = requests.get(
            ENTSOE_BASE_URL,
            params=params,
            timeout=10,
        )
        resp.raise_for_status()

        # Parse XML response to extract current hour's price
        import xml.etree.ElementTree as ET

        root = ET.fromstring(resp.text)
        ns = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"}
        hour = dt.hour

        # Walk TimeSeries → Period → Point
        for ts in root.findall(".//ns:TimeSeries", ns):
            for period in ts.findall("ns:Period", ns):
                for point in period.findall("ns:Point", ns):
                    pos = point.find("ns:position", ns)
                    val = point.find("ns:price.amount", ns)
                    if pos is not None and val is not None:
                        if int(pos.text) == hour + 1:  # ENTSO-E is 1-indexed
                            return round(float(val.text), 2)
        return None

    except Exception as exc:
        log.warning("ENTSO-E API error — falling back to synthetic: %s", exc)
        return None


# ─── Record builder ───────────────────────────────────────────────────────────


def build_record(price: float, dt: datetime, mode: str) -> dict:
    return {
        "source": "entso_e",
        "mode": mode,  # "real" or "synthetic"
        "event_timestamp": dt.isoformat(),
        "ingestion_timestamp": datetime.now(UTC).isoformat(),
        "price_area": "GB",
        "currency": "GBP",
        "unit": "MWh",
        "spot_price": price,
        "resolution": "PT60M",
    }


# ─── Main loop ────────────────────────────────────────────────────────────────


def main() -> None:
    token = os.environ.get("ENTSOE_API_TOKEN")
    mode = "real" if token else "synthetic"

    log.info(
        "ENTSO-E Producer starting | mode=%s | interval=%ds",
        mode,
        POLL_INTERVAL_S,
    )
    if not token:
        log.info(
            "No ENTSOE_API_TOKEN set — running in synthetic mode. "
            "Register free at transparency.entsoe.eu for real GB prices."
        )

    client = KinesisClient()
    client.wait_for_stream()

    while True:
        now = datetime.now(UTC)
        price = None

        if token:
            price = fetch_real_price(token, now)
            if price:
                log.info("ENTSO-E real price | £%.2f/MWh", price)
            else:
                log.warning("Real price unavailable — using synthetic fallback.")

        if price is None:
            price = synthetic_price(now)
            log.info("Synthetic price | £%.2f/MWh | hour=%d", price, now.hour)
            mode = "synthetic"

        record = build_record(price, now, mode)
        client.put_record(record, partition_key=PARTITION_KEY)

        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    main()
