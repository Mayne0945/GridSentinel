"""
GridSentinel — Temporal Alignment + BFT Pipeline
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

import boto3
from botocore.config import Config

from bft.gatekeeper import BFTGatekeeper
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-36s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("ingestion.consumer_align")

WINDOW_SECONDS = 300  # 5-minute canonical windows
POLL_INTERVAL_S = 1  # How often to poll Kinesis shard
LATE_SLACK_S = settings.kinesis.late_record_slack_s  # 10s watermark


# ─── Kinesis shard reader ─────────────────────────────────────────────────────


def get_all_shard_iterators(client: Any, stream_name: str) -> list[str]:
    """
    Grab LATEST iterators for ALL shards.
    """
    try:
        resp = client.list_shards(StreamName=stream_name)
        shards = resp.get("Shards", [])
        if not shards:
            log.warning("Stream is ACTIVE but list_shards returned an empty list!")
            return []

        iterators = []
        for shard in shards:
            iter_resp = client.get_shard_iterator(
                StreamName=stream_name,
                ShardId=shard["ShardId"],
                ShardIteratorType="LATEST",
            )
            iterators.append(iter_resp["ShardIterator"])

        log.info("Successfully locked onto all %d shards.", len(iterators))
        return iterators
    except Exception as exc:
        log.error("CRITICAL: Failed to get shard iterators: %s", exc)
        raise  # Crash loudly so we can see it in the logs


def wait_for_stream(client: Any, stream_name: str, max_wait_s: int = 60) -> None:
    waited = 0
    while waited < max_wait_s:
        try:
            resp = client.describe_stream_summary(StreamName=stream_name)
            status = resp["StreamDescriptionSummary"]["StreamStatus"]
            if status == "ACTIVE":
                log.info("Stream %s ACTIVE.", stream_name)
                return
        except Exception:
            pass
        log.info("Waiting for stream %s...", stream_name)
        time.sleep(3)
        waited += 3
    raise RuntimeError(f"Stream {stream_name} not ACTIVE after {max_wait_s}s")


# ─── Window arithmetic ────────────────────────────────────────────────────────


def window_start_for(ts: datetime) -> datetime:
    epoch = ts.timestamp()
    floored = (epoch // WINDOW_SECONDS) * WINDOW_SECONDS
    return datetime.fromtimestamp(floored, tz=UTC)


# ─── Per-window state ─────────────────────────────────────────────────────────


class WindowBuffer:
    def __init__(self, window_start: datetime) -> None:
        self.window_start: datetime = window_start
        self.window_end: datetime = window_start + timedelta(seconds=WINDOW_SECONDS)
        self.bus_records: defaultdict[str, list[dict]] = defaultdict(list)
        self.depot_meter: dict | None = None
        self.entso_price: dict | None = None
        self.weather: dict | None = None
        self.late_drops: int = 0
        self.record_count: int = 0

    def is_closed(self, now: datetime) -> bool:
        deadline = self.window_end + timedelta(seconds=LATE_SLACK_S)
        return now >= deadline

    def add(self, record: dict, event_ts: datetime, now: datetime) -> bool:
        deadline = self.window_end + timedelta(seconds=LATE_SLACK_S)
        if event_ts >= self.window_end and now > deadline:
            self.late_drops += 1
            return False

        source = record.get("source", "")
        self.record_count += 1

        if source == "fleet_sim":
            bus_id = record.get("bus_id")
            if bus_id:
                self.bus_records[bus_id].append(record)
        elif source == "depot_meter":
            self.depot_meter = record
        elif source == "entso_e":
            self.entso_price = record
        elif source == "open_meteo":
            self.weather = record

        return True


# ─── Aggregation ──────────────────────────────────────────────────────────────


def aggregate_window(
    buf: WindowBuffer,
    depot_id: int,
    prev_entso: dict | None,
    prev_weather: dict | None,
) -> dict:
    entso = buf.entso_price or prev_entso or {}
    spot_price = entso.get("spot_price", 0.0)

    wx = buf.weather or prev_weather or {}
    temperature_c = wx.get("temperature_c", 21.0)
    solar_irradiance_wm2 = wx.get("solar_irradiance_wm2", 0.0)
    wind_speed_kmh = wx.get("wind_speed_kmh", 10.0)

    meter = buf.depot_meter or {}
    meter_kw = meter.get("aggregate_power_kw", 0.0)
    chargers = meter.get("active_chargers", 0)

    buses = []
    for bus_id, records in buf.bus_records.items():
        if not records:
            continue
        soc_values = [r.get("soc_pct", 0.0) for r in records]
        power_values = [r.get("power_kw", 0.0) for r in records]
        last = records[-1]

        buses.append(
            {
                "bus_id": bus_id,
                "depot_id": depot_id,
                "mean_soc_pct": round(sum(soc_values) / len(soc_values), 2),
                "mean_power_kw": round(sum(power_values) / len(power_values), 2),
                "sum_power_kwh": round(sum(power_values) * (5.0 / 60.0), 3),
                "soh_pct": last.get("soh_pct", 100.0),
                "status": last.get("status", "idle"),
                "ambient_temperature_c": last.get("ambient_temperature_c", temperature_c),
                "record_count": len(records),
                "is_byzantine": last.get("is_byzantine", False),
            }
        )

    return {
        "canonical_timestamp": buf.window_start.isoformat(),
        "depot_id": depot_id,
        "spot_price": spot_price,
        "temperature_c": temperature_c,
        "solar_irradiance_wm2": solar_irradiance_wm2,
        "wind_speed_kmh": wind_speed_kmh,
        "depot_meter_kw": meter_kw,
        "active_chargers": chargers,
        "buses": buses,
        "record_count": buf.record_count,
        "late_drops": buf.late_drops,
    }


# ─── Main consumer loop ───────────────────────────────────────────────────────


def run(depot_id: int) -> None:
    cfg = settings.kinesis
    stream_name = cfg.stream_name

    log.info(
        "Align+BFT consumer | depot=%d | stream=%s | window=%ds",
        depot_id,
        stream_name,
        WINDOW_SECONDS,
    )

    kinesis = boto3.client(
        "kinesis",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL", "http://localstack:4566"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "eu-west-1"),
        aws_access_key_id="test",
        aws_secret_access_key="test",
        config=Config(retries={"max_attempts": 3, "mode": "adaptive"}),
    )
    wait_for_stream(kinesis, stream_name)

    iterators = get_all_shard_iterators(kinesis, stream_name)
    bft = BFTGatekeeper(depot_id=depot_id)

    prev_entso: dict | None = None
    prev_weather: dict | None = None
    windows: dict[datetime, WindowBuffer] = {}

    total_windows = 0
    total_late_drop = 0

    while True:
        try:
            now = datetime.now(UTC)
            new_iterators = []

            for iterator in iterators:
                resp = kinesis.get_records(ShardIterator=iterator, Limit=500)
                nxt = resp.get("NextShardIterator")
                if nxt:
                    new_iterators.append(nxt)

                for raw in resp.get("Records", []):
                    try:
                        record = json.loads(raw["Data"])
                        source = record.get("source", "")

                        if source in ("fleet_sim", "depot_meter"):
                            if record.get("depot_id") != depot_id:
                                continue

                        ts_str = record.get("event_timestamp") or record.get("ingestion_timestamp")
                        if not ts_str:
                            continue
                        event_ts = datetime.fromisoformat(ts_str)
                        if event_ts.tzinfo is None:
                            event_ts = event_ts.replace(tzinfo=UTC)

                        w_start = window_start_for(event_ts)
                        if w_start not in windows:
                            windows[w_start] = WindowBuffer(w_start)

                        accepted = windows[w_start].add(record, event_ts, now)
                        if not accepted:
                            total_late_drop += 1

                        if source == "entso_e":
                            prev_entso = record
                        elif source == "open_meteo":
                            prev_weather = record

                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue

            iterators = new_iterators

            closed = [ws for ws, buf in windows.items() if buf.is_closed(now)]
            for ws in sorted(closed):
                buf = windows.pop(ws)
                snapshot = aggregate_window(buf, depot_id, prev_entso, prev_weather)
                total_windows += 1

                log.info(
                    "Window closed | ts=%s | buses=%d | records=%d",
                    ws.strftime("%H:%M"),
                    len(snapshot["buses"]),
                    snapshot["record_count"],
                )
                bft.process(snapshot)

        except Exception as exc:
            log.error("Consumer error | depot=%d | %s", depot_id, exc)
            time.sleep(5)

        time.sleep(POLL_INTERVAL_S)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depot-id", type=int, default=int(os.environ.get("DEPOT_ID", 0)))
    args = parser.parse_args()
    run(depot_id=args.depot_id)


if __name__ == "__main__":
    main()
