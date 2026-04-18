"""
GridSentinel — Stream Monitor
================================
Live terminal view of all records flowing through the Kinesis stream.
Shows fleet telemetry, prices, and weather interleaved in real time.

Usage:
    python -m ingestion.monitor
    python -m ingestion.monitor --depot 0        # filter to one depot
    python -m ingestion.monitor --source fleet   # filter by source

Output format:
    [09:15:03] fleet      d0_b042  │ SoC: 72.4%  Pwr: -85.0kW  [CLEAN]  🟢
    [09:15:04] entso_e    GB       │ £87.43/MWh                          📈
    [09:15:04] open_meteo JHB      │ 28.4°C  Solar: 612W/m²  Wind: 14km/h ☀️
    [09:15:05] fleet      d0_b007  │ SoC: 51.2%  Pwr:   0.0kW  [BYZNT]  🔴
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import boto3
from botocore.config import Config

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

STREAM_NAME = os.environ.get("KINESIS_STREAM_NAME", "gridsentinel-telemetry")
ENDPOINT    = os.environ.get("AWS_ENDPOINT_URL",    "http://localstack:4566")
REGION      = os.environ.get("AWS_DEFAULT_REGION",  "eu-west-1")

# ANSI colours
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def get_client() -> boto3.client:
    return boto3.client(
        "kinesis",
        endpoint_url=ENDPOINT,
        region_name=REGION,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "test"),
        config=Config(retries={"max_attempts": 2}),
    )


def get_shard_iterators(client) -> list[str]:
    """Get LATEST shard iterators for all shards in the stream."""
    resp    = client.describe_stream_summary(StreamName=STREAM_NAME)
    n_shards= resp["StreamDescriptionSummary"]["OpenShardCount"]

    resp    = client.list_shards(StreamName=STREAM_NAME)
    shards  = resp["Shards"]

    iterators = []
    for shard in shards:
        resp = client.get_shard_iterator(
            StreamName=STREAM_NAME,
            ShardId=shard["ShardId"],
            ShardIteratorType="LATEST",
        )
        iterators.append(resp["ShardIterator"])
    return iterators


def format_record(record: dict, depot_filter: int | None, source_filter: str | None) -> str | None:
    """
    Format a single Kinesis record for display.
    Returns None if the record should be filtered out.
    """
    source    = record.get("source", "unknown")
    now_str   = datetime.now(timezone.utc).strftime("%H:%M:%S")

    # Apply filters
    if source_filter and source not in source_filter:
        return None

    if source == "fleet_sim":
        depot_id = record.get("depot_id")
        if depot_filter is not None and depot_id != depot_filter:
            return None

        bus_id    = record.get("bus_id", "?")
        soc       = record.get("soc_pct", 0.0)
        power     = record.get("power_kw", 0.0)
        is_byz    = record.get("is_byzantine", False)
        status    = record.get("status", "")

        byz_tag   = f"{RED}[BYZNT]{RESET}" if is_byz else f"{GREEN}[CLEAN]{RESET}"
        icon      = "🔴" if is_byz else "🟢"
        pwr_str   = f"{power:+7.1f}kW"

        return (
            f"[{CYAN}{now_str}{RESET}] "
            f"{YELLOW}fleet    {RESET} "
            f"{bus_id:<12} │ "
            f"SoC:{soc:6.1f}%  Pwr:{pwr_str}  "
            f"{byz_tag} {icon}"
        )

    elif source == "depot_meter":
        depot_id = record.get("depot_id")
        if depot_filter is not None and depot_id != depot_filter:
            return None
        power    = record.get("aggregate_power_kw", 0.0)
        chargers = record.get("active_chargers", 0)
        return (
            f"[{CYAN}{now_str}{RESET}] "
            f"{BOLD}meter    {RESET} "
            f"depot_{depot_id:<8}  │ "
            f"Grid: {power:+8.1f}kW  Chargers: {chargers}"
        )

    elif source == "entso_e":
        if source_filter and "entso" not in source_filter:
            return None
        price = record.get("spot_price", 0.0)
        mode  = record.get("mode", "")
        tag   = f"({mode})" if mode == "synthetic" else ""
        return (
            f"[{CYAN}{now_str}{RESET}] "
            f"{BOLD}entso_e  {RESET} "
            f"{'GB':<12} │ "
            f"£{price:7.2f}/MWh {tag} 📈"
        )

    elif source == "open_meteo":
        if source_filter and "weather" not in source_filter and "meteo" not in source_filter:
            return None
        temp  = record.get("temperature_c", 0.0)
        solar = record.get("solar_irradiance_wm2", 0.0)
        wind  = record.get("wind_speed_kmh", 0.0)
        mode  = record.get("mode", "")
        tag   = f"({mode})" if mode == "synthetic" else ""
        icon  = "☀️" if solar > 100 else "🌥️"
        return (
            f"[{CYAN}{now_str}{RESET}] "
            f"{BOLD}weather  {RESET} "
            f"{'JHB':<12} │ "
            f"{temp:.1f}°C  Solar:{solar:6.0f}W/m²  Wind:{wind:.1f}km/h {tag} {icon}"
        )

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="GridSentinel stream monitor")
    parser.add_argument("--depot",  type=int,  default=None, help="Filter by depot_id")
    parser.add_argument("--source", type=str,  default=None, help="Filter by source (fleet/entso/weather)")
    args = parser.parse_args()

    client = get_client()

    print(f"\n{BOLD}GridSentinel | Live Stream Monitor{RESET}")
    print(f"Stream: {STREAM_NAME} @ {ENDPOINT}")
    if args.depot is not None:
        print(f"Filter: depot={args.depot}")
    if args.source:
        print(f"Filter: source={args.source}")
    print("─" * 75)

    try:
        iterators = get_shard_iterators(client)
    except Exception as exc:
        print(f"{RED}Could not connect to stream: {exc}{RESET}")
        print("Is the stream ACTIVE? Run: docker compose up localstack fleet-sim")
        sys.exit(1)

    record_count = 0
    import datetime
    from collections import defaultdict

    try:
        while True:
            new_iterators = []
            # Reset the batch counters for this 1-second tick
            batch_counts = defaultdict(int)

            for iterator in iterators:
                try:
                    # Bump limit slightly to catch bursts efficiently
                    resp        = client.get_records(ShardIterator=iterator, Limit=1000)
                    new_iter    = resp.get("NextShardIterator")
                    if new_iter:
                        new_iterators.append(new_iter)

                    for raw in resp.get("Records", []):
                        try:
                            record  = json.loads(raw["Data"])
                            # Instead of formatting and printing, just tally the source
                            source = record.get("source", "unknown")
                            batch_counts[source] += 1
                            record_count += 1
                        except json.JSONDecodeError:
                            pass

                except Exception as exc:
                    print(f"{RED}Shard read error: {exc}{RESET}", file=sys.stderr)

            iterators = new_iterators
            
            # Print a single, high-signal heartbeat if data arrived in this tick
            if batch_counts:
                now = datetime.datetime.now().strftime("%H:%M:%S")
                stats = " │ ".join(f"{k.upper()}: {v}" for k, v in batch_counts.items())
                print(f"[{now}] ⚡ PIPELINE HEARTBEAT │ {stats}")

            time.sleep(1)

    except KeyboardInterrupt:
        print(f"\n{BOLD}Monitor stopped.{RESET} {record_count} total records seen.")


if __name__ == "__main__":
    main()