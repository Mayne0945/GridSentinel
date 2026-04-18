"""
GridSentinel — Shared Kinesis Client
=======================================
Synchronous boto3 Kinesis client shared by the ENTSO-E and weather
producers. These are poll-based producers (not async event loops),
so sync boto3 is the right tool — no aioboto3 needed here.

The fleet_sim uses aioboto3 because it runs 500 concurrent coroutines.
These producers poll an external API once per interval and emit a handful
of records — synchronous is cleaner and simpler.

Partition key strategy:
  - ENTSO-E and weather records use partition key "shared"
  - This lands them on a single shard alongside their source label
  - The alignment module reads all shards and identifies source by record.source
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

import boto3
from botocore.config import Config

log = logging.getLogger(__name__)


class KinesisClient:
    """
    Thin synchronous Kinesis client wrapper.
    One instance per producer process.
    """

    MAX_BATCH = 500

    def __init__(self) -> None:
        self._stream_name = os.environ.get(
            "KINESIS_STREAM_NAME", "gridsentinel-telemetry"
        )
        endpoint = os.environ.get("AWS_ENDPOINT_URL", "http://localstack:4566")
        region   = os.environ.get("AWS_DEFAULT_REGION", "eu-west-1")

        self._client = boto3.client(
            "kinesis",
            endpoint_url=endpoint,
            region_name=region,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "test"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "test"),
            config=Config(retries={"max_attempts": 3, "mode": "adaptive"}),
        )
        log.info(
            "KinesisClient ready | stream=%s | endpoint=%s",
            self._stream_name, endpoint,
        )

    def wait_for_stream(self, max_wait_s: int = 60, poll_s: int = 3) -> None:
        """
        Block until stream is ACTIVE. Same race-condition guard as kinesis_writer.py.
        Call this once at producer startup.
        """
        import time
        waited = 0
        while waited < max_wait_s:
            try:
                resp   = self._client.describe_stream_summary(
                    StreamName=self._stream_name
                )
                status = resp["StreamDescriptionSummary"]["StreamStatus"]
                if status == "ACTIVE":
                    log.info("Stream %s ACTIVE.", self._stream_name)
                    return
                log.info("Stream status: %s — waiting...", status)
            except Exception:
                log.info("Stream not found yet — retrying in %ds...", poll_s)
            time.sleep(poll_s)
            waited += poll_s
        raise RuntimeError(
            f"Stream '{self._stream_name}' not ACTIVE after {max_wait_s}s."
        )

    def put_record(self, record: dict, partition_key: str) -> None:
        """Emit a single record. Used by producers that emit one record at a time."""
        payload = json.dumps(record, default=_json_default).encode("utf-8")
        try:
            self._client.put_record(
                StreamName=self._stream_name,
                Data=payload,
                PartitionKey=partition_key,
            )
            log.debug("PUT | key=%s | source=%s", partition_key, record.get("source"))
        except Exception as exc:
            log.error("Kinesis put_record error | key=%s | %s", partition_key, exc)

    def put_records(self, records: list[dict], partition_key: str) -> None:
        """Emit a batch of records (max 500 per Kinesis limit)."""
        if not records:
            return
        batch = [
            {
                "Data": json.dumps(r, default=_json_default).encode("utf-8"),
                "PartitionKey": partition_key,
            }
            for r in records
        ]
        for i in range(0, len(batch), self.MAX_BATCH):
            chunk = batch[i : i + self.MAX_BATCH]
            try:
                resp   = self._client.put_records(
                    StreamName=self._stream_name, Records=chunk
                )
                failed = resp.get("FailedRecordCount", 0)
                if failed:
                    log.warning(
                        "Kinesis partial failure | key=%s | failed=%d/%d",
                        partition_key, failed, len(chunk),
                    )
                else:
                    log.debug(
                        "PUT_RECORDS OK | key=%s | count=%d", partition_key, len(chunk)
                    )
            except Exception as exc:
                log.error(
                    "Kinesis put_records error | key=%s | %s", partition_key, exc
                )


def _json_default(obj: Any) -> str:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Not JSON serialisable: {type(obj)}")