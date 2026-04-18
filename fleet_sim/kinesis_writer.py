"""
GridSentinel — Kinesis Writer
================================
Async Kinesis producer for fleet telemetry.

Partitioning strategy:
  All records from one depot use the same partition key (f"depot_{id}").
  This guarantees all buses from one depot land on the same shard,
  enabling in-order per-depot processing by the alignment module.

At 100 buses × 5s intervals = 20 records/second per depot.
One Kinesis shard handles 1,000 records/second → 50× headroom per depot.

Retry: adaptive backoff via botocore.config (max 3 attempts).
Failed records in a PutRecords batch are logged and counted as a metric.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import aioboto3
from botocore.config import Config

from config.settings import settings

log = logging.getLogger(__name__)

# Kinesis hard limits
MAX_BATCH_SIZE = 500  # PutRecords max records per call
MAX_RECORD_BYTES = 1_000_000  # 1 MB per record


class KinesisWriter:
    """
    Async Kinesis producer. One instance shared across all depots.

    Usage:
        writer = KinesisWriter()
        await writer.start()
        await writer.put_records(records, partition_key)
        await writer.stop()
    """

    def __init__(self) -> None:
        cfg = settings.kinesis
        self._stream_name = cfg.stream_name
        self._endpoint_url = "http://localstack:4566"
        self._session = aioboto3.Session()
        self._client: Any = None

    async def start(self) -> None:
        """
        Initialise the boto3 async client and wait for the stream to be ACTIVE.

        LocalStack's healthcheck passes before the init script finishes
        creating the Kinesis stream. This loop retries until the stream
        exists and is ACTIVE — so fleet-sim never crashes on a race condition.
        """
        self._client = await self._session.client(
            "kinesis",
            endpoint_url=self._endpoint_url,
            region_name="eu-west-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
            config=Config(retries={"max_attempts": 3, "mode": "adaptive"}),
        ).__aenter__()
        log.info(
            "KinesisWriter connected | stream=%s | endpoint=%s",
            self._stream_name,
            self._endpoint_url,
        )
        await self._wait_for_stream()

    async def stop(self) -> None:
        """Close the Kinesis client gracefully."""
        if self._client:
            await self._client.__aexit__(None, None, None)
            log.info("KinesisWriter stopped.")

    async def put_records(
        self,
        records: list[Any],
        partition_key: str,
    ) -> None:
        """
        Emit a batch of records to Kinesis.
        """
        if not records or self._client is None:
            return

        kinesis_records = []
        for record in records:
            try:
                # Unpack Pydantic models safely before standard JSON serialization
                if hasattr(record, "model_dump"):
                    safe_record = record.model_dump()
                elif hasattr(record, "dict"):
                    safe_record = record.dict()
                else:
                    safe_record = record

                payload = json.dumps(safe_record, default=_json_default).encode("utf-8")

                if len(payload) > MAX_RECORD_BYTES:
                    log.warning(
                        "Record exceeds 1MB limit — skipped | key=%s | size=%d",
                        partition_key,
                        len(payload),
                    )
                    continue

                kinesis_records.append(
                    {
                        "Data": payload,
                        "PartitionKey": partition_key,
                    }
                )
            except Exception as e:
                log.error(f"Failed to serialize record: {e}")
                continue

        # Split into batches of ≤ 500 (Kinesis PutRecords limit)
        for i in range(0, len(kinesis_records), MAX_BATCH_SIZE):
            batch = kinesis_records[i : i + MAX_BATCH_SIZE]
            await self._put_batch(batch, partition_key)

    async def _wait_for_stream(
        self,
        max_wait_s: int = 60,
        poll_interval_s: int = 3,
    ) -> None:
        """
        Poll until the stream is ACTIVE or max_wait_s exceeded.

        Handles the LocalStack race condition where the healthcheck passes
        before the init script finishes creating the stream.
        """
        import asyncio

        waited = 0
        while waited < max_wait_s:
            try:
                resp = await self._client.describe_stream_summary(StreamName=self._stream_name)
                status = resp["StreamDescriptionSummary"]["StreamStatus"]
                if status == "ACTIVE":
                    log.info("Stream %s is ACTIVE — ready to write.", self._stream_name)
                    return
                log.info("Stream status: %s — waiting...", status)
            except Exception:
                log.info(
                    "Stream %s not found yet — retrying in %ds...",
                    self._stream_name,
                    poll_interval_s,
                )
            await asyncio.sleep(poll_interval_s)
            waited += poll_interval_s

        raise RuntimeError(
            f"Stream '{self._stream_name}' not ACTIVE after {max_wait_s}s. "
            "Check LocalStack init script."
        )

    async def _put_batch(
        self,
        batch: list[dict],
        partition_key: str,
    ) -> None:
        """Internal: send one batch of ≤ 500 records."""
        try:
            response = await self._client.put_records(
                StreamName=self._stream_name,
                Records=batch,
            )
            failed_count = response.get("FailedRecordCount", 0)

            if failed_count:
                log.warning(
                    "Kinesis partial failure | key=%s | failed=%d/%d",
                    partition_key,
                    failed_count,
                    len(batch),
                )
            else:
                log.debug(
                    "Kinesis OK | key=%s | records=%d",
                    partition_key,
                    len(batch),
                )

        except Exception as exc:
            log.error(
                "Kinesis put_records error | key=%s | error=%s",
                partition_key,
                exc,
            )


def _json_default(obj: Any) -> str:
    """JSON serialiser for datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
