#!/bin/bash
# scripts/localstack-init.sh
# Runs inside LocalStack container on startup.
# Creates the Kinesis stream with correct shard count from env.

set -e

STREAM_NAME="${KINESIS_STREAM_NAME:-gridsentinel-telemetry}"
SHARD_COUNT="${KINESIS_SHARDS:-5}"
REGION="${DEFAULT_REGION:-eu-west-1}"

echo "[LocalStack Init] Creating Kinesis stream: $STREAM_NAME ($SHARD_COUNT shards)"

awslocal kinesis create-stream \
  --stream-name "$STREAM_NAME" \
  --shard-count "$SHARD_COUNT" \
  --region "$REGION"

echo "[LocalStack Init] Stream created. Waiting for ACTIVE state..."

awslocal kinesis wait stream-exists \
  --stream-name "$STREAM_NAME" \
  --region "$REGION"

echo "[LocalStack Init] ✓ $STREAM_NAME is ACTIVE with $SHARD_COUNT shards"
