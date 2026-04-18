"""
GridSentinel — BFT Main
=========================
Entry point for standalone BFT operation.

In production, the BFT Gatekeeper is called directly by consumer_align.py
within the same process — no separate container needed.

This module exists for:
  1. Direct testing: python -m bft.main --depot-id 0
  2. Future: standalone BFT service if the pipeline is split
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-32s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("bft.main")


def main() -> None:
    parser = argparse.ArgumentParser(description="GridSentinel BFT standalone")
    parser.add_argument(
        "--depot-id", type=int,
        default=int(os.environ.get("DEPOT_ID", 0)),
    )
    args = parser.parse_args()

    log.info(
        "BFT standalone mode | depot=%d | "
        "Note: in production BFT runs inside consumer_align process.",
        args.depot_id,
    )

    # In standalone mode, run the full consumer_align pipeline
    # which calls BFTGatekeeper internally
    from ingestion.consumer_align import run
    run(depot_id=args.depot_id)


if __name__ == "__main__":
    main()