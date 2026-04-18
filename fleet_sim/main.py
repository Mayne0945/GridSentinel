"""
GridSentinel — Fleet Simulator Entry Point
============================================
Starts all depots as concurrent asyncio tasks.

Each depot instance owns:
  - N buses (default 100) as asyncio coroutines
  - A Kinesis producer partitioned to its shard
  - A telemetry emission loop (every 5 seconds)

In single-depot Docker mode, the DEPOT_ID environment variable
selects which depot this container runs. All depots share the
same stream but different partition keys.

Usage:
    # All depots in one process (dev / local testing):
    python -m fleet_sim

    # Single depot (Docker Compose — one container per depot):
    DEPOT_ID=2 python -m fleet_sim --single-depot
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

from config.settings import settings
from fleet_sim.depot import Depot
from fleet_sim.kinesis_writer import KinesisWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("fleet_sim.main")


async def run_all_depots() -> None:
    """
    Launch all depots in a single process.
    Used for local development / testing.
    """
    cfg = settings.fleet
    log.info(
        "GridSentinel Fleet Simulator | %d depots × %d buses = %d total",
        cfg.depots,
        cfg.buses_per_depot,
        cfg.size,
    )

    writer = KinesisWriter()
    await writer.start()

    depots: list[Depot] = [Depot(depot_id=i, writer=writer) for i in range(cfg.depots)]

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _handle_signal() -> None:
        log.info("Shutdown signal received — stopping gracefully...")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_signal)

    tasks = [asyncio.create_task(d.run(), name=f"depot_{d.depot_id}") for d in depots]

    log.info("All depot tasks launched. Fleet is live.")
    await stop_event.wait()

    log.info("Cancelling depot tasks...")
    for task in tasks:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)
    await writer.stop()
    log.info("Fleet simulator stopped cleanly.")


async def run_single_depot(depot_id: int) -> None:
    """
    Launch a single depot.
    Used in Docker Compose mode where each container = one depot.
    """
    cfg = settings.fleet
    log.info(
        "GridSentinel Fleet Simulator | Depot %d | %d buses",
        depot_id,
        cfg.buses_per_depot,
    )

    writer = KinesisWriter()
    await writer.start()

    depot = Depot(depot_id=depot_id, writer=writer)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _handle_signal() -> None:
        log.info("Depot %d — shutdown signal received.", depot_id)
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_signal)

    task = asyncio.create_task(depot.run(), name=f"depot_{depot_id}")

    log.info("Depot %d live. Waiting for stop signal.", depot_id)
    await stop_event.wait()

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await writer.stop()
    log.info("Depot %d stopped cleanly.", depot_id)


def main() -> None:
    single_depot_mode = "--single-depot" in sys.argv
    depot_id_env = os.environ.get("DEPOT_ID")

    if single_depot_mode and depot_id_env is not None:
        try:
            depot_id = int(depot_id_env)
        except ValueError:
            log.error("DEPOT_ID env var must be an integer, got: %s", depot_id_env)
            sys.exit(1)
        asyncio.run(run_single_depot(depot_id))
    else:
        asyncio.run(run_all_depots())


if __name__ == "__main__":
    main()
