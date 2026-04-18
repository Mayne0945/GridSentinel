"""
GridSentinel — Depot Orchestrator
=====================================
One Depot instance = one Docker service (gs-depot-0 through gs-depot-4).

Responsibilities:
  1. Manages N buses (default 100) as concurrent asyncio coroutines.
  2. Emits per-bus BusTelemetry to Kinesis every 5 seconds.
  3. Emits DepotMeterReading (ground-truth aggregate) alongside telemetry.
  4. Exposes inject_byzantine() for chaos toggle integration.
  5. Accepts V2G commands from the MPC coordinator.

The depot meter is the BFT layer's ground-truth anchor:
  - It sums actual (not reported) bus.power_kw
  - If Byzantine buses over-report draw, the meter reveals the true total
  - If Byzantine buses under-report (flatline), the meter shows real draw
"""
from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timezone
from typing import List

from fleet_sim.bus import EVBus
from fleet_sim.kinesis_writer import KinesisWriter
from fleet_sim.models import AttackType, BusTelemetry, DepotMeterReading
from fleet_sim.route_loader import load_routes
from config.settings import settings

log = logging.getLogger(__name__)


class Depot:
    """
    Orchestrates one depot of up to 100 EV buses.

    Each depot maps to:
      - One Docker Compose service (fleet-sim reads DEPOT_ID env var)
      - One Kinesis shard (partition key = f"depot_{depot_id}")
      - One BFT align-bft service consuming that shard
      - One MPC depot solver
      - One Digital Twin instance
    """

    TELEMETRY_INTERVAL_S: int = 5

    def __init__(self, depot_id: int, writer: KinesisWriter) -> None:
        self.depot_id = depot_id
        self.writer   = writer
        self.buses:   List[EVBus] = []
        self._byzantine_ids: set[str] = set()

    # ─── Fleet initialisation ─────────────────────────────────────────────────

    def build_fleet(self) -> None:
        """
        Instantiate all buses for this depot.

        Buses are spread across the full SoH range to represent a realistic
        mixed-age fleet (some new buses at 100% SoH, some older at 78%).
        """
        cfg    = settings.fleet
        deg    = settings.degradation
        routes = load_routes(settings.schedule.source)
        n      = cfg.buses_per_depot

        log.info(
            "Depot %d — building fleet of %d buses (SoH %.0f%%–%.0f%%)",
            self.depot_id,
            n,
            deg.oldest_bus_soh * 100,
            deg.new_bus_soh * 100,
        )

        for i in range(n):
            bus_id = f"d{self.depot_id:02d}_b{i:03d}"

            # Evenly distribute cycle counts across the fleet
            # Bus 0 is the newest (0 cycles), Bus N-1 is the oldest (2000 cycles)
            cycle_count = int(
                deg.cycle_count_range[0]
                + (i / max(n - 1, 1))
                * (deg.cycle_count_range[1] - deg.cycle_count_range[0])
            )

            route = routes[i % len(routes)]
            bus   = EVBus(
                bus_id=bus_id,
                depot_id=self.depot_id,
                route=route,
                cycle_count=cycle_count,
            )
            self.buses.append(bus)

        log.info("Depot %d — fleet built: %d buses ready.", self.depot_id, len(self.buses))

    # ─── Byzantine injection (chaos toggle) ───────────────────────────────────

    def inject_byzantine(
        self,
        attack_type: AttackType,
        pct: float = 0.10,
    ) -> List[str]:
        """
        Inject a Byzantine attack on pct% of this depot's fleet.

        Args:
            attack_type: One of FLATLINE, SPIKE, COORDINATED, REPLAY
            pct:         Fraction of fleet to compromise (default 10%)

        Returns:
            List of bus_ids that were compromised.

        The coordinated attack targets buses simultaneously so they all
        lie in the same direction — designed to fool naive averaging.
        The MAD filter is robust to this because it uses the median, not
        the mean, as its central estimate.
        """
        n_targets = max(1, int(len(self.buses) * pct))
        targets   = random.sample(self.buses, n_targets)
        affected  = []

        for bus in targets:
            bus.inject_byzantine(attack_type)
            self._byzantine_ids.add(bus.bus_id)
            affected.append(bus.bus_id)

        log.warning(
            "Depot %d — Byzantine injection: %s on %d buses (%d%% of fleet)",
            self.depot_id,
            attack_type.value,
            n_targets,
            int(pct * 100),
        )
        return affected

    def clear_byzantine(self) -> None:
        """Remove all Byzantine behaviour from this depot's fleet."""
        for bus in self.buses:
            if bus.bus_id in self._byzantine_ids:
                bus.clear_byzantine()
        cleared = len(self._byzantine_ids)
        self._byzantine_ids.clear()
        log.info("Depot %d — cleared Byzantine state on %d buses.", self.depot_id, cleared)

    # ─── MPC command passthrough ──────────────────────────────────────────────

    def apply_v2g_commands(self, commands: List[dict]) -> None:
        """
        Apply V2G dispatch commands from the MPC depot solver.

        Expected command structure:
          {"bus_id": "d00_b042", "action": "discharge", "power_kw": 85.0}
        """
        bus_map = {b.bus_id: b for b in self.buses}
        for cmd in commands:
            bus = bus_map.get(cmd.get("bus_id"))
            if not bus:
                continue
            action = cmd.get("action")
            if action == "discharge":
                bus.set_v2g_command(cmd.get("power_kw", 0.0))
            elif action in ("charge", "hold"):
                bus.cancel_v2g()

    # ─── Depot meter (ground truth anchor) ───────────────────────────────────

    def _depot_meter_reading(self) -> DepotMeterReading:
        """
        Ground-truth aggregate power at the main grid meter.

        CRITICAL: Uses bus.power_kw (actual internal state), NOT
        the reported value from bus.snapshot(). This is the honest
        measurement the BFT uses to cross-validate charger sensors.

        Byzantine detection via cross-validation:
          If Σ(reported power) >> depot_meter.aggregate_power_kw:
            → Buses are over-reporting draw (spike / coordinated attack)
          If Σ(reported power) ≈ 0 but meter shows real draw:
            → Buses are flatline-attacking (reporting zero while charging)
        """
        # Sum ACTUAL power (positive = charging, negative = V2G/driving)
        # Meter only sees grid-side: charging buses add load, V2G exports reduce it
        actual_grid_kw = sum(b.power_kw for b in self.buses)
        active_chargers = sum(1 for b in self.buses if b.power_kw > 0)

        return DepotMeterReading(
            depot_id=self.depot_id,
            event_timestamp=datetime.now(timezone.utc),
            aggregate_power_kw=round(actual_grid_kw, 2),
            active_chargers=active_chargers,
        )

    # ─── Telemetry emission loop ──────────────────────────────────────────────

    async def _emit_telemetry(self) -> None:
        """
        Every 5 seconds: snapshot all buses + depot meter → Kinesis.

        All records share the same partition key (depot_{id}) so they
        land on the same Kinesis shard and arrive in order to the
        alignment module.
        """
        partition_key = f"depot_{self.depot_id}"

        while True:
            records: list[dict] = []

            for bus in self.buses:
                snap = bus.snapshot()
                records.append(snap.model_dump(mode="json"))

            # Depot meter record always appended last
            meter = self._depot_meter_reading()
            records.append(meter.model_dump(mode="json"))

            await self.writer.put_records(
                records=records,
                partition_key=partition_key,
            )

            await asyncio.sleep(self.TELEMETRY_INTERVAL_S)

    # ─── Main coroutine ───────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Launch all bus coroutines + telemetry emitter.
        Call build_fleet() before run().
        """
        if not self.buses:
            self.build_fleet()

        log.info("Depot %d — starting %d bus coroutines.", self.depot_id, len(self.buses))

        bus_tasks       = [asyncio.create_task(b.run(), name=b.bus_id) for b in self.buses]
        telemetry_task  = asyncio.create_task(self._emit_telemetry(), name=f"depot_{self.depot_id}_emit")

        try:
            await asyncio.gather(*bus_tasks, telemetry_task)
        except asyncio.CancelledError:
            log.info("Depot %d — shutting down.", self.depot_id)
            raise