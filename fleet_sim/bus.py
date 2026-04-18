"""
GridSentinel — EVBus State Machine
=====================================
Each bus runs as an asyncio coroutine progressing through:

    IDLE → DRIVING → IDLE → CHARGING → (repeat)
    IDLE → GRID_EXPORT (when MPC dispatches V2G command)

Byzantine injection corrupts only the *reported* values in BusTelemetry.
The bus's internal state (self.soc_pct, self.power_kw) remains honest.
This is the invariant the BFT layer detects against.

The depot meter sums actual (not reported) bus.power_kw values —
if Byzantine buses report high power draw but actual draw is normal,
the depot meter exposes the lie.
"""

from __future__ import annotations

import asyncio
import math
import random
from collections import deque
from datetime import UTC, datetime

from config.settings import settings
from fleet_sim.battery_physics import (
    charging_power_kw,
    energy_consumed_kwh,
    soh_fade_per_full_cycle,
)
from fleet_sim.models import AttackType, BusStatus, BusTelemetry

# ─── Depot home coordinates (JHB depots) ─────────────────────────────────────
# Real Johannesburg locations aligned with Rea Vaya BRT corridor

DEPOT_HOME: list[tuple[float, float]] = [
    (-26.2041, 28.0473),  # 0 — Johannesburg Park Station (main BRT hub)
    (-26.1929, 28.0305),  # 1 — Braamfontein Civic Centre
    (-26.2300, 28.0126),  # 2 — Nasrec / Soweto Interchange
    (-26.1067, 28.0568),  # 3 — Sandton Interchange
    (-26.2485, 28.1210),  # 4 — East Rand Terminus
]


class EVBus:
    """
    Single EV bus. Runs as a long-lived asyncio coroutine.

    State machine:
        IDLE  →  DRIVING  →  IDLE  →  CHARGING  →  IDLE ...
        IDLE  →  GRID_EXPORT  (injected by MPC via set_v2g_command)

    The bus holds honest internal state. Byzantine corruption only
    affects snapshot() — the values written to Kinesis.
    """

    TELEMETRY_INTERVAL_S: int = 5

    def __init__(
        self,
        bus_id: str,
        depot_id: int,
        route: dict,
        cycle_count: int,
    ) -> None:
        self.bus_id = bus_id
        self.depot_id = depot_id
        self.route = route

        cfg = settings.fleet
        deg = settings.degradation

        # ── Battery state (ground truth — never corrupted) ────────────────
        self.battery_capacity_kwh: float = cfg.battery_capacity_kwh
        self.soc_pct: float = random.uniform(55.0, 90.0)
        self.soh_pct: float = (
            deg.new_bus_soh
            - (cycle_count / max(deg.cycle_count_range[1], 1))
            * (deg.new_bus_soh - deg.oldest_bus_soh)
        ) * 100.0

        self.max_charge_rate_kw: float = cfg.max_charge_rate_kw
        self.max_discharge_rate_kw: float = cfg.max_charge_rate_kw * 0.90  # V2G slightly lower
        self.power_kw: float = 0.0  # +ve = charging, -ve = discharging/driving

        # ── Operational state ─────────────────────────────────────────────
        self.status: BusStatus = BusStatus.IDLE
        self.odometer_km: float = 0.0
        self.required_soc_at_departure: float = 85.0  # % — set by scheduler

        # ── Location (starts at home depot) ──────────────────────────────
        home = DEPOT_HOME[depot_id % len(DEPOT_HOME)]
        self.location_lat: float = home[0]
        self.location_lon: float = home[1]

        # ── Byzantine injection ───────────────────────────────────────────
        self._is_byzantine: bool = False
        self._attack_type: AttackType | None = None
        self._flatline_soc: float | None = None
        # Replay buffer: stores (soc_pct, power_kw) tuples from 24h ago
        # 288 windows = 24h × 12 five-minute windows/h
        self._replay_buffer: deque[tuple[float, float]] = deque(maxlen=288)

        # ── MPC V2G command ───────────────────────────────────────────────
        self._v2g_power_kw: float = 0.0  # Set by MPC coordinator
        self._v2g_active: bool = False

    # ─── Byzantine injection API ──────────────────────────────────────────────

    def inject_byzantine(self, attack_type: AttackType) -> None:
        """
        Activate Byzantine behaviour. Does NOT alter internal state.
        Only snapshot() returns corrupted values.
        """
        self._is_byzantine = True
        self._attack_type = attack_type
        if attack_type == AttackType.FLATLINE:
            # Freeze reported SoC at its current value
            self._flatline_soc = self.soc_pct

    def clear_byzantine(self) -> None:
        self._is_byzantine = False
        self._attack_type = None
        self._flatline_soc = None

    # ─── MPC command injection ────────────────────────────────────────────────

    def set_v2g_command(self, power_kw: float) -> None:
        """
        MPC coordinator injects a V2G discharge command (negative kW).
        Bus will export this power until cancelled or SoC hits minimum.
        """
        self._v2g_power_kw = abs(power_kw)
        self._v2g_active = True

    def cancel_v2g(self) -> None:
        self._v2g_active = False
        self._v2g_power_kw = 0.0

    # ─── Reported values (may be Byzantine) ──────────────────────────────────

    def _reported_soc(self) -> float:
        """Returns SoC to write to Kinesis — may be corrupted."""
        if not self._is_byzantine:
            return round(self.soc_pct, 2)

        if self._attack_type == AttackType.FLATLINE:
            # Bus appears stuck at its last honest SoC reading
            return round(self._flatline_soc or self.soc_pct, 2)

        if self._attack_type == AttackType.SPIKE:
            # Reports an impossibly high SoC jump
            return min(100.0, round(self.soc_pct + random.uniform(35.0, 55.0), 2))

        if self._attack_type == AttackType.COORDINATED:
            # All coordinated buses report same inflated value — designed to
            # fool simple cluster averaging. MAD filter catches this.
            return min(100.0, round(self.soc_pct + 48.0, 2))

        if self._attack_type == AttackType.REPLAY:
            if self._replay_buffer:
                old_soc, _ = random.choice(list(self._replay_buffer))
                return round(old_soc, 2)

        return round(self.soc_pct, 2)

    def _reported_power_kw(self) -> float:
        """Returns power draw to write to Kinesis — may be corrupted."""
        if not self._is_byzantine:
            return round(self.power_kw, 2)

        if self._attack_type == AttackType.SPIKE:
            return round(self.power_kw * 10.0, 2)

        if self._attack_type == AttackType.COORDINATED:
            return round(self.power_kw * 8.0, 2)

        if self._attack_type == AttackType.REPLAY:
            if self._replay_buffer:
                _, old_power = random.choice(list(self._replay_buffer))
                return round(old_power, 2)

        if self._attack_type == AttackType.FLATLINE:
            # Flatline reports zero power even when charging
            return 0.0

        return round(self.power_kw, 2)

    # ─── Ambient temperature (diurnal model) ──────────────────────────────────

    @staticmethod
    def _ambient_temperature_c() -> float:
        """
        Johannesburg diurnal temperature model.
        JHB summer: ~14°C nights, ~28°C afternoons.
        JHB is at 1750m elevation — moderate temperatures year-round.

        Uses a sinusoidal model: T(h) = 21 + 9·sin(π·(h−6)/12)
        Peak ~30°C at 14:00 local, trough ~12°C at 02:00 local.
        UTC offset: +2 hours (SAST)
        """
        hour_utc = datetime.now(UTC).hour
        hour_local = (hour_utc + 2) % 24
        return 21.0 + 9.0 * math.sin(math.pi * (hour_local - 6) / 12.0)

    # ─── State machine steps ──────────────────────────────────────────────────

    async def _step_drive(self) -> None:
        """
        Simulate one route leg (out-and-back).
        Blocks for the real route duration (with traffic variance).
        """
        route = self.route
        dt_s = self.TELEMETRY_INTERVAL_S

        distance_km = route["distance_km"]
        elevation_gain_m = route.get("elevation_gain_m", 25.0)
        elevation_loss_m = route.get("elevation_loss_m", 25.0)
        lat_delta = route.get("lat_delta", 0.10)

        # Traffic variance ±15%
        travel_time_min = route["duration_min"] * random.uniform(0.85, 1.15)
        total_s = travel_time_min * 60.0

        ambient_c = self._ambient_temperature_c()
        self.status = BusStatus.DRIVING

        total_energy_kwh = energy_consumed_kwh(
            distance_km,
            elevation_gain_m,
            elevation_loss_m,
            ambient_c,
            self.soh_pct / 100.0,
        )
        energy_per_step_kwh = total_energy_kwh * (dt_s / total_s)
        soc_at_departure = self.soc_pct
        elapsed_s = 0.0

        home = DEPOT_HOME[self.depot_id % len(DEPOT_HOME)]

        while elapsed_s < total_s:
            soc_delta = (energy_per_step_kwh / self.battery_capacity_kwh) * 100.0
            self.soc_pct = max(10.0, self.soc_pct - soc_delta)
            self.power_kw = -(energy_per_step_kwh * 3600.0 / dt_s)  # kW, negative

            # Linear position interpolation (out and back)
            progress = min(elapsed_s / total_s, 1.0)
            # Out in first half, back in second half
            if progress <= 0.5:
                frac = progress * 2.0
            else:
                frac = (1.0 - progress) * 2.0
            self.location_lat = home[0] + lat_delta * frac
            self.odometer_km += distance_km * (dt_s / total_s)

            # Store honest reading in replay buffer
            self._replay_buffer.append((self.soc_pct, self.power_kw))

            elapsed_s += dt_s
            await asyncio.sleep(dt_s)

        # Apply SoH fade for this cycle
        dod = abs(soc_at_departure - self.soc_pct) / 100.0
        fade = soh_fade_per_full_cycle(dod, ambient_c, self.soh_pct / 100.0)
        self.soh_pct = max(0.0, self.soh_pct - fade * 100.0)

        self.status = BusStatus.IDLE
        self.power_kw = 0.0

    async def _step_charge(self) -> None:
        """Charge bus until required SoC for departure."""
        ambient_c = self._ambient_temperature_c()
        self.status = BusStatus.CHARGING
        dt_s = self.TELEMETRY_INTERVAL_S

        while self.soc_pct < self.required_soc_at_departure:
            charge_kw = charging_power_kw(self.soc_pct, self.max_charge_rate_kw)
            if charge_kw <= 0.0:
                break  # Battery full

            energy_kwh = charge_kw * (dt_s / 3600.0)
            soc_gain = (energy_kwh / self.battery_capacity_kwh) * 100.0
            self.soc_pct = min(100.0, self.soc_pct + soc_gain)
            self.power_kw = charge_kw

            self._replay_buffer.append((self.soc_pct, self.power_kw))
            await asyncio.sleep(dt_s)

        self.status = BusStatus.IDLE
        self.power_kw = 0.0

    async def _step_v2g(self) -> None:
        """
        V2G discharge: export power to grid at MPC-commanded rate.
        Continues until: V2G cancelled, SoC hits minimum, or 5-min window ends.
        """
        cfg = settings.fleet
        soc_min = 20.0  # Never discharge below 20%
        dt_s = self.TELEMETRY_INTERVAL_S
        ambient_c = self._ambient_temperature_c()
        self.status = BusStatus.GRID_EXPORT

        window_steps = 60  # 5-minute window = 60 × 5-second steps

        for _ in range(window_steps):
            if not self._v2g_active or self.soc_pct <= soc_min:
                break

            discharge_kw = min(self._v2g_power_kw, self.max_discharge_rate_kw)
            energy_kwh = discharge_kw * (dt_s / 3600.0)
            soc_loss = (energy_kwh / self.battery_capacity_kwh) * 100.0
            self.soc_pct = max(soc_min, self.soc_pct - soc_loss)
            self.power_kw = -discharge_kw  # Negative = exporting

            self._replay_buffer.append((self.soc_pct, self.power_kw))
            await asyncio.sleep(dt_s)

        self.status = BusStatus.IDLE
        self.power_kw = 0.0

    # ─── Main coroutine ───────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Main bus lifecycle coroutine. Runs indefinitely until cancelled.

        Cycle: charge to required SoC → idle briefly → drive → return → repeat
        V2G commands from MPC interrupt the idle phase.
        """
        # Stagger bus start times to avoid simultaneous charge/departure spikes
        await asyncio.sleep(random.uniform(0, 30))

        while True:
            # Charging phase — ensure bus is ready for departure
            await self._step_charge()

            # Idle before departure (simulate depot dwell time)
            idle_s = random.uniform(60, 600)  # 1–10 min pre-departure idle
            elapsed = 0.0
            while elapsed < idle_s:
                if self._v2g_active:
                    await self._step_v2g()
                await asyncio.sleep(self.TELEMETRY_INTERVAL_S)
                elapsed += self.TELEMETRY_INTERVAL_S

            # Drive route
            await self._step_drive()

            # Post-return idle (turnaround time)
            turnaround_s = random.uniform(300, 900)  # 5–15 min
            elapsed = 0.0
            while elapsed < turnaround_s:
                if self._v2g_active:
                    await self._step_v2g()
                await asyncio.sleep(self.TELEMETRY_INTERVAL_S)
                elapsed += self.TELEMETRY_INTERVAL_S

    # ─── Telemetry snapshot ───────────────────────────────────────────────────

    def snapshot(self) -> BusTelemetry:
        """
        Current bus state as a Kinesis-ready record.
        soc_pct and power_kw may be Byzantine-corrupted.
        soh_pct, ambient_temperature_c, location are always honest.
        """
        now = datetime.now(UTC)
        return BusTelemetry(
            bus_id=self.bus_id,
            depot_id=self.depot_id,
            event_timestamp=now,
            ingestion_timestamp=now,
            soc_pct=self._reported_soc(),
            soh_pct=round(self.soh_pct, 3),
            power_kw=self._reported_power_kw(),
            status=self.status,
            location_lat=round(self.location_lat, 6),
            location_lon=round(self.location_lon, 6),
            odometer_km=round(self.odometer_km, 2),
            ambient_temperature_c=round(self._ambient_temperature_c(), 1),
            is_byzantine=self._is_byzantine,
            attack_type=self._attack_type,
        )
