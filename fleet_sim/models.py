"""
GridSentinel — Fleet Telemetry Models
======================================
Pydantic v2 models for all messages flowing through the system.

Every record that enters Kinesis carries:
  - source          → identifies producer type
  - event_timestamp → when the event happened (used by watermark alignment)
  - ingestion_timestamp → when it entered the stream

The distinction matters: late-arriving records are accepted into the
correct canonical window via event_timestamp, not wall clock.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

# ─── Enumerations ─────────────────────────────────────────────────────────────


class BusStatus(str, Enum):
    CHARGING = "charging"
    DRIVING = "driving"
    IDLE = "idle"
    GRID_EXPORT = "grid_export"  # Active V2G discharge to grid


class AttackType(str, Enum):
    FLATLINE = "flatline"  # Reports constant SoC regardless of activity
    SPIKE = "spike"  # Reports 10× normal power draw
    COORDINATED = "coordinated"  # Fleet-wide lie in the same direction (fools averaging)
    REPLAY = "replay"  # Replays yesterday's legitimate readings


# ─── Primary telemetry record ─────────────────────────────────────────────────


class BusTelemetry(BaseModel):
    """
    5-second telemetry snapshot from one bus.

    soc_pct and power_kw are the *reported* values — may be Byzantine.
    The BFT layer's job is to detect when these diverge from ground truth.
    """

    # Identity
    bus_id: str
    depot_id: int

    # Timestamps (ISO-8601, UTC)
    event_timestamp: datetime
    ingestion_timestamp: datetime

    # Reported state (potentially corrupted by Byzantine injection)
    soc_pct: float = Field(..., ge=0.0, le=100.0, description="State of Charge %")
    soh_pct: float = Field(..., ge=0.0, le=100.0, description="State of Health %")
    power_kw: float = Field(
        ...,
        description="Net power kW. Positive = charging from grid. Negative = driving/V2G.",
    )
    status: BusStatus

    # Location
    location_lat: float
    location_lon: float
    odometer_km: float

    # Environment (used by MPC temperature-aware degradation cost)
    ambient_temperature_c: float

    # Byzantine metadata (ground truth — not visible to BFT, used only for scoring)
    is_byzantine: bool = False
    attack_type: AttackType | None = None

    # Kinesis routing
    source: str = "fleet_sim"


# ─── Depot meter (ground-truth cross-validation anchor) ──────────────────────


class DepotMeterReading(BaseModel):
    """
    Aggregate power reading from the depot's main grid meter.

    This is the BFT layer's ground-truth anchor. It measures total current
    at the substation connection point — a single, harder-to-compromise
    measurement that the BFT uses to cross-validate charger sensor reports.

    If chargers report low aggregate draw but the depot meter shows high
    current: the charger cluster is lying.
    """

    depot_id: int
    event_timestamp: datetime
    aggregate_power_kw: float = Field(..., description="True grid import kW (honest)")
    active_chargers: int
    source: str = "depot_meter"


# ─── Canonical 5-minute aligned snapshot ─────────────────────────────────────


class CanonicalBusWindow(BaseModel):
    """
    Output of the temporal alignment module — one record per bus
    per 5-minute canonical window.

    Produced by consumer_align.py after aggregating 5-second records.
    This is what the BFT layer and MPC consume.
    """

    canonical_timestamp: datetime
    bus_id: str
    depot_id: int
    mean_soc_pct: float
    mean_soh_pct: float
    sum_power_kwh: float  # Energy in this window (kWh)
    mean_power_kw: float
    status: BusStatus
    ambient_temperature_c: float
    record_count: int  # How many 5-sec records contributed (quality indicator)
    is_byzantine: bool = False
