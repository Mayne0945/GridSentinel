"""
GridSentinel — Central Configuration
========================================
Pydantic v2 settings model. Reads from config/fleet.yaml.

Single source of truth for all services. Every module reads from this —
bus.py, battery_physics.py, bft/main.py, mpc/depot/solver.py, etc.

If you change a value in fleet.yaml, all downstream components pick it up
on next restart — no code changes required.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

# ─── Nested config models ──────────────────────────────────────────────────────


class TempMultiplierConfig(BaseModel):
    """Arrhenius-informed temperature coefficients for degradation cost."""

    below_15c: float = 1.2
    range_15_35c: float = 1.0
    range_35_45c: float = 1.4
    above_45c: float = 2.0


class FleetConfig(BaseModel):
    size: int = 500
    depots: int = 5
    buses_per_depot: int = 100
    battery_capacity_kwh: float = 300.0
    max_charge_rate_kw: float = 150.0
    depot_chargers: int = 40

    @model_validator(mode="after")
    def validate_fleet_size(self) -> FleetConfig:
        if self.size != self.depots * self.buses_per_depot:
            raise ValueError(
                f"fleet.size ({self.size}) must equal "
                f"fleet.depots ({self.depots}) × fleet.buses_per_depot ({self.buses_per_depot})"
            )
        return self


class ScheduleConfig(BaseModel):
    source: str = "tfl"  # "tfl" or "rea_vaya"
    shifts: int = 3
    earliest_departure: str = "05:00"
    latest_return: str = "23:00"


class DegradationConfig(BaseModel):
    new_bus_soh: float = 1.00
    oldest_bus_soh: float = 0.78
    cycle_count_range: list[int] = Field(default=[0, 2000])
    temp_multiplier: TempMultiplierConfig = Field(default_factory=TempMultiplierConfig)


class MpcConfig(BaseModel):
    horizon_steps: int = 48  # 4h at 5-min resolution
    solver_backend: str = "OSQP"
    max_solve_time_s: float = 2.0


class KinesisConfig(BaseModel):
    stream_name: str = "gridsentinel-telemetry"
    shards: int = 5  # Must equal fleet.depots
    late_record_slack_s: int = 10  # Watermark window


class DigitalTwinConfig(BaseModel):
    v_min_pu: float = 0.95
    v_max_pu: float = 1.05
    transformer_limit_pct: float = 0.80


class BftConfig(BaseModel):
    """
    Contextual Trust Ledger parameters.

    Three decay tiers distinguish isolated noise from coordinated attacks:
      - trust_decay_minor:       Isolated spike. Likely transient sensor noise.
      - trust_decay_standard:    Spike that also fails depot meter cross-validation.
      - trust_decay_coordinated: Three or more buses flagged simultaneously.

    Recovery:
      - trust_recovery_per_window: Slow recovery per clean 5-min window.
      - trust_recovery_burst:      Bonus after 5 consecutive clean readings.
    """

    mad_threshold_k: float = 3.0
    trust_decay_minor: float = 0.02
    trust_decay_standard: float = 0.10
    trust_decay_coordinated: float = 0.20
    trust_recovery_per_window: float = 0.01
    trust_recovery_burst: float = 0.05
    trust_blacklist_threshold: float = 0.50


# ─── Root settings ─────────────────────────────────────────────────────────────


class Settings(BaseModel):
    fleet: FleetConfig
    schedule: ScheduleConfig
    degradation: DegradationConfig
    mpc: MpcConfig
    kinesis: KinesisConfig
    digital_twin: DigitalTwinConfig
    bft: BftConfig


# ─── Loader ───────────────────────────────────────────────────────────────────


def _find_config_file() -> Path:
    """Locate fleet.yaml relative to this file or common project roots."""
    candidates = [
        Path(__file__).parent / "fleet.yaml",  # config/fleet.yaml
        Path(__file__).parent.parent / "config" / "fleet.yaml",
        Path("/app/config/fleet.yaml"),  # Docker mount
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "fleet.yaml not found. Expected at config/fleet.yaml "
        "or /app/config/fleet.yaml in Docker."
    )


@lru_cache(maxsize=1)
def _load_settings() -> Settings:
    config_path = _find_config_file()
    with config_path.open("r") as f:
        raw = yaml.safe_load(f)
    return Settings(**raw)


# Module-level singleton — import this everywhere
settings: Settings = _load_settings()


# Add this at the absolute bottom of config/settings.py
def get_config():
    """Adapter for Claude's scripts that expect a get_config() factory."""
    return settings
