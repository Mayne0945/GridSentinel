"""
GridSentinel — Battery Physics Engine
=======================================
All functions used by the bus state machine and the MPC objective.

Coefficient sources:
  - NASA PCoE Battery Dataset (capacity fade vs cycle count)
  - Argonne National Laboratory ANL-20/38 (degradation cost per kWh)
  - Arrhenius equation (temperature-accelerated degradation)
  - IEC 61851 CC-CV charging curve model

Units:
  - Energy  → kWh
  - Power   → kW
  - SoC/SoH → % (0–100)
  - Temperature → °C
  - Cost    → USD/kWh (for MPC objective)
"""

from __future__ import annotations

from config.settings import settings

# ─── Drive cycle energy consumption ──────────────────────────────────────────

# Base energy consumption for a fully-loaded double-decker e-bus
# at 20°C, flat road, nominal SoH.  Source: TFL e-bus trial data (2022).
BASE_CONSUMPTION_KWH_PER_KM: float = 1.50

# Gravitational potential → kinetic energy for climb/descent
# E = m·g·h  →  300kg × 9.81 / 1000 × 3600⁻¹  ≈ 0.0027 kWh/m
ELEVATION_PENALTY_KWH_PER_M: float = 0.0027

# Regenerative braking recovery efficiency (modern e-bus: ~55–65%)
REGENERATION_FACTOR: float = 0.60


def temperature_efficiency_factor(ambient_c: float) -> float:
    """
    Energy consumption multiplier due to ambient temperature.

    Cold batteries have higher internal resistance → more energy wasted as heat.
    Hot batteries run close to nominal efficiency but suffer faster degradation
    (modelled separately in degradation_cost_per_kwh).

    Returns a multiplier applied to base energy consumption:
      > 1.0 → less efficient → more kWh consumed per km
      = 1.0 → nominal
    """
    if ambient_c < 0.0:
        return 1.35  # Severe cold — significant range penalty, cabin heating load
    elif ambient_c < 10.0:
        return 1.20
    elif ambient_c < 20.0:
        return 1.08
    elif ambient_c <= 35.0:
        return 1.00  # Nominal operating window
    elif ambient_c <= 45.0:
        return 1.05  # Mild overconsumption + HVAC cooling load
    else:
        return 1.12  # High ambient — aggressive thermal management draw


def energy_consumed_kwh(
    distance_km: float,
    elevation_gain_m: float,
    elevation_loss_m: float,
    ambient_c: float,
    soh: float,  # 0.0–1.0 (not %)
) -> float:
    """
    Net energy consumed for a route leg.

    SoH affects *usable capacity* (how much range remains), not
    energy-per-km efficiency directly — a degraded bus needs the
    same kWh to move the same mass the same distance, but has
    less reserve to draw from. The MPC accounts for this via
    SOC bounds on the degraded battery.

    Args:
        distance_km:      Route leg distance
        elevation_gain_m: Total ascent metres (climb cost)
        elevation_loss_m: Total descent metres (regen recovery)
        ambient_c:        Ambient temperature
        soh:              State of Health 0.0–1.0

    Returns:
        Net kWh consumed (always positive — energy out of battery)
    """
    base = distance_km * BASE_CONSUMPTION_KWH_PER_KM
    climb_cost = elevation_gain_m * ELEVATION_PENALTY_KWH_PER_M
    regen = elevation_loss_m * ELEVATION_PENALTY_KWH_PER_M * REGENERATION_FACTOR
    temp_adj = temperature_efficiency_factor(ambient_c)

    return max(0.0, (base + climb_cost - regen) * temp_adj)


# ─── Charging curve ───────────────────────────────────────────────────────────


def charging_power_kw(soc_pct: float, max_charge_rate_kw: float) -> float:
    """
    Non-linear CC-CV charging curve (IEC 61851 model).

    Constant-Current (CC) phase: full rate up to ~80% SoC.
    Constant-Voltage (CV) phase: tapering from 80% → 100%.

    This is why charging the last 20% takes as long as the first 80%.
    The MPC uses this to avoid scheduling buses for V2G when they're
    already in the CV phase — the charge rate is too slow to be worth it.

    Returns:
        Power accepted by the battery in kW (positive)
    """
    if soc_pct >= 98.0:
        return 0.0  # Fully charged
    elif soc_pct >= 90.0:
        return max_charge_rate_kw * 0.08  # CV trickle
    elif soc_pct >= 80.0:
        # Linear taper: 100% at 80% SoC → 8% at 90% SoC
        taper = 1.0 - ((soc_pct - 80.0) / 10.0) * 0.92
        return max_charge_rate_kw * taper
    else:
        return max_charge_rate_kw  # Full CC rate


# ─── Degradation model (used by both bus physics and MPC objective) ───────────


def degradation_cost_per_kwh(soc_pct: float, ambient_c: float) -> float:
    """
    Battery degradation cost in USD/kWh throughput.

    This is the same function used in the MPC objective:

        c_deg(SoC, T) = c_deg_base(SoC) × temperature_multiplier(T)

    Base cost (Argonne ANL degradation studies):
      - Mid-range cycling (20–80% SoC): ~$0.03/kWh
      - Full-depth cycling (0–100% SoC): ~$0.12/kWh

    Temperature multiplier (Arrhenius-informed, from fleet.yaml):
      - < 15°C: 1.2  (cold → higher internal resistance → faster fade)
      - 15–35°C: 1.0 (nominal)
      - 35–45°C: 1.4 (accelerated degradation onset)
      - > 45°C: 2.0  (aggressive zone — deep cycling here is expensive)

    Effect on MPC: on a hot Johannesburg afternoon (35°C+), the objective
    function naturally penalises deep V2G discharge without any hardcoded
    temperature rule. The math does the work.

    Returns:
        USD per kWh of battery throughput
    """
    cfg = settings.degradation
    tm = cfg.temp_multiplier

    # Piecewise base cost
    base_cost = 0.03 if 20.0 <= soc_pct <= 80.0 else 0.12

    # Temperature multiplier from config
    if ambient_c < 15.0:
        multiplier = tm.below_15c
    elif ambient_c <= 35.0:
        multiplier = tm.range_15_35c
    elif ambient_c <= 45.0:
        multiplier = tm.range_35_45c
    else:
        multiplier = tm.above_45c

    return base_cost * multiplier


def soh_fade_per_full_cycle(
    depth_of_discharge: float,  # 0.0–1.0 (fraction of full capacity)
    ambient_c: float,
    current_soh: float,  # 0.0–1.0
) -> float:
    """
    SoH capacity fade for one full equivalent cycle.

    Combines:
      - Linear base fade: 1.0 → 0.78 over 2000 nominal cycles (NASA PCoE)
      - Wöhler depth-of-discharge exponent: shallower cycles → longer life
      - Arrhenius temperature acceleration

    Args:
        depth_of_discharge: DoD as fraction (0.6 = 60% swing)
        ambient_c:          Temperature during cycle
        current_soh:        Current SoH (fade accelerates at low SoH)

    Returns:
        SoH fraction to subtract (e.g., 0.00011 per cycle at nominal)
    """
    deg = settings.degradation
    total_nominal_cycles = deg.cycle_count_range[1]

    # Base fade per full cycle at nominal conditions
    soh_range = deg.new_bus_soh - deg.oldest_bus_soh  # 0.22
    base_fade = soh_range / total_nominal_cycles  # ≈ 0.00011

    # Wöhler exponent: DoD^1.5 means deep cycles hurt disproportionately
    # Full DoD (1.0) → 1.0× penalty; 50% DoD → ~0.35× penalty
    dod_factor = depth_of_discharge**1.5

    # Arrhenius temperature acceleration
    if ambient_c < 15.0:
        temp_factor = 1.15  # Cold degrades slightly faster (Li plating risk)
    elif ambient_c <= 35.0:
        temp_factor = 1.00
    elif ambient_c <= 45.0:
        temp_factor = 1.35
    else:
        temp_factor = 1.80  # High temps → exponential fade onset

    return base_fade * dod_factor * temp_factor
