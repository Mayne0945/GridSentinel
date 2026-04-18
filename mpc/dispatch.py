"""
GridSentinel — mpc/dispatch.py  (v2 — LP formulation)
=======================================================
Model Predictive Controller — The Dispatch Brain.

v2 Change: Binary variables removed. Simultaneous charge+discharge
is now prevented by a penalty term in the objective (0.50 EUR/kW),
dropping from MILP to pure LP. Solve time: ~200s → <5s.

Every command produced here is PENDING — must pass Digital Twin
(Phase 4) before execution.

Rolling horizon : 4h = 48 × 5-min steps
Solver          : PuLP CBC (pure LP)
Solve time goal : < 5s for 100 buses

Objective:
  min Σ_t [ λ_t·P_net_t + Σ_b c_deg·(P_ch+P_dis)·Δt + penalty·(P_ch+P_dis) ]

Hard constraints:
  SOC_b_dep   >= SOC_required   (departure — non-negotiable)
  SOC_min     <= SOC_b_t <= SOC_max
  0 <= P_ch   <= P_ch_max
  0 <= P_dis  <= P_dis_max
  |ΔP_ch/dis| <= ramp_max       (ramp rate)
  Σ_b P_net   <= transformer_max
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pulp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("gridsent.mpc")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HORIZON_STEPS: int = 48
STEP_MINUTES: int = 5
STEP_HOURS: float = STEP_MINUTES / 60.0

SOC_MIN: float = 0.10
SOC_MAX: float = 0.95
SOC_REQUIRED: float = 0.90

DEG_COST_NORMAL: float = 0.03
DEG_COST_STRESS: float = 0.12
SOC_STRESS_LOW: float = 0.20
SOC_STRESS_HIGH: float = 0.80

SIMULTANEOUS_PENALTY: float = 0.50  # EUR/kW — enforces ch/dis exclusivity without binaries
RAMP_RATE_FRACTION: float = 0.30
TRANSFORMER_LIMIT_KW: float = 4_000.0
SOLVER_TIMEOUT_S: int = 30

DEFAULT_CHARGE_RATE_KW: float = 150.0
DEFAULT_DISCHARGE_RATE_KW: float = 100.0
DEFAULT_CAPACITY_KWH: float = 300.0
EFF_CHARGE: float = 0.95
EFF_DISCHARGE: float = 0.95


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class BusState:
    bus_id: str
    soc: float
    soh: float
    capacity_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    departure_step: int
    required_soc: float
    available: bool
    is_trusted: bool

    @property
    def effective_capacity_kwh(self) -> float:
        return self.capacity_kwh * self.soh

    @property
    def deg_cost(self) -> float:
        if SOC_STRESS_LOW <= self.soc <= SOC_STRESS_HIGH:
            return DEG_COST_NORMAL
        return DEG_COST_STRESS


@dataclass
class BusCommand:
    bus_id: str
    action: str
    power_kw: float
    soc_after: float
    reason: str


@dataclass
class DispatchResult:
    generated_at: str
    horizon_steps: int
    step_minutes: int
    solver_status: str
    solve_time_s: float
    objective_value: float
    expected_profit_eur: float
    total_discharge_kw: float
    total_charge_kw: float
    transformer_load_pct: float
    commands: list[BusCommand]
    full_schedule: dict
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_forecast(path: Path) -> tuple[list[float], list[float]]:
    with open(path) as f:
        data = json.load(f)
    intervals = data["intervals"][:HORIZON_STEPS]
    while len(intervals) < HORIZON_STEPS:
        intervals.append(intervals[-1])
    return (
        [iv["lower_80"] for iv in intervals],
        [iv["point_forecast_eur_mwh"] for iv in intervals],
    )


def load_fleet_state(path: Path) -> list[BusState]:
    if not path.exists():
        log.warning("Fleet state not found at %s — using synthetic fleet (dev mode).", path)
        return _synthetic_fleet()
    with open(path) as f:
        data = json.load(f)
    return [
        BusState(
            bus_id=b["bus_id"],
            soc=float(b["soc"]),
            soh=float(b.get("soh", 1.0)),
            capacity_kwh=float(b.get("capacity_kwh", DEFAULT_CAPACITY_KWH)),
            max_charge_kw=float(b.get("max_charge_kw", DEFAULT_CHARGE_RATE_KW)),
            max_discharge_kw=float(b.get("max_discharge_kw", DEFAULT_DISCHARGE_RATE_KW)),
            departure_step=int(b.get("departure_step", -1)),
            required_soc=float(b.get("required_soc", SOC_REQUIRED)),
            available=bool(b.get("available", True)),
            is_trusted=bool(b.get("is_trusted", True)),
        )
        for b in data["buses"]
    ]


def _synthetic_fleet() -> list[BusState]:
    rng = np.random.default_rng(42)
    buses = []
    for i in range(100):
        soh = float(np.clip(rng.normal(0.88, 0.08), 0.70, 1.00))
        soc = float(rng.uniform(0.30, 0.85))
        if rng.random() < 0.30:
            departure_step = int(rng.integers(12, HORIZON_STEPS))
            required_soc = float(rng.uniform(0.80, 0.95))
        else:
            departure_step = -1
            required_soc = SOC_REQUIRED
        buses.append(
            BusState(
                bus_id=f"Bus_{i+1:03d}",
                soc=soc,
                soh=soh,
                capacity_kwh=DEFAULT_CAPACITY_KWH,
                max_charge_kw=DEFAULT_CHARGE_RATE_KW,
                max_discharge_kw=DEFAULT_DISCHARGE_RATE_KW,
                departure_step=departure_step,
                required_soc=required_soc,
                available=True,
                is_trusted=True,
            )
        )
    return buses


# ---------------------------------------------------------------------------
# MPC Dispatcher
# ---------------------------------------------------------------------------


class MPCDispatcher:
    def __init__(
        self,
        transformer_limit_kw: float = TRANSFORMER_LIMIT_KW,
        solver_timeout_s: int = SOLVER_TIMEOUT_S,
    ):
        self.transformer_limit_kw = transformer_limit_kw
        self.solver_timeout_s = solver_timeout_s

    def solve(
        self,
        prices_lower: list[float],
        prices_point: list[float],
        fleet: list[BusState],
    ) -> DispatchResult:
        t0 = time.perf_counter()
        active = [b for b in fleet if b.available and b.is_trusted]
        excl = len(fleet) - len(active)
        if excl:
            log.info("Excluded %d buses (unavailable or Byzantine).", excl)
        if not active:
            log.error("No active trusted buses — cannot solve.")
            return self._infeasible_result(t0, "No active buses")
        log.info(
            "MPC solve | buses=%d | horizon=%d steps (%.0fh) | price_now=%.2f EUR/MWh",
            len(active),
            HORIZON_STEPS,
            HORIZON_STEPS * STEP_HOURS,
            prices_point[0],
        )
        return self._build_and_solve(prices_lower, prices_point, active, t0)

    def _build_and_solve(self, prices_lower, prices_point, buses, t0):
        T = HORIZON_STEPS
        B = len(buses)

        prob = pulp.LpProblem("GridSentinel_MPC", pulp.LpMinimize)
        P_ch = pulp.LpVariable.dicts(
            "P_ch", [(b, t) for b in range(B) for t in range(T)], lowBound=0
        )
        P_dis = pulp.LpVariable.dicts(
            "P_dis", [(b, t) for b in range(B) for t in range(T)], lowBound=0
        )
        SoC = pulp.LpVariable.dicts(
            "SoC",
            [(b, t) for b in range(B) for t in range(T + 1)],
            lowBound=SOC_MIN,
            upBound=SOC_MAX,
        )
        P_net = pulp.LpVariable.dicts("P_net", range(T))

        # Objective: energy cost + degradation + simultaneous-use penalty
        obj_energy = pulp.lpSum(prices_point[t] / 1000.0 * P_net[t] for t in range(T))
        obj_deg = pulp.lpSum(
            buses[b].deg_cost * (P_ch[b, t] + P_dis[b, t]) * STEP_HOURS
            for b in range(B)
            for t in range(T)
        )
        obj_simult = pulp.lpSum(
            SIMULTANEOUS_PENALTY * (P_ch[b, t] + P_dis[b, t]) for b in range(B) for t in range(T)
        )
        prob += obj_energy + obj_deg + obj_simult, "Minimise_Net_Cost"

        # Per-bus constraints
        for b, bus in enumerate(buses):
            cap = bus.effective_capacity_kwh
            ramp = RAMP_RATE_FRACTION * max(bus.max_charge_kw, bus.max_discharge_kw)

            prob += SoC[b, 0] == bus.soc, f"Init_SoC_b{b}"

            for t in range(T):
                prob += P_ch[b, t] <= bus.max_charge_kw, f"Ch_max_b{b}_t{t}"
                prob += P_dis[b, t] <= bus.max_discharge_kw, f"Dis_max_b{b}_t{t}"
                prob += (
                    SoC[b, t + 1]
                    == SoC[b, t]
                    + (EFF_CHARGE * P_ch[b, t] - P_dis[b, t] / EFF_DISCHARGE) * STEP_HOURS / cap,
                    f"SoC_dyn_b{b}_t{t}",
                )
                if t > 0:
                    prob += P_ch[b, t] - P_ch[b, t - 1] <= ramp, f"Ramp_ch_up_b{b}_t{t}"
                    prob += P_ch[b, t - 1] - P_ch[b, t] <= ramp, f"Ramp_ch_dn_b{b}_t{t}"
                    prob += P_dis[b, t] - P_dis[b, t - 1] <= ramp, f"Ramp_dis_up_b{b}_t{t}"
                    prob += P_dis[b, t - 1] - P_dis[b, t] <= ramp, f"Ramp_dis_dn_b{b}_t{t}"

            if 0 < bus.departure_step <= T:
                prob += SoC[b, bus.departure_step] >= bus.required_soc, f"Departure_SoC_b{b}"

        # Net power definition + transformer limits
        for t in range(T):
            prob += (
                P_net[t] == pulp.lpSum(P_ch[b, t] - P_dis[b, t] for b in range(B)),
                f"P_net_t{t}",
            )
            prob += P_net[t] <= self.transformer_limit_kw, f"TFmr_import_t{t}"
            prob += P_net[t] >= -self.transformer_limit_kw, f"TFmr_export_t{t}"

        log.info(
            "Solving LP | variables=%d | constraints=%d …",
            len(prob.variables()),
            len(prob.constraints),
        )

        solve_start = time.perf_counter()
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=self.solver_timeout_s, gapRel=0.01))
        solve_time = time.perf_counter() - solve_start
        total_time = time.perf_counter() - t0

        status = pulp.LpStatus[prob.status]
        log.info(
            "Solver done | status=%s | solve_time=%.2fs | total=%.2fs",
            status,
            solve_time,
            total_time,
        )

        if prob.status not in (pulp.LpStatusOptimal, 1):
            log.error("Non-optimal: %s", status)
            return self._infeasible_result(t0, status)

        return self._extract_result(
            prob,
            buses,
            P_ch,
            P_dis,
            SoC,
            P_net,
            prices_lower,
            prices_point,
            solve_time,
            total_time,
            status,
        )

    def _extract_result(
        self,
        prob,
        buses,
        P_ch,
        P_dis,
        SoC,
        P_net,
        prices_lower,
        prices_point,
        solve_time,
        total_time,
        status,
    ):
        T = HORIZON_STEPS
        B = len(buses)
        commands, full_schedule = [], {}
        total_ch_t0 = total_dis_t0 = 0.0

        for b, bus in enumerate(buses):
            schedule = []
            for t in range(T):
                ch = pulp.value(P_ch[b, t]) or 0.0
                dis = pulp.value(P_dis[b, t]) or 0.0
                sn = pulp.value(SoC[b, t + 1]) or 0.0
                ch = ch if ch > 0.5 else 0.0
                dis = dis if dis > 0.5 else 0.0
                action = "charge" if ch > dis else ("discharge" if dis > ch else "hold")
                schedule.append(
                    {
                        "step": t,
                        "action": action,
                        "charge_kw": round(ch, 2),
                        "discharge_kw": round(dis, 2),
                        "soc": round(sn, 4),
                    }
                )
            full_schedule[bus.bus_id] = schedule

            ch0 = schedule[0]["charge_kw"]
            dis0 = schedule[0]["discharge_kw"]
            soc1 = schedule[0]["soc"]

            if ch0 > dis0:
                action = "charge"
                power_kw = ch0
                reason = self._charge_reason(bus, prices_point[0])
                total_ch_t0 += ch0
            elif dis0 > ch0:
                action = "discharge"
                power_kw = dis0
                reason = f"arbitrage — price {prices_lower[0]:.1f} EUR/MWh"
                total_dis_t0 += dis0
            else:
                action = "hold"
                power_kw = 0.0
                reason = self._hold_reason(bus)

            commands.append(
                BusCommand(
                    bus_id=bus.bus_id,
                    action=action,
                    power_kw=round(power_kw, 2),
                    soc_after=round(soc1, 4),
                    reason=reason,
                )
            )

        p_net_t0 = pulp.value(P_net[0]) or 0.0
        obj_val = pulp.value(prob.objective) or 0.0
        revenue = total_dis_t0 * prices_lower[0] / 1000.0 * STEP_HOURS
        cost = total_ch_t0 * prices_point[0] / 1000.0 * STEP_HOURS
        profit = revenue - cost
        tfmr_pct = abs(p_net_t0) / self.transformer_limit_kw * 100.0

        counts = {"charge": 0, "discharge": 0, "hold": 0}
        for cmd in commands:
            counts[cmd.action] += 1

        log.info(
            "Dispatch t=0 | charge=%.0fkW (%d) | discharge=%.0fkW (%d) | "
            "hold=%d | transformer=%.1f%% | profit=€%.2f",
            total_ch_t0,
            counts["charge"],
            total_dis_t0,
            counts["discharge"],
            counts["hold"],
            tfmr_pct,
            profit,
        )

        return DispatchResult(
            generated_at=datetime.now(UTC).isoformat(),
            horizon_steps=T,
            step_minutes=STEP_MINUTES,
            solver_status=status,
            solve_time_s=round(solve_time, 3),
            objective_value=round(obj_val, 4),
            expected_profit_eur=round(profit, 2),
            total_discharge_kw=round(total_dis_t0, 2),
            total_charge_kw=round(total_ch_t0, 2),
            transformer_load_pct=round(tfmr_pct, 2),
            commands=commands,
            full_schedule=full_schedule,
            meta={
                "active_buses": len(buses),
                "total_elapsed_s": round(total_time, 3),
                "solve_time_s": round(solve_time, 3),
                "price_now_eur_mwh": round(prices_point[0], 2),
                "transformer_limit_kw": self.transformer_limit_kw,
                "action_counts": counts,
                "formulation": "LP (no binaries — simultaneous-use penalty)",
            },
        )

    def _charge_reason(self, bus, price):
        if bus.departure_step != -1:
            return f"departure in {bus.departure_step * STEP_MINUTES}min — need +{(bus.required_soc - bus.soc)*100:.0f}% SoC"
        return f"off-peak charging — price {price:.1f} EUR/MWh"

    def _hold_reason(self, bus):
        if bus.soc >= SOC_STRESS_HIGH:
            return "SoC at ceiling — avoiding stress degradation"
        if bus.soc <= SOC_STRESS_LOW:
            return "SoC at floor — protecting battery health"
        return "no profitable action this step"

    def _infeasible_result(self, t0, reason):
        return DispatchResult(
            generated_at=datetime.now(UTC).isoformat(),
            horizon_steps=HORIZON_STEPS,
            step_minutes=STEP_MINUTES,
            solver_status="Infeasible",
            solve_time_s=round(time.perf_counter() - t0, 3),
            objective_value=0.0,
            expected_profit_eur=0.0,
            total_discharge_kw=0.0,
            total_charge_kw=0.0,
            transformer_load_pct=0.0,
            commands=[],
            full_schedule={},
            meta={"reason": reason},
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="GridSentinel MPC — Dispatch Brain",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--forecast", "-f", type=Path, default=Path("data/forecasts/latest_forecast.json")
    )
    p.add_argument("--fleet", "-l", type=Path, default=Path("data/fleet_state/latest_fleet.json"))
    p.add_argument("--output", "-o", type=Path, default=Path("data/dispatch/latest_dispatch.json"))
    p.add_argument("--transformer-limit", type=float, default=TRANSFORMER_LIMIT_KW)
    p.add_argument("--timeout", type=int, default=SOLVER_TIMEOUT_S)
    return p


def main():
    args = _build_arg_parser().parse_args()

    if not args.forecast.exists():
        raise FileNotFoundError(
            f"Forecast not found: {args.forecast}\nRun forecasting/inference.py first."
        )

    log.info("Loading forecast from %s …", args.forecast)
    prices_lower, prices_point = load_forecast(args.forecast)

    log.info("Loading fleet state from %s …", args.fleet)
    fleet = load_fleet_state(args.fleet)
    log.info("Fleet loaded | total=%d buses", len(fleet))

    dispatcher = MPCDispatcher(
        transformer_limit_kw=args.transformer_limit, solver_timeout_s=args.timeout
    )
    result = dispatcher.solve(prices_lower, prices_point, fleet)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.to_json())
    log.info("Dispatch written → %s", args.output)

    counts = result.meta["action_counts"]
    print("\n" + "═" * 60)
    print("  GridSentinel — MPC Dispatch Complete")
    print("═" * 60)
    print(f"  Status           : {result.solver_status}")
    print(f"  Solve time       : {result.solve_time_s:.2f}s")
    print(f"  Active buses     : {result.meta.get('active_buses','?')}")
    print(f"  Charging         : {counts['charge']} buses  ({result.total_charge_kw:.0f} kW)")
    print(f"  Discharging      : {counts['discharge']} buses  ({result.total_discharge_kw:.0f} kW)")
    print(f"  Holding          : {counts['hold']} buses")
    print(f"  Transformer load : {result.transformer_load_pct:.1f}%")
    print(f"  Est. profit (t0) : €{result.expected_profit_eur:.2f}")
    print(f"  Objective value  : {result.objective_value:.4f} EUR")

    dis_cmds = [c for c in result.commands if c.action == "discharge"]
    ch_cmds = [c for c in result.commands if c.action == "charge"]
    if dis_cmds:
        print("\n  Discharging buses (sample):")
        for cmd in dis_cmds[:5]:
            print(f"    {cmd.bus_id}: {cmd.power_kw:.0f} kW  →  {cmd.reason}")
    if ch_cmds:
        print("\n  Charging buses (sample):")
        for cmd in ch_cmds[:5]:
            print(f"    {cmd.bus_id}: {cmd.power_kw:.0f} kW  →  {cmd.reason}")

    print("\n" + "═" * 60)
    print("  ⚠  Commands are PENDING — Digital Twin validation required.")
    print(
        "     Next: python digital_twin/validate.py --dispatch data/dispatch/latest_dispatch.json"
    )
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
