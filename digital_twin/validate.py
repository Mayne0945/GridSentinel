"""
GridSentinel - digital_twin/validate.py
Physics-Based Grid Safety Layer - The Digital Twin.

Every MPC dispatch command is PENDING until it passes through here.
No hardware executes a command that has not been physically validated.

Two validation paths:
  1. Linearized DistFlow (every 5-min cycle - fast)
  2. Full AC Newton-Raphson via pandapower (every 30 min - deep check)

Outcomes: APPROVED | CURTAILED | REJECTED

Usage:
    python digital_twin/validate.py --dispatch data/dispatch/latest_dispatch.json
    python digital_twin/validate.py --dispatch data/dispatch/latest_dispatch.json --no-pandapower
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("gridsent.digital_twin")

# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------
V_NOM_KV = 11.0
V_NOM_PU = 1.0
V_MIN_PU = 0.95
V_MAX_PU = 1.05

# Feeder impedances (ohm) - pre-computed from R/X per km x length
R1_OHM = 0.25 * 2.0  # seg1: substation -> mid (2 km)
X1_OHM = 0.10 * 2.0
R2_OHM = 0.25 * 1.5  # seg2: mid -> depot (1.5 km)
X2_OHM = 0.10 * 1.5
R_TOTAL_OHM = R1_OHM + R2_OHM
X_TOTAL_OHM = X1_OHM + X2_OHM

S_BASE_MVA = 10.0
Z_BASE_OHM = (V_NOM_KV**2) / S_BASE_MVA  # 12.1 ohm

TRANSFORMER_MVA = 4.0
TRANSFORMER_KW = TRANSFORMER_MVA * 1000.0

POWER_FACTOR = 0.95
TAN_PHI = float(np.tan(np.arccos(POWER_FACTOR)))

CURTAIL_STEP = 0.10
CURTAIL_MIN_KW = 10.0

AC_CHECK_INTERVAL_S = 1800.0

APPROVED = "APPROVED"
CURTAILED = "CURTAILED"
REJECTED = "REJECTED"


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class CommandValidation:
    bus_id: str
    original_action: str
    original_power_kw: float
    validated_action: str
    validated_power_kw: float
    outcome: str
    reason: str


@dataclass
class ValidationResult:
    generated_at: str
    validation_method: str
    solver_status: str
    validation_time_s: float
    v_depot_pu: float
    v_mid_pu: float
    transformer_load_kw: float
    transformer_load_pct: float
    approved_count: int
    curtailed_count: int
    rejected_count: int
    total_discharge_kw: float
    total_charge_kw: float
    expected_profit_eur: float
    validations: list
    meta: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Digital Twin
# ---------------------------------------------------------------------------


class DigitalTwin:
    def __init__(self, transformer_limit_kw=TRANSFORMER_KW, use_pandapower=True):
        self.transformer_limit_kw = transformer_limit_kw
        self.use_pandapower = use_pandapower
        self._last_ac_check_time = 0.0
        self._pandapower_net = None

        if use_pandapower:
            self._pandapower_net = self._build_pandapower_network()

    def validate(self, dispatch):
        t0 = time.perf_counter()
        commands = dispatch.get("commands", [])
        price_now = dispatch.get("meta", {}).get("price_now_eur_mwh", 0.0)

        log.info("Digital Twin validation | commands=%d", len(commands))

        total_dis = sum(c["power_kw"] for c in commands if c["action"] == "discharge")
        total_ch = sum(c["power_kw"] for c in commands if c["action"] == "charge")
        net_kw = total_ch - total_dis

        log.info(
            "Aggregate | charge=%.0fkW | discharge=%.0fkW | net=%.0fkW", total_ch, total_dis, net_kw
        )

        v_depot_pu, v_mid_pu = self._distflow_voltage(net_kw)
        tfmr_pct = abs(net_kw) / self.transformer_limit_kw * 100.0

        log.info(
            "DistFlow | V_mid=%.4f pu | V_depot=%.4f pu | Transformer=%.1f%%",
            v_mid_pu,
            v_depot_pu,
            tfmr_pct,
        )

        validations, net_kw_safe = self._validate_commands(commands, net_kw, v_depot_pu, v_mid_pu)

        method = "distflow_linear"
        now = time.time()
        if (
            self._pandapower_net is not None
            and (now - self._last_ac_check_time) >= AC_CHECK_INTERVAL_S
        ):
            v_ac_depot, v_ac_mid = self._ac_power_flow(net_kw_safe)
            if v_ac_depot is not None:
                v_depot_pu = v_ac_depot
                v_mid_pu = v_ac_mid
                method = "ac_newton_raphson"
                self._last_ac_check_time = now
                log.info("AC power flow | V_mid=%.4f pu | V_depot=%.4f pu", v_mid_pu, v_depot_pu)

        final_dis = sum(
            v.validated_power_kw for v in validations if v.validated_action == "discharge"
        )
        final_ch = sum(v.validated_power_kw for v in validations if v.validated_action == "charge")
        final_net = final_ch - final_dis

        step_hours = 5 / 60.0
        profit_approx = (
            final_dis * price_now / 1000.0 * step_hours - final_ch * price_now / 1000.0 * step_hours
        )

        counts = {
            APPROVED: sum(1 for v in validations if v.outcome == APPROVED),
            CURTAILED: sum(1 for v in validations if v.outcome == CURTAILED),
            REJECTED: sum(1 for v in validations if v.outcome == REJECTED),
        }

        if counts[REJECTED] == 0 and counts[CURTAILED] == 0:
            solver_status = "Safe"
        elif counts[REJECTED] == len(commands):
            solver_status = "Unsafe"
        else:
            solver_status = "Curtailed"

        elapsed = time.perf_counter() - t0
        log.info(
            "Validation complete | %s | approved=%d | curtailed=%d | rejected=%d | elapsed=%.3fs",
            solver_status,
            counts[APPROVED],
            counts[CURTAILED],
            counts[REJECTED],
            elapsed,
        )

        return ValidationResult(
            generated_at=datetime.now(UTC).isoformat(),
            validation_method=method,
            solver_status=solver_status,
            validation_time_s=round(elapsed, 4),
            v_depot_pu=round(v_depot_pu, 6),
            v_mid_pu=round(v_mid_pu, 6),
            transformer_load_kw=round(abs(final_net), 2),
            transformer_load_pct=round(abs(final_net) / self.transformer_limit_kw * 100, 2),
            approved_count=counts[APPROVED],
            curtailed_count=counts[CURTAILED],
            rejected_count=counts[REJECTED],
            total_discharge_kw=round(final_dis, 2),
            total_charge_kw=round(final_ch, 2),
            expected_profit_eur=round(profit_approx, 2),
            validations=validations,
            meta={
                "original_discharge_kw": round(total_dis, 2),
                "original_charge_kw": round(total_ch, 2),
                "original_net_kw": round(net_kw, 2),
                "v_limits_pu": [V_MIN_PU, V_MAX_PU],
                "transformer_limit_kw": self.transformer_limit_kw,
                "pandapower_available": self._pandapower_net is not None,
            },
        )

    def _distflow_voltage(self, net_kw):
        p_pu = net_kw / 1000.0 / S_BASE_MVA
        q_pu = p_pu * TAN_PHI
        dv1 = (R1_OHM / Z_BASE_OHM * p_pu + X1_OHM / Z_BASE_OHM * q_pu) / V_NOM_PU
        dv2 = (R2_OHM / Z_BASE_OHM * p_pu + X2_OHM / Z_BASE_OHM * q_pu) / V_NOM_PU
        v_mid = V_NOM_PU - dv1
        v_dep = v_mid - dv2
        return v_dep, v_mid

    def _validate_commands(self, commands, net_kw, v_depot_pu, v_mid_pu):
        if abs(net_kw) > self.transformer_limit_kw:
            log.warning(
                "Transformer overload: %.0fkW > %.0fkW - curtailing",
                abs(net_kw),
                self.transformer_limit_kw,
            )
            commands, net_kw = self._curtail_to_transformer_limit(commands, net_kw)
            v_depot_pu, v_mid_pu = self._distflow_voltage(net_kw)

        iterations = 0
        while (v_depot_pu < V_MIN_PU or v_depot_pu > V_MAX_PU) and iterations < 20:
            iterations += 1
            if v_depot_pu < V_MIN_PU:
                log.warning(
                    "Undervoltage: V_depot=%.4f pu - curtailing charge (iter %d)",
                    v_depot_pu,
                    iterations,
                )
                commands, net_kw = self._curtail_action(commands, "charge", net_kw)
            else:
                log.warning(
                    "Overvoltage: V_depot=%.4f pu - curtailing discharge (iter %d)",
                    v_depot_pu,
                    iterations,
                )
                commands, net_kw = self._curtail_action(commands, "discharge", net_kw)
            v_depot_pu, v_mid_pu = self._distflow_voltage(net_kw)

        validations = []
        for cmd in commands:
            orig_power = cmd.get("_original_power_kw", cmd["power_kw"])
            orig_action = cmd.get("_original_action", cmd["action"])
            val_power = cmd["power_kw"]
            val_action = cmd["action"]

            if val_action == "hold" and orig_action != "hold":
                outcome = REJECTED
                reason = f"Rejected: below {CURTAIL_MIN_KW}kW minimum threshold"
            elif abs(val_power - orig_power) > 0.1:
                outcome = CURTAILED
                reason = (
                    f"Curtailed {orig_power:.0f}kW to {val_power:.0f}kW "
                    f"- V_depot={v_depot_pu:.4f} pu"
                )
            else:
                outcome = APPROVED
                reason = (
                    f"Approved - V_depot={v_depot_pu:.4f} pu within [{V_MIN_PU:.2f},{V_MAX_PU:.2f}]"
                )

            validations.append(
                CommandValidation(
                    bus_id=cmd["bus_id"],
                    original_action=orig_action,
                    original_power_kw=round(orig_power, 2),
                    validated_action=val_action,
                    validated_power_kw=round(val_power, 2),
                    outcome=outcome,
                    reason=reason,
                )
            )

        return validations, net_kw

    def _curtail_to_transformer_limit(self, commands, net_kw):
        commands = self._tag_original(commands)
        excess = abs(net_kw) - self.transformer_limit_kw
        dis_cmds = [c for c in commands if c["action"] == "discharge"]
        total_dis = sum(c["power_kw"] for c in dis_cmds)
        if total_dis > 0:
            scale = max(0.0, (total_dis - excess) / total_dis)
            for c in dis_cmds:
                new_p = c["power_kw"] * scale
                c["action"] = "hold" if new_p < CURTAIL_MIN_KW else "discharge"
                c["power_kw"] = 0.0 if new_p < CURTAIL_MIN_KW else new_p
        net_kw = sum(
            c["power_kw"] if c["action"] == "charge" else -c["power_kw"]
            for c in commands
            if c["action"] in ("charge", "discharge")
        )
        return commands, net_kw

    def _curtail_action(self, commands, action, net_kw):
        commands = self._tag_original(commands)
        for c in commands:
            if c["action"] == action:
                new_p = c["power_kw"] * (1.0 - CURTAIL_STEP)
                c["action"] = "hold" if new_p < CURTAIL_MIN_KW else action
                c["power_kw"] = 0.0 if new_p < CURTAIL_MIN_KW else new_p
        net_kw = sum(
            c["power_kw"] if c["action"] == "charge" else -c["power_kw"]
            for c in commands
            if c["action"] in ("charge", "discharge")
        )
        return commands, net_kw

    @staticmethod
    def _tag_original(commands):
        for c in commands:
            if "_original_power_kw" not in c:
                c["_original_power_kw"] = c["power_kw"]
                c["_original_action"] = c["action"]
        return commands

    def _build_pandapower_network(self):
        try:
            import pandapower as pp

            net = pp.create_empty_network()
            b0 = pp.create_bus(net, vn_kv=33.0, name="HV_substation")
            b1 = pp.create_bus(net, vn_kv=V_NOM_KV, name="LV_substation")
            b2 = pp.create_bus(net, vn_kv=V_NOM_KV, name="mid_feeder")
            b3 = pp.create_bus(net, vn_kv=V_NOM_KV, name="ev_depot")
            pp.create_ext_grid(net, bus=b0, vm_pu=1.0, name="Grid")
            pp.create_transformer_from_parameters(
                net,
                hv_bus=b0,
                lv_bus=b1,
                sn_mva=S_BASE_MVA,
                vn_hv_kv=33.0,
                vn_lv_kv=V_NOM_KV,
                vkr_percent=1.0,
                vk_percent=6.0,
                pfe_kw=0,
                i0_percent=0,
                name="HV_MV_transformer",
            )
            pp.create_line_from_parameters(
                net,
                from_bus=b1,
                to_bus=b2,
                length_km=2.0,
                r_ohm_per_km=0.25,
                x_ohm_per_km=0.10,
                c_nf_per_km=0,
                max_i_ka=0.5,
                name="feeder_seg1",
            )
            pp.create_line_from_parameters(
                net,
                from_bus=b2,
                to_bus=b3,
                length_km=1.5,
                r_ohm_per_km=0.25,
                x_ohm_per_km=0.10,
                c_nf_per_km=0,
                max_i_ka=0.5,
                name="feeder_seg2",
            )
            pp.create_load(
                net, bus=b3, p_mw=0.0, q_mvar=0.0, name="ev_depot_load", controllable=True
            )
            log.info("Pandapower network built | buses=%d | lines=%d", len(net.bus), len(net.line))
            return net
        except ImportError:
            log.warning("pandapower not installed - using DistFlow only")
            return None
        except Exception as exc:
            log.error("Pandapower network build failed: %s", exc)
            return None

    def _ac_power_flow(self, net_kw):
        if self._pandapower_net is None:
            return None, None
        try:
            import pandapower as pp

            net = self._pandapower_net
            p_mw = net_kw / 1000.0
            q_mvar = p_mw * TAN_PHI
            net.load.at[0, "p_mw"] = p_mw
            net.load.at[0, "q_mvar"] = q_mvar
            pp.runpp(net, algorithm="nr", numba=False, verbose=False)
            v_mid = float(net.res_bus.at[2, "vm_pu"])
            v_dep = float(net.res_bus.at[3, "vm_pu"])
            return v_dep, v_mid
        except Exception as exc:
            log.error("AC power flow failed: %s - falling back to DistFlow", exc)
            return None, None


# ---------------------------------------------------------------------------
# Loader + CLI
# ---------------------------------------------------------------------------


def load_dispatch(path):
    with open(path) as f:
        return json.load(f)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="GridSentinel Digital Twin - Physics Validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dispatch", "-d", type=Path, default=Path("data/dispatch/latest_dispatch.json")
    )
    p.add_argument(
        "--output", "-o", type=Path, default=Path("data/validated/latest_validated.json")
    )
    p.add_argument("--transformer-limit", type=float, default=TRANSFORMER_KW)
    p.add_argument("--no-pandapower", action="store_true")
    return p


def main():
    args = _build_arg_parser().parse_args()

    if not args.dispatch.exists():
        raise FileNotFoundError(f"Dispatch not found: {args.dispatch}\nRun mpc/dispatch.py first.")

    log.info("Loading dispatch from %s ...", args.dispatch)
    dispatch = load_dispatch(args.dispatch)
    log.info(
        "Dispatch loaded | commands=%d | status=%s",
        len(dispatch.get("commands", [])),
        dispatch.get("solver_status", "unknown"),
    )

    twin = DigitalTwin(
        transformer_limit_kw=args.transformer_limit, use_pandapower=not args.no_pandapower
    )
    result = twin.validate(dispatch)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.to_json())
    log.info("Validation written -> %s", args.output)

    icon = (
        "checkmark"
        if result.solver_status == "Safe"
        else ("lightning" if result.solver_status == "Curtailed" else "x")
    )
    print("\n" + "=" * 62)
    print("  GridSentinel - Digital Twin Validation Complete")
    print("=" * 62)
    print(f"  Status           : {result.solver_status}")
    print(f"  Method           : {result.validation_method}")
    print(f"  Validation time  : {result.validation_time_s:.4f}s")
    print(
        f"  V_depot          : {result.v_depot_pu:.4f} pu  (limits: [{V_MIN_PU:.2f}, {V_MAX_PU:.2f}])"
    )
    print(f"  V_mid-feeder     : {result.v_mid_pu:.4f} pu")
    print(f"  Transformer load : {result.transformer_load_pct:.1f}%")
    print()
    print(f"  Approved         : {result.approved_count} commands")
    print(f"  Curtailed        : {result.curtailed_count} commands")
    print(f"  Rejected         : {result.rejected_count} commands")
    print()
    print(f"  Final discharge  : {result.total_discharge_kw:.0f} kW")
    print(f"  Final charge     : {result.total_charge_kw:.0f} kW")
    print(f"  Est. profit (t0) : EUR {result.expected_profit_eur:.2f}")

    curtailed = [v for v in result.validations if v.outcome == CURTAILED]
    if curtailed:
        print("\n  Curtailed buses (sample):")
        for v in curtailed[:5]:
            print(f"    {v.bus_id}: {v.original_power_kw:.0f}kW -> {v.validated_power_kw:.0f}kW")

    rejected = [v for v in result.validations if v.outcome == REJECTED]
    if rejected:
        print("\n  Rejected buses (sample):")
        for v in rejected[:5]:
            print(f"    {v.bus_id}: {v.reason}")

    print("\n" + "=" * 62)
    if result.solver_status == "Safe":
        print("  All commands APPROVED - safe to execute.")
    elif result.solver_status == "Curtailed":
        print("  Some commands curtailed - modified commands safe to execute.")
    else:
        print("  Grid unsafe - no commands should execute this cycle.")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
