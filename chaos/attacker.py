"""
chaos/attacker.py
-----------------
Byzantine attack injector. Modifies the synthetic snapshot that the BFT
gatekeeper reads, simulating compromised bus sensors.

Attack types:
  flatline    — buses report constant SoC regardless of activity
  spike       — buses report 10x normal power draw
  coordinated — pct% of fleet lies simultaneously in same direction
  replay      — buses replay yesterday's legitimate readings

Usage:
  python chaos/attacker.py --attack coordinated --pct 0.10
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger(__name__)

DISPATCH_PATH  = Path("data/dispatch/latest_dispatch.json")
SNAPSHOT_PATH  = Path("data/aligned/latest_snapshot.json")


def _base_buses() -> list[dict]:
    """Build a clean bus list from latest dispatch."""
    if not DISPATCH_PATH.exists():
        log.error("No dispatch found — run mpc/dispatch.py first")
        return []
    dispatch = json.loads(DISPATCH_PATH.read_text())
    return [
        {
            "bus_id":       cmd["bus_id"],
            "mean_soc_pct": float(random.uniform(30, 90)),
            "mean_power_kw": cmd.get("power_kw", 0.0),
            "status":       cmd.get("action", "hold"),
        }
        for cmd in dispatch.get("commands", [])
    ]


def _build_snapshot(buses: list[dict]) -> dict:
    return {
        "canonical_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spot_price":          50.0,
        "price_metadata":      {"confidence": 1.0},
        "depot_meter_kw":      sum(b["mean_power_kw"] for b in buses),
        "buses":               buses,
    }


def flatline_attack(buses: list[dict], pct: float) -> list[dict]:
    n = max(1, int(len(buses) * pct))
    targets = random.sample(buses, n)
    target_ids = {b["bus_id"] for b in targets}
    log.warning("FLATLINE attack | %d buses | %s", n, sorted(target_ids)[:5])
    for b in buses:
        if b["bus_id"] in target_ids:
            b["mean_soc_pct"] = 42.0   # constant, never changes
    return buses


def spike_attack(buses: list[dict], pct: float) -> list[dict]:
    n = max(1, int(len(buses) * pct))
    targets = random.sample(buses, n)
    target_ids = {b["bus_id"] for b in targets}
    log.warning("SPIKE attack | %d buses | %s", n, sorted(target_ids)[:5])
    for b in buses:
        if b["bus_id"] in target_ids:
            b["mean_power_kw"] = b["mean_power_kw"] * 10 + 500
    return buses


def coordinated_attack(buses: list[dict], pct: float) -> list[dict]:
    n = max(1, int(len(buses) * pct))
    targets = random.sample(buses, n)
    target_ids = {b["bus_id"] for b in targets}
    log.warning("COORDINATED attack | %d buses | %s", n, sorted(target_ids)[:5])
    for b in buses:
        if b["bus_id"] in target_ids:
            # All lie in the same direction — designed to fool simple averaging
            b["mean_soc_pct"]  = 95.0
            b["mean_power_kw"] = -500.0   # claim discharging when not
    return buses


def replay_attack(buses: list[dict], pct: float) -> list[dict]:
    n = max(1, int(len(buses) * pct))
    targets = random.sample(buses, n)
    target_ids = {b["bus_id"] for b in targets}
    log.warning("REPLAY attack | %d buses | %s", n, sorted(target_ids)[:5])
    # Replay: freeze values from first pass (simulate yesterday's readings)
    frozen = {b["bus_id"]: (b["mean_soc_pct"], b["mean_power_kw"]) for b in buses
              if b["bus_id"] in target_ids}
    for b in buses:
        if b["bus_id"] in target_ids:
            b["mean_soc_pct"], b["mean_power_kw"] = frozen[b["bus_id"]]
    return buses


ATTACKS = {
    "flatline":    flatline_attack,
    "spike":       spike_attack,
    "coordinated": coordinated_attack,
    "replay":      replay_attack,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="GridSentinel Byzantine Attack Injector")
    parser.add_argument("--attack", choices=list(ATTACKS), default="coordinated")
    parser.add_argument("--pct",    type=float, default=0.10,
                        help="Fraction of fleet to compromise (0.01–0.50)")
    parser.add_argument("--cycles", type=int,   default=0,
                        help="Number of attack cycles (0 = run once)")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Seconds between cycles")
    args = parser.parse_args()

    attack_fn = ATTACKS[args.attack]
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    def run_cycle() -> None:
        buses    = _base_buses()
        if not buses:
            return
        attacked = attack_fn(buses, args.pct)
        snapshot = _build_snapshot(attacked)
        SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2))
        log.info("Snapshot written → %s | attack=%s | compromised=%.0f%%",
                 SNAPSHOT_PATH, args.attack, args.pct * 100)

    if args.cycles == 0:
        run_cycle()
    else:
        for i in range(args.cycles):
            log.info("Attack cycle %d/%d", i + 1, args.cycles)
            run_cycle()
            if i < args.cycles - 1:
                time.sleep(args.interval)

    log.info("Attack complete.")


if __name__ == "__main__":
    main()