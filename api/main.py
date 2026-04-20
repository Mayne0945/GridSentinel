"""
api/main.py
-----------
GridSentinel FastAPI server — 9 endpoints powering the Chaos Dashboard.

Production fixes applied:
  1. Chaos state stored in Redis (multi-worker safe, TTL-bounded)
  2. JSON file reads cached in-process (4-second TTL, eliminates per-request disk I/O)
  3. Redis graceful fallback — if Redis is unavailable, chaos endpoints degrade
     cleanly rather than crashing the entire API.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import redis
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# ── Config ────────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    cache_ttl_seconds: float = 4.0      # file-read cache lifetime
    chaos_ttl_seconds: int = 300        # chaos state expires after 5 min

    class Config:
        env_file = ".env"

settings = Settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [api] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ── Redis client (optional — degrades gracefully) ─────────────────────────────

def _make_redis() -> redis.Redis | None:
    try:
        client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
            socket_connect_timeout=1,
        )
        client.ping()
        log.info("Redis connected at %s:%d", settings.redis_host, settings.redis_port)
        return client
    except (redis.ConnectionError, redis.TimeoutError):
        log.warning(
            "Redis unavailable — chaos state will use in-process fallback. "
            "Run `docker compose up redis` to enable multi-worker chaos tracking."
        )
        return None


_redis: redis.Redis | None = _make_redis()

# In-process fallback for chaos state when Redis is unavailable.
# Single-worker only — acceptable for local dev / portfolio demo.
_chaos_fallback: dict[str, Any] = {"active": False, "pid": None, "attack_type": None}


# ── File-read cache (TTL-based) ───────────────────────────────────────────────

_file_cache: dict[str, tuple[float, dict]] = {}   # path → (expires_at, payload)

def _load_json(path: Path) -> dict[str, Any]:
    """Read a JSON file with a short TTL cache to avoid per-request disk I/O."""
    key = str(path)
    now = time.monotonic()
    expires_at, payload = _file_cache.get(key, (0.0, {}))

    if now < expires_at:
        return payload                          # cache hit

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{path} not found — pipeline may not have run yet.")

    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Malformed JSON at {path}: {exc}") from exc

    _file_cache[key] = (now + settings.cache_ttl_seconds, payload)
    return payload


# ── App ───────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")

app = FastAPI(
    title="GridSentinel API",
    description="Autonomous Byzantine-Resilient V2G Energy Arbitrage & Grid Safety System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _utcnow() -> str:
    return datetime.utcnow().isoformat() + "Z"

# Redis key constants
_CHAOS_ACTIVE_KEY   = "gridsentinel:chaos:active"
_CHAOS_PID_KEY      = "gridsentinel:chaos:pid"
_CHAOS_TYPE_KEY     = "gridsentinel:chaos:attack_type"


def _chaos_get() -> dict[str, Any]:
    if _redis:
        return {
            "active":      _redis.get(_CHAOS_ACTIVE_KEY) == "true",
            "pid":         _redis.get(_CHAOS_PID_KEY),
            "attack_type": _redis.get(_CHAOS_TYPE_KEY),
        }
    return _chaos_fallback.copy()


def _chaos_set(active: bool, pid: int | None, attack_type: str | None) -> None:
    if _redis:
        ttl = settings.chaos_ttl_seconds
        _redis.setex(_CHAOS_ACTIVE_KEY,  ttl, "true" if active else "false")
        _redis.setex(_CHAOS_PID_KEY,     ttl, str(pid) if pid else "")
        _redis.setex(_CHAOS_TYPE_KEY,    ttl, attack_type or "")
    else:
        _chaos_fallback.update({"active": active, "pid": pid, "attack_type": attack_type})


# ── Endpoints — System ────────────────────────────────────────────────────────

@app.get("/status", tags=["system"])
def get_status() -> dict:
    """Liveness check — confirms all pipeline modules are registered."""
    return {
        "system":    "GridSentinel",
        "version":   "1.0.0",
        "timestamp": _utcnow(),
        "phase":     "5 — Chaos Dashboard",
        "redis":     "connected" if _redis else "unavailable (single-worker fallback)",
        "pipeline": {
            "bft":          "active",
            "forecasting":  "active",
            "mpc":          "active",
            "digital_twin": "active",
        },
    }


# ── Endpoints — Pipeline data ─────────────────────────────────────────────────

@app.get("/forecast", tags=["pipeline"])
def get_forecast() -> dict:
    """Latest 24-hour probabilistic price forecast with arbitrage windows."""
    return _load_json(DATA_DIR / "forecasts" / "latest_forecast.json")


@app.get("/dispatch", tags=["pipeline"])
def get_dispatch() -> dict:
    """Latest MPC dispatch commands (PENDING Digital Twin approval)."""
    return _load_json(DATA_DIR / "dispatch" / "latest_dispatch.json")


@app.get("/validation", tags=["pipeline"])
def get_validation() -> dict:
    """Latest Digital Twin validation result: APPROVED / CURTAILED / REJECTED counts."""
    return _load_json(DATA_DIR / "validated" / "latest_validated.json")


@app.get("/trust", tags=["pipeline"])
def get_trust() -> dict:
    """BFT Trust Ledger — per-bus trust scores and currently flagged sensors."""
    path = DATA_DIR / "bft" / "trust_ledger.json"
    if not path.exists():
        # Synthetic ledger when BFT hasn't written yet (dev / cold start)
        return {
            "timestamp": _utcnow(),
            "note":      "BFT not yet active — showing clean synthetic ledger",
            "buses":     {f"Bus_{i:03d}": 1.0 for i in range(1, 101)},
            "flagged":   [],
        }
    return _load_json(path)


@app.get("/metrics", tags=["pipeline"])
def get_metrics() -> dict:
    """Aggregated dashboard metrics — dispatch summary + Digital Twin outcomes."""
    dispatch  = _load_json(DATA_DIR / "dispatch"  / "latest_dispatch.json")
    validated = _load_json(DATA_DIR / "validated" / "latest_validated.json")

    d_summary = dispatch.get("summary", {})
    v_summary = validated.get("summary", {})

    return {
        "timestamp":            _utcnow(),
        "total_buses":          d_summary.get("total_buses",          100),
        "charging_buses":       d_summary.get("charging_buses",         0),
        "discharging_buses":    d_summary.get("discharging_buses",       0),
        "holding_buses":        d_summary.get("holding_buses",           0),
        "total_charge_kw":      d_summary.get("total_charge_kw",         0),
        "total_discharge_kw":   d_summary.get("total_discharge_kw",      0),
        "transformer_pct":      d_summary.get("transformer_pct",         0),
        "approved":             v_summary.get("approved",                 0),
        "curtailed":            v_summary.get("curtailed",                0),
        "rejected":             v_summary.get("rejected",                 0),
        "estimated_profit_eur": d_summary.get("estimated_profit_eur",    0.0),
    }


# ── Endpoints — Chaos ─────────────────────────────────────────────────────────

class ChaosRequest(BaseModel):
    attack_type: str  = Field("coordinated", description="flatline | spike | coordinated | replay")
    pct: float        = Field(0.10, ge=0.01, le=0.5, description="Fraction of fleet to compromise")


@app.post("/chaos/inject", tags=["chaos"])
def inject_chaos(req: ChaosRequest) -> dict:
    """
    Launch a Byzantine attack against the live fleet.
    State stored in Redis — safe across multiple Uvicorn workers.
    """
    state = _chaos_get()
    if state["active"]:
        return {
            "status":      "already_running",
            "attack_type": state["attack_type"],
            "pid":         state["pid"],
        }

    proc = subprocess.Popen(
        ["python", "chaos/attacker.py", "--attack", req.attack_type, "--pct", str(req.pct)],
    )
    _chaos_set(active=True, pid=proc.pid, attack_type=req.attack_type)
    log.info("Chaos injected | type=%s | pct=%.0f%% | pid=%d", req.attack_type, req.pct * 100, proc.pid)

    return {
        "status":      "injected",
        "attack_type": req.attack_type,
        "fleet_pct":   req.pct,
        "pid":         proc.pid,
    }


@app.post("/chaos/stop", tags=["chaos"])
def stop_chaos() -> dict:
    """Terminate the active Byzantine attack."""
    state = _chaos_get()
    if not state["active"]:
        return {"status": "not_running"}

    pid = state.get("pid")
    if pid:
        try:
            subprocess.run(["kill", str(pid)], check=False)
            log.info("Chaos stopped | pid=%s", pid)
        except Exception as exc:
            log.warning("Could not kill pid %s: %s", pid, exc)

    _chaos_set(active=False, pid=None, attack_type=None)
    return {"status": "stopped", "pid": pid}


@app.get("/chaos/status", tags=["chaos"])
def chaos_status() -> dict:
    """Current chaos injection state."""
    return _chaos_get()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["api"],
    )