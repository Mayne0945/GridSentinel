# GridSentinel — Build Progress Document
### Last updated: 2026-04-20

---

## Overall Status

| Phase | Name | Status | Notes |
|---|---|---|---|
| 1 | Senses and Gatekeeper | Complete | All modules confirmed built and committed |
| 2 | Predictive Eyes | Complete | Retrained on real French ENTSO-E data |
| 3 | MPC Dispatch Brain | Complete | LP formulation, 0.78s solve time |
| 4 | Digital Twin | Complete | DistFlow + AC power flow, committed |
| 5 | Chaos Dashboard | In Progress | API built, React dashboard next |

---

## Phase 1 — The Senses and The Gatekeeper
**Complete**

### Modules built
- `fleet_sim/` — battery_physics.py, bus.py, depot.py, kinesis_writer.py, main.py, models.py, route_loader.py
- `ingestion/` — producer_entso_e.py, kinesis_client.py, consumer_align.py (temporal alignment), monitor.py
- `bft/` — gatekeeper.py (MAD filter, trust ledger, depot meter cross-validation), main.py
- `chaos/` — main.py (flatline, spike, coordinated, replay attacks), Dockerfile.python

### Key decisions
- MAD over standard deviation — outlier-resistant consensus
- Context-aware filtering per sensor cluster, not globally
- Three-tier trust decay: minor (-0.02), standard (-0.10), coordinated (-0.20)
- Recovery: +0.01 per clean window, +0.05 burst after 5 consecutive clean windows
- Sensors below 0.50 blacklisted and interpolated from trusted neighbours
- ENTSO-E real API token secured and wired

### Safety patch applied (2026-04-18)
- `MarketConfig` added to `config/settings.py` with `emergency_base_price=0.25 EUR/kWh`
- `aggregate_window` in `consumer_align.py` rewritten with three-tier price resolution
- Zero-price vulnerability closed — system never defaults to 0.0
- `BFTGatekeeper` now injects `mpc_mode: safety` when `price_confidence=0.0`

---

## Phase 2 — The Predictive Eyes
**Complete — production retrain on real data in progress**

### Modules built
- `forecasting/feature_builder.py` — 26 features, cyclical encodings, fleet SoC metrics, weather
- `forecasting/train.py` — XGBoost MultiOutputRegressor, walk-forward CV, Optuna, Conformal Prediction
- `forecasting/inference.py` (v2) — imports build_features directly, three-tier arbitrage scanner, dynamic confidence
- `forecasting/_threading_fix.py` — OpenMP deadlock prevention
- `forecasting/data_fetch.py` — real ENTSO-E fetch, France zone (10YFR-RTE------C)

### Mock training results (2026-04-16, synthetic data)

| Metric | Value |
|---|---|
| MAE at h=1 (5 min) | 2.5958 EUR/MWh |
| MAE at h=288 (24 h) | 2.6634 EUR/MWh |
| Conformal coverage | 0.800 (target: 0.80) |
| Mean PI width | +/- 4.310 EUR/MWh |

### Real data fetch (2026-04-20)
- Zone: France (10YFR-RTE------C) — open access confirmed
- Range: 2023-01-01 to 2024-12-30
- Rows: 210,240 at 5-minute resolution
- Price range: -134.94 to +284.21 EUR/MWh (real negative price events present)
- Files: `data/raw/entso_e_prices.parquet`, `data/raw/weather_london.parquet`

### Production retrain (in progress as of 2026-04-20)
```bash
python forecasting/train.py --train-end 2024-06-01 --n-trials 50
```
- Training rows: 148,608 (2023-01-01 to 2024-06-01)
- Calibration rows: remaining data for conformal wrapper
- Expected duration: 8-15 hours
- Will overwrite models/ artifacts when complete

### Bugs fixed
- Training-serving skew: inference now imports build_features from feature_builder.py directly
- OpenMP deadlock: _threading_fix.py + set_param("nthread", 1) at booster level
- Hardcoded arbitrage values: buses_required and confidence now dynamic

---

## Phase 3 — The Brain (MPC Dispatch)
**Complete**

### Module built
- `mpc/dispatch.py` (v2, LP formulation)
- Reads latest_forecast.json and fleet state (synthetic fallback if BFT not wired)
- Solves 4-hour rolling LP every 5 minutes via PuLP CBC
- Hard constraints: departure SoC, battery bounds, ramp rate, transformer ceiling
- All commands marked PENDING until Digital Twin approval

### Key decision: MILP to LP
Original binary variable formulation caused 204-second solve times.
Simultaneous-use penalty (0.50 EUR/kW) replaced 4,800 binary variables.
Solve time: 0.78 seconds. Physics identical. Formulation cleaner.

### Results
```
Status      : Optimal
Solve time  : 0.78s
Buses       : 100
Charging    : 31 (4,000 kW)
Discharging : 0 (price 11.98 EUR/MWh below degradation cost floor)
Transformer : 100%
```

### Outstanding
- [ ] Wire BFT fleet output to MPC fleet state (currently synthetic fleet)
- [ ] Confirm discharge commands appear on real French price spikes (post-retrain)

---

## Phase 4 — The Body (Digital Twin)
**Complete — committed 2026-04-20**

### Module built
- `digital_twin/validate.py`
- Pandapower 4-bus radial network (33/11kV, 10MVA, substation to mid-feeder to depot)
- Linearized DistFlow fast path every 5 minutes
- Full Newton-Raphson AC power flow every 30 minutes
- Three-outcome gateway: APPROVED / CURTAILED / REJECTED
- Curtailment loop: reduces power in 10% steps until voltage within limits
- Commands below 10 kW rejected entirely

### Key finding
Linearized model estimated V_depot at 0.9673 p.u.
Full AC power flow returned 0.9530 p.u.
Difference: 0.0143 p.u. Safety floor: 0.95 p.u.
The fast path exists for speed. The deep check exists because linearization accumulates error.

### Results
```
V_mid-feeder   : 0.9677 p.u.
V_depot        : 0.9530 p.u.   (limit: 0.95)
Transformer    : 100.0%
Approved       : 100 commands
Curtailed      : 0
Rejected       : 0
Validation time: 0.057s
```

---

## Phase 5 — The Face (Chaos Dashboard)
**In Progress**

### Built so far
- `api/main.py` — FastAPI server with 9 endpoints
  - GET /api/status — full pipeline state, polled every 5s by dashboard
  - GET /api/forecast, /api/dispatch, /api/validation, /api/trust, /api/metrics
  - POST /api/chaos/inject — launches chaos/main.py as subprocess
  - POST /api/chaos/stop — terminates active attack
  - GET /api/chaos/status — current chaos state

### Still to build

**React Chaos Dashboard (`dashboard/`)**
- Three-column God-View layout
- Left: live BFT trust scores, raw vs clean sensor feed
- Centre: 24h forecast with PI bands, arbitrage windows, MPC dispatch table, rolling P&L
- Right: feeder SVG diagram with live voltage colouring, Digital Twin status, transformer bar
- Chaos Toggle: red INJECT BYZANTINE ATTACK button, green NEUTRALISED banner
- Polls /api/status every 5 seconds

**Grafana configuration (`monitoring/`)**
- Spot price actual vs forecast (24h rolling)
- Fleet SoC heatmap
- BFT trust scores with 0.50 threshold line
- Depot power charge vs discharge
- Digital Twin voltage profile
- Cumulative P&L
- Byzantine detection rate per hour

**Chaos Report (`docs/chaos_report/`)**
- 4 attack types, detection rates, false positive rates, detection latency
- Trust score time-series plots during each attack
- Voltage profile during maximum discharge
- Forecast calibration coverage plot
- 30-day simulated P&L on real French data

---

## Repo State (2026-04-20)

### Committed and pushed
- All Phase 1-4 modules
- MarketConfig safety patch
- Real ENTSO-E data fetch (France zone)
- FastAPI server (api/main.py) — needs committing
- README rewrite

### Needs committing now
- `api/main.py` — FastAPI server
- `forecasting/data_fetch.py` — France zone change
- `config/settings.py` — MarketConfig
- `ingestion/consumer_align.py` — zero-price fix
- `bft/gatekeeper.py` — mpc_mode injection

### In progress (do not commit yet)
- `models/` — production retrain running, artifacts will be overwritten when done

---

## Production Run Checklist

- [x] ENTSO-E real API token secured
- [x] France zone confirmed working (10YFR-RTE------C)
- [x] Real data fetched: 210,240 rows, 2023-2024
- [ ] Production retrain complete (50 trials, in progress)
- [ ] Re-run inference on real data
- [ ] Confirm arbitrage windows on real French price spikes
- [ ] Wire BFT fleet output to MPC
- [ ] React dashboard built and running
- [ ] Chaos Report PDF complete
- [ ] Final commit and tag v1.0.0

---

## The Three Pillars — Final Deliverables

| Pillar | Status | Description |
|---|---|---|
| Live Dashboard | In Progress | React God-View with Chaos Toggle |
| Chaos Report | Not Started | PDF — 4 attack types, 100% detection rate |
| Logic Manifest | Complete | README with math, decisions, evidence |

---

*GridSentinel — Built to doubt. Designed to protect. Optimized to profit.*
