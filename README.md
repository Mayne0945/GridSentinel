# GridSentinel

**Autonomous Byzantine-Resilient V2G Energy Arbitrage & Grid Safety System**

> *Built to doubt. Designed to protect. Optimized to profit.*

GridSentinel orchestrates a fleet of 100 electric vehicle buses as distributed grid assets — simultaneously protecting the pipeline from corrupted sensor data, forecasting energy prices 24 hours ahead with statistical certainty, executing profitable V2G arbitrage trades, and validating every dispatch command against a physics-based digital twin before a single watt flows.

---

## What Makes This Different

Most EV fleet management systems assume their sensors are honest. GridSentinel doesn't.

**1. Byzantine Fault Detection at the data layer.**
Every sensor reading passes through a Median Absolute Deviation consensus filter before it reaches the forecasting engine. Hacked chargers, faulty meters, and coordinated replay attacks are caught and quarantined. The system has never been silently corrupted in testing — every attack is logged, flagged, and neutralised.

**2. Physics validation before hardware execution.**
Every MPC dispatch command is marked `PENDING` until a Pandapower digital twin approves it. The twin runs a linearized DistFlow voltage check every 5 minutes and a full Newton-Raphson AC power flow every 30 minutes. The system cannot issue a command that would blow a transformer or cause a voltage violation.

**3. Chaos-tested under adversarial conditions.**
Four attack vectors. Ten percent fleet compromise. One hundred percent detection rate. See the [Chaos Report](docs/chaos_report/) for evidence.

---

## Architecture

```
[Data Sources]
ENTSO-E Prices  ──►
Weather API     ──►  Multi-Source    ──►  Byzantine    ──►  Forecasting   ──►  MPC        ──►  Digital   ──►  Hardware
Fleet Telemetry ──►  Ingestor            Gatekeeper        Engine             Dispatch        Twin
(fleet_sim)          (Kinesis)           (MAD Filter)      (XGBoost+CP)       (LP, PuLP)      (Pandapower)
                          │                   │                  │                  │               │
                          ▼                   ▼                  ▼                  ▼               ▼
                   Temporal            Trust Ledger        24h PI Forecast    Dispatch JSON   APPROVED /
                   Alignment           Flagged Sensors     Arbitrage Windows  (PENDING)       CURTAILED /
                   (5-min windows)     Clean Truth                                            REJECTED
```

---

## Pipeline Stages

### 1 — The Senses (Multi-Source Ingestor)
Three Kinesis producers stream data every 5 seconds — ENTSO-E day-ahead prices, Open-Meteo weather (solar irradiance, temperature, wind), and fleet telemetry from 100 EV buses via `fleet_sim`. A temporal alignment module reconciles the three streams (hourly, 15-min, and 5-second resolution respectively) into a unified 5-minute canonical window. Late records up to 200ms drift are accepted; anything beyond a 10-second watermark is dropped and logged.

**Market safety circuit breaker:** If the ENTSO-E API goes dark, the pipeline does not inject `0.0` as the spot price. A three-tier resolution (live → Last Known Good → emergency price) ensures the MPC always receives a physically meaningful signal. A zero price would be a silent command to charge all buses simultaneously — a grid event, not a billing issue.

### 2 — The Filter (Byzantine Gatekeeper)
The BFT Gatekeeper applies a Median Absolute Deviation consensus filter to every sensor cluster in every 5-minute window.

**The Math:**

$$\text{MAD} = \text{median}(|x_i - \text{median}(X)|)$$

A sensor $x_i$ is flagged Byzantine if:

$$\frac{|x_i - \text{median}(X)|}{1.4826 \times \text{MAD}} > k \quad (k = 3)$$

The 1.4826 consistency constant makes MAD equivalent to $\sigma$ for Gaussian data. MAD is used instead of standard deviation because a hacked sensor sending a $1{,}000{,}000$ kW reading pulls the mean toward it — the median stays anchored.

**Three-tier contextual decay:**
- `minor` — isolated spike, likely transient noise → trust decays 0.02
- `standard` — MAD flag + depot meter cross-validation failure → decays 0.10
- `coordinated` — 3+ buses flagged simultaneously → decays 0.20

Sensors below trust score 0.50 are blacklisted and their readings replaced with weighted interpolation from trusted neighbours. Clean Truth is written to Redis (TTL 90s) for MPC consumption.

### 3 — The Eyes (Probabilistic Forecasting)
An XGBoost `MultiOutputRegressor` trains 288 separate models — one per 5-minute horizon step across a 24-hour window. Walk-forward cross-validation prevents data leakage. Hyperparameters are tuned via Optuna.

A Conformal Prediction wrapper produces calibrated 80% prediction intervals:

$$\hat{y} \pm q_{1-\alpha}$$

Where $q_{1-\alpha}$ is the $(1-\alpha)$ quantile of calibration residuals. At $\alpha = 0.20$, empirical coverage on held-out data is exactly **0.800** — the intervals are not just wide, they are statistically honest.

**Training results (mock data, 2023 Q1):**

| Metric | Value |
|---|---|
| Final MAE (h=1, 5min) | 2.5958 EUR/MWh |
| Final MAE (h=288, 24h) | 2.6634 EUR/MWh |
| Conformal coverage | 0.800 (target: 0.80) ✓ |
| Mean PI width | ±4.310 EUR/MWh |

The arbitrage window scanner identifies charge/discharge pairs where the discharge **lower bound** exceeds the charge **upper bound** — meaning the trade is profitable even in the worst-case scenario of both intervals simultaneously.

### 4 — The Brain (MPC Dispatch)
A rolling 4-hour Model Predictive Controller solves a Linear Program every 5 minutes across all 100 buses.

**Objective — Minimise Net Cost:**

$$\min \sum_{t=1}^{T} \left[ \lambda_t \cdot P_t^{net} + \sum_b c_{deg}(SoC_b) \cdot (P_t^{ch} + P_t^{dis}) \cdot \Delta t + \text{penalty} \cdot (P_t^{ch} + P_t^{dis}) \right]$$

Where $c_{deg}$ is the degradation cost — 0.03 EUR/kWh in the 20–80% SoC comfort zone, 0.12 EUR/kWh outside it. The `penalty` term (0.50 EUR/kW) replaces binary charge/discharge exclusivity constraints, dropping the problem from MILP to LP and reducing solve time from **204 seconds to 0.78 seconds**.

**Hard constraints:**

| Constraint | Description |
|---|---|
| $SoC_{b,t_{dep}} \geq SoC_{required}$ | Bus must reach required SoC before departure — non-negotiable |
| $SoC_{min} \leq SoC_{b,t} \leq SoC_{max}$ | Battery bounds (10%–95%) |
| $\|P_{b,t} - P_{b,t-1}\| \leq \Delta P_{max}$ | Ramp rate (30% of rated power per step) |
| $\sum_b P_{b,t} \leq P_{transformer}$ | Depot transformer ceiling (4 MVA) |

All commands are marked `PENDING`. Nothing executes without Digital Twin approval.

### 5 — The Body (Digital Twin)
Every dispatch command passes through a Pandapower 4-bus radial distribution grid model before execution.

**Linearized DistFlow approximation (fast path, every 5 min):**

$$V_{depot} \approx V_{substation} - \sum_{lines}(R \cdot P + X \cdot Q)$$

Checked against: $0.95 \leq V_{depot} \leq 1.05$ p.u.

**Full Newton-Raphson AC power flow (deep check, every 30 min)** via Pandapower catches non-linear effects the linearized model misses.

**Validation results (synthetic fleet, 4 MVA transformer):**

```
V_mid-feeder  : 0.9677 p.u.
V_depot       : 0.9530 p.u.   (limit: 0.95 — within bounds)
Transformer   : 100.0%
Approved      : 100 commands
Curtailed     : 0
Rejected      : 0
Validation time: 0.057s
```

The depot voltage sitting at 0.9530 p.u. under 4,000 kW of charging load is physically correct — and sits 0.003 p.u. above the safety limit. The twin is not conservative by accident.

---

## Quick Start

```bash
# Clone
git clone https://github.com/Mayne0945/GridSentinel.git
cd GridSentinel

# Install dependencies
poetry install

# Start full stack (LocalStack Kinesis + all services)
docker-compose up

# Run the pipeline manually (single-shot)
python fleet_sim/main.py --buses 100 --duration 24h
python forecasting/inference.py --input data/clean_truth/latest.parquet
python mpc/dispatch.py --forecast data/forecasts/latest_forecast.json
python digital_twin/validate.py --dispatch data/dispatch/latest_dispatch.json
```

---

## Running the Chaos Demo

```bash
# Terminal 1 — start the pipeline
docker-compose up

# Terminal 2 — inject a coordinated Byzantine attack (10% fleet)
python chaos/main.py --attack coordinated --pct 10

# Watch the BFT terminal — trust scores drop in real time
# Clean Truth remains unaffected — the attack never reaches the MPC
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Ingestion | AWS Kinesis (LocalStack for dev) |
| Fleet Simulation | Python, stochastic modelling (Rea Vaya / TFL schedules) |
| BFT Filter | Python, MAD consensus, Redis pub/sub |
| Forecasting | XGBoost, MAPIE (Conformal Prediction), Optuna |
| MPC Optimizer | PuLP (CBC solver, pure LP) |
| Digital Twin | Pandapower, linearized DistFlow |
| Time-Series Storage | InfluxDB |
| Monitoring | Grafana |
| Dashboard | React (Phase 5) |
| Orchestration | Docker Compose |
| CI/CD | GitHub Actions (Ruff lint, Mypy, pytest, Docker build) |
| Config | Pydantic v2, YAML |

---

## How the Byzantine Filter Works (Plain English)

Imagine 100 chargers all reporting their power draw. If one charger has been hacked and reports 0 kW while actually drawing 150 kW, a simple average gets pulled toward the lie. But the **median** of 100 readings doesn't care — it stays anchored to what most sensors report.

GridSentinel goes further. It measures how far each sensor deviates from the median, in units of the median deviation itself. A reading that's 3 standard deviations away from the cluster gets flagged. Its trust score drops. If it keeps lying, it gets blacklisted and its readings are replaced with interpolated values from its honest neighbours — and it never reaches the brain.

---

## How Grid Safety Works (Plain English)

Before any discharge command executes, GridSentinel asks the virtual grid: "If I push 850 kW back onto the feeder right now, what happens to the voltage at the depot bus?"

Using the linearized DistFlow equation, it estimates the voltage drop across the feeder cable (resistance × active power + reactance × reactive power, divided by the nominal voltage squared). If the depot voltage would drop below 0.95 per-unit — undervoltage, the classic risk of too much load on a weak feeder — the command is curtailed until the voltage is safe.

Every 30 minutes, a full Newton-Raphson AC power flow runs to catch anything the linear approximation missed. No command executes without physics sign-off.

---

## Project Structure

```
GridSentinel/
├── fleet_sim/          # Stochastic EV fleet simulator
├── ingestion/          # Kinesis producers, temporal alignment, market safety
├── bft/                # Byzantine Fault Detection gatekeeper
├── forecasting/        # XGBoost + Conformal Prediction engine
├── mpc/                # Model Predictive Controller (LP)
├── digital_twin/       # Pandapower grid safety validator
├── chaos/              # Attacker scripts (4 attack types)
├── dashboard/          # React God-View (Phase 5)
├── monitoring/         # Grafana + InfluxDB configuration
├── api/                # REST API (chaos toggle endpoint)
├── config/             # Pydantic v2 settings, fleet.yaml
├── data/               # Runtime data (forecasts, dispatch, validated)
├── models/             # Trained model artifacts
├── tests/              # Unit + integration tests
└── docs/               # Chaos Report, architecture diagrams
```

---

## CI Status

![CI](https://github.com/Mayne0945/GridSentinel/actions/workflows/ci.yml/badge.svg)

All jobs passing: Lint & Type Check · Unit Tests · Integration Tests · Docker Build

---

## Roadmap

- [x] Phase 1 — Multi-source ingestion, temporal alignment, Byzantine fault detection
- [x] Phase 2 — Probabilistic forecasting with calibrated uncertainty intervals
- [x] Phase 3 — LP dispatch with departure constraints and degradation cost
- [x] Phase 4 — Physics-based Digital Twin validation
- [x] Market safety circuit breaker (zero-price vulnerability patched)
- [ ] Phase 5 — React Chaos Dashboard with live God-View and Chaos Toggle
- [ ] Production retrain on full 2023–2024 real ENTSO-E data
- [ ] Wire BFT fleet output to MPC fleet state (currently synthetic fleet)

---

*GridSentinel is a portfolio project demonstrating production-grade distributed systems, probabilistic ML, and power systems engineering. It is not affiliated with any utility or grid operator.*