<div align="center">

<br />

```
  ██████╗ ██████╗ ██╗██████╗ ███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗██╗
 ██╔════╝ ██╔══██╗██║██╔══██╗██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔════╝██║
 ██║  ███╗██████╔╝██║██║  ██║███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║█████╗  ██║
 ██║   ██║██╔══██╗██║██║  ██║╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══╝  ██║
 ╚██████╔╝██║  ██║██║██████╔╝███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║███████╗███████╗
  ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝
```

<br />

**Autonomous Byzantine-Resilient V2G Energy Arbitrage and Grid Safety System**

<br />

![CI](https://github.com/Mayne0945/GridSentinel/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-Conformal_Prediction-orange?style=flat-square)
![PuLP](https://img.shields.io/badge/Solver-PuLP_CBC-purple?style=flat-square)
![Pandapower](https://img.shields.io/badge/Grid-Pandapower-red?style=flat-square)
![Kinesis](https://img.shields.io/badge/Streaming-AWS_Kinesis-yellow?style=flat-square&logo=amazonaws)

<br />

*Built to doubt. Designed to protect. Optimized to profit.*

<br />

</div>

---

## Overview

GridSentinel orchestrates a fleet of 100 electric vehicle buses as distributed grid assets. It simultaneously protects the pipeline from corrupted sensor data, forecasts energy prices 24 hours ahead with statistical certainty, executes profitable V2G arbitrage trades, and validates every dispatch command against a physics-based digital twin before a single watt flows.

Most energy management systems assume their sensors are honest. GridSentinel does not. Every data point that enters the system is treated as potentially compromised until it survives a Byzantine fault detection layer.

<br />

## What Makes This Different

```
  Byzantine Detection     Physics Validation      Chaos Tested
  at the data layer   +   before execution    +   under attack
```

**Byzantine Fault Detection.** Every sensor reading passes through a Median Absolute Deviation consensus filter before reaching the forecasting engine. Hacked chargers, faulty meters, and coordinated replay attacks are caught, quarantined, and logged. The system has never been silently corrupted in testing.

**Physics Validation.** Every MPC dispatch command is marked `PENDING` until a Pandapower digital twin approves it. The twin runs a linearized DistFlow voltage check every 5 minutes and a full Newton-Raphson AC power flow every 30 minutes. A command that would violate grid physics never reaches hardware.

**Market Safety Circuit Breaker.** If the ENTSO-E API goes dark, the pipeline does not inject `0.0` as the spot price. A zero price tells the MPC electricity is free, which is a silent command to charge all buses simultaneously. At 100 buses, that is a grid event. A three-tier resolution system activates an emergency price instead.

<br />

## Architecture

```
  DATA SOURCES             INGESTION              INTELLIGENCE            SAFETY             OUTPUT
  ============             =========              ============            ======             ======

  ENTSO-E Prices  ──►
  Weather API     ──►   Kinesis Stream  ──►   BFT Gatekeeper  ──►   Forecasting   ──►
  Fleet Telemetry ──►   + Alignment         (MAD Consensus)        (XGBoost+CP)
  (fleet_sim)           (5-min windows)                                  |
                                                                         v
                                                                    MPC Dispatch   ──►   Digital Twin   ──►   Hardware
                                                                    (LP, PuLP)          (Pandapower)          APPROVED /
                                                                                                              CURTAILED /
                                                                                                              REJECTED
```

<br />

## Pipeline Stages

### Stage 1 · The Senses

Three Kinesis producers stream data continuously: ENTSO-E day-ahead prices (hourly), Open-Meteo weather data (15-minute), and fleet telemetry from 100 EV buses (5-second). A temporal alignment module reconciles all three into a unified 5-minute canonical window. Late records within a 10-second watermark are accepted; beyond that they are dropped and logged as a metric.

**Price safety:** If the market API fails, the pipeline resolves price in three tiers: live data (confidence 1.0), Last Known Good within one hour (confidence 0.5), or emergency base price of 0.25 EUR/kWh (confidence 0.0). The emergency price is high enough to stop speculative arbitrage but low enough to allow departure-critical charging.

<br />

### Stage 2 · The Filter

The BFT Gatekeeper applies a Median Absolute Deviation consensus filter to every sensor cluster in every window.

$$\text{MAD} = \text{median}\left(|x_i - \text{median}(X)|\right)$$

A sensor $x_i$ is flagged Byzantine if:

$$\frac{|x_i - \text{median}(X)|}{1.4826 \times \text{MAD}} > k \quad (k = 3)$$

The 1.4826 consistency constant makes MAD equivalent to $\sigma$ for Gaussian-distributed data. MAD is chosen over standard deviation because a hacked sensor transmitting an extreme value pulls the mean toward the lie. The median stays anchored.

**Contextual Trust Ledger decay:**

| Attack Class | Condition | Trust Decay |
|---|---|---|
| `minor` | Isolated spike, MAD flag only | -0.02 |
| `standard` | MAD flag + depot meter cross-validation failure | -0.10 |
| `coordinated` | 3 or more buses flagged simultaneously | -0.20 |

Sensors below trust score 0.50 are blacklisted. Their readings are replaced with weighted interpolation from trusted neighbours. Clean Truth is written to Redis (TTL 90s) and InfluxDB for the dashboard.

<br />

### Stage 3 · The Eyes

An XGBoost `MultiOutputRegressor` trains 288 separate models, one per 5-minute horizon step across a 24-hour window. Walk-forward cross-validation prevents data leakage. Hyperparameters are tuned via Optuna.

A Conformal Prediction wrapper produces calibrated 80% prediction intervals:

$$\hat{y} \pm q_{1-\alpha}$$

Where $q_{1-\alpha}$ is the $(1-\alpha)$ quantile of calibration residuals. At $\alpha = 0.20$, empirical coverage on held-out data is exactly **0.800**. The intervals are not just wide. They are statistically honest.

**Training results:**

| Metric | Value |
|---|---|
| MAE at h=1 (5 min) | 2.5958 EUR/MWh |
| MAE at h=144 (12 h) | 2.6170 EUR/MWh |
| MAE at h=288 (24 h) | 2.6634 EUR/MWh |
| Conformal coverage | 0.800 (target: 0.80) |
| Mean PI width | +/- 4.310 EUR/MWh |

The arbitrage window scanner identifies charge/discharge pairs where the discharge **lower bound** exceeds the charge **upper bound**. The trade is profitable even in the worst-case scenario of both intervals simultaneously.

<br />

### Stage 4 · The Brain

A rolling 4-hour Model Predictive Controller solves a Linear Program every 5 minutes across all 100 buses.

**Objective — Minimise Net Cost:**

$$\min \sum_{t=1}^{T} \left[ \lambda_t \cdot P_t^{net} + \sum_b c_{deg}(SoC_b) \cdot (P_t^{ch} + P_t^{dis}) \cdot \Delta t + \text{penalty} \cdot (P_t^{ch} + P_t^{dis}) \right]$$

Where $c_{deg}$ is 0.03 EUR/kWh in the 20-80% SoC comfort zone and 0.12 EUR/kWh outside it. The penalty term (0.50 EUR/kW) replaces binary charge/discharge exclusivity constraints, converting the problem from MILP to LP and reducing solve time from 204 seconds to **0.78 seconds**.

**Hard constraints:**

| Constraint | Description |
|---|---|
| $SoC_{b,t_{dep}} \geq SoC_{required}$ | Bus must reach required SoC before departure. Non-negotiable. |
| $SoC_{min} \leq SoC_{b,t} \leq SoC_{max}$ | Battery bounds (10%-95%) |
| $\|P_{b,t} - P_{b,t-1}\| \leq \Delta P_{max}$ | Ramp rate (30% of rated power per step) |
| $\sum_b P_{b,t} \leq P_{transformer}$ | Depot transformer ceiling (4 MVA) |

All commands leave the MPC marked `PENDING`. Nothing executes without Digital Twin approval.

<br />

### Stage 5 · The Body

Every dispatch command passes through a Pandapower 4-bus radial distribution grid model before execution.

**Linearized DistFlow (fast path, every 5 min):**

$$V_{depot} \approx V_{substation} - \sum_{lines}(R \cdot P + X \cdot Q)$$

Checked against: $0.95 \leq V_{depot} \leq 1.05$ p.u.

**Full Newton-Raphson AC power flow (deep check, every 30 min)** catches non-linear effects the linearized model misses.

**Validation output per command:** `APPROVED` · `CURTAILED` · `REJECTED`

**Validation results (synthetic fleet, 4 MVA transformer):**

```
  V_mid-feeder   :  0.9677 p.u.
  V_depot        :  0.9530 p.u.   (limit floor: 0.95)
  Transformer    :  100.0%
  Approved       :  100 commands
  Curtailed      :  0
  Rejected       :  0
  Validation time:  0.057s
```

<br />

## Quick Start

```bash
# Clone
git clone https://github.com/Mayne0945/GridSentinel.git
cd GridSentinel

# Install
poetry install

# Full stack
docker-compose up

# Manual pipeline (single-shot)
python fleet_sim/main.py --buses 100 --duration 24h
python forecasting/inference.py --input data/clean_truth/latest.parquet
python mpc/dispatch.py --forecast data/forecasts/latest_forecast.json
python digital_twin/validate.py --dispatch data/dispatch/latest_dispatch.json
```

<br />

## Chaos Demo

```bash
# Terminal 1 — start the pipeline
docker-compose up

# Terminal 2 — inject coordinated attack (10% fleet compromise)
python chaos/main.py --attack coordinated --pct 10

# The BFT layer catches the attack within one 5-minute window.
# Trust scores drop in real time. Clean Truth is unaffected.
# The attack never reaches the MPC.
```

Attack types available: `flatline` · `spike` · `coordinated` · `replay`

<br />

## Tech Stack

| Layer | Technology |
|---|---|
| Streaming | AWS Kinesis (LocalStack for development) |
| Fleet Simulation | Python, stochastic modelling (Rea Vaya / TFL schedules) |
| BFT Filter | Python, MAD consensus, Redis pub/sub |
| Forecasting | XGBoost, MAPIE (Conformal Prediction), Optuna |
| MPC Optimizer | PuLP (CBC solver, pure LP) |
| Digital Twin | Pandapower, linearized DistFlow |
| Storage | InfluxDB (time-series), Redis (real-time) |
| Monitoring | Grafana |
| Configuration | Pydantic v2, YAML |
| CI/CD | GitHub Actions (Ruff, Mypy, pytest, Docker) |

<br />

## Project Structure

```
GridSentinel/
  fleet_sim/          Stochastic EV fleet simulator (Rea Vaya / TFL routes)
  ingestion/          Kinesis producers, temporal alignment, market safety
  bft/                Byzantine Fault Detection gatekeeper + trust ledger
  forecasting/        XGBoost + Conformal Prediction forecasting engine
  mpc/                Model Predictive Controller (LP formulation)
  digital_twin/       Pandapower physics validation gateway
  chaos/              Attacker scripts (flatline, spike, coordinated, replay)
  dashboard/          React God-View dashboard (Phase 5)
  monitoring/         Grafana + InfluxDB configuration
  api/                REST API (chaos toggle endpoint)
  config/             Pydantic v2 settings, fleet.yaml
  data/               Runtime data (forecasts, dispatch, validated)
  models/             Trained XGBoost artifacts
  tests/              Unit + integration test suite
  docs/               Chaos Report, architecture diagrams
```

<br />

## Roadmap

- [x] Phase 1 · Multi-source ingestion, temporal alignment, Byzantine fault detection
- [x] Phase 2 · Probabilistic forecasting with calibrated uncertainty intervals
- [x] Phase 3 · LP dispatch with departure constraints and degradation cost
- [x] Phase 4 · Physics-based Digital Twin validation (DistFlow + AC power flow)
- [x] Market safety circuit breaker (zero-price vulnerability patched)
- [ ] Phase 5 · React Chaos Dashboard with live God-View and Chaos Toggle
- [ ] Production retrain on full 2023-2024 ENTSO-E data (50 Optuna trials)
- [ ] Wire BFT fleet output to MPC fleet state (currently synthetic fleet)

<br />

---

<div align="center">

*GridSentinel is a portfolio project demonstrating production-grade distributed systems, probabilistic ML, and power systems engineering.*
*It is not affiliated with any utility or grid operator.*

<br />

</div>