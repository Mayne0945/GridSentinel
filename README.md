<div align="center">

<br />

<h1>GridSentinel</h1>

<p>
  <img src="https://github.com/Mayne0945/GridSentinel/actions/workflows/ci.yml/badge.svg" alt="CI" />
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.12" />
  &nbsp;
  <img src="https://img.shields.io/badge/XGBoost-Conformal_Prediction-EA580C?style=flat-square" alt="XGBoost" />
  &nbsp;
  <img src="https://img.shields.io/badge/Solver-cvxpy_OSQP-6B21A8?style=flat-square" alt="cvxpy" />
  &nbsp;
  <img src="https://img.shields.io/badge/Grid-Pandapower-DC2626?style=flat-square" alt="Pandapower" />
  &nbsp;
  <img src="https://img.shields.io/badge/Streaming-AWS_Kinesis-F59E0B?style=flat-square&logo=amazonaws&logoColor=white" alt="Kinesis" />
</p>

<br />

<p><em>A sensor that lies is not a sensor. It is a weapon.</em></p>

<br />

</div>

---

Grids are about to absorb tens of millions of EVs. Most charging optimization systems assume the data feeding them is honest.

**GridSentinel does not.**

Every sensor reading is treated as potentially compromised until it survives a Byzantine fault detection layer. Only verified data reaches the forecasting engine. Only physics-validated commands reach hardware. The system has been attacked with four adversarial patterns at 10% fleet compromise. Detection rate: 100%. The clean truth stream has never been corrupted.

<br />

---

<br />

## The Pipeline

```
  ┌─────────────────────────────────────────────────────────────────────────────────────┐
  │                                                                                     │
  │   ENTSO-E ──►                                                                       │
  │   Weather ──►  Temporal Alignment  ──►  Byzantine Filter  ──►  Probabilistic        │
  │   Fleet   ──►  5-min Kinesis             MAD Consensus         Forecasting          │
  │                windows                   Trust Ledger          XGBoost + CP         │
  │                                          Clean Truth           24h PI               │
  │                                                                     │               │
  │                                                                     ▼               │
  │                                              Digital Twin  ◄──  MPC Dispatch        │
  │                                              Pandapower         LP Solver           │
  │                                              DistFlow + AC      0.78s               │
  │                                              0.057s                                 │
  │                                                  │                                  │
  │                                     APPROVED / CURTAILED / REJECTED                 │
  │                                                                                     │
  └─────────────────────────────────────────────────────────────────────────────────────┘
```

Six stages. Each with a defined contract. Each failing independently without corrupting the next.

<br />

---

<br />

## Four Decisions That Define This System

<br />

### 1 · We Deleted 4,800 Variables and the Solver Got 260x Faster

The original MPC dispatch used binary variables to enforce charge/discharge exclusivity: one binary per bus per time step, across 100 buses and 48 horizon steps. That is 4,800 binary variables. The problem was a Mixed-Integer Linear Program. Solve time: **204 seconds**. A 5-minute cycle budget.

The insight was that simultaneous charge and discharge is self-penalizing when the objective already includes a degradation cost term. Charging and immediately discharging costs battery life on both directions and produces zero net energy. Adding a 0.50 EUR/kW simultaneous-use penalty to the objective makes overlap strictly worse than doing nothing. The binaries became unnecessary.

We removed them entirely. The problem became a pure LP.

Solve time: **0.78 seconds**.

The physics are identical. The formulation is cleaner. The system now re-solves every 5 minutes with 174 seconds to spare.

<br />

### 2 · The Linear Model Was Optimistic. The AC Model Was Not.

The linearized DistFlow approximation estimated depot voltage at **0.9673 p.u.** under full charging load. The full Newton-Raphson AC power flow returned **0.9530 p.u.** A difference of 0.0143 p.u. The safety floor is 0.95 p.u.

Without the full AC check, the system believed it had 0.0173 p.u. of headroom. It had 0.003.

GridSentinel runs both. The linearized model runs every 5 minutes for speed. The full AC power flow runs every 30 minutes for accuracy. When they disagree, the AC result wins. The fast path exists to catch obvious violations instantly. The deep check exists because linearization is an approximation, and approximations accumulate error in exactly the conditions where precision matters most.

<br />

### 3 · A Zero Default Price Is Not a Bug. It Is a Grid Event.

The original ingestion layer defaulted to `spot_price = 0.0` when the ENTSO-E market API was unavailable. A zero spot price tells the MPC that electricity is free. The MPC responds rationally: it charges everything, immediately, at maximum power. At 100 buses drawing 150 kW each, that is 15 MW of unplanned load on a distribution network designed for a fraction of that.

The fix was not to add a null check. The fix was to design a three-tier resolution system that makes the failure mode explicit and safe.

| Tier | Condition | Confidence | Behaviour |
|---|---|---|---|
| Live | Market API responding | 1.0 | Full arbitrage optimization |
| Last Known Good | API down, cache age < 1h | 0.5 | Arbitrage continues with caution |
| Emergency | No data or cache stale | 0.0 | MPC shifts to safety mode. Departure-critical charging only. |

The emergency base price of 0.25 EUR/kWh is not arbitrary. It is calibrated to be above the degradation cost floor (0.03 EUR/kWh), making speculative charging unprofitable. The system fails expensive. Not free.

The BFT Gatekeeper reads `price_confidence` from every snapshot and injects `mpc_mode: safety` into the clean truth when confidence reaches zero. The MPC never sees the raw failure. It sees a mode flag.

<br />

### 4 · The Training Pipeline and the Serving Pipeline Were Using Different Features

The inference engine contained its own `build_feature_matrix` function. The training pipeline used `build_features` from `feature_builder.py`. Both functions built a 26-column feature matrix. The column names were different. `sin_hour` in one. `hour_sin` in the other. Fleet features present in training, absent in serving.

The model loaded. It ran. It produced predictions. They were wrong. Silently.

This is training-serving skew. It is one of the most common and least visible failure modes in production ML systems. The model does not crash. It does not warn you. It returns a number that looks plausible and is quietly disconnected from reality.

The architectural fix: inference imports `build_features` directly from `feature_builder.py`. One function. One definition. One source of truth. If the feature contract ever drifts, the system raises a hard `ValueError` listing exactly which columns diverged, with the instruction to retrain or revert.

The bug is now structurally impossible.

<br />

---

<br />

## The Math

### Byzantine Fault Detection

$$\text{MAD} = \text{median}\left(|x_i - \text{median}(X)|\right)$$

$$\text{Flag Byzantine if} \quad \frac{|x_i - \text{median}(X)|}{1.4826 \times \text{MAD}} > 3$$

The 1.4826 consistency constant preserves equivalence with $\sigma$ for Gaussian-distributed clean data. MAD is chosen over standard deviation because a single extreme outlier pulls $\sigma$ toward the lie. The median does not move.

A second validation cross-checks the charger cluster against the depot master meter. If reported aggregate power diverges more than 20% from the substation measurement, the cluster fails regardless of its MAD score. The meter is the ground truth anchor.

**Trust Ledger decay:** -0.02 for isolated noise, -0.10 for confirmed Byzantine, -0.20 for coordinated attacks involving 3 or more buses simultaneously. Recovery is slow by design: +0.01 per clean window, with a +0.05 burst after 5 consecutive clean readings.

<br />

### Probabilistic Forecasting

$$\hat{y} \pm q_{1-\alpha} \quad \text{where} \quad q_{1-\alpha} = \text{quantile}_{1-\alpha}\left(\{|y_i - \hat{y}_i|\}_{i=1}^{n}\right)$$

288 XGBoost models. One per 5-minute horizon step. Walk-forward cross-validation. Optuna hyperparameter search. Conformal Prediction calibration for 80% coverage intervals.

| Metric | Value | What It Means |
|---|---|---|
| MAE at h=1 (5 min) | 2.5958 EUR/MWh | Near-term forecast error |
| MAE at h=288 (24 h) | 2.6634 EUR/MWh | Long-range forecast error |
| Horizon degradation | 0.068 EUR/MWh | How much accuracy we lose over 24 hours |
| Conformal coverage | 0.800 | Exactly 80% of true prices fall inside the interval |

The horizon degradation figure is the result that matters. Most forecasting models degrade significantly at longer horizons. GridSentinel loses 0.068 EUR/MWh across a full 24-hour window. That near-flatness is what makes a 4-hour rolling MPC viable. The optimizer trusts the long-range price signal almost as much as the short one.

<br />

### MPC Dispatch

$$\min \sum_{t=1}^{T} \left[ \lambda_t \cdot P_t^{net} + \sum_b c_{deg}(SoC_b) \cdot (P_t^{ch} + P_t^{dis}) \cdot \Delta t + \text{penalty} \cdot (P_t^{ch} + P_t^{dis}) \right]$$

$\lambda_t$ is the conservative lower-bound price from the prediction interval. If the trade is not profitable under the worst-case price, it does not happen.

$c_{deg}$ is 0.03 EUR/kWh in the 20-80% SoC comfort zone and 0.12 EUR/kWh outside it. The optimizer knows that deep cycling kills batteries faster. It balances this against profit, automatically.

Hard constraints: departure SoC (non-negotiable regardless of profit), battery bounds, ramp rate, transformer ceiling.

<br />

### Grid Safety

$$V_{depot} \approx V_{substation} - \sum_{lines}(R \cdot P + X \cdot Q)$$

Validated against $0.95 \leq V_{depot} \leq 1.05$ p.u. at every dispatch cycle. Commands exceeding the voltage envelope are curtailed in 10% steps. Commands below 10 kW are rejected. The hardware never receives a command that has not passed this gate.

<br />

---

<br />

## Chaos Engineering

```bash
# Coordinated attack: 10% fleet compromise
python chaos/attacker.py --attack coordinated --pct 10

# All four adversarial patterns
python chaos/attacker.py --attack flatline      # sensors frozen at last reading
python chaos/attacker.py --attack spike         # extreme isolated outliers
python chaos/attacker.py --attack coordinated   # synchronized multi-bus spoofing
python chaos/attacker.py --attack replay        # yesterday's legitimate data, injected today
```

The coordinated attack is the hardest to detect. Ten buses simultaneously reporting a plausible but fabricated SoC is designed to fool simple averaging. MAD catches it because ten buses lying in the same direction creates asymmetry in the cluster that the median does not share.

Detection rate across all four patterns: **100%**. The attack is caught within one 5-minute window. The MPC never sees the corrupted data.

<br />

---

<br />

## Quick Start

```bash
git clone https://github.com/Mayne0945/GridSentinel.git
cd GridSentinel
poetry install

# Full stack with LocalStack Kinesis
docker-compose up

# Manual end-to-end
python fleet_sim/main.py --buses 100 --duration 24h
python forecasting/inference.py --input data/clean_truth/latest.parquet
python mpc/dispatch.py --forecast data/forecasts/latest_forecast.json
python digital_twin/validate.py --dispatch data/dispatch/latest_dispatch.json
```

<br />

## Tech Stack

| Layer | Technology |
|---|---|
| Streaming | AWS Kinesis (LocalStack for development) |
| Fleet Simulation | Python, stochastic modelling on Rea Vaya and TFL route schedules |
| BFT Filter | Python, MAD consensus filter, Redis pub/sub |
| Forecasting | XGBoost, MAPIE (Conformal Prediction), Optuna |
| MPC Optimizer | cvxpy with OSQP backend, pure LP formulation |
| Digital Twin | Pandapower, linearized DistFlow |
| Storage | InfluxDB for time-series, Redis for real-time clean truth |
| Observability | Grafana |
| Configuration | Pydantic v2, YAML |
| CI/CD | GitHub Actions: Ruff, Mypy, pytest, Docker build |

<br />

## Project Structure

```
GridSentinel/
  fleet_sim/       Stochastic EV fleet simulator on real transit route schedules
  ingestion/       Kinesis producers, temporal alignment, market safety circuit breaker
  bft/             Byzantine Fault Detection gatekeeper and contextual trust ledger
  forecasting/     XGBoost forecasting engine with Conformal Prediction calibration
  mpc/             Model Predictive Controller, LP formulation, dispatch output
  digital_twin/    Pandapower physics validation gateway
  chaos/           Adversarial attack scripts for BFT verification
  dashboard/       React God-View dashboard (Phase 5)
  monitoring/      Grafana and InfluxDB configuration
  api/             REST API for chaos toggle and dashboard integration
  config/          Pydantic v2 settings and fleet.yaml
  data/            Runtime artifacts: forecasts, dispatch commands, validated output
  models/          Trained XGBoost model artifacts and conformal quantiles
  tests/           Unit and integration test suite
  docs/            Chaos Report and architecture documentation
```

<br />

## Roadmap

- [x] Phase 1 · Multi-source ingestion, temporal alignment, Byzantine fault detection
- [x] Phase 2 · Probabilistic forecasting with calibrated 80% prediction intervals
- [x] Phase 3 · LP dispatch with hard departure constraints and degradation cost
- [x] Phase 4 · Physics-based Digital Twin with DistFlow and AC power flow validation
- [x] Market safety circuit breaker closing the zero-price vulnerability
- [x] Phase 5 · React Chaos Dashboard with live God-View and Chaos Toggle
- [ ] Production retrain on full 2023-2024 ENTSO-E data with 50 Optuna trials
- [ ] Wire BFT fleet output to MPC fleet state replacing synthetic fleet

<br />

---

<div align="center">
<br />
<sub>GridSentinel is a portfolio project demonstrating production-grade distributed systems, applied mathematics, and power systems engineering. Not affiliated with any utility or grid operator.</sub>
<br /><br />
</div>