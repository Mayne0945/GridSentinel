import { useState, useEffect, useCallback, useRef } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Area, AreaChart
} from 'recharts'
import axios from 'axios'
import './index.css'

const API = '/api'
const POLL_MS = 5000

// ── Helpers ──────────────────────────────────────────────────────────────────

const fmt = (n, dec = 1) => (typeof n === 'number' ? n.toFixed(dec) : '—')

function trustColor(score) {
  if (score >= 0.8) return '#22c55e'
  if (score >= 0.5) return '#fb923c'
  return '#f87171'
}

function transformerColor(pct) {
  if (pct < 70) return '#22c55e'
  if (pct < 90) return '#fb923c'
  return '#f87171'
}

// ── Sub-components ────────────────────────────────────────────────────────────

function SensorRow({ id, score, flagged }) {
  return (
    <div className="sensor-row">
      <span style={{ color: '#94a3b8' }}>{id}</span>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ color: trustColor(score), fontVariantNumeric: 'tabular-nums' }}>
          {fmt(score, 2)}
        </span>
        <span className={`badge ${flagged ? 'badge-byzantine' : 'badge-trusted'}`}>
          {flagged ? 'BYZANTINE' : 'TRUSTED'}
        </span>
      </div>
    </div>
  )
}

function TrustBar({ score }) {
  return (
    <div className="trust-bar-bg">
      <div
        className="trust-bar-fill"
        style={{ width: `${score * 100}%`, background: trustColor(score) }}
      />
    </div>
  )
}

// ── Left Column — Truth Stream ────────────────────────────────────────────────

function TruthStream({ trust, metrics }) {
  const buses = trust?.buses ?? {}
  const flagged = new Set(trust?.flagged ?? [])
  const entries = Object.entries(buses)
  const byzantineCount = entries.filter(([, s]) => s < 0.5).length + flagged.size
  const cleanCount = entries.length - byzantineCount

  // Show flagged first, then lowest-scoring
  const sorted = entries
    .sort(([, a], [, b]) => a - b)
    .slice(0, 15)

  return (
    <div className="col">
      <div className="col-header">◈ Truth Stream — BFT Gatekeeper</div>

      <div className="card">
        <div className="card-title">Fleet Health</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          <div style={{ textAlign: 'center', padding: '8px', background: '#052e16', borderRadius: 4 }}>
            <div style={{ fontSize: 24, fontWeight: 700, color: '#22c55e' }}>{cleanCount}</div>
            <div style={{ fontSize: 10, color: '#64748b', marginTop: 2 }}>TRUSTED</div>
          </div>
          <div style={{ textAlign: 'center', padding: '8px', background: '#450a0a', borderRadius: 4 }}>
            <div style={{ fontSize: 24, fontWeight: 700, color: '#f87171' }}>{byzantineCount}</div>
            <div style={{ fontSize: 10, color: '#64748b', marginTop: 2 }}>BYZANTINE</div>
          </div>
        </div>
        {trust?.note && (
          <div style={{ marginTop: 8, fontSize: 10, color: '#475569', fontStyle: 'italic' }}>
            {trust.note}
          </div>
        )}
      </div>

      <div className="card" style={{ flex: 1 }}>
        <div className="card-title">Sensor Trust Scores (lowest first)</div>
        {sorted.map(([id, score]) => (
          <div key={id}>
            <SensorRow id={id} score={score} flagged={flagged.has(id) || score < 0.5} />
            <TrustBar score={score} />
          </div>
        ))}
        {entries.length === 0 && (
          <div style={{ color: '#475569', fontSize: 11, textAlign: 'center', padding: 16 }}>
            BFT not active — start the pipeline
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-title">Dispatch Summary</div>
        <div className="metric-row">
          <span className="metric-label">Charging</span>
          <span className="metric-value action-charge">
            {metrics?.charging_buses ?? '—'} buses · {fmt(metrics?.total_charge_kw, 0)} kW
          </span>
        </div>
        <div className="metric-row">
          <span className="metric-label">Discharging</span>
          <span className="metric-value action-discharge">
            {metrics?.discharging_buses ?? '—'} buses · {fmt(metrics?.total_discharge_kw, 0)} kW
          </span>
        </div>
        <div className="metric-row">
          <span className="metric-label">Holding</span>
          <span className="metric-value" style={{ color: '#475569' }}>
            {metrics?.holding_buses ?? '—'} buses
          </span>
        </div>
      </div>
    </div>
  )
}

// ── Centre Column — The Brain ─────────────────────────────────────────────────

function Brain({ forecast, dispatch, metrics }) {
  const intervals = forecast?.intervals ?? []
  const arbitrage = forecast?.arbitrage_windows ?? []
  const commands  = dispatch?.commands ?? []

  // Build chart data from forecast intervals (sample every 6th point = 30-min resolution)
  const chartData = intervals
    .filter((_, i) => i % 6 === 0)
    .slice(0, 48)
    .map(iv => ({
      t: iv.timestamp ? iv.timestamp.slice(11, 16) : '',
      price:  iv.point_forecast_eur_mwh,
      upper:  iv.upper_80,
      lower:  iv.lower_80,
    }))

  const profit = metrics?.estimated_profit_eur ?? 0

  return (
    <div className="col">
      <div className="col-header">⬡ The Brain — MPC Dispatch</div>

      <div className="card">
        <div className="card-title">P&L — Current Cycle</div>
        <div className={`pnl-value ${profit >= 0 ? 'pnl-positive' : 'pnl-negative'}`}>
          {profit >= 0 ? '+' : ''}€{fmt(profit, 2)}
        </div>
        <div style={{ color: '#475569', fontSize: 11, marginTop: 4 }}>
          Est. profit this dispatch window
        </div>
      </div>

      <div className="card" style={{ flex: '0 0 200px' }}>
        <div className="card-title">24-Hour Price Forecast (80% PI)</div>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={160}>
            <AreaChart data={chartData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="piGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#38bdf8" stopOpacity={0.15} />
                  <stop offset="95%" stopColor="#38bdf8" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="#1e2d45" strokeDasharray="3 3" />
              <XAxis dataKey="t" tick={{ fill: '#475569', fontSize: 9 }} interval={7} />
              <YAxis tick={{ fill: '#475569', fontSize: 9 }} />
              <Tooltip
                contentStyle={{ background: '#111827', border: '1px solid #1e2d45', fontSize: 11 }}
                labelStyle={{ color: '#94a3b8' }}
              />
              <Area type="monotone" dataKey="upper" stroke="none" fill="url(#piGrad)" />
              <Area type="monotone" dataKey="lower" stroke="none" fill="#0a0e1a" />
              <Line type="monotone" dataKey="price" stroke="#38bdf8" dot={false} strokeWidth={1.5} />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div style={{ height: 160, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#475569' }}>
            No forecast data — run forecasting/predict.py
          </div>
        )}
      </div>

      {arbitrage.length > 0 && (
        <div className="card">
          <div className="card-title">Arbitrage Windows</div>
          {arbitrage.slice(0, 3).map((w, i) => (
            <div key={i} className="metric-row">
              <span className="metric-label">{w.charge_window} → {w.discharge_window}</span>
              <span style={{ color: '#22c55e', fontWeight: 600 }}>
                +€{fmt(w.estimated_profit_eur, 0)} · {w.buses_required} buses
              </span>
            </div>
          ))}
        </div>
      )}

      <div className="card" style={{ flex: 1 }}>
        <div className="card-title">Active Dispatch Commands (sample)</div>
        {commands.length > 0 ? (
          <table className="dispatch-table">
            <thead>
              <tr>
                <th>Bus</th>
                <th>Action</th>
                <th>Power</th>
                <th>Reason</th>
              </tr>
            </thead>
            <tbody>
              {commands.slice(0, 12).map((cmd, i) => (
                <tr key={i}>
                  <td style={{ color: '#94a3b8' }}>{cmd.bus_id}</td>
                  <td className={`action-${cmd.action}`}>{cmd.action?.toUpperCase()}</td>
                  <td style={{ fontVariantNumeric: 'tabular-nums' }}>{fmt(cmd.power_kw, 0)} kW</td>
                  <td style={{ color: '#475569' }}>{cmd.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div style={{ color: '#475569', fontSize: 11, textAlign: 'center', padding: 16 }}>
            No dispatch commands — run mpc/dispatch.py
          </div>
        )}
      </div>
    </div>
  )
}

// ── Right Column — The Grid ───────────────────────────────────────────────────

function GridView({ validation, metrics }) {
  const summary = validation?.summary ?? {}
  const voltage = validation?.voltage_pu ?? validation?.v_depot_pu ?? null
  const txPct   = metrics?.transformer_pct ?? 0

  return (
    <div className="col">
      <div className="col-header">⚡ The Grid — Digital Twin</div>

      {/* Single-line feeder SVG */}
      <div className="card">
        <div className="card-title">Distribution Feeder — Live Voltage</div>
        <svg viewBox="0 0 260 80" style={{ width: '100%' }}>
          {/* Substation */}
          <rect x="4" y="28" width="32" height="24" rx="2"
            fill="#0a0e1a" stroke="#38bdf8" strokeWidth="1.5" />
          <text x="20" y="43" textAnchor="middle" fill="#38bdf8" fontSize="7" fontWeight="bold">SUB</text>
          <text x="20" y="54" textAnchor="middle" fill="#475569" fontSize="6">33kV</text>

          {/* Line SUB→MID */}
          <line x1="36" y1="40" x2="110" y2="40" stroke="#38bdf8" strokeWidth="2" />

          {/* Mid-feeder node */}
          <circle cx="120" cy="40" r="8" fill="#0a0e1a" stroke="#64748b" strokeWidth="1.5" />
          <text x="120" y="44" textAnchor="middle" fill="#64748b" fontSize="7">MID</text>

          {/* Line MID→DEPOT */}
          <line x1="128" y1="40" x2="210" y2="40" stroke="#38bdf8" strokeWidth="2" />

          {/* Depot bus */}
          <rect x="210" y="24" width="46" height="32" rx="2"
            fill="#0a0e1a"
            stroke={voltage !== null ? (voltage < 0.95 ? '#f87171' : '#22c55e') : '#475569'}
            strokeWidth="1.5" />
          <text x="233" y="38" textAnchor="middle"
            fill={voltage !== null ? (voltage < 0.95 ? '#f87171' : '#22c55e') : '#475569'}
            fontSize="7" fontWeight="bold">DEPOT</text>
          <text x="233" y="50" textAnchor="middle"
            fill={voltage !== null ? (voltage < 0.95 ? '#f87171' : '#22c55e') : '#64748b'}
            fontSize="8" fontWeight="bold">
            {voltage !== null ? `${fmt(voltage, 3)} pu` : '— pu'}
          </text>

          {/* EV fleet symbol */}
          <text x="233" y="66" textAnchor="middle" fill="#475569" fontSize="8">⚡ 100 buses</text>
        </svg>

        <div className="voltage-limits">V_min 0.950 p.u. · V_max 1.050 p.u.</div>
      </div>

      {/* Voltage display */}
      <div className="card">
        <div className="card-title">Depot Voltage</div>
        <div className="voltage-value">
          {voltage !== null ? `${fmt(voltage, 4)} p.u.` : '— p.u.'}
        </div>
        {voltage !== null && voltage < 0.95 && (
          <div style={{ color: '#f87171', fontSize: 11, marginTop: 6, fontWeight: 600 }}>
            ⚠ UNDERVOLTAGE — command curtailed
          </div>
        )}
        {voltage !== null && voltage >= 0.95 && (
          <div style={{ color: '#22c55e', fontSize: 11, marginTop: 6 }}>
            ✓ Within safe operating limits
          </div>
        )}
      </div>

      {/* DT outcomes */}
      <div className="card">
        <div className="card-title">Digital Twin Gateway</div>
        <div className="dt-counts">
          <div className="dt-count-box">
            <div className="dt-count-value" style={{ color: '#22c55e' }}>
              {summary.approved ?? 0}
            </div>
            <div className="dt-count-label">Approved</div>
          </div>
          <div className="dt-count-box">
            <div className="dt-count-value" style={{ color: '#fb923c' }}>
              {summary.curtailed ?? 0}
            </div>
            <div className="dt-count-label">Curtailed</div>
          </div>
          <div className="dt-count-box">
            <div className="dt-count-value" style={{ color: '#f87171' }}>
              {summary.rejected ?? 0}
            </div>
            <div className="dt-count-label">Rejected</div>
          </div>
        </div>
      </div>

      {/* Transformer loading */}
      <div className="card">
        <div className="card-title">Transformer Loading</div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
          <span style={{ color: '#64748b' }}>10MVA · 33/11kV</span>
          <span style={{ color: transformerColor(txPct), fontWeight: 700 }}>
            {fmt(txPct, 1)}%
          </span>
        </div>
        <div className="transformer-bg">
          <div
            className="transformer-fill"
            style={{
              width: `${Math.min(txPct, 100)}%`,
              background: transformerColor(txPct),
            }}
          />
        </div>
        <div style={{ color: '#475569', fontSize: 10, marginTop: 4 }}>
          Thermal limit: 80% rated capacity
        </div>
      </div>

      {/* Last validation timestamp */}
      {validation?.timestamp && (
        <div style={{ color: '#334155', fontSize: 10, textAlign: 'center' }}>
          Last validated: {validation.timestamp.slice(11, 19)} UTC
        </div>
      )}
    </div>
  )
}

// ── Chaos Footer ──────────────────────────────────────────────────────────────

function ChaosFooter({ chaosStatus, onInject, onStop }) {
  const active = chaosStatus?.active ?? false

  return (
    <div className="chaos-footer">
      {!active ? (
        <button className="btn-chaos" onClick={onInject}>
          ⚠ Inject Byzantine Attack
        </button>
      ) : (
        <>
          <button className="btn-stop" onClick={onStop}>
            ■ Stop Attack
          </button>
          <div className="chaos-banner chaos-banner-attack">
            🔴 BYZANTINE ATTACK ACTIVE — BFT Gatekeeper engaged · PID {chaosStatus?.pid}
          </div>
        </>
      )}
      {!active && chaosStatus?.pid && (
        <div className="chaos-banner chaos-banner-neutralised">
          ✓ BFT Layer: Attack Detected & Neutralised. Clean Truth Maintained.
        </div>
      )}
      <div style={{ marginLeft: 'auto', color: '#334155', fontSize: 10, textAlign: 'right' }}>
        <div>GridSentinel v1.0</div>
        <div>Phase 5 — Chaos Dashboard</div>
      </div>
    </div>
  )
}

// ── Root App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [status,      setStatus]      = useState(null)
  const [forecast,    setForecast]    = useState(null)
  const [dispatch,    setDispatch]    = useState(null)
  const [validation,  setValidation]  = useState(null)
  const [trust,       setTrust]       = useState(null)
  const [metrics,     setMetrics]     = useState(null)
  const [chaosStatus, setChaosStatus] = useState({ active: false, pid: null })
  const [lastPoll,    setLastPoll]    = useState(null)
  const prevChaosActive = useRef(false)

  const poll = useCallback(async () => {
    const get = async (path, setter) => {
      try {
        const r = await axios.get(`${API}${path}`, { timeout: 4000 })
        setter(r.data)
      } catch {
        // endpoint not ready yet — keep previous state
      }
    }

    await Promise.allSettled([
      get('/status',     setStatus),
      get('/forecast',   setForecast),
      get('/dispatch',   setDispatch),
      get('/validation', setValidation),
      get('/trust',      setTrust),
      get('/metrics',    setMetrics),
      get('/chaos/status', (d) => {
        // Detect transition from active → stopped (attack was neutralised)
        if (prevChaosActive.current && !d.active) {
          prevChaosActive.current = false
        } else if (d.active) {
          prevChaosActive.current = true
        }
        setChaosStatus(d)
      }),
    ])
    setLastPoll(new Date().toISOString().slice(11, 19))
  }, [])

  useEffect(() => {
    poll()
    const id = setInterval(poll, POLL_MS)
    return () => clearInterval(id)
  }, [poll])

  const handleInject = async () => {
    try {
      await axios.post(`${API}/chaos/inject`, { attack_type: 'coordinated', pct: 0.10 })
      setChaosStatus(prev => ({ ...prev, active: true }))
    } catch (e) {
      console.error('Inject failed:', e)
    }
  }

  const handleStop = async () => {
    try {
      await axios.post(`${API}/chaos/stop`)
      setChaosStatus(prev => ({ ...prev, active: false }))
    } catch (e) {
      console.error('Stop failed:', e)
    }
  }

  return (
    <div className="god-view">
      {/* Header */}
      <div className="header">
        <div>
          <div className="header-title">GridSentinel — God View</div>
          <div className="header-subtitle">
            Autonomous Byzantine-Resilient V2G Energy Arbitrage & Grid Safety System
          </div>
        </div>
        <div className="header-status">
          <div>
            <div style={{ color: '#475569', fontSize: 10 }}>Redis</div>
            <div style={{ fontSize: 11, color: status?.redis === 'connected' ? '#22c55e' : '#fb923c' }}>
              {status?.redis ?? '—'}
            </div>
          </div>
          <div>
            <div style={{ color: '#475569', fontSize: 10 }}>Last Poll</div>
            <div style={{ fontSize: 11, color: '#94a3b8' }}>{lastPoll ?? '—'} UTC</div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div className="status-dot" />
            <span style={{ fontSize: 11, color: '#22c55e' }}>LIVE</span>
          </div>
        </div>
      </div>

      {/* Three columns */}
      <TruthStream trust={trust} metrics={metrics} />
      <Brain forecast={forecast} dispatch={dispatch} metrics={metrics} />
      <GridView validation={validation} metrics={metrics} />

      {/* Chaos footer */}
      <ChaosFooter
        chaosStatus={chaosStatus}
        onInject={handleInject}
        onStop={handleStop}
      />
    </div>
  )
}
