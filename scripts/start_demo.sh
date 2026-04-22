#!/usr/bin/env bash
# =============================================================================
# GridSentinel — Demo Start Script
# =============================================================================
# Starts the full demo stack in the correct order:
#   1. FastAPI backend        → http://localhost:8000
#   2. React God-View         → http://localhost:5173
#   3. BFT Gatekeeper watcher (background)
#
# Usage:
#   chmod +x scripts/start_demo.sh
#   ./scripts/start_demo.sh
#
# To run the chaos demo after startup:
#   python chaos/attacker.py --attack coordinated --pct 0.10 --cycles 999 --interval 5
#
# To stop everything:
#   ./scripts/stop_demo.sh
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         GridSentinel — Starting Demo         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════╝${NC}"
echo ""

# ---------------------------------------------------------------------------
# Check venv
# ---------------------------------------------------------------------------
if [ ! -f ".venv/bin/activate" ]; then
    echo -e "${RED}[ERROR] .venv not found. Run: poetry install${NC}"
    exit 1
fi

source .venv/bin/activate
echo -e "${GREEN}[OK]${NC} Virtual environment activated"

# ---------------------------------------------------------------------------
# Check required data files exist
# ---------------------------------------------------------------------------
MISSING=0

if [ ! -f "data/forecasts/latest_forecast.json" ]; then
    echo -e "${YELLOW}[WARN]${NC} No forecast found — run: python forecasting/inference.py --input data/clean_truth/latest.parquet"
    MISSING=1
fi

if [ ! -f "data/dispatch/latest_dispatch.json" ]; then
    echo -e "${YELLOW}[WARN]${NC} No dispatch found — run: python mpc/dispatch.py"
    MISSING=1
fi

if [ ! -f "data/validated/latest_validated.json" ]; then
    echo -e "${YELLOW}[WARN]${NC} No validation found — run: python digital_twin/validate.py"
    MISSING=1
fi

if [ $MISSING -eq 1 ]; then
    echo ""
    echo -e "${YELLOW}Some data files are missing. The dashboard will show partial data.${NC}"
    echo -e "${YELLOW}Run the pipeline first for the full demo experience.${NC}"
    echo ""
fi

# ---------------------------------------------------------------------------
# PID file directory
# ---------------------------------------------------------------------------
mkdir -p .demo_pids

# ---------------------------------------------------------------------------
# 1. Start FastAPI backend
# ---------------------------------------------------------------------------
echo -e "${BLUE}[1/3]${NC} Starting FastAPI backend..."

if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}      Port 8000 already in use — skipping${NC}"
else
    python api/main.py &
    API_PID=$!
    echo $API_PID > .demo_pids/api.pid
    sleep 2

    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        echo -e "${GREEN}[OK]${NC} API running → http://localhost:8000"
        echo -e "      Docs    → http://localhost:8000/docs"
    else
        echo -e "${RED}[ERROR]${NC} API failed to start. Check api/main.py"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 2. Start BFT Gatekeeper watcher
# ---------------------------------------------------------------------------
echo -e "${BLUE}[2/3]${NC} Starting BFT Gatekeeper watcher..."

if [ -f ".demo_pids/bft.pid" ] && kill -0 "$(cat .demo_pids/bft.pid)" 2>/dev/null; then
    echo -e "${YELLOW}      BFT watcher already running (PID $(cat .demo_pids/bft.pid))${NC}"
else
    python -m bft.gatekeeper --watch > .demo_pids/bft.log 2>&1 &
    BFT_PID=$!
    echo $BFT_PID > .demo_pids/bft.pid
    sleep 1
    echo -e "${GREEN}[OK]${NC} BFT Gatekeeper watching (PID $BFT_PID)"
    echo -e "      Logs    → tail -f .demo_pids/bft.log"
fi

# ---------------------------------------------------------------------------
# 3. Start React dashboard
# ---------------------------------------------------------------------------
echo -e "${BLUE}[3/3]${NC} Starting React God-View dashboard..."

if lsof -Pi :5173 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}      Port 5173 already in use — skipping${NC}"
else
    cd dashboard/god-view
    npm run dev -- --host > "$REPO_ROOT/.demo_pids/dashboard.log" 2>&1 &
    DASH_PID=$!
    echo $DASH_PID > "$REPO_ROOT/.demo_pids/dashboard.pid"
    cd "$REPO_ROOT"
    sleep 3
    echo -e "${GREEN}[OK]${NC} Dashboard running → http://localhost:5173"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           GridSentinel is Live               ║${NC}"
echo -e "${BLUE}╠══════════════════════════════════════════════╣${NC}"
echo -e "${BLUE}║${NC}  God-View Dashboard  →  http://localhost:5173  ${BLUE}║${NC}"
echo -e "${BLUE}║${NC}  API + Docs          →  http://localhost:8000  ${BLUE}║${NC}"
echo -e "${BLUE}║${NC}  BFT Logs            →  .demo_pids/bft.log     ${BLUE}║${NC}"
echo -e "${BLUE}╠══════════════════════════════════════════════╣${NC}"
echo -e "${BLUE}║${NC}  Chaos demo:                                   ${BLUE}║${NC}"
echo -e "${BLUE}║${NC}  python chaos/attacker.py --attack coordinated ${BLUE}║${NC}"
echo -e "${BLUE}║${NC}    --pct 0.10 --cycles 999 --interval 5        ${BLUE}║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════╝${NC}"
echo ""
echo -e "To stop all services: ${YELLOW}./scripts/stop_demo.sh${NC}"
echo ""