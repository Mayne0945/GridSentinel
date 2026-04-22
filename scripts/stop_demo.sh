#!/usr/bin/env bash
# =============================================================================
# GridSentinel — Demo Stop Script
# =============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "Stopping GridSentinel demo services..."
echo ""

stop_pid() {
    local name=$1
    local pidfile=".demo_pids/$2.pid"

    if [ -f "$pidfile" ]; then
        PID=$(cat "$pidfile")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID" 2>/dev/null
            echo -e "${GREEN}[STOPPED]${NC} $name (PID $PID)"
        else
            echo -e "  $name — already stopped"
        fi
        rm -f "$pidfile"
    else
        echo -e "  $name — no PID file found"
    fi
}

stop_pid "FastAPI backend"       "api"
stop_pid "BFT Gatekeeper"        "bft"
stop_pid "React dashboard"       "dashboard"

# Also kill any orphaned attacker processes
pkill -f "chaos/attacker.py" 2>/dev/null && \
    echo -e "${GREEN}[STOPPED]${NC} Chaos attacker" || true

echo ""
echo "All services stopped."
echo ""