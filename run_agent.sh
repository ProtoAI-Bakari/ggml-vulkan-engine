#!/bin/bash
# run_agent.sh — Self-restarting agent wrapper
# The Python agent can die, update itself, or be killed — this loop restarts it
# Usage: ./run_agent.sh [agent_file] [log_file]
AGENT=${1:-OMNIAGENT_v4_focused.py}
LOG=${2:-LOGS/main_trace.log}
cd ~/AGENT
mkdir -p LOGS

echo "🔄 Agent launcher: $AGENT → $LOG"
echo "   Ctrl+C once = restart agent | Ctrl+C twice fast = kill launcher"

LAST_SIGINT=0
FIRST_RUN=1
trap 'NOW=$(date +%s); if [ $((NOW - LAST_SIGINT)) -lt 2 ]; then echo "💀 Launcher killed"; exit 0; fi; LAST_SIGINT=$NOW; echo "🔄 Restarting agent..."; kill %1 2>/dev/null' INT

while true; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚀 [$(date '+%H:%M:%S')] Starting $AGENT"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    source ~/.venv-vLLM_0.17.1_Stable/bin/activate 2>/dev/null
    # First run: interactive (no auto-go). Restarts: auto-go from GO_PROMPT.md
    if [ "$FIRST_RUN" -eq 1 ]; then
        python3 "$AGENT" 2>&1 | tee -a "$LOG"
        FIRST_RUN=0
    else
        echo "🔁 Auto-GO: loading GO_PROMPT.md on restart"
        python3 "$AGENT" --auto-go 2>&1 | tee -a "$LOG"
    fi
    EXIT_CODE=$?
    echo ""
    echo "⚠️  [$(date '+%H:%M:%S')] Agent exited (code $EXIT_CODE). Restarting in 3s..."
    sleep 3
done
