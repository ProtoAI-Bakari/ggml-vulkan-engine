#!/bin/bash
# claim_task.sh — Claim a task via central API (fallback to local edit)
# API runs on sys1 at 10.255.255.128:9091 for atomic cross-node claims.
TASK=$1; AGENT=$2
[ -z "$TASK" ] || [ -z "$AGENT" ] && echo "Usage: $0 T07 'AgentName'" && exit 1

TASK_API="http://10.255.255.128:9091"

# --- Try central API first (atomic, no race conditions) ---
if command -v curl >/dev/null 2>&1; then
    RESP=$(curl -s --max-time 3 -X POST "$TASK_API/claim" -d "task=$TASK&agent=$AGENT" 2>/dev/null)
    if [ -n "$RESP" ]; then
        OK=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('ok',''))" 2>/dev/null)
        MSG=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('msg',''))" 2>/dev/null)
        if [ "$OK" = "True" ]; then
            echo "CLAIMED $TASK for $AGENT (via API)"
            exit 0
        else
            echo "$MSG"
            exit 1
        fi
    fi
    echo "Task API unreachable. Retrying..." >&2
    sleep 3
    RESP=$(curl -s --max-time 5 -X POST "$TASK_API/claim" -d "task=$TASK&agent=$AGENT" 2>/dev/null)
    if [ -n "$RESP" ]; then
        OK=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('ok',''))" 2>/dev/null)
        MSG=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('msg',''))" 2>/dev/null)
        [ "$OK" = "True" ] && echo "CLAIMED $TASK for $AGENT (via API, retry)" && exit 0
        echo "$MSG"; exit 1
    fi
    echo "FAILED: Task API unreachable. Use execute_bash to work on your existing task."
    exit 1
fi

echo "FAILED: curl not available. Use execute_bash to work on your existing task."
exit 1
