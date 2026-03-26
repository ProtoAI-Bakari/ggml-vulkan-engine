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
    echo "WARNING: Task API unreachable, falling back to local edit" >&2
fi

# --- Fallback: local file edit (original behavior) ---
for F in ~/AGENT/TASK_QUEUE_v5.md /Users/z/AGENT/TASK_QUEUE_v5.md; do
    [ -f "$F" ] && QUEUE="$F" && break
done
[ -z "$QUEUE" ] && echo "NO QUEUE FILE FOUND" && exit 1

# Check if task is READY
grep -q "$TASK.*\[READY\]" "$QUEUE" || { echo "NOT READY — $TASK is not available"; exit 1; }

# Check if this agent already has a task
EXISTING=$(grep "IN_PROGRESS by $AGENT" "$QUEUE" | head -1 | grep -oE 'T[0-9]+' | head -1)
if [ -n "$EXISTING" ]; then
    echo "BLOCKED — $AGENT already has $EXISTING. Complete it first."
    exit 1
fi

# Claim it (local sed — NOT atomic across nodes)
sed -i.bak "/$TASK/s/\[READY\]/[IN_PROGRESS by $AGENT]/" "$QUEUE" 2>/dev/null || \
sed -i '' "/$TASK/s/\[READY\]/[IN_PROGRESS by $AGENT]/" "$QUEUE"
echo "CLAIMED $TASK for $AGENT (local fallback)"
