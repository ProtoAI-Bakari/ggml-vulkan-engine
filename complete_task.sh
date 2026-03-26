#!/bin/bash
# complete_task.sh — Mark a task DONE via central API (fallback to local edit)
# API runs on sys1 at 10.255.255.128:9091 for atomic cross-node updates.
TASK=$1; AGENT=${2:-unknown}

TASK_API="http://10.255.255.128:9091"

# --- Try central API first (atomic, no race conditions) ---
if command -v curl >/dev/null 2>&1; then
    RESP=$(curl -s --max-time 3 -X POST "$TASK_API/complete" -d "task=$TASK&agent=$AGENT" 2>/dev/null)
    if [ -n "$RESP" ]; then
        OK=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('ok',''))" 2>/dev/null)
        MSG=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('msg',''))" 2>/dev/null)
        if [ "$OK" = "True" ]; then
            echo "COMPLETED $TASK (via API)"
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
    [ -f "$F" ] && FILE="$F" && break
done
[ -z "$FILE" ] && echo "NO QUEUE FILE FOUND" && exit 1

# Use flock if available for atomic write
if command -v flock >/dev/null 2>&1; then
    (
        flock -w 5 200 || { echo "LOCK TIMEOUT"; exit 1; }
        sed -i.bak "/$TASK/s/\[IN_PROGRESS by [^]]*\]/[DONE by $AGENT]/" "$FILE"
    ) 200>"$FILE.lock"
else
    sed -i.bak "/$TASK/s/\[IN_PROGRESS by [^]]*\]/[DONE by $AGENT]/" "$FILE" 2>/dev/null || \
    sed -i '' "/$TASK/s/\[IN_PROGRESS by [^]]*\]/[DONE by $AGENT]/" "$FILE"
fi
echo "COMPLETED $TASK (local fallback)"
