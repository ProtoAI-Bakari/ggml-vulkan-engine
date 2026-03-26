#!/bin/bash
# complete_task.sh — Mark a task DONE (uses v5 queue, matching claim_task.sh)
TASK=$1; AGENT=${2:-unknown}

# Find the queue file (same search order as claim_task.sh)
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
echo "COMPLETED $TASK"
