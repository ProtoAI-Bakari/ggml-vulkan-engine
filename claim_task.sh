#!/bin/bash
# claim_task.sh — Claim a task (cross-platform, no flock)
TASK=$1; AGENT=$2
[ -z "$TASK" ] || [ -z "$AGENT" ] && echo "Usage: $0 T07 'AgentName'" && exit 1

# Find the queue file
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

# Claim it (atomic sed)
sed -i.bak "/$TASK/s/\[READY\]/[IN_PROGRESS by $AGENT]/" "$QUEUE" 2>/dev/null || \
sed -i '' "/$TASK/s/\[READY\]/[IN_PROGRESS by $AGENT]/" "$QUEUE"
echo "CLAIMED $TASK for $AGENT"
