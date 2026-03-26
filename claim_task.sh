#!/bin/bash
# claim_task.sh — Atomically claim ONE task (enforces single-task per agent)
TASK=$1; AGENT=$2; LOCK=~/AGENT/.task_lock
[ -z "$TASK" ] || [ -z "$AGENT" ] && echo "Usage: $0 T07 'AgentName'" && exit 1
exec 200>$LOCK; flock -n 200 || { echo "LOCKED"; exit 1; }

# Check if THIS agent already has a task in progress
for FILE in ~/AGENT/TASK_QUEUE_v4.md ~/AGENT/TASK_QUEUE_v5.md; do
  [ -f "$FILE" ] || continue
  EXISTING=$(grep "IN_PROGRESS by $AGENT" "$FILE" | head -1 | grep -oP 'T\d+' | head -1)
  if [ -n "$EXISTING" ]; then
    echo "BLOCKED — $AGENT already has $EXISTING in progress. Complete it first."
    flock -u 200; exit 1
  fi
done

# Check if task is already taken
for FILE in ~/AGENT/TASK_QUEUE_v4.md ~/AGENT/TASK_QUEUE_v5.md; do
  [ -f "$FILE" ] || continue
  if grep -q "$TASK.*\[IN_PROGRESS" "$FILE" 2>/dev/null; then
    OWNER=$(grep "$TASK" "$FILE" | head -1 | grep -oP 'IN_PROGRESS by [^]]*')
    echo "TAKEN — $OWNER (in $(basename $FILE))"
    flock -u 200; exit 1
  fi
done

# Check task exists and is READY
FOUND=0
for FILE in ~/AGENT/TASK_QUEUE_v4.md ~/AGENT/TASK_QUEUE_v5.md; do
  [ -f "$FILE" ] || continue
  if grep -q "$TASK.*\[READY\]" "$FILE" 2>/dev/null; then
    FOUND=1
  fi
done
[ "$FOUND" -eq 0 ] && echo "NOT FOUND — $TASK is not READY" && flock -u 200 && exit 1

# Claim in both files
for FILE in ~/AGENT/TASK_QUEUE_v4.md ~/AGENT/TASK_QUEUE_v5.md; do
  [ -f "$FILE" ] || continue
  sed -i "/$TASK/s/\[READY\]/[IN_PROGRESS by $AGENT]/" "$FILE"
done
echo "CLAIMED $TASK for $AGENT"
flock -u 200
