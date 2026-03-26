#!/bin/bash
# claim_task.sh — Atomically claim a task across ALL queue files
TASK=$1; AGENT=$2; LOCK=~/AGENT/.task_lock
[ -z "$TASK" ] || [ -z "$AGENT" ] && echo "Usage: $0 T07 'AgentName'" && exit 1
exec 200>$LOCK; flock -n 200 || { echo "LOCKED"; exit 1; }

# Check BOTH queue files
for FILE in ~/AGENT/TASK_QUEUE_v4.md ~/AGENT/TASK_QUEUE_v5.md; do
  [ -f "$FILE" ] || continue
  if grep -q "$TASK.*\[IN_PROGRESS" "$FILE" 2>/dev/null; then
    OWNER=$(grep "$TASK" "$FILE" | head -1 | grep -oP 'IN_PROGRESS by [^]]*')
    echo "TAKEN — $OWNER (in $(basename $FILE))"
    flock -u 200; exit 1
  fi
done

# Claim in both files
for FILE in ~/AGENT/TASK_QUEUE_v4.md ~/AGENT/TASK_QUEUE_v5.md; do
  [ -f "$FILE" ] || continue
  sed -i "/$TASK/s/\[READY\]/[IN_PROGRESS by $AGENT]/" "$FILE"
done
echo "CLAIMED $TASK for $AGENT"
flock -u 200
