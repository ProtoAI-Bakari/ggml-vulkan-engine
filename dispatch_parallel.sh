#!/bin/bash
# dispatch_parallel.sh — Send task to BOTH brains in parallel, collect responses
TASK="$1"
[ -z "$TASK" ] && echo "Usage: $0 'task'" && exit 1

echo "=== Dispatching to BOTH brains in parallel ==="
echo "Task: $TASK"
echo ""

# Run both in parallel
(echo "=== CODER BRAIN (Sys4) ===" && ~/AGENT/dispatch_agent.sh coder "$TASK") > /tmp/coder_response.txt 2>&1 &
PID1=$!
(echo "=== REASONING BRAIN (.11) ===" && ~/AGENT/dispatch_agent.sh brain "$TASK") > /tmp/brain_response.txt 2>&1 &
PID2=$!

wait $PID1 $PID2

cat /tmp/coder_response.txt
echo ""
cat /tmp/brain_response.txt
