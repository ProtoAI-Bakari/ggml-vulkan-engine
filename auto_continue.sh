#!/bin/bash
# Run this in a SEPARATE terminal alongside the agent.
# It watches for agent idle and reminds you to type 'go'.
# Or pipe it to the agent's terminal.

echo '[AUTO-CONTINUE] Watching for agent idle...'
echo 'When you see the prompt, just type: go'
echo 'Or paste: Read ~/AGENT/TASK_QUEUE.md and continue the next READY task. Do not stop.'
echo ''

while true; do
    inotifywait -q -e modify ~/AGENT/SYS0-LeadAgent-TaskAgent1-comms-bridge.md 2>/dev/null
    echo ''
    echo '⚡ NEW BRIDGE UPDATE — Tell the agent:'
    echo '   "Check bridge. New update from Sys12. Continue next task."'
    echo ''
done
