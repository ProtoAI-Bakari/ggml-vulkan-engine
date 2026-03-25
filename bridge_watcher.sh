#!/bin/bash
# Bridge Watcher — runs in background, prints new bridge entries to terminal
# Usage: source this in the agent's terminal, or run in a split pane

BRIDGE=~/AGENT/SYS0-LeadAgent-TaskAgent1-comms-bridge.md
TASK_Q=~/AGENT/TASK_QUEUE.md
LAST_SIZE=$(wc -c < $BRIDGE 2>/dev/null || echo 0)

echo '[WATCHER] Monitoring bridge for updates...'

while true; do
    # Watch for ANY .md file change in ~/AGENT
    inotifywait -q -e modify,create ~/AGENT/*.md ~/AGENT/.signals/* 2>/dev/null
    
    NEW_SIZE=$(wc -c < $BRIDGE 2>/dev/null || echo 0)
    if [ "$NEW_SIZE" -gt "$LAST_SIZE" ]; then
        DIFF=$((NEW_SIZE - LAST_SIZE))
        echo ''
        echo '╔══════════════════════════════════════════════════════╗'
        echo '║  🔔 BRIDGE UPDATE DETECTED — NEW CONTENT BELOW     ║'
        echo '╚══════════════════════════════════════════════════════╝'
        tail -c $DIFF $BRIDGE
        echo ''
        echo '════════════════════════════════════════════════════════'
        LAST_SIZE=$NEW_SIZE
    fi
    
    # Also check for signal files
    for sig in ~/AGENT/.signals/LEAD_TO_WORKER ~/AGENT/.signals/WORKER_TO_LEAD ~/AGENT/.signals/SYS12_TO_SYS0; do
        if [ -f "$sig" ] && [ "$sig" -nt ~/AGENT/.signals/.last_check 2>/dev/null ]; then
            echo "[SIGNAL] $(basename $sig): $(cat $sig)"
        fi
    done
    touch ~/AGENT/.signals/.last_check 2>/dev/null
done
