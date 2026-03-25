#!/bin/bash
LOGF=/home/z/AGENT/monitor_status.log
while true; do
    echo "========== $(date) ==========" >> "$LOGF"
    echo "AGENT:" >> "$LOGF"
    pgrep -af v44_GPU >> "$LOGF" 2>&1 || echo "NOT RUNNING" >> "$LOGF"
    echo "SERVER TPS:" >> "$LOGF"
    grep "generation throughput" /home/z/AGENT/LOGS/.latest 2>/dev/null | tail -3 >> "$LOGF"
    echo "BRIDGE (last 2):" >> "$LOGF"
    tail -2 /home/z/AGENT/bridge/agent_to_claude.jsonl >> "$LOGF" 2>/dev/null
    echo "TRACE (last line):" >> "$LOGF"
    tail -1 /home/z/AGENT/v44_GPU_trace.log >> "$LOGF" 2>/dev/null
    echo "MEMORY:" >> "$LOGF"
    free -h | head -2 >> "$LOGF"
    echo "" >> "$LOGF"
    sleep 300
done
