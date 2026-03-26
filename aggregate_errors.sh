#!/bin/bash
# Aggregate errors from all agent nodes into one file
echo "=== ERROR AGGREGATE $(date) ===" > ~/AGENT/LOGS/all_errors.log
for ip in 2 3 4 5 6 7; do
  echo "--- sys$ip ---" >> ~/AGENT/LOGS/all_errors.log
  sshpass -f ~/DEV/authpass ssh -o StrictHostKeyChecking=no -o ConnectTimeout=3 z@10.255.255.$ip \
    "strings ~/AGENT/LOGS/agent_trace.log 2>/dev/null | grep -E 'Error|SELF-CORRECTION|TIMEOUT|STREAM ERROR' | tail -10" \
    >> ~/AGENT/LOGS/all_errors.log 2>/dev/null
done
echo "--- sys1 ---" >> ~/AGENT/LOGS/all_errors.log
strings ~/AGENT/LOGS/main_trace.log 2>/dev/null | grep -E 'Error|SELF-CORRECTION|TIMEOUT|STREAM ERROR' | tail -10 >> ~/AGENT/LOGS/all_errors.log
echo "Aggregated to ~/AGENT/LOGS/all_errors.log"
