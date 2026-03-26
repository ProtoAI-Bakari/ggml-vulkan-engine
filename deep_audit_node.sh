#!/bin/bash
# Deep audit a single MLX node. Usage: deep_audit_node.sh <name> <ip>
NAME=$1; IP=$2
PASS_FILE=~/DEV/authpass
SSH="sshpass -f $PASS_FILE ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 z@$IP"

echo "══════════════════════════════════════════════"
echo "  DEEP AUDIT: $NAME ($IP) — $(date '+%Y-%m-%d %H:%M:%S')"
echo "══════════════════════════════════════════════"

# 1. Agent process
echo -e "\n[PROC]"
PROCS=$($SSH "ps -ef | grep -E 'OMNIAGENT|mlx_lm|python3.*serve' | grep -v grep" 2>&1)
echo "$PROCS"
AGENT_PID=$($SSH "pgrep -f OMNIAGENT | head -1" 2>/dev/null)
if [ -z "$AGENT_PID" ]; then
  echo "  !! AGENT NOT RUNNING !!"
else
  echo "  Agent PID: $AGENT_PID"
fi

# 2. Memory
echo -e "\n[MEM]"
$SSH "vm_stat 2>/dev/null | awk '/Pages free/{free=\$NF} /Pages active/{active=\$NF} END{printf \"Free: %.1fGB  Active: %.1fGB\n\", free*16384/1073741824, active*16384/1073741824}'" 2>&1

# 3. Disk  
echo -e "\n[DISK]"
$SSH "df -h / | tail -1 | awk '{printf \"Used: %s/%s (%s)  Avail: %s\n\", \$3, \$2, \$5, \$4}'" 2>&1

# 4. Model server
echo -e "\n[MODEL]"
MODEL=$(curl -s --max-time 5 http://$IP:8000/v1/models 2>/dev/null)
if echo "$MODEL" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'OK: {d[\"data\"][0][\"id\"]}')" 2>/dev/null; then
  :
else
  echo "  !! MODEL SERVER DOWN !!"
fi

# 5. Agent metrics from log
echo -e "\n[AGENT]"
$SSH "
  LOG=~/AGENT/LOGS/agent_trace.log
  echo \"Turn: \$(strings \$LOG 2>/dev/null | grep -oE 'Turn [0-9]+' | tail -1)\"
  echo \"TPS: \$(strings \$LOG 2>/dev/null | grep -oE '[0-9]+\.[0-9]+ t/s' | tail -1)\"
  echo \"TTFT: \$(strings \$LOG 2>/dev/null | grep -oE 'TTFT [0-9]+ms' | tail -1)\"
  echo \"Task: \$(strings \$LOG 2>/dev/null | grep -oE 'CLAIMED T[0-9]+' | tail -1)\"
  echo \"Log size: \$(ls -lh \$LOG 2>/dev/null | awk '{print \$5}')\"
  echo \"Errors (last 200 lines): \$(strings \$LOG 2>/dev/null | tail -200 | grep -ciE 'error|traceback|exception')\"
  echo \"Parse fails: \$(strings \$LOG 2>/dev/null | tail -200 | grep -c 'SELF-CORRECTION')\"
  echo \"Claim loops: \$(strings \$LOG 2>/dev/null | tail -200 | grep -c 'ALREADY_DONE')\"
  echo \"Stream errs: \$(strings \$LOG 2>/dev/null | tail -200 | grep -c 'STREAM ERROR')\"
" 2>&1

# 6. Last productive actions  
echo -e "\n[PRODUCTIVE]"
$SSH "strings ~/AGENT/LOGS/agent_trace.log 2>/dev/null | grep -E 'write_file|execute_bash|git commit|Successfully wrote|EXIT CODE|push_changes' | tail -5" 2>&1

# 7. Last 5 tool calls
echo -e "\n[TOOLS]"
$SSH "strings ~/AGENT/LOGS/agent_trace.log 2>/dev/null | grep -oE 'EXECUTING\]: [a-z_]+' | tail -5" 2>&1

# 8. Timestamps — is agent fresh?
echo -e "\n[FRESHNESS]"
$SSH "strings ~/AGENT/LOGS/agent_trace.log 2>/dev/null | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}' | tail -1" 2>&1

# 9. Git status
echo -e "\n[GIT]"
$SSH "cd ~/AGENT && git log --oneline -3 2>/dev/null || echo 'no git'" 2>&1

echo -e "\n══════════════════════════════════════════════"
