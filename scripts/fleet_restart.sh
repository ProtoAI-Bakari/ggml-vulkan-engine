#!/bin/bash
# Fleet restart: sync code + restart all agents
# Usage: ./scripts/fleet_restart.sh [node1 node2 ...]

PASSFILE=~/DEV/authpass
SSH="sshpass -f $PASSFILE ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5"
SCP="sshpass -f $PASSFILE scp -o StrictHostKeyChecking=no"
NODES="${@:-2 3 4 5 6 7}"

echo "=== FLEET RESTART ==="

# 1. Kill all agents
for ip in $NODES; do
  $SSH z@10.255.255.$ip "pkill -9 -f OMNIAGENT 2>/dev/null" &
done
wait
echo "Agents killed"

# 2. Sync code
FILES="OMNIAGENT_v4_focused.py GO_PROMPT.md TASK_QUEUE_v5.md claim_task.sh complete_task.sh KNOWLEDGE_BASE.md"
for ip in $NODES; do
  (
    for f in $FILES; do
      [ -f ~/AGENT/$f ] && $SCP ~/AGENT/$f z@10.255.255.$ip:~/AGENT/ 2>/dev/null
    done
    # Patch agent name
    $SSH z@10.255.255.$ip "cd ~/AGENT && ~/.pyenv/versions/3.12.10/bin/python3 -c \"c=open('OMNIAGENT_v4_focused.py').read();c=c.replace('default=\\\"OmniAgent [Main]\\\"','default=\\\"OmniAgent [sys$ip]\\\"');open('OMNIAGENT_v4_focused.py','w').write(c)\"" 2>/dev/null
    echo "sys$ip: synced"
  ) &
done
wait
echo "Code synced"

# 3. Launch agents staggered
for ip in $NODES; do
  $SSH z@10.255.255.$ip "cd ~/AGENT && nohup ~/.pyenv/versions/3.12.10/bin/python3 OMNIAGENT_v4_focused.py --auto-go --name 'OmniAgent [sys$ip]' > LOGS/agent_trace.log 2>&1 &" 2>/dev/null
  echo "sys$ip: launched"
  sleep 2
done

echo "=== FLEET RESTART COMPLETE ==="
