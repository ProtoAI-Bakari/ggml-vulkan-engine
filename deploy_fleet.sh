#!/bin/bash
# deploy_fleet.sh — Deploy MLX servers to sys2-7
# Usage: ./deploy_fleet.sh [deploy|start|stop|status|mounts]
set -e

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=5"
PASSFILE=~/DEV/authpass

declare -A HOSTS
HOSTS[sys2]="10.255.255.2"
HOSTS[sys3]="10.255.255.3"
HOSTS[sys5]="10.255.255.5"
HOSTS[sys6]="10.255.255.6"
HOSTS[sys7]="10.255.255.7"

ALL_SYS="sys2 sys3 sys5 sys6 sys7"

ssh_cmd() {
    sshpass -f "$PASSFILE" ssh $SSH_OPTS "z@$1" "$2"
}

scp_cmd() {
    sshpass -f "$PASSFILE" scp $SSH_OPTS "$1" "z@$2:$3"
}

deploy() {
    for sys in $ALL_SYS; do
        ip="${HOSTS[$sys]}"
        echo "━━━ Deploying to $sys ($ip) ━━━"

        ssh_cmd "$ip" "mkdir -p ~/AGENT/LOGS"
        scp_cmd ~/AGENT/mlx_server_${sys}.py "$ip" "~/AGENT/mlx_server_${sys}.py"

        for f in GO_PROMPT.md brain_bridge.py; do
            [ -f ~/AGENT/$f ] && scp_cmd ~/AGENT/$f "$ip" "~/AGENT/$f"
        done

        ssh_cmd "$ip" "cat > ~/AGENT/launch_mlx_${sys}.sh << 'SCRIPT'
#!/bin/bash
cd ~/AGENT
export PATH=\"\$HOME/Library/Python/3.9/bin:\$PATH\"
echo \"Starting MLX server for ${sys}...\"
PYTHON=\$HOME/.pyenv/versions/3.12.10/bin/python3
[ -x \$PYTHON ] || PYTHON=python3
nohup \$PYTHON mlx_server_${sys}.py > LOGS/${sys}_mlx.log 2>&1 &
echo \$! > .${sys}_mlx.pid
echo \"PID: \$(cat .${sys}_mlx.pid)\"
echo \"Log: ~/AGENT/LOGS/${sys}_mlx.log\"
SCRIPT
chmod +x ~/AGENT/launch_mlx_${sys}.sh"

        ssh_cmd "$ip" "cat > ~/AGENT/stop_mlx_${sys}.sh << 'SCRIPT'
#!/bin/bash
pkill -f mlx_server_${sys} 2>/dev/null && echo 'Stopped ${sys}' || echo '${sys} not running'
SCRIPT
chmod +x ~/AGENT/stop_mlx_${sys}.sh"

        echo "  Deployed mlx_server_${sys}.py"
    done
    echo ""
    echo "Done! Run: ./deploy_fleet.sh start"
}

start() {
    local targets="${1:-$ALL_SYS}"
    for sys in $targets; do
        ip="${HOSTS[$sys]}"
        echo "━━━ Starting $sys ($ip) ━━━"
        ssh_cmd "$ip" "cd ~/AGENT && export PATH=\$HOME/Library/Python/3.9/bin:\$PATH && bash launch_mlx_${sys}.sh"
    done
    echo ""
    echo "Waiting 30s for models to load..."
    sleep 30
    status
}

stop() {
    local targets="${1:-$ALL_SYS}"
    for sys in $targets; do
        ip="${HOSTS[$sys]}"
        echo "Stopping $sys ($ip)..."
        ssh_cmd "$ip" "pkill -f mlx_server_${sys} 2>/dev/null; echo done"
    done
}

mounts() {
    PASS=$(cat "$PASSFILE")
    for sys in $ALL_SYS; do
        ip="${HOSTS[$sys]}"
        echo "━━━ Mounting repos on $sys ($ip) ━━━"
        sshpass -f "$PASSFILE" ssh -tt $SSH_OPTS "z@$ip" "
echo '$PASS' | sudo -S mkdir -p ~/repo ~/slowrepo ~/vmrepo 2>/dev/null
echo '$PASS' | sudo -S umount ~/repo 2>/dev/null; echo '$PASS' | sudo -S mount -t nfs -o resvport,nfsvers=3 10.255.255.10:/repo ~/repo 2>&1
echo '$PASS' | sudo -S umount ~/slowrepo 2>/dev/null; echo '$PASS' | sudo -S mount -t nfs -o resvport,nfsvers=3 10.255.255.10:/slowrepo ~/slowrepo 2>&1
echo '$PASS' | sudo -S umount ~/vmrepo 2>/dev/null; echo '$PASS' | sudo -S mount -t nfs -o resvport,nfsvers=3 10.255.255.13:/vmrepo ~/vmrepo 2>&1
ls ~/repo/models/ >/dev/null 2>&1 && echo '$sys: repos OK' || echo '$sys: repos FAIL'
" 2>&1 | grep -E "OK|FAIL|mount_nfs"
    done
}

status() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "           FLEET STATUS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    for entry in "cluster:10.255.255.11:8000" "sys2:10.255.255.2:8000" "sys3:10.255.255.3:8000" "sys4:10.255.255.4:8000" "sys5:10.255.255.5:8000" "sys6:10.255.255.6:8000" "sys7:10.255.255.7:8000" "fast:localhost:8081"; do
        name="${entry%%:*}"; rest="${entry#*:}"; ip="${rest%%:*}"; port="${rest##*:}"
        result=$(curl -s --connect-timeout 2 "http://$ip:$port/health" 2>/dev/null)
        if [ -n "$result" ]; then
            model=$(echo "$result" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('model','?'))" 2>/dev/null)
            echo "  $name ($ip): UP | $model"
        else
            echo "  $name ($ip): DOWN"
        fi
    done
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

case "${1:-status}" in
    deploy) deploy ;;
    start)  start "$2" ;;
    stop)   stop "$2" ;;
    status) status ;;
    mounts) mounts ;;
    all)    deploy; start ;;
    *)      echo "Usage: $0 {deploy|start|stop|status|mounts|all} [sys2|sys3|...]" ;;
esac
