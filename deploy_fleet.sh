#!/bin/bash
# deploy_fleet.sh — Deploy MLX servers to sys5, sys6, sys7
# Usage: ./deploy_fleet.sh [deploy|start|stop|status]
set -e

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=5"
PASSFILE=~/DEV/authpass

declare -A HOSTS
HOSTS[sys5]="10.255.255.5"
HOSTS[sys6]="10.255.255.6"
HOSTS[sys7]="10.255.255.7"

ssh_cmd() {
    sshpass -f "$PASSFILE" ssh $SSH_OPTS "z@$1" "$2"
}

scp_cmd() {
    sshpass -f "$PASSFILE" scp $SSH_OPTS "$1" "z@$2:$3"
}

deploy() {
    for sys in sys5 sys6 sys7; do
        ip="${HOSTS[$sys]}"
        echo "━━━ Deploying to $sys ($ip) ━━━"

        # Create AGENT dir
        ssh_cmd "$ip" "mkdir -p ~/AGENT/LOGS"

        # Copy server file
        scp_cmd ~/AGENT/mlx_server_${sys}.py "$ip" "~/AGENT/mlx_server_${sys}.py"

        # Copy shared files
        for f in GO_PROMPT.md brain_bridge.py; do
            if [ -f ~/AGENT/$f ]; then
                scp_cmd ~/AGENT/$f "$ip" "~/AGENT/$f"
            fi
        done

        # Create launch script
        ssh_cmd "$ip" "cat > ~/AGENT/launch_mlx_${sys}.sh << 'SCRIPT'
#!/bin/bash
cd ~/AGENT
export PATH=\"\$HOME/Library/Python/3.9/bin:\$PATH\"
echo \"Starting MLX server for ${sys}...\"
nohup python3 mlx_server_${sys}.py > LOGS/${sys}_mlx.log 2>&1 &
echo \$! > .${sys}_mlx.pid
echo \"PID: \$(cat .${sys}_mlx.pid)\"
echo \"Log: ~/AGENT/LOGS/${sys}_mlx.log\"
SCRIPT
chmod +x ~/AGENT/launch_mlx_${sys}.sh"

        # Create stop script
        ssh_cmd "$ip" "cat > ~/AGENT/stop_mlx_${sys}.sh << 'SCRIPT'
#!/bin/bash
if [ -f ~/AGENT/.${sys}_mlx.pid ]; then
    kill \$(cat ~/AGENT/.${sys}_mlx.pid) 2>/dev/null && echo 'Stopped' || echo 'Not running'
    rm -f ~/AGENT/.${sys}_mlx.pid
else
    echo 'No PID file'
fi
SCRIPT
chmod +x ~/AGENT/stop_mlx_${sys}.sh"

        echo "  Deployed mlx_server_${sys}.py + launch/stop scripts"
    done
    echo ""
    echo "Done! Run: ./deploy_fleet.sh start"
}

start() {
    for sys in sys5 sys6 sys7; do
        ip="${HOSTS[$sys]}"
        echo "━━━ Starting $sys ($ip) ━━━"
        ssh_cmd "$ip" "cd ~/AGENT && bash launch_mlx_${sys}.sh"
    done
    echo ""
    echo "Waiting 30s for models to load..."
    sleep 30
    status
}

stop() {
    for sys in sys5 sys6 sys7; do
        ip="${HOSTS[$sys]}"
        echo "Stopping $sys ($ip)..."
        ssh_cmd "$ip" "cd ~/AGENT && bash stop_mlx_${sys}.sh 2>/dev/null || pkill -f mlx_server_${sys} 2>/dev/null; echo done"
    done
}

status() {
    echo "━━━ Fleet Status ━━━"
    for sys in sys5 sys6 sys7; do
        ip="${HOSTS[$sys]}"
        result=$(curl -s --connect-timeout 3 "http://$ip:8000/health" 2>/dev/null || echo '{"status":"DOWN"}')
        echo "  $sys ($ip): $result"
    done
    # Also check existing brains
    for check in "sys4:10.255.255.4:8000" "sys0-fast:127.0.0.1:8081"; do
        name="${check%%:*}"; rest="${check#*:}"; ip="${rest%%:*}"; port="${rest##*:}"
        result=$(curl -s --connect-timeout 3 "http://$ip:$port/health" 2>/dev/null || echo '{"status":"DOWN"}')
        echo "  $name ($ip:$port): $result"
    done
}

case "${1:-status}" in
    deploy) deploy ;;
    start)  start ;;
    stop)   stop ;;
    status) status ;;
    all)    deploy; start ;;
    *)      echo "Usage: $0 {deploy|start|stop|status|all}" ;;
esac
