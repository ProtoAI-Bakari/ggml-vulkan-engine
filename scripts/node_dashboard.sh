#!/bin/bash
# node_dashboard.sh — 3-pane tmux layout for a fleet node
# Usage: ssh into node, then: bash ~/AGENT/scripts/node_dashboard.sh
# Top: mactop | Middle: MLX server log | Bottom: agent trace

SYS=$(hostname -s | tr 'A-Z' 'a-z' | grep -o 'sys[0-9]*' || echo "unknown")

tmux new-session -d -s dashboard -x 200 -y 50

# Top pane: mactop (or htop if mactop not available)
tmux send-keys "mactop 2>/dev/null || htop 2>/dev/null || top" C-m

# Split horizontal for middle pane: MLX server log
tmux split-window -v -p 66
tmux send-keys "tail -f ~/AGENT/LOGS/${SYS}_mlx.log 2>/dev/null || echo 'No MLX log yet — start server with: agent'" C-m

# Split again for bottom pane: agent trace or shell
tmux split-window -v -p 50
tmux send-keys "echo 'Ready. Commands: agent (foreground) | agentd (daemon) | agentstop | agentstatus'" C-m

# Select middle pane
tmux select-pane -t 1

tmux attach -t dashboard
