#!/bin/bash
# sys10_aliases.sh — Paste these into Sys10's .zshrc or .bashrc
# Each creates a 3-pane tmux session: mactop | server log | shell

cat << 'ALIASES'
# ═══ ProtoAI-Bakari Fleet Dashboard Aliases ═══
# Usage: xd5 (opens sys5 dashboard), xd6, xd7, etc.

PASSFILE=~/DEV/authpass  # copy authpass to Sys10 too

xd() {
    local name=$1 ip=$2
    tmux kill-session -t $name 2>/dev/null
    tmux new-session -d -s $name -x 200 -y 50
    tmux send-keys "sshpass -f $PASSFILE ssh -o StrictHostKeyChecking=no z@$ip 'mactop 2>/dev/null || top'" C-m
    tmux split-window -v -p 66
    tmux send-keys "sshpass -f $PASSFILE ssh -o StrictHostKeyChecking=no z@$ip 'tail -f ~/AGENT/LOGS/${name}_mlx.log'" C-m
    tmux split-window -v -p 50
    tmux send-keys "sshpass -f $PASSFILE ssh -o StrictHostKeyChecking=no z@$ip" C-m
    tmux select-pane -t 1
    tmux attach -t $name
}

alias xd2='xd sys2 10.255.255.2'
alias xd3='xd sys3 10.255.255.3'
alias xd4='xd sys4 10.255.255.4'
alias xd5='xd sys5 10.255.255.5'
alias xd6='xd sys6 10.255.255.6'
alias xd7='xd sys7 10.255.255.7'

# Open ALL dashboards at once (one tmux session per node)
alias xdall='for n in xd2 xd3 xd4 xd5 xd6 xd7; do $n &; sleep 1; done'
ALIASES
