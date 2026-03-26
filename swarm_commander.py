#!/usr/bin/env python3
"""
ProtoAI-Bakari Agentic AI Cluster Orchestrator v1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Live dashboard + command center for the hybrid AI micro-cluster.
  macOS MLX (sys2-7) | Asahi Linux Vulkan (sys0) | CUDA cluster (.11)

Usage: python3 swarm_commander.py
"""

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box

# ── Fleet Configuration ─────────────────────────────────────────────────────
PASSFILE = os.path.expanduser("~/DEV/authpass")
SSH_OPTS = "-o StrictHostKeyChecking=no -o ConnectTimeout=3"

NODES = {
    "sys0":  {"ip": "127.0.0.1",      "port": 8081, "role": "AGENT-HOST",  "chip": "M1 Ultra",  "ram": "128G", "os": "Asahi Linux"},
    "sys2":  {"ip": "10.255.255.2",    "port": 8000, "role": "ARCHITECT",   "chip": "M2 Ultra",  "ram": "192G", "os": "macOS"},
    "sys3":  {"ip": "10.255.255.3",    "port": 8000, "role": "ENGINEER",    "chip": "M2 Ultra",  "ram": "192G", "os": "macOS"},
    "sys4":  {"ip": "10.255.255.4",    "port": 8000, "role": "CODER",       "chip": "M1 Ultra",  "ram": "128G", "os": "macOS"},
    "sys5":  {"ip": "10.255.255.5",    "port": 8000, "role": "DESIGNER",    "chip": "M1 Ultra",  "ram": "128G", "os": "macOS"},
    "sys6":  {"ip": "10.255.255.6",    "port": 8000, "role": "REVIEWER",    "chip": "M1 Ultra",  "ram": "128G", "os": "macOS"},
    "sys7":  {"ip": "10.255.255.7",    "port": 8000, "role": "FAST-CODER",  "chip": "M1 Ultra",  "ram": "128G", "os": "macOS"},
    "cluster": {"ip": "10.255.255.11", "port": 8000, "role": "GPU-CLUSTER", "chip": "8x3090",    "ram": "256G", "os": "Linux"},
}

AGENTS = {
    "agent1": {"script": "OMNIAGENT_v4_focused.py",         "log": "LOGS/main_trace.log",     "name": "OmniAgent [Main]"},
    "agent2": {"script": "agents/OMNIAGENT_v4_sys4.py",     "log": "LOGS/sys4_trace.log",     "name": "OmniAgent [Sys4]"},
    "agent3": {"script": "agents/OMNIAGENT_v4_cluster2.py", "log": "LOGS/cluster2_trace.log", "name": "OmniAgent [Cluster2]"},
    "agent4": {"script": "OMNIAGENT_v4_focused.py",         "log": "LOGS/agent4_trace.log",   "name": "OmniAgent [Worker4]"},
    "agent5": {"script": "OMNIAGENT_v4_focused.py",         "log": "LOGS/agent5_trace.log",   "name": "OmniAgent [Worker5]"},
    "agent6": {"script": "OMNIAGENT_v4_focused.py",         "log": "LOGS/agent6_trace.log",   "name": "OmniAgent [Worker6]"},
}

console = Console()

# ── Utilities ────────────────────────────────────────────────────────────────

def ssh_cmd(ip, cmd, timeout=5):
    """Run command on remote macOS node via sshpass."""
    if ip in ("127.0.0.1", "localhost"):
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            return r.stdout.strip()
        except Exception:
            return ""
    try:
        full = f"sshpass -f {PASSFILE} ssh {SSH_OPTS} z@{ip} \"{cmd}\""
        r = subprocess.run(full, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""

def health_check(name, node):
    """Check brain server health endpoint."""
    import requests
    ip, port = node["ip"], node["port"]
    try:
        r = requests.get(f"http://{ip}:{port}/health", timeout=3)
        d = r.json()
        model = d.get("model", "?").split("/")[-1][:35]
        role = d.get("role", node["role"])
        active = d.get("active", 0)
        return {"status": "UP", "model": model, "role": role, "active": active}
    except Exception:
        return {"status": "DOWN", "model": "-", "role": node["role"], "active": 0}

def get_mem_usage(ip):
    """Get memory usage % from remote node."""
    if ip in ("127.0.0.1", "localhost"):
        try:
            import psutil
            return f"{psutil.virtual_memory().percent:.0f}%"
        except Exception:
            return "?"
    out = ssh_cmd(ip, "ps -A -o %mem | awk '{s+=$1} END {printf \"%.0f%%\", s}'", timeout=3)
    return out if out else "?"

def count_tasks():
    """Count tasks from TASK_QUEUE_v5.md."""
    path = os.path.expanduser("~/AGENT/TASK_QUEUE_v5.md")
    done = ready = progress = 0
    try:
        for line in open(path):
            if "[DONE" in line: done += 1
            elif "[IN_PROGRESS" in line: progress += 1
            elif "[READY]" in line: ready += 1
    except FileNotFoundError:
        pass
    return done, progress, ready

def check_agents():
    """Check which agents are running on sys0."""
    running = []
    try:
        r = subprocess.run("ps aux | grep OMNIAGENT | grep -v grep", shell=True,
                          capture_output=True, text=True, timeout=3)
        for line in r.stdout.strip().split("\n"):
            if "focused" in line: running.append("agent1")
            elif "sys4" in line: running.append("agent2")
            elif "cluster2" in line: running.append("agent3")
    except Exception:
        pass
    return running

# ── Fleet Operations ─────────────────────────────────────────────────────────

def poll_fleet():
    """Poll all nodes in parallel, return status dict."""
    status = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {}
        for name, node in NODES.items():
            futures[pool.submit(health_check, name, node)] = name
        for f in as_completed(futures):
            name = futures[f]
            try:
                status[name] = f.result()
            except Exception:
                status[name] = {"status": "ERR", "model": "-", "role": NODES[name]["role"], "active": 0}
    return status

def launch_brain(name):
    """Start MLX server on a remote node."""
    node = NODES.get(name)
    if not node or name in ("sys0", "cluster"):
        console.print(f"[red]Cannot launch brain on {name}[/red]")
        return
    ip = node["ip"]
    console.print(f"[yellow]Starting brain on {name} ({ip})...[/yellow]")
    # Check if already running — don't kill existing foreground servers
    import requests
    try:
        r = requests.get(f"http://{ip}:{node['port']}/health", timeout=2)
        if r.status_code == 200:
            console.print(f"[green]{name}: already running (skipped)[/green]")
            return
    except Exception:
        pass
    ssh_cmd(ip, f"cd ~/AGENT && "
               f"nohup ~/.pyenv/versions/3.12.10/bin/python3 mlx_server_{name}.py "
               f"> LOGS/{name}_mlx.log 2>&1 &", timeout=10)
    console.print(f"[green]{name}: brain launching[/green]")

def stop_brain(name):
    """Stop MLX server on a remote node."""
    node = NODES.get(name)
    if not node:
        return
    ssh_cmd(node["ip"], "pkill -f mlx_server", timeout=5)
    console.print(f"[red]{name}: brain stopped[/red]")

def launch_agent(agent_id):
    """Start an agent on sys0 in a tmux session."""
    agent = AGENTS.get(agent_id)
    if not agent:
        console.print(f"[red]Unknown agent: {agent_id}[/red]")
        return
    script = agent["script"]
    log = agent["log"]
    # Launch in tmux so it persists
    subprocess.Popen(
        f"tmux new-session -d -s {agent_id} 'cd ~/AGENT && source ~/.venv-vLLM_0.17.1_Stable/bin/activate && "
        f"python3 {script} 2>&1 | tee -a {log}'",
        shell=True
    )
    console.print(f"[green]{agent_id}: launched in tmux session '{agent_id}'[/green]")
    console.print(f"[dim]  Attach: tmux attach -t {agent_id}[/dim]")

def stop_agent(agent_id):
    """Stop an agent tmux session."""
    subprocess.run(f"tmux kill-session -t {agent_id} 2>/dev/null", shell=True)
    console.print(f"[red]{agent_id}: stopped[/red]")

def goal_all():
    """GO ALL — launch all brains and agents."""
    console.print("[bold magenta]━━━ GOAL: LAUNCHING ENTIRE SWARM ━━━[/bold magenta]")
    # Launch all brains in parallel
    with ThreadPoolExecutor(max_workers=6) as pool:
        for name in ["sys2", "sys3", "sys5", "sys6", "sys7"]:  # sys4 usually already running
            pool.submit(launch_brain, name)
    console.print("[yellow]Waiting 20s for models to load...[/yellow]")
    time.sleep(20)
    # Launch agents
    for aid in AGENTS:
        launch_agent(aid)
    console.print("[bold green]━━━ SWARM LAUNCHED ━━━[/bold green]")

def stop_all():
    """Stop all brains and agents."""
    console.print("[bold red]━━━ STOPPING ENTIRE SWARM ━━━[/bold red]")
    for name in NODES:
        if name not in ("sys0", "cluster", "sys4"):
            stop_brain(name)
    for aid in AGENTS:
        stop_agent(aid)
    console.print("[bold red]━━━ SWARM STOPPED ━━━[/bold red]")

def council_query(question):
    """Send a question to architect, engineer, designer — collect answers."""
    import requests
    leads = {
        "ARCHITECT": ("10.255.255.2", 8000),
        "ENGINEER":  ("10.255.255.3", 8000),
        "DESIGNER":  ("10.255.255.5", 8000),
    }
    console.print(f"\n[bold cyan]━━━ COUNCIL QUERY ━━━[/bold cyan]")
    console.print(f"[white]{question}[/white]\n")

    def ask_lead(role, ip, port):
        try:
            r = requests.post(f"http://{ip}:{port}/v1/chat/completions",
                json={"messages": [{"role": "user", "content": question}], "max_tokens": 500, "stream": False},
                timeout=120)
            d = r.json()
            content = d["choices"][0]["message"]["content"]
            # Strip thinking tags
            import re
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            tps = d.get("x_metrics", {}).get("tps", 0)
            return role, content[:600], tps
        except Exception as e:
            return role, f"ERROR: {e}", 0

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(ask_lead, role, ip, port) for role, (ip, port) in leads.items()]
        for f in as_completed(futures):
            role, answer, tps = f.result()
            color = {"ARCHITECT": "red", "ENGINEER": "yellow", "DESIGNER": "cyan"}.get(role, "white")
            console.print(Panel(answer, title=f"[bold {color}]{role}[/bold {color}] ({tps:.0f} TPS)", border_style=color))

# ── Display ──────────────────────────────────────────────────────────────────

def build_dashboard(fleet_status):
    """Build the rich dashboard table."""
    now = datetime.now().strftime("%H:%M:%S")
    done, progress, ready = count_tasks()
    running_agents = check_agents()

    # Node table
    table = Table(title=f"SWARM STATUS — {now}", box=box.DOUBLE_EDGE, show_lines=False,
                  title_style="bold white on blue", padding=(0, 1))
    table.add_column("Node", style="bold", width=8)
    table.add_column("Role", width=12)
    table.add_column("Model", width=36)
    table.add_column("Chip", width=10)
    table.add_column("RAM", width=5, justify="right")
    table.add_column("ST", width=4, justify="center")
    table.add_column("Act", width=3, justify="center")

    for name, node in NODES.items():
        st = fleet_status.get(name, {})
        status = st.get("status", "?")
        model = st.get("model", "-")
        role = st.get("role", node["role"])
        active = st.get("active", 0)

        if status == "UP":
            st_str = "[green]UP[/green]"
        elif status == "DOWN":
            st_str = "[red]DN[/red]"
        else:
            st_str = "[yellow]??[/yellow]"

        role_colors = {
            "ARCHITECT": "red", "ENGINEER": "yellow", "CODER": "green",
            "DESIGNER": "cyan", "REVIEWER": "bright_green", "FAST-CODER": "magenta",
            "AGENT-HOST": "blue", "GPU-CLUSTER": "bright_red",
        }
        rc = role_colors.get(role, "white")

        table.add_row(name, f"[{rc}]{role}[/{rc}]", model, node["chip"], node["ram"],
                      st_str, str(active) if active else "")

    # Summary
    up_count = sum(1 for s in fleet_status.values() if s.get("status") == "UP")
    total = len(NODES)
    agent_str = ", ".join(running_agents) if running_agents else "none"

    summary = (
        f"[bold]Brains:[/bold] {up_count}/{total} online  |  "
        f"[bold]Agents:[/bold] {len(running_agents)} running ({agent_str})  |  "
        f"[bold]Tasks:[/bold] {done} done / {progress} active / {ready} ready"
    )

    return table, summary

def show_dashboard():
    """Show a one-shot dashboard."""
    fleet_status = poll_fleet()
    table, summary = build_dashboard(fleet_status)
    console.print()
    console.print(table)
    console.print(f"  {summary}")
    console.print()

def show_help():
    help_text = """[bold cyan]Commands:[/bold cyan]
  [bold]fleet[/bold] / [bold]status[/bold]    — Refresh dashboard
  [bold]goal[/bold]              — GO ALL: launch all brains + agents
  [bold]stop[/bold]              — Stop all brains + agents
  [bold]launch[/bold] <node>     — Start brain on node (sys2-sys7)
  [bold]kill[/bold] <node>       — Stop brain on node
  [bold]agent[/bold] <1|2|3>     — Start agent on sys0 (in tmux)
  [bold]agentstop[/bold] <1|2|3> — Stop agent
  [bold]council[/bold] <question> — Ask architect+engineer+designer
  [bold]logs[/bold] <node>       — Tail node's MLX log (Ctrl+C to stop)
  [bold]tasks[/bold]             — Show task queue summary
  [bold]swap[/bold] <node> <model> — Change model on node (TODO v2)
  [bold]q[/bold] / [bold]quit[/bold]          — Exit"""
    console.print(Panel(help_text, title="[bold]ProtoAI-Bakari Commander[/bold]", border_style="blue"))

# ── Main Loop ────────────────────────────────────────────────────────────────

def main():
    console.print(Panel(
        "[bold white]ProtoAI-Bakari Agentic AI Cluster Orchestrator v1[/bold white]\n"
        "[dim]macOS MLX  ·  Asahi Vulkan  ·  CUDA Cluster  ·  6 Specialized Brains[/dim]",
        border_style="bright_blue", padding=(1, 2)
    ))

    show_dashboard()
    show_help()

    while True:
        try:
            cmd = console.input("\n[bold blue]commander>[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Exiting commander.[/dim]")
            break

        if not cmd:
            continue

        parts = cmd.split(maxsplit=1)
        action = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if action in ("q", "quit", "exit"):
            break
        elif action in ("fleet", "status", "s", "f"):
            show_dashboard()
        elif action == "goal":
            goal_all()
            time.sleep(5)
            show_dashboard()
        elif action == "stop":
            stop_all()
        elif action == "launch":
            if arg:
                launch_brain(arg)
            else:
                console.print("[red]Usage: launch <node>[/red]")
        elif action == "kill":
            if arg:
                stop_brain(arg)
            else:
                console.print("[red]Usage: kill <node>[/red]")
        elif action in ("agent",):
            aid = f"agent{arg}" if arg.isdigit() else arg
            launch_agent(aid)
        elif action == "agentstop":
            aid = f"agent{arg}" if arg.isdigit() else arg
            stop_agent(aid)
        elif action == "council":
            if arg:
                council_query(arg)
            else:
                console.print("[red]Usage: council <question>[/red]")
        elif action == "logs":
            if arg:
                node = NODES.get(arg)
                if node and arg != "sys0":
                    console.print(f"[dim]Tailing {arg} logs (Ctrl+C to stop)...[/dim]")
                    try:
                        os.system(f"sshpass -f {PASSFILE} ssh {SSH_OPTS} z@{node['ip']} 'tail -f ~/AGENT/LOGS/{arg}_mlx.log'")
                    except KeyboardInterrupt:
                        pass
                elif arg == "sys0":
                    console.print("[dim]Tailing sys0 agent logs...[/dim]")
                    try:
                        os.system("tail -f ~/AGENT/LOGS/main_trace.log")
                    except KeyboardInterrupt:
                        pass
                else:
                    console.print(f"[red]Unknown node: {arg}[/red]")
            else:
                console.print("[red]Usage: logs <node>[/red]")
        elif action == "tasks":
            done, progress, ready = count_tasks()
            console.print(f"  [green]Done:[/green] {done}  [yellow]Active:[/yellow] {progress}  [white]Ready:[/white] {ready}")
            try:
                # Show first few active/ready tasks
                for line in open(os.path.expanduser("~/AGENT/TASK_QUEUE_v5.md")):
                    if "[IN_PROGRESS" in line or "[READY]" in line:
                        console.print(f"  {line.strip()}")
            except FileNotFoundError:
                pass
        elif action == "help":
            show_help()
        else:
            console.print(f"[red]Unknown command: {action}[/red]. Type 'help' for commands.")

if __name__ == "__main__":
    main()
