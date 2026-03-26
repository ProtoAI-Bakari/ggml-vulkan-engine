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
    # ── Apple Silicon MLX Fleet ──
    "sys1":      {"ip": "127.0.0.1",      "port": 8081, "role": "AGENT-HOST",  "chip": "M1 Ultra",  "ram": "128G", "os": "Asahi"},
    "sys2":      {"ip": "10.255.255.2",    "port": 8000, "role": "ARCHITECT",   "chip": "M2 Ultra",  "ram": "192G", "os": "macOS"},
    "sys3":      {"ip": "10.255.255.3",    "port": 8000, "role": "ENGINEER",    "chip": "M2 Ultra",  "ram": "192G", "os": "macOS"},
    "sys4":      {"ip": "10.255.255.4",    "port": 8000, "role": "CODER",       "chip": "M1 Ultra",  "ram": "128G", "os": "macOS"},
    "sys5":      {"ip": "10.255.255.5",    "port": 8000, "role": "DESIGNER",    "chip": "M1 Ultra",  "ram": "128G", "os": "macOS"},
    "sys6":      {"ip": "10.255.255.6",    "port": 8000, "role": "REVIEWER",    "chip": "M1 Ultra",  "ram": "128G", "os": "macOS"},
    "sys7":      {"ip": "10.255.255.7",    "port": 8000, "role": "FAST-CODER",  "chip": "M1 Ultra",  "ram": "128G", "os": "macOS"},
    # ── CUDA Cluster (8x3090 TP8 over 100G RoCEv2) ──
    "cuda-sys1": {"ip": "10.255.255.11", "port": 8000, "role": "CUDA-BRAIN",   "chip": "2x3090",  "ram": "256G", "os": "Linux"},
    "cuda-sys2": {"ip": "10.255.255.12", "port": 0,    "role": "CUDA-WORK",    "chip": "2x3090",  "ram": "128G", "os": "Linux"},
    "cuda-sys3": {"ip": "10.255.255.13", "port": 0,    "role": "CUDA-WORK",    "chip": "2x3090",  "ram": "64G",  "os": "Linux"},
    "cuda-sys4": {"ip": "10.255.255.14", "port": 0,    "role": "CUDA-WORK",    "chip": "2x3090",  "ram": "64G",  "os": "Linux"},
    # ── Standalone GPU ──
    "gpu-10":    {"ip": "10.255.255.10", "port": 8000, "role": "HYPER-CODER",  "chip": "1x4090",  "ram": "64G",  "os": "Linux"},
}
# Aliases so old names still work
NODE_ALIASES = {
    "mlx-0": "sys1", "mlx-1": "sys1", "sys0": "sys1",
    "mlx-2": "sys2", "mlx-3": "sys3", "mlx-4": "sys4",
    "mlx-5": "sys5", "mlx-6": "sys6", "mlx-7": "sys7",
    "cuda-1": "cuda-sys1", "cuda-2": "cuda-sys2",
    "cuda-3": "cuda-sys3", "cuda-4": "cuda-sys4",
    "z4090": "gpu-10",
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

def _auth_for_ip(ip):
    """Return SSH auth prefix based on IP range."""
    octets = ip.split(".")
    if len(octets) == 4:
        last = int(octets[3])
        if 10 <= last <= 18:
            return "sshpass -p z"
    return f"sshpass -f {PASSFILE}"

def ssh_cmd(ip, cmd, timeout=5):
    """Run command on remote node via sshpass (auto-detects auth, single-quote safe)."""
    if ip in ("127.0.0.1", "localhost"):
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            return r.stdout.strip()
        except Exception:
            return ""
    try:
        auth = _auth_for_ip(ip)
        full = f"{auth} ssh {SSH_OPTS} z@{ip} '{cmd}'"
        r = subprocess.run(full, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""

def health_check(name, node):
    """Check brain server health endpoint, or ping for cluster nodes."""
    import requests
    ip, port = node["ip"], node["port"]
    if port == 0:
        # Ping-only node (no HTTP server)
        try:
            r = subprocess.run(f"ping -c1 -W1 {ip}", shell=True, capture_output=True, timeout=3)
            return {"status": "UP" if r.returncode == 0 else "DOWN", "model": "(no LLM)", "role": node["role"], "active": 0}
        except Exception:
            return {"status": "DOWN", "model": "-", "role": node["role"], "active": 0}
    try:
        r = requests.get(f"http://{ip}:{port}/health", timeout=3)
        d = r.json()
        model = d.get("model", "?").split("/")[-1][:35]
        role = d.get("role", node["role"])
        active = d.get("active", 0)
        return {"status": "UP", "model": model, "role": role, "active": active}
    except Exception:
        pass
    # Try /v1/models (vLLM servers like CUDA)
    try:
        r = requests.get(f"http://{ip}:{port}/v1/models", timeout=3)
        d = r.json()
        models = d.get("data", [])
        if models:
            model = models[0].get("id", "?").split("/")[-1][:35]
            return {"status": "UP", "model": model, "role": node["role"], "active": 0}
    except Exception:
        pass
    # Try ping as fallback
    try:
        r = subprocess.run(f"ping -c1 -W1 {ip}", shell=True, capture_output=True, timeout=3)
        if r.returncode == 0:
            return {"status": "PING", "model": "(reachable, no LLM)", "role": node["role"], "active": 0}
    except Exception:
        pass
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
    """Count tasks from TASK_QUEUE_v5.md with details."""
    path = os.path.expanduser("~/AGENT/TASK_QUEUE_v5.md")
    done = ready = progress = 0
    in_progress_list = []
    try:
        for line in open(path):
            if "[DONE" in line:
                done += 1
            elif "[IN_PROGRESS" in line:
                progress += 1
                import re
                m = re.search(r'(T\d+):.*\[IN_PROGRESS by ([^\]]+)\]', line)
                if m:
                    in_progress_list.append((m.group(1), m.group(2)))
            elif "[READY]" in line:
                ready += 1
    except FileNotFoundError:
        pass
    return done, progress, ready, in_progress_list

def reconcile_tasks():
    """Reconcile task queue with git history — mark committed tasks DONE, release stale claims."""
    import re
    queue_path = os.path.expanduser("~/AGENT/TASK_QUEUE_v5.md")

    # Get tasks with actual git commits
    try:
        r = subprocess.run("cd ~/AGENT && git log --oneline | grep -oP 'T\\d+' | sort -u",
                          shell=True, capture_output=True, text=True, timeout=10)
        committed = set(r.stdout.strip().split("\n")) if r.stdout.strip() else set()
    except Exception:
        committed = set()

    if not committed:
        console.print("[yellow]No git commits found to reconcile[/yellow]")
        return

    try:
        lines = open(queue_path).readlines()
    except FileNotFoundError:
        console.print("[red]Task queue not found[/red]")
        return

    fixed = []
    changes = 0
    for line in lines:
        m = re.match(r'(### (T\d+): \[)IN_PROGRESS by [^\]]+(\].*)', line)
        if m:
            task_id = m.group(2)
            if task_id in committed:
                fixed.append(f"{m.group(1)}DONE{m.group(3)}\n")
                changes += 1
                continue
        fixed.append(line)

    if changes > 0:
        open(queue_path, "w").writelines(fixed)
        console.print(f"[green]Reconciled: {changes} tasks marked DONE (verified via git commits)[/green]")
    else:
        console.print("[dim]Task queue already in sync with git[/dim]")

    done = sum(1 for l in fixed if "[DONE]" in l)
    ready = sum(1 for l in fixed if "[READY]" in l)
    prog = sum(1 for l in fixed if "[IN_PROGRESS" in l)
    console.print(f"  [green]DONE:[/green] {done}  [yellow]IN_PROGRESS:[/yellow] {prog}  [white]READY:[/white] {ready}")

def agent_activity():
    """Show what each agent is doing right now."""
    import re
    console.print("\n[bold cyan]━━━ AGENT ACTIVITY (Local) ━━━[/bold cyan]")
    logs = {
        "A1 Main":     "LOGS/main_trace.log",
        "A2 Sys4":     "LOGS/sys4_trace.log",
        "A3 Clust":    "LOGS/cluster2_trace.log",
        "A4 Work4":    "LOGS/agent4_trace.log",
        "A5 Work5":    "LOGS/agent5_trace.log",
        "A6 Work6":    "LOGS/agent6_trace.log",
    }

    # Remote distributed agents
    REMOTE_AGENTS = {
        "R2 ARCH":  ("10.255.255.2",  "sys2"),
        "R3 ENGR":  ("10.255.255.3",  "sys3"),
        "R5 DSGN":  ("10.255.255.5",  "sys5"),
        "R6 REVW":  ("10.255.255.6",  "sys6"),
        "R7 FAST":  ("10.255.255.7",  "sys7"),
    }

    # Also get task assignments from queue
    queue_path = os.path.expanduser("~/AGENT/TASK_QUEUE_v5.md")
    agent_tasks = {}
    try:
        for line in open(queue_path):
            m = re.search(r'(T\d+):.*\[IN_PROGRESS by ([^\]]+)\]', line)
            if m:
                task_id, agent_name = m.group(1), m.group(2)
                # Map agent names to our short names
                for short, pattern in [("A1 Main", "Main"), ("A2 Sys4", "Sys4"), ("A3 Clust", "Cluster"),
                                       ("A4 Work4", "Worker4"), ("A5 Work5", "Worker5"), ("A6 Work6", "Worker6")]:
                    if pattern in agent_name:
                        agent_tasks[short] = task_id
    except FileNotFoundError:
        pass

    try:
        term_width = os.get_terminal_size().columns
    except Exception:
        term_width = 160

    for name, logfile in logs.items():
        path = os.path.expanduser(f"~/AGENT/{logfile}")
        task = agent_tasks.get(name, "?")
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 8192))
                chunk = f.read().decode("utf-8", errors="replace")
            clean = re.sub(r'\x1b\[[0-9;]*m', '', chunk)

            # Find task from log
            if task == "?":
                claims = re.findall(r'CLAIMED (T\d+)', clean)
                if claims:
                    task = claims[-1]
                else:
                    trefs = re.findall(r'\b(T\d{2,})\b', clean)
                    if trefs:
                        task = trefs[-1]

            # Count turns and tools used
            turns = clean.count("[Thinking...]")
            tools_used = re.findall(r'EXECUTING\]: (\w+)', clean)
            tool_count = len(tools_used)
            last_3_tools = tools_used[-3:] if tools_used else []

            # Last meaningful action (find tool calls, not garbled stream)
            actions = []
            for match in re.finditer(r'\[EXECUTING\]: (\w+)', clean):
                actions.append(match.group(1))
            brain_calls = len(re.findall(r'Pinging|ARCHITECT|ENGINEER|DESIGNER|REVIEWER|CODER', clean))

            mtime = os.path.getmtime(path)
            age = int(time.time() - mtime)
            if age < 60:
                status = "[green]ACT[/green]"
            elif age < 300:
                status = "[yellow]SLW[/yellow]"
            else:
                status = "[red]OLD[/red]"

            task_str = f"[bold yellow]{task}[/bold yellow]" if task != "?" else "[dim]?[/dim]"
            tools_str = " → ".join(last_3_tools[-3:]) if last_3_tools else "starting"
            brain_str = f" [cyan]🧠{brain_calls}[/cyan]" if brain_calls else ""

            console.print(f"  {status} {name} [{task_str}] T{turns} 🔧{tool_count}{brain_str} | {tools_str}")
        except FileNotFoundError:
            console.print(f"  [red]DEAD[/red] {name} [--] No log file")
        except Exception as e:
            console.print(f"  [red]ERR[/red]  {name} [--] {e}")

    # Remote distributed agents
    console.print(f"\n[bold cyan]━━━ DISTRIBUTED AGENTS ━━━[/bold cyan]")
    with ThreadPoolExecutor(max_workers=5) as pool:
        def check_remote(rname, ip, node_label):
            try:
                out = ssh_cmd(ip, "tail -c 8192 ~/AGENT/LOGS/agent_trace.log 2>/dev/null", timeout=5)
                if not out:
                    return f"  [red]DEAD[/red] {rname} [{node_label}] No log"
                clean = re.sub(r'\x1b\[[0-9;]*m', '', out)
                turns = clean.count("[Thinking...]")
                tools = re.findall(r'EXECUTING\]: (\w+)', clean)
                tool_count = len(tools)
                last_3 = tools[-3:] if tools else []
                brains = len(re.findall(r'Pinging|ARCHITECT|ENGINEER|DESIGNER|REVIEWER|CODER', clean))
                claims = re.findall(r'CLAIMED (T\d+)', clean)
                task = claims[-1] if claims else "?"
                task_str = f"[bold yellow]{task}[/bold yellow]" if task != "?" else "[dim]?[/dim]"
                tools_str = " → ".join(last_3) if last_3 else "starting"
                brain_str = f" [cyan]🧠{brains}[/cyan]" if brains else ""
                return f"  [green]ACT[/green] {rname} [{task_str}] T{turns} 🔧{tool_count}{brain_str} | {tools_str}"
            except Exception as e:
                return f"  [red]ERR[/red]  {rname} [{node_label}] {e}"

        futures = {pool.submit(check_remote, rn, ip, nl): rn for rn, (ip, nl) in REMOTE_AGENTS.items()}
        for f in as_completed(futures):
            console.print(f.result())

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

# Map node names to server script file names
NODE_TO_SERVER = {
    "sys2": "sys2", "sys3": "sys3", "sys4": "sys4",
    "sys5": "sys5", "sys6": "sys6", "sys7": "sys7",
}

def launch_brain(name):
    """Start MLX server on a remote node."""
    node = NODES.get(name)
    if not node or name in ("sys1",) or name.startswith("cuda"):
        console.print(f"[red]Cannot launch brain on {name} (use manually)[/red]")
        return
    ip = node["ip"]
    port = node.get("port", 8000)
    console.print(f"[yellow]Starting brain on {name} ({ip})...[/yellow]")
    # Check if already running — don't kill existing foreground servers
    import requests
    try:
        r = requests.get(f"http://{ip}:{port}/health", timeout=2)
        if r.status_code == 200:
            console.print(f"[green]{name}: already running (skipped)[/green]")
            return
    except Exception:
        pass
    server_name = NODE_TO_SERVER.get(name, name)
    ssh_cmd(ip, f"cd ~/AGENT && "
               f"nohup ~/.pyenv/versions/3.12.10/bin/python3 mlx_server_{server_name}.py "
               f"> LOGS/{server_name}_mlx.log 2>&1 &", timeout=10)
    console.print(f"[green]{name}: brain launching[/green]")

def stop_brain(name):
    """Stop MLX server on a remote node."""
    node = NODES.get(name)
    if not node:
        return
    ssh_cmd(node["ip"], "pkill -f mlx_server", timeout=5)
    console.print(f"[red]{name}: brain stopped[/red]")

def launch_agent(agent_id, auto_go=False):
    """Start an agent on sys0 in a tmux session with optional auto-go."""
    agent = AGENTS.get(agent_id)
    if not agent:
        console.print(f"[red]Unknown agent: {agent_id}[/red]")
        return
    # Check if already running (tmux OR bare process)
    r = subprocess.run(f"tmux has-session -t {agent_id} 2>/dev/null", shell=True)
    if r.returncode == 0:
        console.print(f"[yellow]{agent_id}: already running in tmux (attach: tmux attach -t {agent_id})[/yellow]")
        return
    # Skip bare-process check — tmux check above is sufficient
    script = agent["script"]
    log = agent["log"]
    name = agent["name"]
    go_flag = " --auto-go" if auto_go else ""
    subprocess.Popen(
        f"tmux new-session -d -s {agent_id} 'cd ~/AGENT && source ~/.venv-vLLM_0.17.1_Stable/bin/activate && "
        f"python3 {script} --name \"{name}\"{go_flag} 2>&1 | tee -a {log}'",
        shell=True
    )
    mode = "[bold green]AUTO-GO[/bold green]" if auto_go else "interactive"
    console.print(f"[green]{agent_id}: launched ({mode}) → tmux attach -t {agent_id}[/green]")

def stop_agent(agent_id):
    """Stop an agent — kills tmux sessions AND any matching processes."""
    if agent_id == "all":
        # Kill ALL agent processes — tmux sessions AND bare terminal agents
        for aid in AGENTS:
            subprocess.run(f"tmux kill-session -t {aid} 2>/dev/null", shell=True)
        # Kill any OMNIAGENT processes not in tmux
        subprocess.run("pkill -f 'OMNIAGENT_v4' 2>/dev/null", shell=True)
        subprocess.run("pkill -f 'run_agent.sh' 2>/dev/null", shell=True)
        # Verify
        r = subprocess.run("ps aux | grep OMNIAGENT | grep -v grep | wc -l",
                          shell=True, capture_output=True, text=True, timeout=3)
        remaining = r.stdout.strip()
        console.print(f"[red]All agents stopped ({remaining} processes remaining)[/red]")
        return
    subprocess.run(f"tmux kill-session -t {agent_id} 2>/dev/null", shell=True)
    # Also kill the specific script if running outside tmux
    agent = AGENTS.get(agent_id)
    if agent:
        subprocess.run(f"pkill -f '{agent['script']}' 2>/dev/null", shell=True)
    console.print(f"[red]{agent_id}: stopped[/red]")

def goal_all():
    """GO ALL — launch all brains, then launch all agents with auto-go."""
    console.print("[bold magenta]━━━ GOAL: LAUNCHING ENTIRE SWARM ━━━[/bold magenta]")
    # Launch all brains in parallel (skips already-running ones)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for name in ["sys2", "sys3", "sys4", "sys5", "sys6", "sys7"]:
            pool.submit(launch_brain, name)
    console.print("[yellow]Brains checked. Launching agents with AUTO-GO...[/yellow]")
    # Launch ALL agents with auto-go (reads GO_PROMPT.md immediately)
    for aid in AGENTS:
        launch_agent(aid, auto_go=True)
    console.print("[bold green]━━━ SWARM LAUNCHED — ALL AGENTS IN AUTO-GO MODE ━━━[/bold green]")
    console.print("[dim]  Agents reading GO_PROMPT.md and claiming tasks autonomously[/dim]")
    console.print("[dim]  Attach to any: tmux attach -t agent1[/dim]")
    console.print("[dim]  List sessions: tmux ls[/dim]")

def stop_all():
    """Stop all brains and agents."""
    console.print("[bold red]━━━ STOPPING ENTIRE SWARM ━━━[/bold red]")
    for name in NODES:
        if name.startswith("sys") and name != "sys1":
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
    done, progress, ready, in_progress_list = count_tasks()
    running_agents = check_agents()

    # Node table
    table = Table(title=f"SWARM STATUS — {now}", box=box.DOUBLE_EDGE, show_lines=False,
                  title_style="bold white on blue", padding=(0, 1))
    table.add_column("Node", style="bold", width=12)
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
        role = node["role"]  # Always use OUR role, not server's
        active = st.get("active", 0)

        if status == "UP":
            st_str = "[green]UP[/green]"
        elif status == "PING":
            st_str = "[yellow]OK[/yellow]"
        elif status == "DOWN":
            st_str = "[red]DN[/red]"
        else:
            st_str = "[yellow]??[/yellow]"

        role_colors = {
            "ARCHITECT": "red", "ENGINEER": "yellow", "CODER": "green",
            "DESIGNER": "cyan", "REVIEWER": "bright_green", "FAST-CODER": "magenta",
            "AGENT-HOST": "blue", "CUDA-BRAIN": "bright_red", "CUDA-WORK": "red",
            "HYPER-CODER": "bright_magenta",
        }
        rc = role_colors.get(role, "white")

        table.add_row(name, f"[{rc}]{role}[/{rc}]", model, node["chip"], node["ram"],
                      st_str, str(active) if active else "")

    # Summary
    up_count = sum(1 for s in fleet_status.values() if s.get("status") == "UP")
    total = len(NODES)
    # Count tmux sessions for agent count
    try:
        r = subprocess.run("tmux ls 2>/dev/null | grep agent | wc -l", shell=True, capture_output=True, text=True, timeout=3)
        tmux_agents = int(r.stdout.strip()) if r.stdout.strip() else 0
    except Exception:
        tmux_agents = 0
    agent_count = max(len(running_agents), tmux_agents)
    agent_str = ", ".join(running_agents) if running_agents else "none"

    # Show active task assignments
    task_str = ""
    if in_progress_list:
        task_str = " | " + " ".join(f"[yellow]{t}→{a.split('[')[-1].rstrip(']') if '[' in a else a}[/yellow]" for t, a in in_progress_list[:6])

    summary = (
        f"[bold]Brains:[/bold] {up_count}/{total} online  |  "
        f"[bold]Agents:[/bold] {agent_count} running  |  "
        f"[bold]Tasks:[/bold] {done} done / {progress} active / {ready} ready"
        f"{task_str}"
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
  [bold]fleet[/bold] / [bold]status[/bold]     — Refresh dashboard
  [bold]activity[/bold]           — Show what each agent is doing RIGHT NOW
  [bold]goal[/bold]               — GO ALL: launch brains + ALL 6 agents (auto-go)
  [bold]go[/bold]                 — Launch ALL agents with auto-go (brains must be up)
  [bold]go[/bold] <1-6>           — Launch specific agent with auto-go
  [bold]stop[/bold]               — Stop all brains + agents
  [bold]kick[/bold] <1-6>         — Kill and restart an agent (triggers auto-go)
  [bold]launch[/bold] <node>      — Start brain on node (sys2-sys7)
  [bold]kill[/bold] <node>        — Stop brain on node
  [bold]agent[/bold] <1-6>        — Start agent interactively (no auto-go)
  [bold]agentstop[/bold] <1-6|all> — Stop agent(s)
  [bold]reconcile[/bold]          — Sync task queue with git (mark committed tasks DONE)
  [bold]council[/bold] <question>  — Ask architect+engineer+designer
  [bold]logs[/bold] <node>        — Tail node's MLX log (Ctrl+C to stop)
  [bold]redirect[/bold] <agent> <task> — Redirect agent to different task (issue #20)
  [bold]tasks[/bold]              — Show task queue summary
  [bold]tmux[/bold]               — List active tmux sessions
  [bold]attach[/bold] <1-6>       — Attach to agent tmux session
  [bold]q[/bold] / [bold]quit[/bold]           — Exit"""
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

    # Enable readline for up-arrow history
    try:
        import readline
        histfile = os.path.expanduser("~/AGENT/.commander_history")
        try:
            readline.read_history_file(histfile)
        except FileNotFoundError:
            pass
        import atexit
        atexit.register(readline.write_history_file, histfile)
    except ImportError:
        pass

    while True:
        try:
            cmd = input("\ncommander> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Exiting commander.[/dim]")
            break

        if not cmd:
            continue

        parts = cmd.split(maxsplit=1)
        action = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        # Resolve aliases (mlx-3 -> sys3, cuda-1 -> cuda-sys1, etc.)
        arg = NODE_ALIASES.get(arg, arg)

        if action in ("q", "quit", "exit"):
            break
        elif action in ("fleet", "status", "s", "f"):
            show_dashboard()
        elif action in ("activity", "act", "a"):
            agent_activity()
        elif action == "reconcile":
            reconcile_tasks()
        elif action == "kick":
            if arg and arg.isdigit():
                aid = f"agent{arg}"
                console.print(f"[yellow]Kicking {aid}...[/yellow]")
                subprocess.run(f"tmux kill-session -t {aid} 2>/dev/null", shell=True)
                time.sleep(2)
                launch_agent(aid, auto_go=True)
                console.print(f"[green]{aid} restarted with AUTO-GO[/green]")
            else:
                console.print("[red]Usage: kick <1-6>[/red]")
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
        elif action == "go":
            # GO — launch agents with auto-go (reads GO_PROMPT.md)
            if arg and arg.isdigit():
                launch_agent(f"agent{arg}", auto_go=True)
            elif not arg:
                console.print("[bold magenta]━━━ GO: ALL AGENTS AUTO-GO ━━━[/bold magenta]")
                for aid in AGENTS:
                    launch_agent(aid, auto_go=True)
                console.print("[bold green]All agents launched with GO_PROMPT.md[/bold green]")
            else:
                console.print("[red]Usage: go [1-6][/red]")
        elif action in ("agent",):
            aid = f"agent{arg}" if arg.isdigit() else arg
            launch_agent(aid)
        elif action == "agentstop":
            aid = f"agent{arg}" if arg.isdigit() else arg
            stop_agent(aid)
        elif action == "tmux":
            os.system("tmux ls 2>/dev/null || echo 'No tmux sessions'")
        elif action == "attach":
            aid = f"agent{arg}" if arg and arg.isdigit() else arg
            if aid:
                os.system(f"tmux attach -t {aid}")
            else:
                console.print("[red]Usage: attach <1-6>[/red]")
        elif action == "council":
            if arg:
                council_query(arg)
            else:
                console.print("[red]Usage: council <question>[/red]")
        elif action == "logs":
            if arg:
                node = NODES.get(arg)
                if not node:
                    console.print(f"[red]Unknown node: {arg}[/red]")
                elif arg in ("sys1",):
                    console.print("[dim]Tailing sys1 agent logs...[/dim]")
                    try:
                        os.system("tail -f ~/AGENT/LOGS/main_trace.log")
                    except KeyboardInterrupt:
                        pass
                elif arg.startswith("cuda-"):
                    # CUDA nodes use password auth
                    console.print(f"[dim]Tailing {arg} vLLM logs (Ctrl+C to stop)...[/dim]")
                    try:
                        os.system(f"sshpass -p z ssh {SSH_OPTS} z@{node['ip']} 'tail -f /tmp/ray/session_latest/logs/serve*.log 2>/dev/null || journalctl -u vllm -f 2>/dev/null || echo No vLLM logs found'")
                    except KeyboardInterrupt:
                        pass
                elif arg.startswith("sys"):
                    sname = NODE_TO_SERVER.get(arg, arg)
                    console.print(f"[dim]Tailing {arg} MLX logs (Ctrl+C to stop)...[/dim]")
                    try:
                        os.system(f"sshpass -f {PASSFILE} ssh {SSH_OPTS} z@{node['ip']} 'tail -f ~/AGENT/LOGS/{sname}_mlx.log'")
                    except KeyboardInterrupt:
                        pass
                else:
                    console.print(f"[dim]Tailing {arg}...[/dim]")
                    try:
                        os.system(f"sshpass -p z ssh {SSH_OPTS} z@{node['ip']} 'tail -f ~/AGENT/LOGS/*.log 2>/dev/null || echo No logs'")
                    except KeyboardInterrupt:
                        pass
            else:
                console.print("[red]Usage: logs <node>[/red]")
        elif action == "redirect":
            # Issue #20: Redirect agent to a different task
            rparts = arg.split(maxsplit=1)
            if len(rparts) < 2:
                console.print("[red]Usage: redirect <agent_name_or_number> <task_id>[/red]")
                console.print("[dim]  e.g.: redirect 1 T55  or  redirect 'OmniAgent [Main]' T55[/dim]")
            else:
                r_agent_arg, r_task = rparts[0], rparts[1].strip().split()[0]
                # Resolve agent number to name
                if r_agent_arg.isdigit():
                    agent_info = AGENTS.get(f"agent{r_agent_arg}")
                    r_agent_name = agent_info["name"] if agent_info else r_agent_arg
                else:
                    r_agent_name = r_agent_arg
                import requests as _req
                try:
                    resp = _req.post("http://10.255.255.128:9091/redirect",
                                     data=f"agent={r_agent_name}&new_task={r_task}&reason=commander redirect",
                                     headers={"Content-Type": "application/x-www-form-urlencoded"},
                                     timeout=5)
                    d = resp.json()
                    if d.get("ok"):
                        console.print(f"[green]Redirected: {d.get('msg')}[/green]")
                    else:
                        console.print(f"[red]Redirect failed: {d.get('msg')}[/red]")
                except Exception as e:
                    console.print(f"[red]Redirect error: {e}[/red]")
        elif action == "tasks":
            done, progress, ready, in_progress_list = count_tasks()
            console.print(f"  [green]Done:[/green] {done}  [yellow]Active:[/yellow] {progress}  [white]Ready:[/white] {ready}")
            try:
                # Show first few active/ready tasks
                for line in open(os.path.expanduser("~/AGENT/TASK_QUEUE_v5.md")):
                    if "[IN_PROGRESS" in line or "[READY]" in line:
                        console.print(f"  {line.strip()}")
            except FileNotFoundError:
                pass
        elif action == "watch":
            # Live agent monitor — refreshes every 5s, Ctrl+C to stop
            console.print("[bold cyan]Live agent monitor (Ctrl+C to stop)...[/bold cyan]")
            try:
                while True:
                    os.system("clear" if os.name == "posix" else "cls")
                    console.print(f"[bold cyan]━━━ LIVE AGENT MONITOR — {datetime.now().strftime('%H:%M:%S')} (Ctrl+C to stop) ━━━[/bold cyan]")
                    agent_activity()
                    time.sleep(5)
            except KeyboardInterrupt:
                console.print("\n[dim]Watch stopped.[/dim]")
        elif action in ("help", "?", "h"):
            show_help()
        else:
            console.print(f"[red]Unknown command: {action}[/red]. Type 'help' for commands.")

if __name__ == "__main__":
    main()
