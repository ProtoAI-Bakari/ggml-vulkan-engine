#!/usr/bin/env python3
"""
ProtoAI-Bakari Cluster Deployer v1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Synchronizes agent code, configs, deps, and launches across the fleet.
Single source of truth for what runs where.

Usage: python3 cluster_deployer.py [command]
"""

import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

# ── Fleet Definition (single source of truth) ───────────────────────────────
NODES = {
    "mlx-2": {"ip": "10.255.255.2",  "auth": "passfile", "role": "ARCHITECT",   "model": "Qwen3-235B-Thinking", "python": "~/.pyenv/versions/3.12.10/bin/python3"},
    "mlx-3": {"ip": "10.255.255.3",  "auth": "passfile", "role": "ENGINEER",    "model": "Qwen3.5-122B",        "python": "~/.pyenv/versions/3.12.10/bin/python3"},
    "mlx-4": {"ip": "10.255.255.4",  "auth": "passfile", "role": "CODER",       "model": "Qwen3-Coder-Next-8b", "python": "~/.pyenv/versions/3.12.10/bin/python3"},
    "mlx-5": {"ip": "10.255.255.5",  "auth": "passfile", "role": "DESIGNER",    "model": "GLM-4.7-Flash-8b",    "python": "~/.pyenv/versions/3.12.10/bin/python3"},
    "mlx-6": {"ip": "10.255.255.6",  "auth": "passfile", "role": "REVIEWER",    "model": "Qwen3.5-122B",        "python": "~/.pyenv/versions/3.12.10/bin/python3"},
    "mlx-7": {"ip": "10.255.255.7",  "auth": "passfile", "role": "FAST-CODER",  "model": "Qwen3-Coder-Next-4b", "python": "~/.pyenv/versions/3.12.10/bin/python3"},
}

PASSFILE = "/home/z/DEV/authpass"
SSH_OPTS = "-o StrictHostKeyChecking=no -o ConnectTimeout=5"

# Files to sync to every node
SYNC_FILES = [
    "OMNIAGENT_v4_focused.py",
    "GO_PROMPT.md",
    "brain_bridge.py",
    "claim_task.sh",
    "complete_task.sh",
    "TASK_QUEUE_v5.md",
    "KNOWLEDGE_BASE.md",
]

PYPI_DEPS = ["requests", "openai", "mlx", "mlx-lm", "fastapi", "uvicorn", "pydantic"]

# ── SSH Helpers ──────────────────────────────────────────────────────────────
def _ssh_cmd(node_name):
    n = NODES[node_name]
    if n["auth"] == "passfile":
        return f"sshpass -f {PASSFILE} ssh {SSH_OPTS} z@{n['ip']}"
    else:
        return f"sshpass -p z ssh {SSH_OPTS} z@{n['ip']}"

def _scp_cmd(node_name):
    n = NODES[node_name]
    if n["auth"] == "passfile":
        return f"sshpass -f {PASSFILE} scp {SSH_OPTS}"
    else:
        return f"sshpass -p z scp {SSH_OPTS}"

def ssh(node, cmd, timeout=15):
    try:
        r = subprocess.run(f'{_ssh_cmd(node)} "{cmd}"', shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"

def scp_to(node, local, remote):
    ip = NODES[node]["ip"]
    auth = f"-f {PASSFILE}" if NODES[node]["auth"] == "passfile" else "-p z"
    subprocess.run(f"sshpass {auth} scp {SSH_OPTS} {local} z@{ip}:{remote}", shell=True, capture_output=True, timeout=30)

# ── Commands ─────────────────────────────────────────────────────────────────

def cmd_status(targets=None):
    """Show deployment status of all nodes."""
    targets = targets or list(NODES.keys())
    table = Table(title="Deployment Status", box=box.SIMPLE)
    table.add_column("Node", style="bold")
    table.add_column("Role")
    table.add_column("MLX Server")
    table.add_column("Agent")
    table.add_column("Agent Files")
    table.add_column("Deps")

    def check_node(name):
        n = NODES[name]
        # MLX server
        try:
            import requests as req
            r = req.get(f"http://{n['ip']}:8000/health", timeout=3)
            mlx = "[green]UP[/green]"
        except:
            mlx = "[red]DOWN[/red]"
        # Agent process
        agent_ps = ssh(name, "pgrep -f 'python3.*OMNIAGENT' | wc -l", timeout=5)
        agent = "[green]RUNNING[/green]" if agent_ps.strip() not in ("0", "", "ERROR") else "[red]STOPPED[/red]"
        # Files
        files = ssh(name, "ls ~/AGENT/OMNIAGENT_v4_focused.py ~/AGENT/GO_PROMPT.md 2>/dev/null | wc -l", timeout=5)
        has_files = "[green]OK[/green]" if files.strip() == "2" else "[red]MISSING[/red]"
        # Deps
        deps = ssh(name, f"{n['python']} -c 'import requests, openai; print(1)' 2>/dev/null", timeout=10)
        has_deps = "[green]OK[/green]" if deps.strip() == "1" else "[red]MISSING[/red]"
        return name, n["role"], mlx, agent, has_files, has_deps

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(check_node, name): name for name in targets}
        results = {}
        for f in as_completed(futures):
            name = futures[f]
            try:
                results[name] = f.result()
            except:
                results[name] = (name, NODES[name]["role"], "?", "?", "?", "?")

    for name in targets:
        if name in results:
            table.add_row(*results[name])
    console.print(table)

def cmd_sync(targets=None):
    """Sync agent files to nodes."""
    targets = targets or list(NODES.keys())
    console.print(f"[bold cyan]Syncing {len(SYNC_FILES)} files to {len(targets)} nodes...[/bold cyan]")

    def sync_node(name):
        n = NODES[name]
        ssh(name, "mkdir -p ~/AGENT/LOGS ~/AGENT/scripts")
        for f in SYNC_FILES:
            local = os.path.expanduser(f"~/AGENT/{f}")
            if os.path.exists(local):
                scp_to(name, local, f"~/AGENT/{f}")
        # Patch agent name for this node (PRIMARY_IP is already 127.0.0.1)
        ssh(name, f"""cd ~/AGENT && {n['python']} -c "
content = open('OMNIAGENT_v4_focused.py').read()
content = content.replace('default=\\"OmniAgent [Main]\\"', 'default=\\"OmniAgent [{name}]\\"')
open('OMNIAGENT_v4_focused.py', 'w').write(content)
print('patched')
" """, timeout=15)
        return f"{name}: synced"

    with ThreadPoolExecutor(max_workers=6) as pool:
        for f in as_completed({pool.submit(sync_node, n): n for n in targets}):
            console.print(f"  [green]{f.result()}[/green]")

def cmd_deps(targets=None):
    """Install Python dependencies on nodes."""
    targets = targets or list(NODES.keys())
    console.print(f"[bold cyan]Installing deps on {len(targets)} nodes...[/bold cyan]")

    def install_node(name):
        n = NODES[name]
        pip = n["python"].replace("python3", "pip3")
        result = ssh(name, f"{pip} install {' '.join(PYPI_DEPS)} 2>&1 | tail -1", timeout=120)
        return f"{name}: {result[:60]}"

    with ThreadPoolExecutor(max_workers=6) as pool:
        for f in as_completed({pool.submit(install_node, n): n for n in targets}):
            console.print(f"  {f.result()}")

def cmd_launch(targets=None):
    """Launch agents on nodes (kills existing first)."""
    targets = targets or list(NODES.keys())
    console.print(f"[bold magenta]Launching agents on {len(targets)} nodes...[/bold magenta]")

    def launch_node(name):
        n = NODES[name]
        # Check MLX server
        try:
            import requests as req
            req.get(f"http://{n['ip']}:8000/health", timeout=3)
        except:
            return f"{name}: [red]MLX server not running — skipped[/red]"
        # Kill existing
        ssh(name, "pkill -f OMNIAGENT 2>/dev/null", timeout=5)
        time.sleep(1)
        # Launch
        ssh(name, f"cd ~/AGENT && nohup {n['python']} OMNIAGENT_v4_focused.py --auto-go --name 'OmniAgent [{name}]' > LOGS/agent_trace.log 2>&1 &", timeout=10)
        return f"{name}: [green]launched[/green]"

    with ThreadPoolExecutor(max_workers=6) as pool:
        for f in as_completed({pool.submit(launch_node, n): n for n in targets}):
            console.print(f"  {f.result()}")

def cmd_stop(targets=None):
    """Stop agents on nodes."""
    targets = targets or list(NODES.keys())
    for name in targets:
        ssh(name, "pkill -f OMNIAGENT 2>/dev/null")
        console.print(f"  {name}: stopped")

def cmd_restart(targets=None):
    """Stop + sync + launch."""
    targets = targets or list(NODES.keys())
    cmd_stop(targets)
    time.sleep(2)
    cmd_sync(targets)
    cmd_launch(targets)

def cmd_logs(node_name):
    """Tail agent log on a node."""
    n = NODES.get(node_name)
    if not n:
        console.print(f"[red]Unknown node: {node_name}[/red]")
        return
    os.system(f'{_ssh_cmd(node_name)} "tail -f ~/AGENT/LOGS/agent_trace.log"')

def cmd_deploy_all():
    """Full deployment: deps + sync + launch on ALL nodes."""
    console.print("[bold magenta]━━━ FULL DEPLOYMENT TO ALL NODES ━━━[/bold magenta]")
    cmd_deps()
    cmd_sync()
    cmd_launch()
    console.print("[bold green]━━━ DEPLOYMENT COMPLETE ━━━[/bold green]")
    cmd_status()

def show_help():
    console.print("""[bold cyan]Cluster Deployer Commands:[/bold cyan]
  [bold]status[/bold]  [nodes]     — Show deployment status
  [bold]sync[/bold]    [nodes]     — Sync agent files to nodes
  [bold]deps[/bold]    [nodes]     — Install Python dependencies
  [bold]launch[/bold]  [nodes]     — Launch agents (auto-go)
  [bold]stop[/bold]    [nodes]     — Stop agents
  [bold]restart[/bold] [nodes]     — Stop + sync + launch
  [bold]deploy[/bold]              — Full deploy to ALL nodes
  [bold]logs[/bold]    <node>      — Tail agent log
  [bold]q[/bold] / [bold]quit[/bold]              — Exit

  [dim]nodes = space-separated list like: mlx-2 mlx-5 mlx-7[/dim]
  [dim]omit nodes to target ALL[/dim]""")

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    console.print("[bold magenta]ProtoAI-Bakari Cluster Deployer v1[/bold magenta]\n")

    if len(sys.argv) > 1:
        # CLI mode
        action = sys.argv[1]
        targets = sys.argv[2:] if len(sys.argv) > 2 else None
        if targets:
            targets = [t for t in targets if t in NODES]
        dispatch = {"status": cmd_status, "sync": cmd_sync, "deps": cmd_deps,
                    "launch": cmd_launch, "stop": cmd_stop, "restart": cmd_restart,
                    "deploy": cmd_deploy_all}
        if action == "logs" and len(sys.argv) > 2:
            cmd_logs(sys.argv[2])
        elif action in dispatch:
            dispatch[action](targets) if action != "deploy" else dispatch[action]()
        else:
            show_help()
        return

    # Interactive mode
    cmd_status()
    show_help()

    while True:
        try:
            cmd = input("\ndeployer> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not cmd:
            continue
        parts = cmd.split()
        action = parts[0].lower()
        targets = [t for t in parts[1:] if t in NODES] or None

        if action in ("q", "quit"):
            break
        elif action == "status":
            cmd_status(targets)
        elif action == "sync":
            cmd_sync(targets)
        elif action == "deps":
            cmd_deps(targets)
        elif action == "launch":
            cmd_launch(targets)
        elif action == "stop":
            cmd_stop(targets)
        elif action == "restart":
            cmd_restart(targets)
        elif action == "deploy":
            cmd_deploy_all()
        elif action == "logs" and len(parts) > 1:
            cmd_logs(parts[1])
        elif action in ("help", "?"):
            show_help()
        else:
            console.print(f"[red]Unknown: {action}[/red]")

if __name__ == "__main__":
    main()
