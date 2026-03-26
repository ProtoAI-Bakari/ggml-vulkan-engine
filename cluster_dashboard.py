#!/usr/bin/env python3
"""
ProtoAI-Bakari Cluster Dashboard v2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Real-time TUI: all nodes, agents, GPU/CPU/RAM, tasks.
Catppuccin Macchiato palette. Live 5s refresh.

Usage:
  python3 cluster_dashboard.py
  python3 cluster_dashboard.py --interval 10

Requires: pip install rich
"""

import os, sys, time, subprocess, re, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box

# ── Catppuccin Macchiato Palette ──────────────────────────────────────────────
class M:
    ROSEWATER = "#f4dbd6"; FLAMINGO = "#f0c6c6"; PINK   = "#f5bde6"
    MAUVE     = "#c6a0f6"; RED      = "#ed8796"; MAROON = "#ee99a0"
    PEACH     = "#f5a97f"; YELLOW   = "#eed49f"; GREEN  = "#a6da95"
    TEAL      = "#8bd5ca"; SKY      = "#91d7e3"; SAPPHIRE = "#7dc4e4"
    BLUE      = "#8aadf4"; LAVENDER = "#b7bdf8"; TEXT   = "#cad3f5"
    SUBTEXT   = "#a5adcb"; OVERLAY  = "#6e738d"; SURFACE = "#363a4f"
    BASE      = "#24273a"; MANTLE   = "#1e2030"; CRUST  = "#181926"

# ── Config ────────────────────────────────────────────────────────────────────
PASSFILE  = os.path.expanduser("~/DEV/authpass")
SSH_OPTS  = "-o StrictHostKeyChecking=no -o ConnectTimeout=5 -o LogLevel=ERROR"
SSH_AGENT_TIMEOUT = 5   # pgrep check timeout

# Node definitions — groups control separator placement
MLX_NODES = {
    "mlx-0":  {"ip": "127.0.0.1",    "port": 8081, "auth": "local",    "role": "AGENT-HOST", "chip": "M1U", "os": "macos"},
    "mlx-2":  {"ip": "10.255.255.2",  "port": 8000, "auth": "passfile", "role": "ARCHITECT",  "chip": "M2U", "os": "macos"},
    "mlx-3":  {"ip": "10.255.255.3",  "port": 8000, "auth": "passfile", "role": "ENGINEER",   "chip": "M2U", "os": "macos"},
    "mlx-4":  {"ip": "10.255.255.4",  "port": 8000, "auth": "passfile", "role": "CODER",      "chip": "M1U", "os": "macos"},
    "mlx-5":  {"ip": "10.255.255.5",  "port": 8000, "auth": "passfile", "role": "DESIGNER",   "chip": "M1U", "os": "macos"},
    "mlx-6":  {"ip": "10.255.255.6",  "port": 8000, "auth": "passfile", "role": "REVIEWER",   "chip": "M1U", "os": "macos"},
    "mlx-7":  {"ip": "10.255.255.7",  "port": 8000, "auth": "passfile", "role": "FAST-CODE",  "chip": "M1U", "os": "macos"},
}
CUDA_NODES = {
    "cuda-1": {"ip": "10.255.255.11", "port": 8000, "auth": "pass_z",   "role": "CUDA-BRAIN", "chip": "2x3090", "os": "linux"},
    "cuda-2": {"ip": "10.255.255.12", "port": 0,    "auth": "pass_z",   "role": "CUDA-WORK",  "chip": "2x3090", "os": "linux"},
    "cuda-3": {"ip": "10.255.255.13", "port": 0,    "auth": "pass_z",   "role": "CUDA-WORK",  "chip": "2x3090", "os": "linux"},
    "cuda-4": {"ip": "10.255.255.14", "port": 0,    "auth": "pass_z",   "role": "CUDA-WORK",  "chip": "2x3090", "os": "linux"},
}
GPU_NODES = {
    "gpu-10": {"ip": "10.255.255.10", "port": 8000, "auth": "pass_z",   "role": "HYPER-CODE", "chip": "4090",   "os": "linux"},
}
ALL_NODES = {**MLX_NODES, **CUDA_NODES, **GPU_NODES}

ROLE_COLORS = {
    "ARCHITECT":  M.RED,     "ENGINEER":  M.YELLOW, "CODER":     M.GREEN,
    "DESIGNER":   M.SKY,     "REVIEWER":  M.TEAL,   "FAST-CODE": M.PINK,
    "AGENT-HOST": M.BLUE,    "CUDA-BRAIN":M.PEACH,  "CUDA-WORK": M.MAROON,
    "HYPER-CODE": M.FLAMINGO,
}

console = Console()

# ── SSH helper ────────────────────────────────────────────────────────────────
def ssh_cmd(ip, cmd, auth="passfile", timeout=5):
    """Run a command locally or via SSH; return stdout string or '' on failure."""
    if ip in ("127.0.0.1", "localhost"):
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            return r.stdout.strip()
        except Exception:
            return ""
    try:
        if auth == "pass_z":
            full = f'sshpass -p z ssh {SSH_OPTS} z@{ip} "{cmd}"'
        else:
            full = f'sshpass -f {PASSFILE} ssh {SSH_OPTS} z@{ip} "{cmd}"'
        r = subprocess.run(full, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""

# ── Per-node metric collection ────────────────────────────────────────────────
def _health_check(ip, port, auth):
    """Return (status, model_name). Status: 'UP' | 'PING' | 'DOWN'."""
    if port > 0:
        try:
            import urllib.request
            with urllib.request.urlopen(f"http://{ip}:{port}/health", timeout=2) as resp:
                import json
                d = json.loads(resp.read())
            return "UP", d.get("model", "").split("/")[-1][:25]
        except Exception:
            pass
    # Fall back to ping
    try:
        r = subprocess.run(
            f"ping -c1 -W1 {ip}", shell=True, capture_output=True, timeout=3
        )
        return ("PING", "") if r.returncode == 0 else ("DOWN", "")
    except Exception:
        return "DOWN", ""

def _mem_pct(ip, auth, node_os):
    """Return memory usage percentage string, e.g. '42%'."""
    if node_os == "linux":
        raw = ssh_cmd(ip, "free -m | awk '/Mem:/{printf \"%.0f\",$3/$2*100}'", auth, timeout=5)
    else:
        # macOS: sysctl for total bytes, ps -A -o rss= summed for used KB
        raw = ssh_cmd(
            ip,
            "python3 -c \""
            "import subprocess as sp; "
            "tot=int(sp.check_output(['sysctl','-n','hw.memsize']).strip()); "
            "used_kb=sum(int(x) for x in sp.check_output('ps -A -o rss=',shell=True).decode().split() if x.strip().isdigit()); "
            "print(int(used_kb*1024/tot*100))"
            "\" 2>/dev/null",
            auth, timeout=7,
        )
    if raw and raw.strip().lstrip("-").isdigit():
        pct = int(raw.strip())
        if pct > 100:
            pct = 100
        color = M.RED if pct >= 85 else M.YELLOW if pct >= 60 else M.GREEN
        return f"[{color}]{pct}%[/]"
    return f"[{M.OVERLAY}]?[/]"

def _disk_avail(ip, auth):
    """Return available disk space string from df -h / output."""
    raw = ssh_cmd(ip, "df -h / 2>/dev/null | tail -1", auth, timeout=4)
    sizes = re.findall(r'[\d.]+[TGMK]i?', raw) if raw else []
    return sizes[2] if len(sizes) >= 3 else (sizes[0] if sizes else "?")

def _agent_count(ip, auth):
    """Return number of OMNIAGENT processes via pgrep."""
    out = ssh_cmd(ip, "pgrep -c -f OMNIAGENT 2>/dev/null || echo 0", auth, timeout=SSH_AGENT_TIMEOUT)
    try:
        return int(out.strip())
    except ValueError:
        return 0

def _req_count(ip, auth):
    """Count completed requests in MLX server logs."""
    out = ssh_cmd(
        ip,
        "grep -ch 'POST.*200' ~/AGENT/LOGS/*_mlx.log 2>/dev/null | awk '{s+=$1}END{print s+0}'",
        auth, timeout=5,
    )
    try:
        return int(out.strip())
    except (ValueError, AttributeError):
        return 0

def poll_node(name, node):
    """Collect all metrics for one node; return a flat dict."""
    ip   = node["ip"]
    auth = node.get("auth", "passfile")
    port = node.get("port", 0)
    nos  = node.get("os", "macos")

    data = {
        "name": name, "role": node["role"], "chip": node["chip"],
        "status": "DOWN", "model": "-", "agent": "-",
        "mem_pct": f"[{M.OVERLAY}]?[/]", "disk": "?", "reqs": 0,
    }

    status, model = _health_check(ip, port, auth)
    data["status"] = status
    if model:
        data["model"] = model

    if status == "DOWN":
        return data

    # MLX nodes: agent presence + request count
    if name.startswith("mlx-"):
        if name != "mlx-0":
            cnt = _agent_count(ip, auth)
            data["agent"] = "RUN" if cnt > 0 else "OFF"
        else:
            # Local node: check directly
            try:
                r = subprocess.run("pgrep -c -f OMNIAGENT", shell=True, capture_output=True, text=True, timeout=3)
                data["agent"] = "RUN" if r.stdout.strip() not in ("0", "") else "OFF"
            except Exception:
                data["agent"] = "OFF"
        data["reqs"] = _req_count(ip, auth)

    data["mem_pct"] = _mem_pct(ip, auth, nos)
    data["disk"]    = _disk_avail(ip, auth)
    return data

# ── Task queue summary ────────────────────────────────────────────────────────
def count_tasks():
    path = os.path.expanduser("~/AGENT/TASK_QUEUE_v5.md")
    done = prog = ready = blocked = 0
    try:
        for line in open(path):
            if "[DONE"        in line: done    += 1
            elif "[IN_PROGRESS" in line: prog  += 1
            elif "[READY]"    in line: ready   += 1
            elif "[BLOCKED"   in line: blocked += 1
    except Exception:
        pass
    return done, prog, ready, blocked

# ── Table builder ─────────────────────────────────────────────────────────────
def _section_sep(table, label, color):
    """Add a full-width separator row labelled with the section name."""
    table.add_row(
        f"[bold {color}]── {label} ──[/]",
        "", "", "", "", "", "", "", "",
        style=f"on {M.MANTLE}",
    )

def _add_node_row(table, d):
    role = d.get("role", "?")
    rc   = ROLE_COLORS.get(role, M.TEXT)
    st   = d.get("status", "?")

    srv_dot = (
        f"[{M.GREEN}]●[/]"  if st == "UP"   else
        f"[{M.YELLOW}]◌[/]" if st == "PING" else
        f"[{M.RED}]✗[/]"
    )

    agt = d.get("agent", "-")
    agt_str = (
        f"[{M.GREEN}]●[/]"   if agt == "RUN" else
        f"[{M.RED}]✗[/]"     if agt == "OFF" else
        f"[{M.OVERLAY}]–[/]"
    )

    reqs = d.get("reqs", 0)
    reqs_str = (
        f"[{M.TEAL}]{reqs}[/]" if reqs > 0 else f"[{M.OVERLAY}]–[/]"
    )

    model_str = d.get("model", "-") or "-"
    if model_str == "-":
        model_str = f"[{M.OVERLAY}]–[/]"

    table.add_row(
        f"[bold {M.BLUE}]{d['name']}[/]",
        f"[{rc}]{role}[/]",
        f"[{M.SUBTEXT}]{d.get('chip','?')}[/]",
        model_str,
        srv_dot, agt_str,
        d.get("mem_pct", "?"),
        d.get("disk", "?"),
        reqs_str,
    )

# ── Main display builder ──────────────────────────────────────────────────────
def build_display():
    now_ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    done, prog, ready, blocked = count_tasks()

    # Parallel poll — all nodes at once
    node_data = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(poll_node, n, nd): n for n, nd in ALL_NODES.items()}
        for f in as_completed(futures):
            n = futures[f]
            try:
                node_data[n] = f.result()
            except Exception:
                node_data[n] = {
                    "name": n, "status": "ERR", "role": ALL_NODES[n]["role"],
                    "chip": ALL_NODES[n]["chip"], "model": "-", "agent": "-",
                    "mem_pct": f"[{M.OVERLAY}]?[/]", "disk": "?", "reqs": 0,
                }

    # Build table
    tbl = Table(
        box=box.ROUNDED,
        show_lines=True,
        show_edge=True,
        padding=(0, 1),
        expand=True,
        border_style=M.SURFACE,
        header_style=f"bold {M.LAVENDER}",
    )
    tbl.add_column("Node",  style=f"bold {M.BLUE}", width=8,  no_wrap=True)
    tbl.add_column("Role",  width=11, no_wrap=True)
    tbl.add_column("Chip",  style=M.SUBTEXT, width=7,  no_wrap=True)
    tbl.add_column("Model", style=M.TEAL,    width=26, no_wrap=True)
    tbl.add_column("Srv",   width=3,  justify="center")
    tbl.add_column("Agt",   width=3,  justify="center")
    tbl.add_column("Mem",   width=6,  justify="right")
    tbl.add_column("Disk",  width=5,  justify="right",  style=M.SUBTEXT)
    tbl.add_column("Reqs",  width=5,  justify="right")

    # ── MLX Fleet ──
    _section_sep(tbl, "MLX Fleet", M.MAUVE)
    for n in MLX_NODES:
        _add_node_row(tbl, node_data.get(n, {"name": n, "status": "ERR", **MLX_NODES[n], "model": "-", "agent": "-", "mem_pct": "?", "disk": "?", "reqs": 0}))

    # ── CUDA Cluster ──
    _section_sep(tbl, "CUDA Cluster", M.PEACH)
    for n in CUDA_NODES:
        _add_node_row(tbl, node_data.get(n, {"name": n, "status": "ERR", **CUDA_NODES[n], "model": "-", "agent": "-", "mem_pct": "?", "disk": "?", "reqs": 0}))

    # ── Standalone GPU ──
    _section_sep(tbl, "Standalone GPU", M.FLAMINGO)
    for n in GPU_NODES:
        _add_node_row(tbl, node_data.get(n, {"name": n, "status": "ERR", **GPU_NODES[n], "model": "-", "agent": "-", "mem_pct": "?", "disk": "?", "reqs": 0}))

    # ── Footer ────────────────────────────────────────────────────────────────
    up     = sum(1 for d in node_data.values() if d.get("status") in ("UP", "PING"))
    agents = sum(1 for d in node_data.values() if d.get("agent") == "RUN")
    total  = len(ALL_NODES)

    foot = Text(justify="left")
    foot.append("  Nodes ", style=M.SUBTEXT)
    foot.append(f"{up}/{total}", style=f"bold {M.GREEN}" if up == total else f"bold {M.YELLOW}")
    foot.append("    Agents ", style=M.SUBTEXT)
    foot.append(str(agents), style=f"bold {M.SKY}")
    foot.append("    Tasks ", style=M.SUBTEXT)
    foot.append(f"{done}✓ ", style=M.GREEN)
    foot.append(f"{prog}⚡ ", style=M.YELLOW)
    foot.append(f"{ready}○ ", style=M.OVERLAY)
    if blocked:
        foot.append(f"{blocked}⛔ ", style=M.RED)
    foot.append(f"    Last refresh: {now_ts}", style=f"dim {M.OVERLAY}")

    title_txt = (
        f"[bold {M.MAUVE}]PROTOAI-BAKARI SWARM[/]  "
        f"[{M.OVERLAY}]v2[/]"
    )

    layout = Layout()
    layout.split_column(
        Layout(
            Panel(tbl, title=title_txt, border_style=M.SURFACE, padding=(0, 0)),
            name="main",
        ),
        Layout(
            Panel(foot, border_style=M.SURFACE, padding=(0, 0), height=3),
            name="footer",
            size=3,
        ),
    )
    return layout

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    import argparse
    p = argparse.ArgumentParser(description="ProtoAI-Bakari Cluster Dashboard v2")
    p.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds")
    args = p.parse_args()

    # Warn if sshpass not found
    if not shutil.which("sshpass"):
        console.print(f"[{M.YELLOW}]Warning: sshpass not found — SSH nodes will show DOWN[/]")

    try:
        with Live(
            build_display(),
            console=console,
            refresh_per_second=1,
            screen=True,
        ) as live:
            while True:
                time.sleep(args.interval)
                live.update(build_display())
    except KeyboardInterrupt:
        console.print(f"\n[{M.SUBTEXT}]Dashboard stopped.[/]")

if __name__ == "__main__":
    main()
