#!/usr/bin/env python3
"""
ProtoAI-Bakari Cluster Dashboard v3
Real-time TUI: all nodes, agents, GPU/CPU/RAM, TPS, tasks.
Catppuccin Macchiato palette. Live 5s refresh.

Usage:
  python3 cluster_dashboard.py
  python3 cluster_dashboard.py --interval 10
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

# -- Catppuccin Macchiato Palette --
class M:
    ROSEWATER = "#f4dbd6"; FLAMINGO = "#f0c6c6"; PINK   = "#f5bde6"
    MAUVE     = "#c6a0f6"; RED      = "#ed8796"; MAROON = "#ee99a0"
    PEACH     = "#f5a97f"; YELLOW   = "#eed49f"; GREEN  = "#a6da95"
    TEAL      = "#8bd5ca"; SKY      = "#91d7e3"; SAPPHIRE = "#7dc4e4"
    BLUE      = "#8aadf4"; LAVENDER = "#b7bdf8"; TEXT   = "#cad3f5"
    SUBTEXT   = "#a5adcb"; OVERLAY  = "#6e738d"; SURFACE = "#363a4f"
    BASE      = "#24273a"; MANTLE   = "#1e2030"; CRUST  = "#181926"

# -- Config --
PASSFILE  = os.path.expanduser("~/DEV/authpass")
SSH_OPTS  = "-o StrictHostKeyChecking=no -o ConnectTimeout=5 -o LogLevel=ERROR"
SSH_AGENT_TIMEOUT = 5

# -- Node definitions --
MLX_NODES = {
    "sys1":   {"ip": "127.0.0.1",    "port": 8080, "auth": "local",    "role": "AGENT-HOST", "chip": "M1U 128G", "os": "linux"},
    "sys2":   {"ip": "10.255.255.2",  "port": 8000, "auth": "passfile", "role": "ARCHITECT",  "chip": "M2U 192G", "os": "macos"},
    "sys3":   {"ip": "10.255.255.3",  "port": 8000, "auth": "passfile", "role": "ENGINEER",   "chip": "M2U 192G", "os": "macos"},
    "sys4":   {"ip": "10.255.255.4",  "port": 8000, "auth": "passfile", "role": "CODER",      "chip": "M1U 128G", "os": "macos"},
    "sys5":   {"ip": "10.255.255.5",  "port": 8000, "auth": "passfile", "role": "DESIGNER",   "chip": "M1U 128G", "os": "macos"},
    "sys6":   {"ip": "10.255.255.6",  "port": 8000, "auth": "passfile", "role": "REVIEWER",   "chip": "M1U 128G", "os": "macos"},
    "sys7":   {"ip": "10.255.255.7",  "port": 8000, "auth": "passfile", "role": "FAST-CODE",  "chip": "M1U 128G", "os": "macos"},
}
CUDA_NODES = {
    "cuda-sys1": {"ip": "10.255.255.11", "port": 8000, "auth": "pass_z", "role": "CUDA-BRAIN", "chip": "2x3090", "os": "linux"},
    "cuda-sys2": {"ip": "10.255.255.12", "port": 0,    "auth": "pass_z", "role": "CUDA-WORK",  "chip": "2x3090", "os": "linux"},
    "cuda-vm3":  {"ip": "10.255.255.13", "port": 0,    "auth": "pass_z", "role": "CUDA-WORK",  "chip": "2x3090", "os": "linux"},
    "cuda-vm4":  {"ip": "10.255.255.14", "port": 0,    "auth": "pass_z", "role": "CUDA-WORK",  "chip": "2x3090", "os": "linux"},
}
GPU_NODES = {
    "z4090":  {"ip": "10.255.255.10", "port": 8000, "auth": "pass_z",   "role": "HYPER-CODE", "chip": "4090",   "os": "linux"},
}
ALL_NODES = {**MLX_NODES, **CUDA_NODES, **GPU_NODES}

ROLE_COLORS = {
    "ARCHITECT":  M.RED,     "ENGINEER":  M.YELLOW, "CODER":     M.GREEN,
    "DESIGNER":   M.SKY,     "REVIEWER":  M.TEAL,   "FAST-CODE": M.PINK,
    "AGENT-HOST": M.BLUE,    "CUDA-BRAIN":M.PEACH,  "CUDA-WORK": M.MAROON,
    "HYPER-CODE": M.FLAMINGO,
}

console = Console()

# -- SSH helper --
def ssh_cmd(ip, cmd, auth="passfile", timeout=5):
    if ip in ("127.0.0.1", "localhost"):
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            return r.stdout.strip()
        except Exception:
            return ""
    try:
        # Use subprocess list form to avoid all shell quoting issues
        if auth == "pass_z":
            ssh_args = ["sshpass", "-p", "z", "ssh"] + SSH_OPTS.split() + [f"z@{ip}", cmd]
        else:
            ssh_args = ["sshpass", "-f", PASSFILE, "ssh"] + SSH_OPTS.split() + [f"z@{ip}", cmd]
        r = subprocess.run(ssh_args, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""

# -- Model architecture detection --
_ARCH_SCRIPT = r"""
import json, glob, sys
name = sys.argv[1] if len(sys.argv) > 1 else ""
cfgs = (glob.glob(f"/vmrepo/models/*{name}*/config.json")
      + glob.glob(f"/Users/z/.cache/huggingface/hub/models--*{name}*/snapshots/*/config.json")
      + glob.glob(f"/home/z/.cache/huggingface/hub/models--*{name}*/snapshots/*/config.json"))
if not cfgs: print("-|-|-|-"); sys.exit(0)
d = json.load(open(cfgs[0]))
tc = d.get("text_config", d)
mt = d.get("model_type", "")
L = tc.get("num_hidden_layers", "-")
H = tc.get("hidden_size", "-")
E = tc.get("num_experts")
A = tc.get("num_experts_per_tok")
at = "SSM" if "mamba" in mt.lower() else "HYB" if "hybrid" in mt.lower() else "XF"
moe = f"E{E}/{A}" if E else ("MoE" if "moe" in mt.lower() else "")
print(f"{L}|{H}|{at}|{moe}")
""".strip()

def _model_arch(ip, auth, model_name):
    """Detect model architecture from config.json on remote node."""
    info = {"layers": "-", "hidden": "-", "arch_type": "-", "experts": "-"}
    if not model_name or model_name == "-":
        return info
    search = model_name.replace("/", "--").replace(" ", "*")
    # Pipe the script via stdin to avoid all quoting issues
    if ip in ("127.0.0.1", "localhost"):
        try:
            r = subprocess.run(["python3", "-c", _ARCH_SCRIPT, search],
                capture_output=True, text=True, timeout=8)
            raw = r.stdout.strip()
        except Exception:
            return info
    else:
        try:
            if auth == "pass_z":
                args = ["sshpass", "-p", "z", "ssh"] + SSH_OPTS.split() + [f"z@{ip}", f"python3 - {search}"]
            else:
                args = ["sshpass", "-f", PASSFILE, "ssh"] + SSH_OPTS.split() + [f"z@{ip}", f"python3 - {search}"]
            r = subprocess.run(args, input=_ARCH_SCRIPT, capture_output=True, text=True, timeout=8)
            raw = r.stdout.strip()
        except Exception:
            return info
    if raw and "|" in raw:
        parts = raw.strip().split("|")
        if len(parts) == 4:
            info["layers"] = parts[0]
            info["hidden"] = parts[1]
            info["arch_type"] = parts[2]
            info["experts"] = parts[3] if parts[3] else "-"
    return info

# -- Per-node metric collection --
def _health_check(ip, port, auth):
    if port > 0:
        import urllib.request, json as _json
        # Try /health first (MLX servers)
        try:
            with urllib.request.urlopen(f"http://{ip}:{port}/health", timeout=2) as resp:
                d = _json.loads(resp.read())
            return "UP", d.get("model", "").split("/")[-1][:25]
        except Exception:
            pass
        # Try /v1/models (vLLM/CUDA servers)
        try:
            with urllib.request.urlopen(f"http://{ip}:{port}/v1/models", timeout=2) as resp:
                d = _json.loads(resp.read())
            models = d.get("data", [])
            if models:
                return "UP", models[0].get("id", "").split("/")[-1][:25]
        except Exception:
            pass
    try:
        r = subprocess.run(f"ping -c1 -W1 {ip}", shell=True, capture_output=True, timeout=3)
        return ("PING", "") if r.returncode == 0 else ("DOWN", "")
    except Exception:
        return "DOWN", ""

def _mem_pct(ip, auth, node_os):
    """Get memory usage %. Uses python3 on remote to avoid awk quoting issues."""
    if node_os == "linux":
        raw = ssh_cmd(ip, "head -3 /proc/meminfo | tr -s ' ' | cut -d' ' -f2 | paste -sd' ' | python3 -c 'T,F,A=map(int,input().split());print(int((T-A)/T*100))'", auth, timeout=5)
    else:
        # macOS: vm_stat active+wired pages (real pressure, not file cache)
        raw = ssh_cmd(ip,
            'vm_stat 2>/dev/null | grep -E "Pages (active|wired)" | grep -oE "[0-9]+"',
            auth, timeout=10)
        if raw:
            try:
                pages = [int(x) for x in raw.strip().split('\n') if x.strip().isdigit()]
                if len(pages) >= 2:
                    used_bytes = sum(pages) * 16384  # macOS page = 16KB
                    tot_raw = ssh_cmd(ip, 'sysctl -n hw.memsize', auth, timeout=5)
                    tot = int(tot_raw.strip())
                    pct = int(used_bytes / tot * 100)
                    color = M.RED if pct >= 85 else M.YELLOW if pct >= 60 else M.GREEN
                    return f"[{color}]{pct}%[/]"
            except (ValueError, ZeroDivisionError):
                pass
        return f"[{M.OVERLAY}]?[/]"
    if raw and raw.strip().lstrip("-").isdigit():
        pct = int(raw.strip())
        if pct > 100: pct = 100
        color = M.RED if pct >= 85 else M.YELLOW if pct >= 60 else M.GREEN
        return f"[{color}]{pct}%[/]"
    return f"[{M.OVERLAY}]?[/]"

def _gpu_pct(ip, auth, node_os, chip):
    """Get GPU utilization %. CUDA=nvidia-smi, macOS=powermetrics proxy, Asahi=vulkan."""
    if "3090" in chip or "4090" in chip:
        raw = ssh_cmd(ip, 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1', auth, timeout=5)
        if raw and raw.strip().isdigit():
            pct = int(raw.strip())
            color = M.GREEN if pct < 50 else M.YELLOW if pct < 85 else M.RED
            return f"[{color}]{pct}%[/]"
    elif node_os == "macos":
        # macOS: sudo powermetrics for GPU active residency %
        raw = ssh_cmd(ip, 'cat ~/DEV/authpass | sudo -S powermetrics --samplers gpu_power -i 500 -n 1 2>/dev/null | grep "GPU HW active residency" | grep -oE "[0-9]+\\.[0-9]+" | head -1', auth, timeout=8)
        if raw:
            try:
                pct = int(float(raw.strip()))
                color = M.GREEN if pct < 50 else M.YELLOW if pct < 85 else M.RED
                return f"[{color}]{pct}%[/]"
            except ValueError:
                pass
    return f"[{M.OVERLAY}]-[/]"

def _disk_avail(ip, auth):
    raw = ssh_cmd(ip, "df -h / 2>/dev/null | tail -1", auth, timeout=4)
    sizes = re.findall(r'[\d.]+[TGMK]i?', raw) if raw else []
    val = sizes[2] if len(sizes) >= 3 else (sizes[0] if sizes else "?")
    return val.replace("Gi", "G").replace("Ti", "T").replace("Mi", "M")  # normalize

def _agent_count(ip, auth):
    out = ssh_cmd(ip, "ps aux | grep OMNIAGENT | grep -v grep | wc -l", auth, timeout=SSH_AGENT_TIMEOUT)
    try: return int(out.strip())
    except ValueError: return 0

def _agent_task(ip, auth):
    """Get task from central API (source of truth) by matching agent name to IP."""
    # Map IP to agent name for lookup
    ip_to_agent = {
        "10.255.255.2": "sys2", "10.255.255.3": "sys3", "10.255.255.4": "sys4",
        "10.255.255.5": "sys5", "10.255.255.6": "sys6", "10.255.255.7": "sys7",
    }
    agent_label = ip_to_agent.get(ip, "")
    if not agent_label:
        return f"[{M.OVERLAY}]-[/]"
    try:
        import urllib.request, json as _json
        with urllib.request.urlopen("http://127.0.0.1:9091/tasks", timeout=2) as r:
            content = r.read().decode()
        # Find IN_PROGRESS tasks for this agent
        import re as _re
        m = _re.search(rf'### (T\d+):.*?\[IN_PROGRESS by [^\]]*{agent_label}[^\]]*\|?\s*(\d+)%', content)
        if m:
            tid = m.group(1)
            pct = m.group(2)
            return f"[{M.YELLOW}]{tid}[/] {pct}%"
        # Check without progress %
        m2 = _re.search(rf'### (T\d+):.*?\[IN_PROGRESS by [^\]]*{agent_label}', content)
        if m2:
            return f"[{M.YELLOW}]{m2.group(1)}[/]"
    except Exception:
        pass
    return f"[{M.OVERLAY}]-[/]"

def _req_count(ip, auth):
    # Count POST 200 requests — use strings for binary logs, search all log locations
    out = ssh_cmd(ip, "strings ~/AGENT/LOGS/*_mlx.log ~/AGENT/LOGS/ggml_server.log 2>/dev/null | grep -c 'POST.*200' || echo 0", auth, timeout=5)
    try: return int(out.strip())
    except (ValueError, AttributeError): return 0

def _tps_recent(ip, auth):
    # Parse "TPS:47.9" from MLX server logs — use strings to handle binary/ANSI logs
    out = ssh_cmd(ip, "strings ~/AGENT/LOGS/*_mlx.log 2>/dev/null | grep -ohE 'TPS:[0-9]+\\.?[0-9]*' | tail -5 | grep -oE '[0-9]+\\.[0-9]+'", auth, timeout=5)
    if not out: return 0.0
    try:
        vals = [float(x) for x in out.strip().split("\n") if x.strip()]
        return sum(vals) / len(vals) if vals else 0.0
    except (ValueError, AttributeError): return 0.0

def poll_node(name, node):
    ip   = node["ip"]
    auth = node.get("auth", "passfile")
    port = node.get("port", 0)
    nos  = node.get("os", "macos")

    data = {
        "name": name, "role": node["role"], "chip": node["chip"],
        "status": "DOWN", "model": "-", "agent": "-",
        "mem_pct": f"[{M.OVERLAY}]?[/]", "disk": "?", "reqs": 0, "tps": 0.0,
        "layers": "-", "hidden": "-", "arch_type": "-", "experts": "-",
        "gpu_pct": f"[{M.OVERLAY}]-[/]", "task": f"[{M.OVERLAY}]-[/]",
    }

    status, model = _health_check(ip, port, auth)
    data["status"] = status
    if model: data["model"] = model
    if status == "DOWN": return data

    # Get model architecture info
    arch = _model_arch(ip, auth, model)
    data.update(arch)

    # Agent presence
    if name.startswith("sys"):
        if name != "sys1":
            cnt = _agent_count(ip, auth)
            data["agent"] = "RUN" if cnt > 0 else "OFF"
        else:
            try:
                r = subprocess.run("pgrep -af OMNIAGENT | wc -l", shell=True, capture_output=True, text=True, timeout=3)
                data["agent"] = "RUN" if r.stdout.strip() not in ("0", "") else "OFF"
            except Exception:
                data["agent"] = "OFF"

    # Reqs + TPS + Task for ALL nodes with servers
    data["reqs"] = _req_count(ip, auth)
    data["tps"] = _tps_recent(ip, auth)
    if name.startswith("sys") and name != "sys1":
        data["task"] = _agent_task(ip, auth)

    data["mem_pct"] = _mem_pct(ip, auth, nos)
    data["gpu_pct"] = _gpu_pct(ip, auth, nos, node.get("chip", ""))
    data["disk"]    = _disk_avail(ip, auth)
    return data

# -- Task queue summary --
def count_tasks():
    path = os.path.expanduser("~/AGENT/TASK_QUEUE_v5.md")
    done = prog = ready = blocked = 0
    try:
        for line in open(path):
            if "[DONE"        in line: done    += 1
            elif "[IN_PROGRESS" in line: prog  += 1
            elif "[READY]"    in line: ready   += 1
            elif "[BLOCKED"   in line: blocked += 1
    except Exception: pass
    return done, prog, ready, blocked

# -- Table builder --
def _section_sep(table, label, color):
    table.add_row(f"[bold {color}]-- {label} --[/]", "", "", "", "", "", "", "", "", "", "", "", "", style=f"on {M.MANTLE}")

def _add_node_row(table, d):
    role = d.get("role", "?")
    rc   = ROLE_COLORS.get(role, M.TEXT)
    st   = d.get("status", "?")

    srv_dot = f"[{M.GREEN}]O[/]" if st == "UP" else f"[{M.YELLOW}]~[/]" if st == "PING" else f"[{M.RED}]X[/]"
    agt = d.get("agent", "-")
    agt_str = f"[{M.GREEN}]O[/]" if agt == "RUN" else f"[{M.RED}]X[/]" if agt == "OFF" else f"[{M.OVERLAY}]-[/]"

    tps = d.get("tps", 0.0)
    tps_str = f"[{M.GREEN}]{tps:.0f}[/]" if tps >= 80 else f"[{M.YELLOW}]{tps:.0f}[/]" if tps >= 30 else f"[{M.RED}]{tps:.0f}[/]" if tps > 0 else f"[{M.OVERLAY}]-[/]"

    reqs = d.get("reqs", 0)
    reqs_str = f"[{M.TEAL}]{reqs}[/]" if reqs > 0 else f"[{M.OVERLAY}]-[/]"
    model_str = d.get("model", "-") or "-"
    if model_str == "-": model_str = f"[{M.OVERLAY}]-[/]"

    # Build arch string: "48L 3072h XF E256/8" or "47L 2048h XF MoE"
    layers = d.get("layers", "-")
    hidden = d.get("hidden", "-")
    at = d.get("arch_type", "-")
    exp = d.get("experts", "-")
    if layers != "-":
        arch_parts = [f"{layers}L", f"{hidden}h"]
        at_color = M.GREEN if at == "XF" else M.SKY if at == "SSM" else M.PINK if at == "HYB" else M.OVERLAY
        arch_parts.append(f"[{at_color}]{at}[/]")
        if exp and exp != "-":
            arch_parts.append(f"[{M.PEACH}]{exp}[/]")
        arch_str = " ".join(arch_parts)
    else:
        arch_str = f"[{M.OVERLAY}]-[/]"

    table.add_row(
        f"[bold {M.BLUE}]{d['name']}[/]",
        f"[{rc}]{role}[/]",
        f"[{M.SUBTEXT}]{d.get('chip','?')}[/]",
        model_str, arch_str,
        srv_dot, agt_str,
        d.get("mem_pct", "?"), d.get("gpu_pct", f"[{M.OVERLAY}]-[/]"),
        d.get("disk", "?"),
        d.get("task", f"[{M.OVERLAY}]-[/]"),
        tps_str, reqs_str,
    )

# -- Main display builder --
def build_display():
    now_ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    done, prog, ready, blocked = count_tasks()

    node_data = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(poll_node, n, nd): n for n, nd in ALL_NODES.items()}
        for f in as_completed(futures):
            n = futures[f]
            try: node_data[n] = f.result()
            except Exception:
                node_data[n] = {"name": n, "status": "ERR", "role": ALL_NODES[n]["role"],
                    "chip": ALL_NODES[n]["chip"], "model": "-", "agent": "-",
                    "mem_pct": f"[{M.OVERLAY}]?[/]", "disk": "?", "reqs": 0, "tps": 0.0}

    tbl = Table(box=box.ROUNDED, show_lines=True, show_edge=True, padding=(0, 1),
                expand=True, border_style=M.SURFACE, header_style=f"bold {M.LAVENDER}")
    tbl.add_column("Node",  style=f"bold {M.BLUE}", width=10, no_wrap=True)
    tbl.add_column("Role",  width=10, no_wrap=True)
    tbl.add_column("Chip",  style=M.SUBTEXT, width=9,  no_wrap=True)
    tbl.add_column("Model", style=M.TEAL,    width=22, no_wrap=True)
    tbl.add_column("Arch",  width=14, no_wrap=True)
    tbl.add_column("Srv",   width=3,  justify="center")
    tbl.add_column("Agt",   width=3,  justify="center")
    tbl.add_column("Mem",   width=5,  justify="right")
    tbl.add_column("GPU",   width=4,  justify="right")
    tbl.add_column("Disk",  width=5,  justify="right",  style=M.SUBTEXT)
    tbl.add_column("Task",  width=8,  no_wrap=True)
    tbl.add_column("TPS",   width=4,  justify="right")
    tbl.add_column("Reqs",  width=4,  justify="right")

    _section_sep(tbl, "MLX Fleet", M.MAUVE)
    for n in MLX_NODES:
        _add_node_row(tbl, node_data.get(n, {"name": n, "status": "ERR", **MLX_NODES[n], "model": "-", "agent": "-", "mem_pct": "?", "disk": "?", "reqs": 0, "tps": 0.0, "layers": "-", "hidden": "-", "arch_type": "-", "experts": "-", "gpu_pct": f"[{M.OVERLAY}]-[/]", "task": f"[{M.OVERLAY}]-[/]"}))

    _section_sep(tbl, "CUDA Cluster", M.PEACH)
    for n in CUDA_NODES:
        _add_node_row(tbl, node_data.get(n, {"name": n, "status": "ERR", **CUDA_NODES[n], "model": "-", "agent": "-", "mem_pct": "?", "disk": "?", "reqs": 0, "tps": 0.0, "layers": "-", "hidden": "-", "arch_type": "-", "experts": "-", "gpu_pct": f"[{M.OVERLAY}]-[/]", "task": f"[{M.OVERLAY}]-[/]"}))

    _section_sep(tbl, "Standalone GPU", M.FLAMINGO)
    for n in GPU_NODES:
        _add_node_row(tbl, node_data.get(n, {"name": n, "status": "ERR", **GPU_NODES[n], "model": "-", "agent": "-", "mem_pct": "?", "disk": "?", "reqs": 0, "tps": 0.0, "layers": "-", "hidden": "-", "arch_type": "-", "experts": "-", "gpu_pct": f"[{M.OVERLAY}]-[/]", "task": f"[{M.OVERLAY}]-[/]"}))

    up     = sum(1 for d in node_data.values() if d.get("status") in ("UP", "PING"))
    agents = sum(1 for d in node_data.values() if d.get("agent") == "RUN")
    total  = len(ALL_NODES)

    foot = Text(justify="left")
    foot.append("  Nodes ", style=M.SUBTEXT)
    foot.append(f"{up}/{total}", style=f"bold {M.GREEN}" if up == total else f"bold {M.YELLOW}")
    foot.append("    Agents ", style=M.SUBTEXT)
    foot.append(str(agents), style=f"bold {M.SKY}")
    foot.append("    Tasks ", style=M.SUBTEXT)
    foot.append(f"{done}D ", style=M.GREEN)
    foot.append(f"{prog}A ", style=M.YELLOW)
    foot.append(f"{ready}R ", style=M.OVERLAY)
    if blocked: foot.append(f"{blocked}B ", style=M.RED)
    foot.append(f"    Last refresh: {now_ts}", style=f"dim {M.OVERLAY}")

    title_txt = f"[bold {M.MAUVE}]PROTOAI-BAKARI SWARM[/]  [{M.OVERLAY}]v3[/]"
    layout = Layout()
    layout.split_column(
        Layout(Panel(tbl, title=title_txt, border_style=M.SURFACE, padding=(0, 0)), name="main"),
        Layout(Panel(foot, border_style=M.SURFACE, padding=(0, 0), height=3), name="footer", size=3),
    )
    return layout

# -- Entry point --
def main():
    import argparse
    p = argparse.ArgumentParser(description="ProtoAI-Bakari Cluster Dashboard v3")
    p.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds")
    args = p.parse_args()

    if not shutil.which("sshpass"):
        console.print(f"[{M.YELLOW}]Warning: sshpass not found[/]")

    try:
        with Live(build_display(), console=console, refresh_per_second=1, screen=True) as live:
            while True:
                time.sleep(args.interval)
                live.update(build_display())
    except KeyboardInterrupt:
        console.print(f"\n[{M.SUBTEXT}]Dashboard stopped.[/]")

if __name__ == "__main__":
    main()
