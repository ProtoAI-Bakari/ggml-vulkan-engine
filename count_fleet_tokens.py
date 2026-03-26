#!/usr/bin/env python3
"""
Fleet Token Counter — aggregates token counts across all nodes.

MLX nodes (sys2-sys7): parses MLX server logs for "Tok:N" lines
CUDA .11:             queries vLLM /metrics endpoint
z4090 .10:            queries vLLM /metrics endpoint
sys1 (local):         parses ggml_server.log for timing lines

Usage:
  python3 count_fleet_tokens.py
  python3 count_fleet_tokens.py --json
"""

import subprocess
import re
import sys
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ─────────────────────────────────────────────────────────────────────
PASSFILE = "/home/z/DEV/authpass"
SSH_OPTS = "-o StrictHostKeyChecking=no -o ConnectTimeout=5 -o LogLevel=ERROR"
TIMEOUT = 10

MLX_NODES = {
    "sys2": {"ip": "10.255.255.2", "log": "~/AGENT/LOGS/sys2_mlx.log"},
    "sys3": {"ip": "10.255.255.3", "log": "~/AGENT/LOGS/sys3_mlx.log"},
    "sys4": {"ip": "10.255.255.4", "log": "~/AGENT/LOGS/sys4_mlx.log"},
    "sys5": {"ip": "10.255.255.5", "log": "~/AGENT/LOGS/sys5_mlx.log"},
    "sys6": {"ip": "10.255.255.6", "log": "~/AGENT/LOGS/sys6_mlx.log"},
    "sys7": {"ip": "10.255.255.7", "log": "~/AGENT/LOGS/sys7_mlx.log"},
}

VLLM_NODES = {
    "cuda.11": {"ip": "10.255.255.11", "port": 8000, "auth": "pass_z"},
    "z4090":   {"ip": "10.255.255.10", "port": 8000, "auth": "pass_z"},
}

SYS1_LOG = "/home/z/AGENT/LOGS/ggml_server.log"


# ── Helpers ────────────────────────────────────────────────────────────────────

def ssh_cmd(ip, cmd, auth="passfile"):
    """Run a command over SSH. Returns stdout or empty string on failure."""
    try:
        if auth == "passfile":
            args = ["sshpass", "-f", PASSFILE, "ssh"] + SSH_OPTS.split() + [f"z@{ip}", cmd]
        else:
            args = ["sshpass", "-p", "z", "ssh"] + SSH_OPTS.split() + [f"z@{ip}", cmd]
        r = subprocess.run(args, capture_output=True, text=True, timeout=TIMEOUT)
        return r.stdout.strip()
    except Exception:
        return ""


def http_get(url, timeout=5):
    """Fetch a URL. Returns body text or empty string on failure."""
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def parse_vllm_metrics(body):
    """Extract prompt_tokens_total, generation_tokens_total, request counts from Prometheus metrics."""
    prompt_tok = 0
    gen_tok = 0
    requests = 0

    for line in body.splitlines():
        if line.startswith("#"):
            continue
        # vllm:prompt_tokens_total{...} VALUE
        m = re.match(r'^vllm:prompt_tokens_total\{.*\}\s+([\d.eE+\-]+)', line)
        if m:
            prompt_tok += int(float(m.group(1)))
        m = re.match(r'^vllm:generation_tokens_total\{.*\}\s+([\d.eE+\-]+)', line)
        if m:
            gen_tok += int(float(m.group(1)))
        # Sum all request_success_total (stop, length, abort, etc.)
        m = re.match(r'^vllm:request_success_total\{.*\}\s+([\d.eE+\-]+)', line)
        if m:
            requests += int(float(m.group(1)))

    return {"prompt_tokens": prompt_tok, "gen_tokens": gen_tok, "requests": requests}


# ── Node collectors ────────────────────────────────────────────────────────────

def collect_mlx(name, node):
    """Collect token stats from an MLX node via SSH log parsing."""
    ip, log = node["ip"], node["log"]
    # Use strings to handle binary logs, grep for Tok: lines
    raw = ssh_cmd(ip, f"strings {log} 2>/dev/null | grep 'Tok:'", auth="passfile")
    if not raw:
        return {"node": name, "status": "DOWN", "gen_tokens": 0, "requests": 0, "avg_tps": 0}

    lines = raw.strip().splitlines()
    total_tok = 0
    tps_vals = []

    for line in lines:
        # Extract Tok:NNN
        tok_m = re.search(r'Tok:(\d+)', line)
        if tok_m:
            total_tok += int(tok_m.group(1))
        # Extract TPS:NN.N
        tps_m = re.search(r'TPS:([\d.]+)', line)
        if tps_m:
            tps_vals.append(float(tps_m.group(1)))

    avg_tps = round(sum(tps_vals) / len(tps_vals), 1) if tps_vals else 0
    return {
        "node": name,
        "status": "OK",
        "gen_tokens": total_tok,
        "requests": len(lines),
        "avg_tps": avg_tps,
    }


def collect_vllm(name, node):
    """Collect token stats from a vLLM node via /metrics endpoint."""
    url = f"http://{node['ip']}:{node['port']}/metrics"
    body = http_get(url)
    if not body:
        return {"node": name, "status": "DOWN", "prompt_tokens": 0, "gen_tokens": 0, "requests": 0}

    stats = parse_vllm_metrics(body)
    return {"node": name, "status": "OK", **stats}


def collect_sys1():
    """Collect token stats from sys1 local ggml_server.log."""
    try:
        with open(SYS1_LOG, "rb") as f:
            raw = f.read()
    except Exception:
        return {"node": "sys1", "status": "DOWN", "gen_tokens": 0, "requests": 0, "avg_tps": 0}

    # Decode safely
    text = raw.decode("utf-8", errors="replace")
    # Format: [Timing @ 12700 tokens] Graph: 591.0us, Compute: 39226.0us, Total: 40199.0us (24.9 TPS)
    timing_lines = re.findall(
        r'\[Timing @ (\d+) tokens\].*?\(([\d.]+) TPS\)', text
    )

    if not timing_lines:
        return {"node": "sys1", "status": "NO_DATA", "gen_tokens": 0, "requests": 0, "avg_tps": 0}

    # The token count in timing lines is cumulative; take the max as total generated
    max_tok = max(int(t[0]) for t in timing_lines)
    tps_vals = [float(t[1]) for t in timing_lines]
    # Recent average TPS (last 50 entries)
    recent_tps = tps_vals[-50:] if len(tps_vals) > 50 else tps_vals
    avg_tps = round(sum(recent_tps) / len(recent_tps), 1)

    # Count distinct "sessions" by detecting resets (token count drops)
    prev = 0
    sessions = 0
    for tok_str, _ in timing_lines:
        tok = int(tok_str)
        if tok < prev:
            sessions += 1
        prev = tok
    sessions += 1  # Count the current session

    return {
        "node": "sys1",
        "status": "OK",
        "gen_tokens": max_tok,
        "requests": sessions,
        "avg_tps": avg_tps,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def collect_all():
    """Collect from all nodes in parallel."""
    results = []
    futures = {}

    with ThreadPoolExecutor(max_workers=10) as pool:
        # MLX nodes
        for name, node in MLX_NODES.items():
            futures[pool.submit(collect_mlx, name, node)] = name
        # vLLM nodes
        for name, node in VLLM_NODES.items():
            futures[pool.submit(collect_vllm, name, node)] = name
        # sys1 local
        futures[pool.submit(collect_sys1)] = "sys1"

        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({"node": futures[fut], "status": f"ERROR: {e}", "gen_tokens": 0, "requests": 0})

    # Sort by node name
    results.sort(key=lambda r: r["node"])
    return results


def print_table(results):
    """Print a clean summary table."""
    # Column widths
    W_NODE = 10
    W_STATUS = 8
    W_GEN = 14
    W_PROMPT = 14
    W_REQ = 10
    W_TPS = 10

    header = (
        f"{'Node':<{W_NODE}} "
        f"{'Status':<{W_STATUS}} "
        f"{'Gen Tokens':>{W_GEN}} "
        f"{'Prompt Tokens':>{W_PROMPT}} "
        f"{'Requests':>{W_REQ}} "
        f"{'Avg TPS':>{W_TPS}}"
    )
    sep = "-" * len(header)

    print()
    print("  FLEET TOKEN SUMMARY")
    print(f"  {sep}")
    print(f"  {header}")
    print(f"  {sep}")

    total_gen = 0
    total_prompt = 0
    total_req = 0

    for r in results:
        node = r["node"]
        status = r.get("status", "?")
        gen_tok = r.get("gen_tokens", 0)
        prompt_tok = r.get("prompt_tokens", 0)
        requests = r.get("requests", 0)
        avg_tps = r.get("avg_tps", "")

        total_gen += gen_tok
        total_prompt += prompt_tok
        total_req += requests

        gen_str = f"{gen_tok:,}" if gen_tok else "-"
        prompt_str = f"{prompt_tok:,}" if prompt_tok else "-"
        req_str = f"{requests:,}" if requests else "-"
        tps_str = f"{avg_tps}" if avg_tps else "-"

        print(
            f"  {node:<{W_NODE}} "
            f"{status:<{W_STATUS}} "
            f"{gen_str:>{W_GEN}} "
            f"{prompt_str:>{W_PROMPT}} "
            f"{req_str:>{W_REQ}} "
            f"{tps_str:>{W_TPS}}"
        )

    print(f"  {sep}")
    print(
        f"  {'TOTAL':<{W_NODE}} "
        f"{'':<{W_STATUS}} "
        f"{total_gen:>{W_GEN},} "
        f"{total_prompt:>{W_PROMPT},} "
        f"{total_req:>{W_REQ},} "
        f"{'':>{W_TPS}}"
    )
    print(f"  {sep}")
    print()


def main():
    json_mode = "--json" in sys.argv
    results = collect_all()

    if json_mode:
        total_gen = sum(r.get("gen_tokens", 0) for r in results)
        total_prompt = sum(r.get("prompt_tokens", 0) for r in results)
        total_req = sum(r.get("requests", 0) for r in results)
        output = {
            "nodes": results,
            "totals": {
                "gen_tokens": total_gen,
                "prompt_tokens": total_prompt,
                "requests": total_req,
            },
        }
        print(json.dumps(output, indent=2))
    else:
        print_table(results)


if __name__ == "__main__":
    main()
