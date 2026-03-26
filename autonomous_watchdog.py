#!/usr/bin/env python3
"""
Autonomous Watchdog v2 — Intelligent fleet monitor with CUDA brain + fix capabilities.
Designed to run on z4090 (10.255.255.10). Monitors 12-node AI cluster.

INTELLIGENCE:
  - Calls CUDA brain (Qwen3.5-122B @ .11) to analyze problems and recommend fixes
  - Can also use local z4090 model (qwen3-coder-30b) for fast triage

FIX CAPABILITIES:
  - Restart stuck/dead agents with fresh context
  - Sync updated code from sys1 to nodes
  - Release stuck IN_PROGRESS tasks
  - Truncate bloated logs
  - Kill runaway processes eating CPU/memory
  - Repair MLX server processes
  - Inject corrective guidance via comms bridge

MONITORING:
  - Deep log analysis (loops, errors, stalls, context overflow)
  - Vulkan Llama-8B coherency testing (100+ token diverse prompts)
  - Git commit tracking
  - Task progress tracking with stall detection
  - MLX server health (TPS, latency, GPU%)

Usage:
    python3 autonomous_watchdog.py                # run 4h on z4090
    python3 autonomous_watchdog.py --duration 1   # 1 hour
    python3 autonomous_watchdog.py --once          # single check
    python3 autonomous_watchdog.py --local         # run locally (sys1)
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
DURATION_HOURS = -1  # -1 = infinite
POLL_INTERVAL = 120         # 2 min between fleet checks
VULKAN_TEST_INTERVAL = 300  # 5 min between coherency tests
GIT_CHECK_INTERVAL = 600    # 10 min between git checks
REPORT_INTERVAL = 1800      # 30 min detailed report
BRAIN_CONSULT_INTERVAL = 900  # 15 min between CUDA brain consultations
MAX_RESTARTS_PER_NODE = 3   # per session
MAX_FIX_ATTEMPTS = 2        # per issue type per node

PASSFILE = Path.home() / "DEV" / "authpass"
SSH_OPTS = "-o StrictHostKeyChecking=no -o ConnectTimeout=8 -o LogLevel=ERROR"
BASE = Path.home() / "AGENT"
REPORT_DIR = BASE / "watchdog_reports"
LOG_FILE = BASE / "LOGS" / "watchdog.log"

# Detect if running on z4090 or sys1
HOSTNAME = subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip()
IS_Z4090 = "z4090" in HOSTNAME or os.path.exists("/slowrepo")  # z4090 marker
IS_LOCAL = not IS_Z4090  # running on sys1

# ═══════════════════════════════════════════════════════════════
# NODE DEFINITIONS
# ═══════════════════════════════════════════════════════════════
NODES = {
    "sys1":  {"ip": "10.255.255.128", "log": "~/AGENT/LOGS/main_trace.log",  "auth": "passfile", "type": "agent-host",
              "chip": "M1U 128G", "model_port": 8080, "agent_cmd": "python3 OMNIAGENT_v4_focused.py --no-tty"},
    "sys2":  {"ip": "10.255.255.2",   "log": "~/AGENT/LOGS/agent_trace.log", "auth": "passfile", "type": "mlx",
              "chip": "M2U 192G", "model_port": 8000, "agent_cmd": "python3 OMNIAGENT_v4_focused.py --no-tty"},
    "sys3":  {"ip": "10.255.255.3",   "log": "~/AGENT/LOGS/agent_trace.log", "auth": "passfile", "type": "mlx",
              "chip": "M2U 192G", "model_port": 8000, "agent_cmd": "python3 OMNIAGENT_v4_focused.py --no-tty"},
    "sys4":  {"ip": "10.255.255.4",   "log": "~/AGENT/LOGS/agent_trace.log", "auth": "passfile", "type": "mlx",
              "chip": "M1U 128G", "model_port": 8000, "agent_cmd": "python3 OMNIAGENT_v4_focused.py --no-tty"},
    "sys5":  {"ip": "10.255.255.5",   "log": "~/AGENT/LOGS/agent_trace.log", "auth": "passfile", "type": "mlx",
              "chip": "M1U 128G", "model_port": 8000, "agent_cmd": "python3 OMNIAGENT_v4_focused.py --no-tty"},
    "sys6":  {"ip": "10.255.255.6",   "log": "~/AGENT/LOGS/agent_trace.log", "auth": "passfile", "type": "mlx",
              "chip": "M1U 128G", "model_port": 8000, "agent_cmd": "python3 OMNIAGENT_v4_focused.py --no-tty"},
    "sys7":  {"ip": "10.255.255.7",   "log": "~/AGENT/LOGS/agent_trace.log", "auth": "passfile", "type": "mlx",
              "chip": "M1U 128G", "model_port": 8000, "agent_cmd": "python3 OMNIAGENT_v4_focused.py --no-tty"},
}

CUDA_NODES = {
    "cuda-sys1": {"ip": "10.255.255.11", "auth": "pass_z", "type": "cuda", "model_port": 8000},
    "cuda-sys2": {"ip": "10.255.255.12", "auth": "pass_z", "type": "cuda", "model_port": 8000},
    "cuda-vm3":  {"ip": "10.255.255.13", "auth": "pass_z", "type": "cuda", "model_port": 8000},
    "cuda-vm4":  {"ip": "10.255.255.14", "auth": "pass_z", "type": "cuda", "model_port": 8000},
}

MLX_NODES = [n for n, d in NODES.items() if d["type"] == "mlx"]  # sys2-sys7
VULKAN_URL = "http://10.255.255.128:8080"
TASK_API = "http://10.255.255.128:9091"
CUDA_BRAIN_URL = "http://10.255.255.11:8000/v1/chat/completions"
LOCAL_BRAIN_URL = "http://10.255.255.10:8000/v1/chat/completions"  # z4090 qwen3-coder-30b
COMMS_BRIDGE_PATH = "~/AGENT/COMMS_BRIDGE.md"

# ═══════════════════════════════════════════════════════════════
# VULKAN TEST PROMPTS (diverse, requiring 100+ coherent tokens)
# ═══════════════════════════════════════════════════════════════
VULKAN_PROMPTS = [
    "A farmer has 3 fields: 2.5 acres, 1.75 acres, and 3.25 acres. He plants wheat on 60% and corn on the rest. How many acres of each? Show your work.",
    "Explain why the square root of 2 is irrational. Give a proof by contradiction.",
    "Describe photosynthesis from start to finish. Include light-dependent and light-independent reactions.",
    "What were the main causes and consequences of the French Revolution? Be specific about dates and key figures.",
    "Write a short story (about 100 words) about a robot who discovers it can dream. Make it emotional.",
    "Explain TCP vs UDP to someone who has never programmed. Use real-world analogies.",
    "Write a Python function for the Sieve of Eratosthenes. Include comments explaining each step.",
    "Write a bash script that monitors disk usage and alerts if any partition exceeds 80%. Include error handling.",
    "Explain how CRISPR gene editing works. Key components and ethical concerns.",
    "Describe the lifecycle of a star from nebula to final stage. How does mass determine its fate?",
    "Translate to French then back to English: 'The quick brown fox jumps over the lazy dog near the riverbank at sunset.'",
    "List the planets in order from the Sun, with one interesting fact about each.",
    "Explain the difference between a linked list and an array. When would you use each? Give code examples.",
    "What is the halting problem? Why is it unsolvable? Explain with a simple example.",
    "Describe three sorting algorithms, their time complexities, and when each is best suited.",
]


# ═══════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════

def log(level, msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    colors = {"INFO": "\033[36m", "WARN": "\033[33m", "ERROR": "\033[31m",
              "OK": "\033[32m", "TEST": "\033[35m", "CRIT": "\033[41;37m",
              "BRAIN": "\033[95m", "FIX": "\033[96m"}
    c = colors.get(level, "")
    r = "\033[0m"
    line = f"[{ts}] [{level:5}] {msg}"
    # Print with colors to terminal
    print(f"{c}{line}{r}", flush=True)
    # Write plain text (no ANSI) to log file — avoids duplication with tee
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a") as f:
            # Strip any ANSI codes that might be in msg
            clean = re.sub(r'\033\[[0-9;]*m', '', line)
            f.write(clean + "\n")
    except:
        pass


def ssh(ip, cmd, auth="passfile", timeout=15):
    """SSH to a node. Handles passfile, pass_z, and local execution."""
    # If running on sys1 and target is sys1 — run locally
    if IS_LOCAL and ip == "10.255.255.128":
        try:
            r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
            return r.returncode, r.stdout.strip()
        except subprocess.TimeoutExpired:
            return 1, "TIMEOUT"
        except Exception as e:
            return 1, str(e)
    try:
        if auth == "passfile":
            args = ["sshpass", "-f", str(PASSFILE), "ssh"] + SSH_OPTS.split() + [f"z@{ip}", cmd]
        elif auth == "pass_z":
            args = ["sshpass", "-p", "z", "ssh"] + SSH_OPTS.split() + [f"z@{ip}", cmd]
        else:
            args = ["ssh"] + SSH_OPTS.split() + [f"z@{ip}", cmd]
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip()
    except subprocess.TimeoutExpired:
        return 1, "TIMEOUT"
    except Exception as e:
        return 1, str(e)


def scp_to(ip, local_path, remote_path, auth="passfile", timeout=30):
    """SCP a file to a remote node."""
    try:
        if auth == "passfile":
            args = ["sshpass", "-f", str(PASSFILE), "scp"] + SSH_OPTS.split()[:4] + [local_path, f"z@{ip}:{remote_path}"]
        elif auth == "pass_z":
            args = ["sshpass", "-p", "z", "scp"] + SSH_OPTS.split()[:4] + [local_path, f"z@{ip}:{remote_path}"]
        else:
            args = ["scp"] + SSH_OPTS.split()[:4] + [local_path, f"z@{ip}:{remote_path}"]
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0
    except:
        return False


def http_get(url, timeout=10):
    import urllib.request, urllib.error
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception as e:
        return 0, str(e)


def http_post_json(url, data, timeout=60):
    import urllib.request, urllib.error
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        try:
            return e.code, e.read().decode()
        except:
            return e.code, ""
    except Exception as e:
        return 0, str(e)


def comms_write(node_ip, auth, msg):
    """Write to a node's comms bridge."""
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    escaped = msg.replace('"', '\\"').replace("'", "\\'")
    ssh(node_ip, f"echo '[{ts}] [WATCHDOG] {escaped}' >> {COMMS_BRIDGE_PATH}", auth, timeout=8)


def comms_write_local(msg):
    """Write to the local (sys1) comms bridge."""
    try:
        ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        comms_path = os.path.expanduser(COMMS_BRIDGE_PATH)
        with open(comms_path, "a") as f:
            f.write(f"\n[{ts}] [WATCHDOG] {msg}\n")
    except:
        # If local write fails, try via SSH
        comms_write("10.255.255.128", "passfile", msg)


# ═══════════════════════════════════════════════════════════════
# LLM BRAIN — CUDA (.11) and LOCAL z4090
# ═══════════════════════════════════════════════════════════════

BRAIN_SYSTEM_PROMPT = """You are the reasoning core of an autonomous cluster watchdog managing a 12-node AI fleet.

Fleet layout:
- sys1 (M1 Ultra 128GB, .128): Primary host. Runs Vulkan Llama-8B on :8080 and task server on :9091. UNTOUCHABLE — never restart sys1 itself.
- sys2-sys7 (Mac Studios, .2-.7): MLX inference + agent nodes. Each runs an MLX model server (:8000) and OMNIAGENT_v4_focused.py.
- z4090 (RTX 4090, .10): Orchestrator host. This is where you run.
- cuda-sys1 (.11): 2x3090 running Qwen3.5-122B-FP8 via vLLM. That's YOU (the brain being queried).
- cuda-sys2/vm3/vm4 (.12-.14): Additional 3090 pairs, currently idle.

Agent architecture:
- Each agent (sys2-sys7) runs OMNIAGENT_v4_focused.py in a tmux session named "agent"
- Agents use their local MLX server as primary brain, can escalate to cuda_brain
- Agents claim tasks from central task API (sys1:9091), one task at a time
- Agent logs go to ~/AGENT/LOGS/agent_trace.log
- Communication via ~/AGENT/COMMS_BRIDGE.md on each node

Common issues and fixes:
1. CLAIM LOOP: Agent keeps trying to claim DONE tasks → restart agent to clear context
2. CONTEXT OVERFLOW: Agent's conversation too long → restart agent (clears history)
3. PARSE FAILURES: Model outputs malformed tool calls → check if model server is healthy, restart agent
4. STALE/NO OUTPUT: Agent making turns but not producing files → check task assignment, may need redirect
5. STREAM ERRORS: Model server down or overloaded → check server process, restart if needed
6. TASK HOPPING: Agent claiming multiple tasks → nudge to focus on one
7. MLX SERVER DOWN: Model not responding → restart MLX server process

Your job: Given diagnostic data, output a JSON array of fix actions. Each action:
{
  "node": "sys5",
  "action": "restart_agent|nudge|restart_server|sync_code|release_task|truncate_log|kill_process",
  "reason": "why",
  "details": {"task_id": "T38", "message": "focus nudge text", "process": "python3"}
}

Rules:
- NEVER recommend modifying CUDA nodes or sys1's agent/server
- Restarts only for sys2-sys7
- Prefer nudge over restart. Prefer restart over kill.
- If multiple nodes have the same issue, it's likely systemic — diagnose root cause first
- Be concise. Output ONLY the JSON array, no explanation."""


CUDA_MODEL_ID = "/vmrepo/models/Qwen3.5-122B-A10B-FP8"
LOCAL_MODEL_ID = "qwen3-coder-30b"


def ask_brain(question, use_cuda=True, max_tokens=2000):
    """Query the CUDA brain (.11) or local z4090 brain for reasoning."""
    url = CUDA_BRAIN_URL if use_cuda else LOCAL_BRAIN_URL
    model_id = CUDA_MODEL_ID if use_cuda else LOCAL_MODEL_ID
    brain_name = "CUDA-122B" if use_cuda else "z4090-30B"

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": BRAIN_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }

    log("BRAIN", f"Consulting {brain_name}...")
    t0 = time.time()
    status, body = http_post_json(url, payload, timeout=120)
    elapsed = time.time() - t0

    if status != 200:
        log("ERROR", f"{brain_name} returned HTTP {status}")
        # Fall back to local brain if CUDA fails
        if use_cuda:
            log("WARN", "Falling back to local z4090 brain...")
            return ask_brain(question, use_cuda=False, max_tokens=max_tokens)
        return None

    try:
        data = json.loads(body)
        content = data["choices"][0]["message"]["content"]
        # Strip thinking tags (Qwen3.5 uses <think>...</think>)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        # Strip "Thinking Process:" preamble before the actual JSON output
        if "Thinking Process:" in content:
            # Find the JSON array/object after the thinking
            json_start = None
            for marker in ['[{', '[\n{', '```json', '```']:
                idx = content.find(marker)
                if idx >= 0:
                    json_start = idx
                    break
            if json_start is not None:
                content = content[json_start:].strip()
            else:
                # No JSON found — strip thinking and keep whatever remains
                content = re.sub(r'Thinking Process:.*?(?:\n\n|\Z)', '', content, flags=re.DOTALL).strip()
        log("BRAIN", f"{brain_name} responded in {elapsed:.1f}s ({len(content)} chars)")
        return content
    except Exception as e:
        log("ERROR", f"Failed to parse {brain_name} response: {e}")
        return None


def parse_brain_actions(response):
    """Parse the brain's JSON response into actionable fix steps."""
    if not response:
        return []
    try:
        # Try to find JSON array in the response
        # Handle markdown code blocks
        cleaned = re.sub(r'```json\s*', '', response)
        cleaned = re.sub(r'```\s*', '', cleaned)

        # Find the JSON array
        match = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if match:
            actions = json.loads(match.group(0))
            if isinstance(actions, list):
                return actions
        # Try parsing the whole thing
        actions = json.loads(cleaned)
        if isinstance(actions, list):
            return actions
        return []
    except (json.JSONDecodeError, ValueError):
        log("WARN", f"Could not parse brain response as JSON: {response[:200]}")
        return []


# ═══════════════════════════════════════════════════════════════
# DEEP LOG ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_node(name, node):
    """Deep log analysis — returns structured diagnostic."""
    ip, logpath, auth = node["ip"], node["log"], node["auth"]
    result = {
        "node": name, "status": "UNKNOWN", "issues": [], "warnings": [],
        "turns": 0, "tps": 0, "ttft": 0, "task": "-", "last_activity": "-",
        "errors": [], "progress_pct": 0, "log_snippet": "",
    }

    # Get last 400 lines of log
    rc, raw = ssh(ip, f"strings {logpath} 2>/dev/null | tail -400", auth)
    if rc != 0 or not raw:
        result["status"] = "DEAD"
        result["issues"].append("No log output — agent may be dead")
        # Check if SSH works
        rc2, _ = ssh(ip, "echo ok", auth, timeout=8)
        if rc2 != 0:
            result["issues"].append("SSH UNREACHABLE — node may be down entirely")
        else:
            # SSH works but no log — check if agent process exists
            rc3, ps = ssh(ip, "pgrep -f OMNIAGENT | head -3", auth, timeout=8)
            if rc3 != 0 or not ps.strip():
                result["issues"].append("AGENT PROCESS MISSING — needs restart")
            else:
                result["issues"].append(f"Agent PID {ps.strip()} exists but log is empty/missing")
        return result

    # Save snippet for brain analysis
    result["log_snippet"] = raw[-2000:]  # last ~2000 chars for brain

    # Basic metrics
    turns = re.findall(r'Turn (\d+)', raw)
    result["turns"] = int(turns[-1]) if turns else 0

    tps_matches = re.findall(r'(\d+\.?\d*)\s*t/s', raw)
    result["tps"] = float(tps_matches[-1]) if tps_matches else 0

    ttft_matches = re.findall(r'TTFT\s*(\d+)ms', raw)
    result["ttft"] = int(ttft_matches[-1]) if ttft_matches else 0

    task_match = re.findall(r'(?:CLAIMED|working on|Task[: ]+)(T\d+)', raw)
    result["task"] = task_match[-1] if task_match else "-"

    progress_match = re.findall(r'(\d+)%', raw[-500:])
    result["progress_pct"] = int(progress_match[-1]) if progress_match else 0

    # Timestamp analysis
    ts_matches = re.findall(r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})', raw)
    if ts_matches:
        result["last_activity"] = ts_matches[-1]
        try:
            last_ts = datetime.fromisoformat(ts_matches[-1].replace(" ", "T"))
            age_minutes = (datetime.now() - last_ts).total_seconds() / 60
            if age_minutes > 15:
                result["warnings"].append(f"STALE: last activity {int(age_minutes)}min ago")
            if age_minutes > 30:
                result["issues"].append(f"VERY STALE: no activity for {int(age_minutes)}min")
        except:
            pass

    # ── ISSUE DETECTION ──

    # Context overflow
    overflow = raw.count("context length") + raw.count("input_tokens") + raw.count("maximum context")
    if overflow >= 2:
        result["issues"].append(f"CONTEXT OVERFLOW x{overflow}")

    # Parse failures
    parse_fails = raw.count("SELF-CORRECTION") + raw.count("parse error") + raw.count("JSON parse")
    if parse_fails >= 3:
        result["issues"].append(f"PARSE FAILURES x{parse_fails}")

    # Task claim issues
    already_done = raw.count("ALREADY_DONE") + raw.count("already done")
    if already_done >= 3:
        result["issues"].append(f"CLAIM LOOP x{already_done}")

    blocked = raw.count("BLOCKED")
    if blocked >= 5:
        result["issues"].append(f"BLOCKED LOOP x{blocked}")

    # Repeated tool calls (stuck)
    tool_calls = re.findall(r'(?:EXECUTING|calling|tool_call)[:\]]\s*(\w+)', raw)
    if len(tool_calls) >= 6:
        last8 = tool_calls[-8:]
        freq = Counter(last8)
        top, count = freq.most_common(1)[0]
        if count >= 6:
            result["issues"].append(f"STUCK: calling {top} {count}/8 times")

    # File read loop
    file_reads = re.findall(r'read_file.*?(?:path|file).*?([^\s"\']+\.(?:py|c|md|h|comp))', raw)
    if file_reads:
        freq = Counter(file_reads)
        top, count = freq.most_common(1)[0]
        if count >= 5:
            result["issues"].append(f"FILE LOOP: reading {top} x{count}")

    # Queue read loop
    queue_reads = raw.count("TASK_QUEUE_v5.md")
    if queue_reads >= 6:
        result["issues"].append(f"QUEUE LOOP x{queue_reads}")

    # Stream/connection errors
    stream_errors = raw.count("STREAM ERROR") + raw.count("ConnectionError") + raw.count("Connection refused")
    if stream_errors >= 3:
        result["issues"].append(f"STREAM/CONN ERRORS x{stream_errors}")

    # Timeout loop
    timeouts = raw.count("TIMEOUT after") + raw.count("timed out")
    if timeouts >= 3:
        result["issues"].append(f"TIMEOUT LOOP x{timeouts}")

    # GO_PROMPT loop
    go_reloads = raw.count("GO_PROMPT")
    if go_reloads >= 8:
        result["issues"].append(f"GO_PROMPT LOOP x{go_reloads}")

    # Wrong paths in tool calls only
    wrong_tool_calls = len(re.findall(r'(?:read_file|write_file|path)["\s:]+/Users/z/', raw))
    wrong_v4 = raw.count("TASK_QUEUE_v4")
    if wrong_tool_calls + wrong_v4 >= 2:
        result["issues"].append(f"WRONG PATHS x{wrong_tool_calls + wrong_v4}")

    # Python/runtime errors
    py_errors = re.findall(r'(Traceback|TypeError|ValueError|KeyError|AttributeError|ImportError|FileNotFoundError)', raw)
    if py_errors:
        freq = Counter(py_errors)
        for err, count in freq.most_common(3):
            if count >= 2:
                result["errors"].append(f"{err} x{count}")

    # Productive output detection — count actual work indicators
    productive = (raw.count("write_file") + raw.count("git commit") + raw.count("git add") +
                  raw.count("COMPLETED") + raw.count("Successfully wrote") +
                  raw.count("execute_bash") + raw.count("push_changes") +
                  raw.count("EXIT CODE: 0"))
    if result["turns"] > 30 and productive == 0:
        result["warnings"].append("NO PRODUCTIVE OUTPUT despite many turns")

    # NEW DETECTION: Zero-token turns (model returning empty responses)
    zero_tok = len(re.findall(r'0 tok \| 0\.0 t/s', raw[-3000:]))
    if zero_tok >= 5:
        result["issues"].append(f"ZERO-TOKEN LOOP x{zero_tok} — model returning empty responses")

    # NEW DETECTION: Mock/stub server responses
    if "simulated inference" in raw or "ggml-vllm-mock" in raw:
        result["issues"].append("MOCK SERVER — agent hitting stub, not real model")

    # NEW DETECTION: STUCK escalation (asking Claude/brain repeatedly)
    stuck_count = raw.count("[STUCK]") + raw.count("Asking Claude")
    if stuck_count >= 3:
        result["issues"].append(f"STUCK ESCALATION x{stuck_count} — agent can't produce tool calls")

    # NEW DETECTION: ALREADY_DONE loop (trying to claim finished tasks)
    already_done = raw[-2000:].count("ALREADY_DONE")
    if already_done >= 3:
        result["issues"].append(f"DONE-TASK LOOP x{already_done} — claiming finished tasks")

    # NEW DETECTION: NOT_FOUND_OR_NOT_READY (hallucinating task IDs)
    not_found = raw[-2000:].count("NOT_FOUND_OR_NOT_READY")
    if not_found >= 3:
        result["issues"].append(f"PHANTOM TASKS x{not_found} — claiming non-existent tasks")

    # NEW DETECTION: Rapid turn cycling (100+ turns in a short log = spinning)
    if result["turns"] > 1000:
        # Check if turns are cycling too fast (>10 turns/sec = empty responses)
        turn_nums = re.findall(r'Turn (\d+)', raw[-2000:])
        if len(turn_nums) >= 2:
            span = int(turn_nums[-1]) - int(turn_nums[0])
            if span > 100:
                result["warnings"].append(f"RAPID CYCLING: {span} turns in recent log window")

    # NEW DETECTION: Agent calling ./claim_task.sh directly (bypassing Python tool)
    direct_claims = raw.count("claim_task.sh")
    if direct_claims >= 5:
        result["warnings"].append(f"DIRECT CLAIM CALLS x{direct_claims} — agent using bash for claims")

    # NEW DETECTION: TAKEN loop (all tasks claimed by others)
    taken = raw[-2000:].count("TAKEN by")
    if taken >= 5:
        result["issues"].append(f"ALL-TAKEN LOOP x{taken} — no tasks available for this agent")

    # NEW DETECTION: Agent not updating progress (IN_PROGRESS but 0% for long time)
    if result["turns"] > 50 and result["progress_pct"] == 0 and result["task"] != "-":
        result["warnings"].append("STALE PROGRESS: task claimed but 0% after 50+ turns")

    # Multi-claim detection
    claimed = re.findall(r'CLAIMED (T\d+)', raw)
    unique = set(claimed[-10:]) if claimed else set()
    if len(unique) > 2:
        result["warnings"].append(f"TASK HOPPING: {len(unique)} different tasks")

    # Log size check
    rc_sz, size_str = ssh(ip, f"wc -c < {logpath} 2>/dev/null", auth, timeout=8)
    if rc_sz == 0 and size_str.strip().isdigit():
        size_mb = int(size_str.strip()) / (1024 * 1024)
        if size_mb > 50:
            result["warnings"].append(f"LOG BLOAT: {size_mb:.0f}MB")
        if size_mb > 200:
            result["issues"].append(f"LOG CRITICAL: {size_mb:.0f}MB — truncation needed")

    # Status
    if result["issues"]:
        result["status"] = "SICK"
    elif result["warnings"]:
        result["status"] = "WARN"
    else:
        result["status"] = "OK"

    return result


def check_model_server(name, node):
    """Check if the MLX/Vulkan model server is responding."""
    ip = node["ip"]
    port = node.get("model_port", 8000)
    url = f"http://{ip}:{port}"

    # Try /v1/models
    status, body = http_get(f"{url}/v1/models", timeout=8)
    if status == 200:
        try:
            data = json.loads(body)
            model_id = data["data"][0]["id"] if data.get("data") else "unknown"
            return {"healthy": True, "model": model_id}
        except:
            return {"healthy": True, "model": "unknown"}

    # Try /health (Vulkan server)
    status2, body2 = http_get(f"{url}/health", timeout=8)
    if status2 == 200:
        return {"healthy": True, "model": "vulkan"}

    return {"healthy": False, "error": f"HTTP {status}/{status2}"}


# ═══════════════════════════════════════════════════════════════
# FIX OPERATIONS
# ═══════════════════════════════════════════════════════════════

def fix_restart_agent(name, node, reason="watchdog auto-restart"):
    """Kill and restart agent in tmux on a remote node."""
    ip, auth = node["ip"], node["auth"]
    if node["type"] not in ("mlx",):
        log("WARN", f"Won't restart {name} — type={node['type']}")
        return False

    log("FIX", f"Restarting {name} agent: {reason}")

    TMUX = "/opt/homebrew/bin/tmux"

    # Kill existing agent process
    ssh(ip, "pkill -f 'python3.*OMNIAGENT' 2>/dev/null", auth, timeout=10)
    time.sleep(2)
    ssh(ip, "pkill -9 -f 'python3.*OMNIAGENT' 2>/dev/null", auth, timeout=10)
    time.sleep(1)

    # Verify kill — check no stale procs + memory released
    rc_ps, ps_out = ssh(ip, "ps -ef | grep OMNIAGENT | grep -v grep | head -5", auth, timeout=8)
    if ps_out.strip():
        log("WARN", f"{name}: stale procs after kill: {ps_out[:100]}")
        # Force kill by PID
        pids = re.findall(r'\s(\d+)\s', ps_out)
        for pid in pids[:5]:
            ssh(ip, f"kill -9 {pid} 2>/dev/null", auth, timeout=5)
        time.sleep(1)
    else:
        log("OK", f"{name}: agent processes cleared")

    # Check memory after kill (macOS: vm_stat; Linux: free -h)
    rc_mem, mem_out = ssh(ip, "vm_stat 2>/dev/null | head -4 || free -h 2>/dev/null | head -3", auth, timeout=8)
    if mem_out:
        log("INFO", f"{name} memory after kill: {mem_out[:120]}")

    # Write restart marker
    ssh(ip, f"echo '[WATCHDOG RESTART {datetime.now().isoformat()}] {reason}' >> ~/AGENT/LOGS/agent_trace.log", auth, timeout=8)

    # Kill old tmux session if any
    TMUX = "/opt/homebrew/bin/tmux"
    ssh(ip, f"{TMUX} kill-session -t agent 2>/dev/null", auth, timeout=8)
    time.sleep(1)

    # Step 1: Sync latest agent code from sys1 before restarting
    log("FIX", f"Syncing latest code to {name}...")
    for f in ["OMNIAGENT_v4_focused.py", "GO_PROMPT.md"]:
        # SCP from sys1 to target via intermediate if needed
        src_ip = "10.255.255.128"
        if IS_Z4090:
            # z4090 -> pull from sys1 to tmp -> push to target
            tmp = f"/tmp/_sync_{f.replace('/','_')}"
            subprocess.run(
                ["sshpass", "-f", str(PASSFILE), "scp"] + SSH_OPTS.split()[:4] +
                [f"z@{src_ip}:~/AGENT/{f}", tmp],
                capture_output=True, timeout=15)
            if os.path.exists(tmp):
                scp_to(ip, tmp, f"~/AGENT/{f}", auth)
                try: os.remove(tmp)
                except: pass
        else:
            # Running on sys1 — direct SCP
            scp_to(ip, os.path.expanduser(f"~/AGENT/{f}"), f"~/AGENT/{f}", auth)

    # Step 2: Ensure all required Python packages are available
    ssh(ip, (
        "python3 -c 'import requests, openai, numpy' 2>/dev/null || "
        "pip3 install requests openai numpy 2>/dev/null"
    ), auth, timeout=60)

    # Step 3: Launch agent with correct --name flag
    ssh(ip, (
        f"cd ~/AGENT && "
        f"nohup python3 OMNIAGENT_v4_focused.py --auto-go --name 'OmniAgent [{name}]' "
        f">> ~/AGENT/LOGS/agent_trace.log 2>&1 &"
    ), auth, timeout=15)
    time.sleep(5)

    rc3, pid = ssh(ip, "pgrep -f OMNIAGENT | head -1", auth, timeout=8)

    if "agent" in (sessions or ""):
        log("OK", f"Agent {name} restarted, tmux session active" + (f", PID={pid.strip()}" if pid.strip() else ""))
        comms_write(ip, auth, f"Agent restarted by watchdog: {reason}")
        return True
    elif rc3 == 0 and pid.strip():
        log("OK", f"Agent {name} restarted, PID={pid.strip()} (tmux check inconclusive)")
        comms_write(ip, auth, f"Agent restarted by watchdog: {reason}")
        return True
    else:
        log("ERROR", f"Agent {name} restart failed: tmux='{sessions[:80]}', pid='{pid[:40]}'")
        return False


def fix_restart_server(name, node, reason="server unresponsive"):
    """Restart the MLX model server on a node."""
    ip, auth = node["ip"], node["auth"]
    if node["type"] not in ("mlx",):
        return False

    log("FIX", f"Restarting MLX server on {name}: {reason}")

    # We don't know the exact server command, so just check if mlx_lm is running
    rc, ps = ssh(ip, "ps aux | grep 'mlx_lm.server\\|vllm.entrypoints' | grep -v grep | head -3", auth, timeout=10)
    if rc != 0 or not ps.strip():
        log("WARN", f"No MLX server process found on {name} — may be started differently")
        return False

    # Kill and rely on whatever service manager restarts it
    # Or: we know the agents USE the server, so if server is down we just flag it
    log("WARN", f"Server on {name} needs manual attention — not auto-restarting model servers")
    return False


def fix_nudge(name, node, message):
    """Send a targeted guidance message to an agent via comms bridge."""
    ip, auth = node["ip"], node["auth"]
    log("FIX", f"Nudging {name}: {message[:80]}")
    comms_write(ip, auth, message)
    return True


def fix_release_task(task_id, reason="stuck"):
    """Release a stuck IN_PROGRESS task back to READY via task API."""
    log("FIX", f"Releasing task {task_id}: {reason}")
    # The task server doesn't have a release endpoint, so we do it via direct file edit on sys1
    cmd = f"cd ~/AGENT && sed -i 's/\\[IN_PROGRESS by [^]]*\\]/[READY]/g' TASK_QUEUE_v5.md"
    # Too dangerous to release ALL — just release the specific task
    cmd = f"cd ~/AGENT && python3 -c \"\nimport re\nwith open('TASK_QUEUE_v5.md', 'r') as f: t = f.read()\nt = re.sub(r'(### {task_id}:.*?)\\[IN_PROGRESS[^\\]]*\\]', r'\\1[READY]', t)\nwith open('TASK_QUEUE_v5.md', 'w') as f: f.write(t)\nprint('Released {task_id}')\n\""
    rc, out = ssh("10.255.255.128", cmd, "passfile", timeout=10)
    if rc == 0:
        log("OK", f"Released {task_id} back to READY")
        return True
    else:
        log("ERROR", f"Failed to release {task_id}: {out[:100]}")
        return False


def fix_truncate_log(name, node):
    """Truncate a bloated agent log, keeping last 10K lines."""
    ip, auth = node["ip"], node["auth"]
    logpath = node["log"]
    log("FIX", f"Truncating log on {name}")
    rc, _ = ssh(ip, f"tail -10000 {logpath} > /tmp/agent_log_trim && mv /tmp/agent_log_trim {logpath}", auth, timeout=15)
    if rc == 0:
        log("OK", f"Truncated log on {name}")
        return True
    return False


def fix_sync_code(name, node, files=None):
    """Sync agent code from sys1 to a node."""
    ip, auth = node["ip"], node["auth"]
    if files is None:
        files = ["OMNIAGENT_v4_focused.py", "TASK_QUEUE_v5.md", "GO_PROMPT.md"]

    log("FIX", f"Syncing {len(files)} files to {name}")
    results = []
    for f in files:
        # First get from sys1
        src = os.path.expanduser(f"~/AGENT/{f}")
        # If we're on z4090, we need to pull from sys1 first then push to target
        if IS_Z4090:
            # SCP from sys1 to local tmp
            tmp = f"/tmp/watchdog_sync_{f.replace('/', '_')}"
            rc_get, _ = subprocess.run(
                ["sshpass", "-f", str(PASSFILE), "scp"] + SSH_OPTS.split()[:4] +
                [f"z@10.255.255.128:~/AGENT/{f}", tmp],
                capture_output=True, text=True, timeout=20
            ).returncode, ""
            if os.path.exists(tmp):
                ok = scp_to(ip, tmp, f"~/AGENT/{f}", auth)
                results.append(f"{f}: {'OK' if ok else 'FAIL'}")
                os.remove(tmp)
            else:
                results.append(f"{f}: FAIL (couldn't get from sys1)")
        else:
            # Running on sys1 — SCP directly
            ok = scp_to(ip, src, f"~/AGENT/{f}", auth)
            results.append(f"{f}: {'OK' if ok else 'FAIL'}")

    log("OK" if all("OK" in r for r in results) else "WARN",
        f"Sync to {name}: {', '.join(results)}")
    return results


def fix_kill_process(name, node, pattern):
    """Kill a specific process pattern on a node."""
    ip, auth = node["ip"], node["auth"]
    log("FIX", f"Killing '{pattern}' on {name}")
    rc, out = ssh(ip, f"pkill -f '{pattern}' 2>/dev/null; echo killed", auth, timeout=10)
    return rc == 0


def execute_fix_actions(actions, watchdog_state):
    """Execute a list of fix actions from the brain."""
    executed = []
    for action in actions:
        node_name = action.get("node", "")
        act = action.get("action", "")
        reason = action.get("reason", "brain recommended")
        details = action.get("details", {})

        if node_name not in NODES:
            log("WARN", f"Unknown node: {node_name}")
            continue

        node = NODES[node_name]

        # Enforce limits
        restart_key = f"{node_name}_restart"
        fix_key = f"{node_name}_{act}"

        if act == "restart_agent":
            if watchdog_state["restart_counts"].get(node_name, 0) >= MAX_RESTARTS_PER_NODE:
                log("WARN", f"Max restarts reached for {node_name} ({MAX_RESTARTS_PER_NODE})")
                continue
            ok = fix_restart_agent(node_name, node, reason)
            if ok:
                watchdog_state["restart_counts"][node_name] = watchdog_state["restart_counts"].get(node_name, 0) + 1
            executed.append({"node": node_name, "action": act, "success": ok, "reason": reason})

        elif act == "nudge":
            msg = details.get("message", f"WATCHDOG: {reason}")
            fix_nudge(node_name, node, msg)
            executed.append({"node": node_name, "action": act, "success": True, "reason": reason})

        elif act == "restart_server":
            fix_restart_server(node_name, node, reason)
            executed.append({"node": node_name, "action": act, "success": False, "reason": "manual"})

        elif act == "sync_code":
            files = details.get("files", None)
            fix_sync_code(node_name, node, files)
            executed.append({"node": node_name, "action": act, "success": True, "reason": reason})

        elif act == "release_task":
            tid = details.get("task_id", "")
            if tid:
                ok = fix_release_task(tid, reason)
                executed.append({"node": node_name, "action": act, "task": tid, "success": ok})

        elif act == "truncate_log":
            ok = fix_truncate_log(node_name, node)
            executed.append({"node": node_name, "action": act, "success": ok})

        elif act == "kill_process":
            pat = details.get("process", "")
            if pat:
                fix_kill_process(node_name, node, pat)
                executed.append({"node": node_name, "action": act, "success": True})

    return executed


# ═══════════════════════════════════════════════════════════════
# VULKAN COHERENCY TESTING
# ═══════════════════════════════════════════════════════════════

def test_vulkan_coherency(prompt_idx=None):
    """Test Vulkan Llama-8B for coherent 100+ token response."""
    import random
    if prompt_idx is None:
        prompt_idx = random.randint(0, len(VULKAN_PROMPTS) - 1)

    prompt = VULKAN_PROMPTS[prompt_idx % len(VULKAN_PROMPTS)]
    payload = {
        "model": "llama",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.7,
    }

    t0 = time.time()
    status, body = http_post_json(f"{VULKAN_URL}/v1/chat/completions", payload, timeout=90)
    elapsed = time.time() - t0

    result = {
        "prompt_idx": prompt_idx, "prompt": prompt[:80],
        "status_code": status, "elapsed_s": round(elapsed, 2),
        "coherent": False, "tokens": 0, "tps": 0,
        "response": "", "issues": [],
    }

    if status != 200:
        result["issues"].append(f"HTTP {status}")
        return result

    try:
        data = json.loads(body)
        content = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        tps = data.get("_tps", 0)

        result["tokens"] = tokens
        result["tps"] = round(tps, 1)
        result["response"] = content[:300]

        issues = []
        if tokens < 50:
            issues.append(f"SHORT: only {tokens} tokens")
        if re.search(r'(.)\1{10,}', content):
            issues.append("GIBBERISH: repeated chars")
        words = content.split()
        if len(words) >= 20:
            trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
            freq = Counter(trigrams)
            top_tg, top_count = freq.most_common(1)[0]
            if top_count >= 4 and top_count > len(trigrams) * 0.15:
                issues.append(f"REPETITION: '{top_tg}' x{top_count}")
        if "<|eot_id|>" in content or "<|end" in content:
            issues.append("STOP TOKEN LEAK")
        if "." not in content and "!" not in content and "?" not in content:
            issues.append("NO SENTENCE ENDINGS")
        if tps > 0 and tps < 5:
            issues.append(f"LOW TPS: {tps}")

        result["issues"] = issues
        result["coherent"] = len(issues) == 0
    except Exception as e:
        result["issues"].append(f"PARSE ERROR: {e}")

    return result


# ═══════════════════════════════════════════════════════════════
# GIT + TASK MONITORING
# ═══════════════════════════════════════════════════════════════

def check_git_activity():
    rc, raw = ssh("10.255.255.128", "cd ~/AGENT && git log --oneline --since='4 hours ago' 2>/dev/null | head -30", "passfile")
    if rc != 0:
        return {"error": "git unreachable", "commits": []}
    commits = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    rc2, diff = ssh("10.255.255.128", "cd ~/AGENT && git diff --stat 2>/dev/null | tail -5", "passfile")
    return {"recent_commits": len(commits), "commits": commits[:20], "uncommitted": diff or "none"}


def check_task_progress():
    status, body = http_get(f"{TASK_API}/tasks/summary", timeout=10)
    if status != 200:
        return {"error": f"HTTP {status}"}
    try:
        return json.loads(body)
    except:
        return {"error": "Invalid JSON"}


def check_task_assignments():
    status, body = http_get(f"{TASK_API}/tasks", timeout=10)
    if status != 200:
        return {}
    assignments = {}
    for line in body.split("\n"):
        m = re.match(r"###\s+(T\d+):.*?\[IN_PROGRESS by ([^\]|]+)", line)
        if m:
            assignments[m.group(1)] = m.group(2).strip()
    return assignments


# ═══════════════════════════════════════════════════════════════
# WATCHDOG MAIN CLASS
# ═══════════════════════════════════════════════════════════════

class AutonomousWatchdog:

    def __init__(self, duration_hours=-1):
        self.duration = float('inf') if duration_hours < 0 else duration_hours * 3600
        self.start_time = time.time()
        self.running = True
        self.last_vulkan_test = 0
        self.last_git_check = 0
        self.last_report = 0
        self.last_brain_consult = 0
        self.vulkan_prompt_idx = 0
        self.consecutive_sick = defaultdict(int)  # track how many times a node stays SICK

        self.state = {
            "restart_counts": defaultdict(int),
            "fix_counts": defaultdict(int),
        }

        self.history = {
            "vulkan_tests": [], "node_checks": [], "restarts": [],
            "alerts": [], "task_snapshots": [], "brain_consultations": [],
            "fixes_executed": [],
        }
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

    def elapsed_str(self):
        e = time.time() - self.start_time
        return f"{int(e//3600)}h{int((e%3600)//60):02d}m"

    def remaining_str(self):
        if self.duration == float('inf'):
            return "INF"
        r = max(0, self.duration - (time.time() - self.start_time))
        return f"{int(r//3600)}h{int((r%3600)//60):02d}m"

    # ──────────── Fleet Check ────────────

    def run_fleet_check(self):
        log("INFO", f"[{self.elapsed_str()}] Fleet check...")
        results = {}
        threads = []

        def check(name, node):
            results[name] = analyze_node(name, node)

        for name, node in NODES.items():
            t = threading.Thread(target=check, args=(name, node), daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=25)

        # Also check model servers
        server_status = {}
        for name, node in NODES.items():
            server_status[name] = check_model_server(name, node)

        # Print summary
        sick = [r for r in results.values() if r["status"] == "SICK"]
        warn = [r for r in results.values() if r["status"] == "WARN"]
        ok = [r for r in results.values() if r["status"] == "OK"]
        dead = [r for r in results.values() if r["status"] == "DEAD"]

        log("INFO", f"Fleet: {len(ok)} OK, {len(warn)} WARN, {len(sick)} SICK, {len(dead)} DEAD")

        for r in results.values():
            name = r["node"]
            srv = server_status.get(name, {})
            srv_icon = "S" if srv.get("healthy") else "X"
            if r["status"] == "OK":
                log("OK", f"  {name:6} [OK/{srv_icon}] task={r['task']} T={r['turns']} tps={r['tps']} ttft={r['ttft']}ms")
            elif r["status"] == "WARN":
                log("WARN", f"  {name:6} [!!/{srv_icon}] task={r['task']} T={r['turns']} tps={r['tps']}")
                for w in r["warnings"]:
                    log("WARN", f"    {w}")
            elif r["status"] == "SICK":
                log("ERROR", f"  {name:6} [SICK/{srv_icon}] task={r['task']} T={r['turns']}")
                for i in r["issues"]:
                    log("ERROR", f"    {i}")
                for e in r["errors"]:
                    log("ERROR", f"    ERR: {e}")
            else:
                log("CRIT", f"  {name:6} [DEAD]")
                for i in r["issues"]:
                    log("CRIT", f"    {i}")

        # Track consecutive sickness
        for name, r in results.items():
            if r["status"] in ("SICK", "DEAD"):
                self.consecutive_sick[name] += 1
            else:
                self.consecutive_sick[name] = 0

        # ── AUTO-REMEDIATION (rule-based, no brain needed) ──
        for r in (dead):
            name = r["node"]
            node = NODES[name]
            if node["type"] != "mlx":
                continue
            if "AGENT PROCESS MISSING" in " ".join(r["issues"]):
                if self.state["restart_counts"].get(name, 0) < MAX_RESTARTS_PER_NODE:
                    fix_restart_agent(name, node, "agent process missing")
                    self.state["restart_counts"][name] = self.state["restart_counts"].get(name, 0) + 1
                    self.history["restarts"].append({
                        "time": datetime.now().isoformat(), "node": name, "reason": "DEAD/missing"})

        # Simple fixes for common issues
        for r in sick:
            name = r["node"]
            node = NODES[name]
            if node["type"] != "mlx":
                continue

            issues_str = " | ".join(r["issues"])

            # Context overflow — restart clears history
            if "CONTEXT OVERFLOW" in issues_str:
                if self.state["restart_counts"].get(name, 0) < MAX_RESTARTS_PER_NODE:
                    fix_restart_agent(name, node, "context overflow — clearing history")
                    self.state["restart_counts"][name] = self.state["restart_counts"].get(name, 0) + 1
                    self.history["restarts"].append({
                        "time": datetime.now().isoformat(), "node": name, "reason": "context overflow"})
                continue

            # Parse failures persistent — restart to clear corrupted context
            if "PARSE FAILURES" in issues_str and self.consecutive_sick[name] >= 2:
                if self.state["restart_counts"].get(name, 0) < MAX_RESTARTS_PER_NODE:
                    fix_restart_agent(name, node, "persistent parse failures — clearing context")
                    self.state["restart_counts"][name] = self.state["restart_counts"].get(name, 0) + 1
                    self.history["restarts"].append({
                        "time": datetime.now().isoformat(), "node": name, "reason": "parse failures"})
                continue

            # Claim loop — restart to reset agent state
            if "CLAIM LOOP" in issues_str and self.consecutive_sick[name] >= 2:
                if self.state["restart_counts"].get(name, 0) < MAX_RESTARTS_PER_NODE:
                    fix_restart_agent(name, node, "claim loop — agent confused about task state")
                    self.state["restart_counts"][name] = self.state["restart_counts"].get(name, 0) + 1
                    self.history["restarts"].append({
                        "time": datetime.now().isoformat(), "node": name, "reason": "claim loop"})
                continue

            # Log bloat — truncate
            if "LOG CRITICAL" in issues_str:
                fix_truncate_log(name, node)
                continue

            # First offense — just nudge
            if self.consecutive_sick[name] <= 1:
                fix_nudge(name, node, f"WATCHDOG: Issues detected: {issues_str[:120]}. Refocus on your current task.")
            # Second offense with same issues — will be escalated to brain

        self.history["node_checks"].append({
            "time": datetime.now().isoformat(),
            "summary": {r["node"]: {"status": r["status"], "task": r["task"], "issues": r["issues"]}
                        for r in results.values()},
        })
        return results, server_status

    # ──────────── Brain Consultation ────────────

    def consult_brain(self, fleet_results, server_status, task_summary):
        """Send diagnostics to CUDA brain for intelligent analysis and fix recommendations."""
        # Only consult if there are real problems
        sick_nodes = {name: r for name, r in fleet_results.items()
                      if r["status"] in ("SICK", "DEAD") and self.consecutive_sick.get(name, 0) >= 2}

        if not sick_nodes:
            log("BRAIN", "No persistent issues — skipping brain consultation")
            return []

        # Build diagnostic summary for the brain
        diagnostic = "FLEET DIAGNOSTIC REPORT\n\n"
        diagnostic += f"Time: {datetime.now().isoformat()}\n"
        diagnostic += f"Watchdog uptime: {self.elapsed_str()}\n\n"

        diagnostic += "NODE STATUS:\n"
        for name, r in fleet_results.items():
            srv = server_status.get(name, {})
            diagnostic += f"\n{name} [{r['status']}] server={'UP' if srv.get('healthy') else 'DOWN'}\n"
            diagnostic += f"  Task: {r['task']}, Turns: {r['turns']}, TPS: {r['tps']}, TTFT: {r['ttft']}ms\n"
            if r["issues"]:
                diagnostic += f"  Issues: {', '.join(r['issues'])}\n"
            if r["warnings"]:
                diagnostic += f"  Warnings: {', '.join(r['warnings'])}\n"
            if r["errors"]:
                diagnostic += f"  Errors: {', '.join(r['errors'])}\n"
            # Include consecutive sick count
            if name in sick_nodes:
                diagnostic += f"  Consecutive sick checks: {self.consecutive_sick[name]}\n"
                diagnostic += f"  Restarts this session: {self.state['restart_counts'].get(name, 0)}\n"

        # Include log snippets for sick nodes only (keep prompt manageable)
        for name, r in sick_nodes.items():
            if r.get("log_snippet"):
                diagnostic += f"\n--- LOG SNIPPET {name} (last 500 chars) ---\n"
                diagnostic += r["log_snippet"][-500:] + "\n"

        diagnostic += f"\nTASK SUMMARY: {json.dumps(task_summary)}\n"

        assignments = check_task_assignments()
        if assignments:
            diagnostic += f"TASK ASSIGNMENTS: {json.dumps(assignments)}\n"

        diagnostic += f"\nRESTART BUDGET REMAINING: {json.dumps({n: MAX_RESTARTS_PER_NODE - self.state['restart_counts'].get(n, 0) for n in MLX_NODES})}\n"

        diagnostic += "\nWhat fix actions should I take? Return a JSON array of actions."

        # Ask the brain
        response = ask_brain(diagnostic, use_cuda=True, max_tokens=1500)

        if not response:
            log("WARN", "Brain consultation returned nothing")
            return []

        actions = parse_brain_actions(response)
        log("BRAIN", f"Brain recommended {len(actions)} actions")
        for a in actions:
            log("BRAIN", f"  -> {a.get('node','?')}: {a.get('action','?')} — {a.get('reason','?')[:80]}")

        # Execute the recommended fixes
        executed = execute_fix_actions(actions, self.state)
        self.history["fixes_executed"].extend(executed)
        self.history["brain_consultations"].append({
            "time": datetime.now().isoformat(),
            "sick_nodes": list(sick_nodes.keys()),
            "recommended": len(actions),
            "executed": len(executed),
            "actions": actions[:5],  # keep history manageable
        })

        return executed

    # ──────────── Vulkan Test ────────────

    def run_vulkan_test(self):
        log("TEST", f"[{self.elapsed_str()}] Vulkan coherency test #{self.vulkan_prompt_idx}...")
        result = test_vulkan_coherency(self.vulkan_prompt_idx)
        self.vulkan_prompt_idx = (self.vulkan_prompt_idx + 1) % len(VULKAN_PROMPTS)

        if result["coherent"]:
            log("OK", f"  PASS: {result['tokens']}tok, {result['tps']}TPS, {result['elapsed_s']}s")
            log("OK", f"  {result['response'][:120]}")
        else:
            issues = ", ".join(result["issues"])
            log("ERROR", f"  FAIL: {issues}")
            log("ERROR", f"  {result['response'][:120]}")
            comms_write_local(f"Vulkan coherency FAILED: {issues}")
            self.history["alerts"].append({
                "time": datetime.now().isoformat(), "type": "vulkan", "issues": result["issues"]})

            # If vulkan fails, try to diagnose
            if "HTTP 0" in issues or "HTTP 5" in issues:
                log("WARN", "Vulkan server may be down — checking...")
                srv = check_model_server("sys1", NODES["sys1"])
                if not srv.get("healthy"):
                    log("CRIT", "VULKAN SERVER DOWN on sys1!")
                    comms_write_local("CRITICAL: Vulkan server down on sys1:8080!")

        self.history["vulkan_tests"].append({
            "time": datetime.now().isoformat(), "coherent": result["coherent"],
            "tokens": result["tokens"], "tps": result["tps"], "issues": result["issues"]})
        return result

    # ──────────── Git + Tasks ────────────

    def run_git_check(self):
        log("INFO", f"[{self.elapsed_str()}] Git activity check...")
        git = check_git_activity()
        log("INFO", f"  {git['recent_commits']} commits in last 4h")
        for c in git["commits"][:5]:
            log("INFO", f"    {c}")
        if git.get("uncommitted") and git["uncommitted"] != "none":
            log("WARN", f"  Uncommitted: {git['uncommitted'][:100]}")
        return git

    def run_task_check(self):
        summary = check_task_progress()
        if "error" in summary:
            log("ERROR", f"Task API: {summary['error']}")
            return summary
        log("INFO", f"  Tasks: {summary.get('DONE',0)}D {summary.get('IN_PROGRESS',0)}A {summary.get('READY',0)}R {summary.get('BLOCKED',0)}B")

        # Check for stale IN_PROGRESS tasks
        assignments = check_task_assignments()
        for tid, agent in sorted(assignments.items()):
            log("INFO", f"    {tid}: {agent}")

        self.history["task_snapshots"].append({
            "time": datetime.now().isoformat(), "summary": summary})

        # Detect if DONE count hasn't changed in 30min (no progress across fleet)
        if len(self.history["task_snapshots"]) >= 3:
            last3 = self.history["task_snapshots"][-3:]
            done_counts = [s["summary"].get("DONE", 0) for s in last3]
            if len(set(done_counts)) == 1:
                log("WARN", f"NO TASK PROGRESS: DONE stuck at {done_counts[0]} for 3 checks")

        return summary

    # ──────────── CUDA Brain Check ────────────

    def check_cuda_health(self):
        """Quick check that CUDA brain is alive."""
        status, _ = http_get(f"http://10.255.255.11:8000/v1/models", timeout=8)
        if status != 200:
            log("CRIT", "CUDA BRAIN (.11) UNREACHABLE!")
            return False
        return True

    # ──────────── Report ────────────

    def write_report(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        vt = self.history["vulkan_tests"]
        report = {
            "generated": datetime.now().isoformat(),
            "elapsed": self.elapsed_str(),
            "remaining": self.remaining_str(),
            "vulkan": {
                "total": len(vt),
                "passed": sum(1 for t in vt if t["coherent"]),
                "failed": sum(1 for t in vt if not t["coherent"]),
                "avg_tps": round(sum(t["tps"] for t in vt) / max(len(vt), 1), 1),
            },
            "fleet": {
                "checks": len(self.history["node_checks"]),
                "restarts": len(self.history["restarts"]),
                "alerts": len(self.history["alerts"]),
                "brain_consultations": len(self.history["brain_consultations"]),
                "fixes_executed": len(self.history["fixes_executed"]),
            },
            "tasks": self.history["task_snapshots"][-1] if self.history["task_snapshots"] else {},
            "restart_details": self.history["restarts"][-10:],
            "fix_details": self.history["fixes_executed"][-10:],
            "brain_details": self.history["brain_consultations"][-5:],
            "alert_details": self.history["alerts"][-10:],
            "restart_budget": {n: MAX_RESTARTS_PER_NODE - self.state["restart_counts"].get(n, 0)
                               for n in MLX_NODES},
        }

        path = REPORT_DIR / f"watchdog_{ts}.json"
        path.write_text(json.dumps(report, indent=2, default=str))
        log("OK", f"Report: {path.name}")

        v = report["vulkan"]
        f = report["fleet"]
        comms_write_local(
            f"Watchdog [{self.elapsed_str()}]: Vulkan {v['passed']}/{v['total']} OK @ {v['avg_tps']}TPS, "
            f"{f['restarts']} restarts, {f['fixes_executed']} fixes, {f['brain_consultations']} brain consults")
        return report

    # ──────────── Main Loop ────────────

    def run(self):
        log("INFO", "=" * 72)
        dur_str = "INFINITE" if self.duration == float('inf') else f"{self.duration // 3600}h"
        log("INFO", f"  AUTONOMOUS WATCHDOG v2 — {dur_str} run")
        log("INFO", f"  Running on: {'z4090' if IS_Z4090 else 'sys1 (local)'}")
        log("INFO", f"  Nodes: {', '.join(NODES.keys())}")
        log("INFO", f"  Vulkan: {VULKAN_URL}")
        log("INFO", f"  Brain: CUDA-122B @ .11 + z4090-30B @ .10")
        log("INFO", f"  Tasks: {TASK_API}")
        log("INFO", "=" * 72)
        comms_write_local(f"Watchdog v2 starting — {dur_str} run, CUDA brain enabled, fix capabilities active")

        # Verify brain is up
        if self.check_cuda_health():
            log("OK", "CUDA brain (.11) is alive")
        else:
            log("WARN", "CUDA brain unavailable — will use z4090 local brain as fallback")

        # Initial checks
        fleet_results, server_status = self.run_fleet_check()
        self.run_vulkan_test()
        task_summary = self.run_task_check()
        self.run_git_check()

        # Initial brain consultation if there are issues
        sick_count = sum(1 for r in fleet_results.values() if r["status"] in ("SICK", "DEAD"))
        if sick_count > 0:
            self.consult_brain(fleet_results, server_status, task_summary)
            self.last_brain_consult = time.time()

        while self.running and (time.time() - self.start_time) < self.duration:
            try:
                now = time.time()

                # Fleet check every POLL_INTERVAL
                fleet_results, server_status = self.run_fleet_check()

                # Vulkan test every VULKAN_TEST_INTERVAL
                if now - self.last_vulkan_test >= VULKAN_TEST_INTERVAL:
                    self.run_vulkan_test()
                    self.last_vulkan_test = now

                # Git + tasks every GIT_CHECK_INTERVAL
                if now - self.last_git_check >= GIT_CHECK_INTERVAL:
                    self.run_git_check()
                    task_summary = self.run_task_check()
                    self.last_git_check = now
                else:
                    task_summary = self.history["task_snapshots"][-1]["summary"] if self.history["task_snapshots"] else {}

                # Brain consultation every BRAIN_CONSULT_INTERVAL (if issues persist)
                if now - self.last_brain_consult >= BRAIN_CONSULT_INTERVAL:
                    sick_count = sum(1 for name in NODES if self.consecutive_sick.get(name, 0) >= 2)
                    if sick_count > 0:
                        self.consult_brain(fleet_results, server_status, task_summary)
                    self.last_brain_consult = now

                # Report every REPORT_INTERVAL
                if now - self.last_report >= REPORT_INTERVAL:
                    self.write_report()
                    self.last_report = now

                # CUDA health check (periodic)
                if int(now) % 600 < POLL_INTERVAL:
                    self.check_cuda_health()

                log("INFO", f"[{self.elapsed_str()}] Next: {POLL_INTERVAL}s. Remaining: {self.remaining_str()}")
                time.sleep(POLL_INTERVAL)

            except KeyboardInterrupt:
                log("INFO", "Shutting down (Ctrl+C)")
                break
            except Exception as e:
                log("ERROR", f"Main loop error: {e}")
                log("ERROR", traceback.format_exc())
                time.sleep(30)

        # Final
        log("INFO", "=" * 72)
        log("INFO", "  WATCHDOG SESSION COMPLETE")
        log("INFO", "=" * 72)
        report = self.write_report()

        v = report["vulkan"]
        f = report["fleet"]
        log("OK", f"Vulkan: {v['passed']}/{v['total']} passed, avg {v['avg_tps']} TPS")
        log("OK", f"Fleet: {f['checks']} checks, {f['restarts']} restarts, {f['fixes_executed']} fixes, {f['brain_consultations']} brain consults")
        comms_write_local("Watchdog v2 session complete.")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Autonomous Watchdog v2")
    parser.add_argument("--duration", type=float, default=-1, help="Hours (-1 = infinite, default)")
    parser.add_argument("--once", action="store_true", help="Single check + exit")
    parser.add_argument("--local", action="store_true", help="Force local mode (sys1)")
    parser.add_argument("--brain-test", action="store_true", help="Test CUDA brain connection")
    args = parser.parse_args()

    global IS_LOCAL, IS_Z4090
    if args.local:
        IS_LOCAL = True
        IS_Z4090 = False

    if args.brain_test:
        log("TEST", "Testing CUDA brain (.11)...")
        resp = ask_brain("Test query: What is 2+2? Reply with just the number.", use_cuda=True, max_tokens=50)
        log("OK" if resp else "ERROR", f"Brain response: {resp}")
        log("TEST", "Testing z4090 brain (.10)...")
        resp2 = ask_brain("Test query: What is 3+3? Reply with just the number.", use_cuda=False, max_tokens=50)
        log("OK" if resp2 else "ERROR", f"Brain response: {resp2}")
        return

    if args.once:
        log("INFO", "Single check mode")
        for name, node in NODES.items():
            r = analyze_node(name, node)
            srv = check_model_server(name, node)
            icon = {"OK": "OK", "WARN": "!!", "SICK": "XX", "DEAD": "DD"}.get(r["status"], "??")
            srv_icon = "S" if srv.get("healthy") else "X"
            print(f"[{icon}/{srv_icon}] {r['node']:6} task={r['task']} T={r['turns']} tps={r['tps']} issues={r['issues']}")
        vr = test_vulkan_coherency()
        print(f"\nVulkan: {'PASS' if vr['coherent'] else 'FAIL'} — {vr['tokens']}tok {vr['tps']}TPS")
        if vr["issues"]:
            print(f"  Issues: {', '.join(vr['issues'])}")
        ts = check_task_progress()
        print(f"\nTasks: {json.dumps(ts)}")
        return

    watchdog = AutonomousWatchdog(duration_hours=args.duration)
    watchdog.run()


if __name__ == "__main__":
    main()
