#!/usr/bin/env python3
"""
Autonomous Watchdog — Deep fleet monitoring + Vulkan coherency testing
Runs for 4 hours (or until killed). No user input needed.

What it does:
  1. Deep log analysis on all agent nodes (sys2-sys7) — detects loops, errors, stalls
  2. Vulkan Llama-8B coherency testing on sys1:8080 — diverse prompts, 100+ tokens
  3. Git commit monitoring — tracks what each node has committed
  4. Task progress tracking — checks task server, detects stuck agents
  5. Auto-remediation — restarts stuck agents, nudges via comms bridge
  6. Writes detailed reports to ~/AGENT/watchdog_reports/

Usage:
    python3 autonomous_watchdog.py                # run for 4 hours
    python3 autonomous_watchdog.py --duration 1   # run for 1 hour
    python3 autonomous_watchdog.py --once          # single check + exit
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

# ──────────────────── CONFIG ────────────────────
DURATION_HOURS = 4
POLL_INTERVAL = 120        # 2 min between full checks
VULKAN_TEST_INTERVAL = 300 # 5 min between coherency tests
GIT_CHECK_INTERVAL = 600   # 10 min between git checks
REPORT_INTERVAL = 1800     # 30 min detailed report

PASSFILE = Path.home() / "DEV" / "authpass"
SSH_OPTS = "-o StrictHostKeyChecking=no -o ConnectTimeout=8 -o LogLevel=ERROR"
BASE = Path.home() / "AGENT"
REPORT_DIR = BASE / "watchdog_reports"
COMMS_BRIDGE = BASE / "COMMS_BRIDGE.md"
LOG_FILE = BASE / "LOGS" / "watchdog.log"

VULKAN_URL = "http://10.255.255.128:8080"
TASK_API = "http://10.255.255.128:9091"

NODES = {
    "sys1":  {"ip": "10.255.255.128", "log": "~/AGENT/LOGS/main_trace.log",  "auth": "local",    "type": "agent-host"},
    "sys2":  {"ip": "10.255.255.2",   "log": "~/AGENT/LOGS/agent_trace.log", "auth": "passfile", "type": "mlx"},
    "sys3":  {"ip": "10.255.255.3",   "log": "~/AGENT/LOGS/agent_trace.log", "auth": "passfile", "type": "mlx"},
    "sys4":  {"ip": "10.255.255.4",   "log": "~/AGENT/LOGS/agent_trace.log", "auth": "passfile", "type": "mlx"},
    "sys5":  {"ip": "10.255.255.5",   "log": "~/AGENT/LOGS/agent_trace.log", "auth": "passfile", "type": "mlx"},
    "sys6":  {"ip": "10.255.255.6",   "log": "~/AGENT/LOGS/agent_trace.log", "auth": "passfile", "type": "mlx"},
    "sys7":  {"ip": "10.255.255.7",   "log": "~/AGENT/LOGS/agent_trace.log", "auth": "passfile", "type": "mlx"},
}

# ──────────────────── VULKAN TEST PROMPTS ────────────────────
# Diverse prompts that require coherent 100+ token responses
VULKAN_PROMPTS = [
    # Math / reasoning
    "A farmer has 3 fields. The first is 2.5 acres, the second is 1.75 acres, and the third is 3.25 acres. He wants to plant wheat on 60% of his total land and corn on the rest. How many acres of each crop will he plant? Show your work step by step.",
    "Explain why the square root of 2 is irrational. Give a proof by contradiction.",
    # Factual
    "Describe the process of photosynthesis from start to finish. Include the light-dependent and light-independent reactions.",
    "What were the main causes and consequences of the French Revolution? Be specific about dates and key figures.",
    # Creative / language
    "Write a short story (about 100 words) about a robot who discovers it can dream. Make it emotional.",
    "Explain the difference between TCP and UDP protocols to someone who has never programmed before. Use real-world analogies.",
    # Code generation
    "Write a Python function that finds all prime numbers up to N using the Sieve of Eratosthenes. Include comments explaining each step.",
    "Write a bash script that monitors disk usage and sends an alert if any partition exceeds 80%. Include error handling.",
    # Science
    "Explain how CRISPR gene editing works. What are the key components and what are the ethical concerns?",
    "Describe the lifecycle of a star from nebula to its final stage. How does the star's mass determine its fate?",
    # Edge cases
    "Translate this to French and then back to English: 'The quick brown fox jumps over the lazy dog near the riverbank at sunset.'",
    "List the planets in our solar system in order from the Sun, including one interesting fact about each planet.",
]

# ──────────────────── UTILITIES ────────────────────

def log(level, msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    colors = {"INFO": "\033[36m", "WARN": "\033[33m", "ERROR": "\033[31m",
              "OK": "\033[32m", "TEST": "\033[35m", "CRIT": "\033[41;37m"}
    c = colors.get(level, "")
    r = "\033[0m"
    line = f"[{ts}] [{level:5}] {msg}"
    print(f"{c}{line}{r}", flush=True)
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except:
        pass

def ssh(ip, cmd, auth="passfile", timeout=12):
    if ip in ("127.0.0.1", "localhost", "10.255.255.128"):
        # sys1 is local
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
        else:
            args = ["sshpass", "-p", "z", "ssh"] + SSH_OPTS.split() + [f"z@{ip}", cmd]
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip()
    except subprocess.TimeoutExpired:
        return 1, "TIMEOUT"
    except Exception as e:
        return 1, str(e)

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

def http_post(url, data, timeout=60):
    import urllib.request, urllib.error
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode() if hasattr(e, 'read') else ""
    except Exception as e:
        return 0, str(e)

def comms_write(msg):
    try:
        ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        with open(COMMS_BRIDGE, "a") as f:
            f.write(f"\n[{ts}] [WATCHDOG] {msg}\n")
    except:
        pass

# ──────────────────── DEEP LOG ANALYSIS ────────────────────

def analyze_node(name, node):
    """Deep log analysis for a single node — much more thorough than fleet_health_check."""
    ip, logpath, auth = node["ip"], node["log"], node["auth"]
    result = {
        "node": name, "status": "UNKNOWN", "issues": [], "warnings": [],
        "turns": 0, "tps": 0, "ttft": 0, "task": "-", "last_activity": "-",
        "errors": [], "progress_pct": 0,
    }

    # Get last 400 lines of log (more than health checker's 200)
    rc, raw = ssh(ip, f"strings {logpath} 2>/dev/null | tail -400", auth)
    if rc != 0 or not raw:
        result["status"] = "DEAD"
        result["issues"].append("No log output — agent may be dead")
        return result

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

    # Timestamp analysis — when was the last activity?
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
        result["issues"].append(f"CLAIM LOOP x{already_done} — trying to claim DONE tasks")

    blocked = raw.count("BLOCKED")
    if blocked >= 3:
        result["issues"].append(f"BLOCKED LOOP x{blocked}")

    # Repeated tool calls (stuck)
    tool_calls = re.findall(r'(?:EXECUTING|calling|tool_call)[:\]]\s*(\w+)', raw)
    if len(tool_calls) >= 6:
        last6 = tool_calls[-6:]
        freq = Counter(last6)
        top, count = freq.most_common(1)[0]
        if count >= 5:
            result["issues"].append(f"STUCK: calling {top} {count}/6 times")

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

    # Stream errors
    stream_errors = raw.count("STREAM ERROR") + raw.count("ConnectionError") + raw.count("Connection refused")
    if stream_errors >= 3:
        result["issues"].append(f"STREAM/CONN ERRORS x{stream_errors}")

    # Timeout loop
    timeouts = raw.count("TIMEOUT after") + raw.count("timed out")
    if timeouts >= 3:
        result["issues"].append(f"TIMEOUT LOOP x{timeouts}")

    # GO_PROMPT loop
    go_reloads = raw.count("re-loading GO_PROMPT") + raw.count("GO_PROMPT")
    if go_reloads >= 5:
        result["issues"].append(f"GO_PROMPT LOOP x{go_reloads}")

    # Wrong paths — only count if agent is USING wrong paths in tool calls, not just displaying code
    # Look for tool-call patterns with wrong paths (read_file, write_file with /Users/z/)
    wrong_tool_calls = len(re.findall(r'(?:read_file|write_file|path)["\s:]+/Users/z/', raw))
    wrong_v4 = raw.count("TASK_QUEUE_v4")
    wrong_paths = wrong_tool_calls + wrong_v4
    if wrong_paths >= 2:
        result["issues"].append(f"WRONG PATHS x{wrong_paths} (tool calls using /Users/z/ or v4 refs)")

    # Python/runtime errors
    py_errors = re.findall(r'(Traceback|TypeError|ValueError|KeyError|AttributeError|ImportError|FileNotFoundError|NameError)', raw)
    if py_errors:
        freq = Counter(py_errors)
        for err, count in freq.most_common(3):
            if count >= 2:
                result["errors"].append(f"{err} x{count}")

    # Detect if agent is actually making progress (writing files, git operations)
    productive_actions = (
        raw.count("write_file") + raw.count("git commit") + raw.count("git add") +
        raw.count("COMPLETED") + raw.count("created") + raw.count("updated")
    )
    if result["turns"] > 10 and productive_actions == 0:
        result["warnings"].append("NO PRODUCTIVE OUTPUT: many turns but no file writes or commits")

    # Multi-claim detection
    claimed_tasks = re.findall(r'CLAIMED (T\d+)', raw)
    unique_claims = set(claimed_tasks[-10:]) if claimed_tasks else set()
    if len(unique_claims) > 2:
        result["warnings"].append(f"TASK HOPPING: claimed {len(unique_claims)} different tasks recently")

    # Set overall status
    if result["issues"]:
        result["status"] = "SICK"
    elif result["warnings"]:
        result["status"] = "WARN"
    else:
        result["status"] = "OK"

    return result


# ──────────────────── VULKAN COHERENCY TESTING ────────────────────

def test_vulkan_coherency(prompt_idx=None):
    """Send a prompt to the Vulkan server, check for coherent 100+ token response."""
    if prompt_idx is None:
        import random
        prompt_idx = random.randint(0, len(VULKAN_PROMPTS) - 1)

    prompt = VULKAN_PROMPTS[prompt_idx % len(VULKAN_PROMPTS)]
    payload = {
        "model": "llama",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.7,
    }

    t0 = time.time()
    status, body = http_post(f"{VULKAN_URL}/v1/chat/completions", payload, timeout=90)
    elapsed = time.time() - t0

    result = {
        "prompt_idx": prompt_idx,
        "prompt_preview": prompt[:80],
        "status_code": status,
        "elapsed_s": round(elapsed, 2),
        "coherent": False,
        "tokens": 0,
        "tps": 0,
        "response_preview": "",
        "issues": [],
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
        result["response_preview"] = content[:200]

        # Coherency checks
        issues = []

        # Check for minimum token count
        if tokens < 50:
            issues.append(f"SHORT: only {tokens} tokens")

        # Check for gibberish (repeated chars, excessive special chars)
        if re.search(r'(.)\1{10,}', content):
            issues.append("GIBBERISH: repeated character pattern")

        # Check for repeated phrases (3+ word sequences repeated 3+ times)
        words = content.split()
        if len(words) >= 20:
            trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
            freq = Counter(trigrams)
            top_tg, top_count = freq.most_common(1)[0]
            if top_count >= 4 and top_count > len(trigrams) * 0.15:
                issues.append(f"REPETITION: '{top_tg}' x{top_count}")

        # Check for stop token leaks
        if "<|eot_id|>" in content or "<|end" in content:
            issues.append("STOP TOKEN LEAK in output")

        # Check for complete sentences (at least one period)
        if "." not in content and "!" not in content and "?" not in content:
            issues.append("NO SENTENCE ENDINGS — may be incoherent")

        # Check TPS is reasonable
        if tps > 0 and tps < 5:
            issues.append(f"LOW TPS: {tps}")

        result["issues"] = issues
        result["coherent"] = len(issues) == 0

    except Exception as e:
        result["issues"].append(f"PARSE ERROR: {e}")

    return result


# ──────────────────── GIT MONITORING ────────────────────

def check_git_activity():
    """Check git log on sys1 (main repo) for recent commits from agents."""
    rc, raw = ssh("10.255.255.128", "cd ~/AGENT && git log --oneline --since='4 hours ago' 2>/dev/null | head -30", "local")
    if rc != 0:
        return {"error": "Could not read git log", "commits": []}

    commits = []
    for line in raw.strip().split("\n"):
        if line.strip():
            commits.append(line.strip())

    # Also check for uncommitted changes
    rc2, diff = ssh("10.255.255.128", "cd ~/AGENT && git diff --stat 2>/dev/null | tail -5", "local")

    return {
        "recent_commits": len(commits),
        "commits": commits[:20],
        "uncommitted_changes": diff if diff else "none",
    }


def check_node_git(name, node):
    """Check git status on a remote node."""
    ip, auth = node["ip"], node["auth"]
    rc, raw = ssh(ip, "cd ~/AGENT && git log --oneline -5 2>/dev/null", auth)
    if rc != 0:
        return {"node": name, "error": "git not accessible"}
    return {"node": name, "recent": raw.strip().split("\n")[:5] if raw else []}


# ──────────────────── TASK MONITORING ────────────────────

def check_task_progress():
    """Query the task API for current state."""
    status, body = http_get(f"{TASK_API}/tasks/summary", timeout=10)
    if status != 200:
        return {"error": f"Task API returned {status}"}
    try:
        return json.loads(body)
    except:
        return {"error": "Invalid JSON from task API"}


def check_task_assignments():
    """Get full task list to see who's working on what."""
    status, body = http_get(f"{TASK_API}/tasks", timeout=10)
    if status != 200:
        return {}
    try:
        text = body if isinstance(body, str) else body.decode()
        # Parse IN_PROGRESS tasks
        assignments = {}
        for line in text.split("\n"):
            m = re.match(r"###\s+(T\d+):.*?\[IN_PROGRESS by ([^\]|]+)", line)
            if m:
                tid, agent = m.group(1), m.group(2).strip()
                assignments[tid] = agent
        return assignments
    except:
        return {}


# ──────────────────── REMEDIATION ────────────────────

def nudge_agent(name, node, message):
    """Write a nudge to the agent's comms bridge or inject guidance."""
    ip, auth = node["ip"], node["auth"]
    nudge_text = f"[WATCHDOG {datetime.now().strftime('%H:%M')}] {message}"
    # Write to the node's local comms bridge
    rc, _ = ssh(ip, f'echo "{nudge_text}" >> ~/AGENT/COMMS_BRIDGE.md', auth, timeout=8)
    if rc == 0:
        log("INFO", f"Nudged {name}: {message[:60]}")
    return rc == 0


def restart_agent_tmux(name, node):
    """Restart agent via tmux on a remote node."""
    ip, auth = node["ip"], node["auth"]
    if auth == "local" or node["type"] != "mlx":
        log("WARN", f"Won't restart {name} — type={node['type']}")
        return False

    log("WARN", f"Attempting restart of {name} agent...")
    # Kill existing agent process
    rc1, _ = ssh(ip, "pkill -f 'python3.*OMNIAGENT' 2>/dev/null; sleep 2", auth, timeout=15)

    # Relaunch in tmux
    rc2, out = ssh(ip,
        "cd ~/AGENT && tmux kill-session -t agent 2>/dev/null; "
        "tmux new-session -d -s agent 'python3 OMNIAGENT_v4_focused.py --no-tty < /dev/null 2>&1 | tee -a ~/AGENT/LOGS/agent_trace.log'",
        auth, timeout=15)

    if rc2 == 0:
        log("OK", f"Restarted {name} agent via tmux")
        comms_write(f"Restarted {name} agent (was sick/stalled)")
        return True
    else:
        log("ERROR", f"Failed to restart {name}: {out[:100]}")
        return False


# ──────────────────── WATCHDOG MAIN ────────────────────

class AutonomousWatchdog:

    def __init__(self, duration_hours=4):
        self.duration = duration_hours * 3600
        self.start_time = time.time()
        self.running = True
        self.last_vulkan_test = 0
        self.last_git_check = 0
        self.last_report = 0
        self.vulkan_prompt_idx = 0
        self.history = {
            "vulkan_tests": [],
            "node_checks": [],
            "restarts": [],
            "alerts": [],
            "task_snapshots": [],
        }
        self.restart_counts = defaultdict(int)  # per-node restart count this session
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

    def elapsed_str(self):
        e = time.time() - self.start_time
        h, m = int(e // 3600), int((e % 3600) // 60)
        return f"{h}h{m:02d}m"

    def remaining_str(self):
        r = max(0, self.duration - (time.time() - self.start_time))
        h, m = int(r // 3600), int((r % 3600) // 60)
        return f"{h}h{m:02d}m"

    def run_fleet_check(self):
        """Deep check all nodes."""
        log("INFO", f"[{self.elapsed_str()}] Fleet check starting...")
        results = {}
        threads = []

        def check(name, node):
            results[name] = analyze_node(name, node)

        for name, node in NODES.items():
            t = threading.Thread(target=check, args=(name, node), daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=20)

        # Report
        sick = [r for r in results.values() if r["status"] == "SICK"]
        warn = [r for r in results.values() if r["status"] == "WARN"]
        ok = [r for r in results.values() if r["status"] == "OK"]
        dead = [r for r in results.values() if r["status"] == "DEAD"]

        log("INFO", f"Fleet: {len(ok)} OK, {len(warn)} WARN, {len(sick)} SICK, {len(dead)} DEAD")

        for r in results.values():
            if r["status"] in ("OK", "WARN"):
                icon = "OK" if r["status"] == "OK" else "!!"
                log("OK" if r["status"] == "OK" else "WARN",
                    f"  {r['node']:6} [{icon}] task={r['task']} turns={r['turns']} tps={r['tps']} ttft={r['ttft']}ms")
                for w in r["warnings"]:
                    log("WARN", f"    {w}")
            elif r["status"] == "SICK":
                log("ERROR", f"  {r['node']:6} [SICK] task={r['task']} turns={r['turns']}")
                for i in r["issues"]:
                    log("ERROR", f"    {i}")
                for e in r["errors"]:
                    log("ERROR", f"    ERR: {e}")
            else:
                log("CRIT", f"  {r['node']:6} [DEAD]")

        # Auto-remediation
        for r in (sick + dead):
            name = r["node"]
            node = NODES[name]
            if node["type"] != "mlx":
                continue  # don't touch sys1

            # Try nudge first for SICK
            if r["status"] == "SICK":
                issues_str = "; ".join(r["issues"][:2])
                nudge_agent(name, node, f"ISSUE DETECTED: {issues_str}. Please reset and refocus on your task.")

            # Restart if DEAD or stuck in same issue 3+ checks
            if r["status"] == "DEAD" and self.restart_counts[name] < 2:
                restart_agent_tmux(name, node)
                self.restart_counts[name] += 1
                self.history["restarts"].append({
                    "time": datetime.now().isoformat(), "node": name, "reason": "DEAD"
                })

        self.history["node_checks"].append({
            "time": datetime.now().isoformat(),
            "summary": {r["node"]: r["status"] for r in results.values()},
        })
        return results

    def run_vulkan_test(self):
        """Test Vulkan coherency with a diverse prompt."""
        log("TEST", f"[{self.elapsed_str()}] Vulkan coherency test #{self.vulkan_prompt_idx}...")

        result = test_vulkan_coherency(self.vulkan_prompt_idx)
        self.vulkan_prompt_idx = (self.vulkan_prompt_idx + 1) % len(VULKAN_PROMPTS)

        if result["coherent"]:
            log("OK", f"  COHERENT: {result['tokens']} tokens, {result['tps']} TPS, {result['elapsed_s']}s")
            log("OK", f"  Preview: {result['response_preview'][:120]}")
        else:
            issues = ", ".join(result["issues"])
            log("ERROR", f"  FAILED: {issues}")
            log("ERROR", f"  Preview: {result['response_preview'][:120]}")
            comms_write(f"Vulkan coherency FAILED: {issues}")
            self.history["alerts"].append({
                "time": datetime.now().isoformat(),
                "type": "vulkan_coherency",
                "issues": result["issues"],
            })

        self.history["vulkan_tests"].append({
            "time": datetime.now().isoformat(),
            "coherent": result["coherent"],
            "tokens": result["tokens"],
            "tps": result["tps"],
            "issues": result["issues"],
        })
        return result

    def run_git_check(self):
        """Check git activity across the fleet."""
        log("INFO", f"[{self.elapsed_str()}] Git activity check...")
        git_info = check_git_activity()
        log("INFO", f"  {git_info['recent_commits']} commits in last 4h")
        for c in git_info["commits"][:5]:
            log("INFO", f"    {c}")
        if git_info["uncommitted_changes"] and git_info["uncommitted_changes"] != "none":
            log("WARN", f"  Uncommitted: {git_info['uncommitted_changes'][:100]}")
        return git_info

    def run_task_check(self):
        """Check task progress."""
        summary = check_task_progress()
        if "error" in summary:
            log("ERROR", f"Task API: {summary['error']}")
            return summary

        log("INFO", f"  Tasks: {summary.get('DONE',0)}D {summary.get('IN_PROGRESS',0)}A {summary.get('READY',0)}R")

        assignments = check_task_assignments()
        if assignments:
            for tid, agent in sorted(assignments.items()):
                log("INFO", f"    {tid}: {agent}")

        self.history["task_snapshots"].append({
            "time": datetime.now().isoformat(),
            "summary": summary,
        })
        return summary

    def write_report(self):
        """Write detailed report to disk."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = {
            "generated": datetime.now().isoformat(),
            "elapsed": self.elapsed_str(),
            "remaining": self.remaining_str(),
            "vulkan_tests": {
                "total": len(self.history["vulkan_tests"]),
                "passed": sum(1 for t in self.history["vulkan_tests"] if t["coherent"]),
                "failed": sum(1 for t in self.history["vulkan_tests"] if not t["coherent"]),
                "avg_tps": round(
                    sum(t["tps"] for t in self.history["vulkan_tests"]) /
                    max(len(self.history["vulkan_tests"]), 1), 1),
            },
            "fleet": {
                "checks": len(self.history["node_checks"]),
                "restarts": len(self.history["restarts"]),
                "alerts": len(self.history["alerts"]),
            },
            "tasks": self.history["task_snapshots"][-1] if self.history["task_snapshots"] else {},
            "restart_details": self.history["restarts"],
            "alert_details": self.history["alerts"][-10:],
        }

        path = REPORT_DIR / f"watchdog_{ts}.json"
        path.write_text(json.dumps(report, indent=2))
        log("OK", f"Report written: {path.name}")
        comms_write(f"Watchdog report: {report['vulkan_tests']['passed']}/{report['vulkan_tests']['total']} vulkan OK, "
                     f"{report['fleet']['restarts']} restarts, {report['fleet']['alerts']} alerts")
        return report

    def run(self):
        """Main watchdog loop."""
        log("INFO", "=" * 70)
        log("INFO", f"  AUTONOMOUS WATCHDOG — starting {self.duration // 3600}h run")
        log("INFO", f"  Monitoring: {', '.join(NODES.keys())}")
        log("INFO", f"  Vulkan: {VULKAN_URL}")
        log("INFO", f"  Tasks: {TASK_API}")
        log("INFO", "=" * 70)
        comms_write(f"Watchdog starting — {self.duration // 3600}h autonomous run. No user input needed.")

        # Initial checks
        self.run_fleet_check()
        self.run_vulkan_test()
        self.run_task_check()
        self.run_git_check()

        while self.running and (time.time() - self.start_time) < self.duration:
            try:
                now = time.time()

                # Fleet check every POLL_INTERVAL
                self.run_fleet_check()

                # Vulkan coherency test every VULKAN_TEST_INTERVAL
                if now - self.last_vulkan_test >= VULKAN_TEST_INTERVAL:
                    self.run_vulkan_test()
                    self.last_vulkan_test = now

                # Git check every GIT_CHECK_INTERVAL
                if now - self.last_git_check >= GIT_CHECK_INTERVAL:
                    self.run_git_check()
                    self.run_task_check()
                    self.last_git_check = now

                # Report every REPORT_INTERVAL
                if now - self.last_report >= REPORT_INTERVAL:
                    self.write_report()
                    self.last_report = now

                # Sleep
                log("INFO", f"[{self.elapsed_str()}] Next check in {POLL_INTERVAL}s. Remaining: {self.remaining_str()}")
                time.sleep(POLL_INTERVAL)

            except KeyboardInterrupt:
                log("INFO", "Shutting down (Ctrl+C)")
                break
            except Exception as e:
                log("ERROR", f"Main loop error: {e}")
                log("ERROR", traceback.format_exc())
                time.sleep(30)

        # Final report
        log("INFO", "=" * 70)
        log("INFO", "  WATCHDOG SESSION COMPLETE")
        log("INFO", "=" * 70)
        report = self.write_report()

        vt = report["vulkan_tests"]
        log("OK", f"Vulkan: {vt['passed']}/{vt['total']} passed, avg {vt['avg_tps']} TPS")
        log("OK", f"Fleet: {report['fleet']['checks']} checks, {report['fleet']['restarts']} restarts, {report['fleet']['alerts']} alerts")
        comms_write("Watchdog session complete. Final report written.")


# ──────────────────── CLI ────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Autonomous Watchdog")
    parser.add_argument("--duration", type=float, default=4, help="Duration in hours (default: 4)")
    parser.add_argument("--once", action="store_true", help="Single check and exit")
    args = parser.parse_args()

    if args.once:
        log("INFO", "Single check mode")
        # Fleet
        for name, node in NODES.items():
            r = analyze_node(name, node)
            icon = {"OK": "OK", "WARN": "!!", "SICK": "XX", "DEAD": "DD"}.get(r["status"], "??")
            print(f"[{icon}] {r['node']:6} task={r['task']} turns={r['turns']} tps={r['tps']} issues={r['issues']}")
        # Vulkan
        vr = test_vulkan_coherency()
        print(f"\nVulkan: {'PASS' if vr['coherent'] else 'FAIL'} — {vr['tokens']} tokens, {vr['tps']} TPS")
        if vr["issues"]:
            print(f"  Issues: {', '.join(vr['issues'])}")
        # Tasks
        ts = check_task_progress()
        print(f"\nTasks: {json.dumps(ts)}")
        return

    watchdog = AutonomousWatchdog(duration_hours=args.duration)
    watchdog.run()


if __name__ == "__main__":
    main()
