#!/usr/bin/env python3
"""
Cluster Orchestrator — Autonomous monitoring and coordination agent
Runs on z4090 (10.255.255.10, RTX 4090). Monitors 12-node AI cluster.
DRY_RUN by default. Drop ORCHESTRATOR_KILL to enter read-only mode.

Usage:
    python3 cluster_orchestrator.py             # start monitoring loop
    python3 cluster_orchestrator.py --command status   # one-shot status
    python3 cluster_orchestrator.py --live       # promote to live mode
    python3 cluster_orchestrator.py --dry-run    # back to dry-run
"""

import argparse
import datetime
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────── HARD-CODED GUARDRAILS ───────────────────────
MAX_RESTARTS_PER_NODE_PER_HOUR = 3
MAX_TOTAL_RESTARTS_PER_HOUR = 10
COOLDOWN_AFTER_RESTART = 300          # seconds
MIN_AGENT_UPTIME_BEFORE_RESTART = 120 # seconds
CIRCUIT_BREAKER_DEATHS = 3
CIRCUIT_BREAKER_WINDOW = 300          # 5 minutes
SYNC_COOLDOWN = 300                   # 5 minutes
POLL_INTERVAL = 60                    # seconds
REPORT_INTERVAL = 3600                # 1 hour
RECONCILE_INTERVAL = 1800             # 30 minutes
HEALTH_TIMEOUT = 8                    # seconds per HTTP check
SSH_TIMEOUT = 15                      # seconds per SSH command

# ─────────────────────── NODE DEFINITIONS ────────────────────────────
NODES: Dict[str, Dict[str, Any]] = {
    "sys1":  {"ip": "10.255.255.128", "port": 8081, "type": "untouchable", "ssh": None},
    "sys2":  {"ip": "10.255.255.2",   "port": 8000, "type": "mlx",  "ssh": "passfile"},
    "sys3":  {"ip": "10.255.255.3",   "port": 8000, "type": "mlx",  "ssh": "passfile"},
    "sys4":  {"ip": "10.255.255.4",   "port": 8000, "type": "mlx",  "ssh": "passfile"},
    "sys5":  {"ip": "10.255.255.5",   "port": 8000, "type": "mlx",  "ssh": "passfile"},
    "sys6":  {"ip": "10.255.255.6",   "port": 8000, "type": "mlx",  "ssh": "passfile"},
    "sys7":  {"ip": "10.255.255.7",   "port": 8000, "type": "mlx",  "ssh": "passfile"},
    "z4090": {"ip": "10.255.255.10",  "port": 8000, "type": "local", "ssh": None},
    "cuda1": {"ip": "10.255.255.11",  "port": 8000, "type": "cuda", "ssh": "pass_z"},
    "cuda2": {"ip": "10.255.255.12",  "port": 8000, "type": "cuda", "ssh": "pass_z"},
    "cuda3": {"ip": "10.255.255.13",  "port": 8000, "type": "cuda", "ssh": "pass_z"},
    "cuda4": {"ip": "10.255.255.14",  "port": 8000, "type": "cuda", "ssh": "pass_z"},
}

MLX_NODES = [n for n, d in NODES.items() if d["type"] == "mlx"]
CUDA_NODES = [n for n, d in NODES.items() if d["type"] == "cuda"]
RESTARTABLE_NODES = MLX_NODES  # only sys2-sys7

# ─────────────────────── PATHS ───────────────────────────────────────
BASE = pathlib.Path.home() / "AGENT"
STATE_DIR = BASE / "orchestrator_state"
REPORT_DIR = BASE / "orchestrator_reports"
CMD_DIR = BASE / "orchestrator_commands"
STATE_FILE = STATE_DIR / "state.json"
AUDIT_LOG = STATE_DIR / "audit_log.jsonl"
ALERTS_LOG = REPORT_DIR / "alerts.jsonl"
COMMS_BRIDGE = BASE / "COMMS_BRIDGE.md"
KILL_SWITCH = BASE / "ORCHESTRATOR_KILL"
TASK_QUEUE = BASE / "TASK_QUEUE_v5.md"
PASSFILE = pathlib.Path.home() / "DEV" / "authpass"

# ─────────────────────── LLM SYSTEM PROMPT ───────────────────────────
LLM_SYSTEM_PROMPT = """You are the reasoning core of an autonomous cluster orchestrator managing a 12-node AI fleet.
Your job: analyze fleet telemetry, diagnose problems, and recommend actions.

Fleet layout:
- sys1 (M1 Ultra 128GB, 10.255.255.128): UNTOUCHABLE primary. Never restart or modify.
- sys2-sys7 (M1/M2 Mac Studios): MLX inference nodes running distributed agents.
- z4090 (RTX 4090, 10.255.255.10): This machine. Orchestrator host.
- cuda1-cuda4 (10.255.255.11-14): 3090 CUDA cluster. READ-ONLY monitoring.

Rules:
- Never recommend modifying CUDA nodes or sys1.
- Restarts only for sys2-sys7, and only if the server process is confirmed dead.
- Rate limits: max 3 restarts/node/hour, max 10 total/hour.
- If 3+ agents die in 5 minutes, recommend READ-ONLY mode (circuit breaker).
- Be concise. Output JSON when asked for structured data.
"""

# ─────────────────────── UTILITIES ───────────────────────────────────

try:
    import requests
except ImportError:
    # Minimal fallback using urllib — requests is strongly preferred
    import urllib.request
    import urllib.error

    class _FakeResponse:
        def __init__(self, code, data):
            self.status_code = code
            self._data = data
        def json(self):
            return json.loads(self._data)
        @property
        def text(self):
            return self._data

    class _Requests:
        @staticmethod
        def get(url, timeout=10):
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = resp.read().decode()
                    return _FakeResponse(resp.status, data)
            except urllib.error.HTTPError as e:
                return _FakeResponse(e.code, "")
            except Exception:
                raise ConnectionError(f"GET {url} failed")

        @staticmethod
        def post(url, json=None, timeout=10):
            import json as _json
            body = _json.dumps(json).encode() if json else b""
            req = urllib.request.Request(url, data=body, method="POST",
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = resp.read().decode()
                    return _FakeResponse(resp.status, data)
            except urllib.error.HTTPError as e:
                return _FakeResponse(e.code, "")
            except Exception:
                raise ConnectionError(f"POST {url} failed")

    requests = _Requests()


def now_ts() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def now_epoch() -> float:
    return time.time()


def log(level: str, msg: str):
    ts = now_ts()
    color = {"INFO": "\033[36m", "WARN": "\033[33m", "ERROR": "\033[31m",
             "CRIT": "\033[41;37m", "OK": "\033[32m", "DRY": "\033[35m"}.get(level, "")
    reset = "\033[0m"
    print(f"{color}[{ts}] [{level:5}]{reset} {msg}", flush=True)


def ssh_cmd(node: str, cmd: str, timeout: int = SSH_TIMEOUT) -> Tuple[int, str]:
    """Execute SSH command on a node. Returns (returncode, stdout+stderr)."""
    info = NODES[node]
    ip = info["ip"]
    if info["ssh"] == "passfile":
        prefix = ["sshpass", "-f", str(PASSFILE), "ssh",
                   "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=8",
                   f"z@{ip}"]
    elif info["ssh"] == "pass_z":
        prefix = ["sshpass", "-p", "z", "ssh",
                   "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=8",
                   f"z@{ip}"]
    else:
        return 1, "No SSH config for this node"
    try:
        result = subprocess.run(prefix + [cmd], capture_output=True, text=True, timeout=timeout)
        return result.returncode, (result.stdout + result.stderr).strip()
    except subprocess.TimeoutExpired:
        return 1, "SSH timeout"
    except Exception as e:
        return 1, str(e)


# ═══════════════════════ ORCHESTRATOR CLASS ══════════════════════════

class ClusterOrchestrator:

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.read_only = False
        self.running = True
        self.state: Dict[str, Any] = {
            "restart_history": defaultdict(list),
            "consecutive_down": defaultdict(int),
            "death_log": [],       # timestamps of agent deaths for circuit breaker
            "last_report_time": 0,
            "last_reconcile_time": 0,
            "last_sync_time": 0,
            "dry_run": dry_run,
            "memory_high_counts": defaultdict(int),
        }
        self._fleet_cache: Dict[str, Any] = {}
        self._ensure_dirs()
        self._load_state()

    # ────────────── Persistence ──────────────

    def _ensure_dirs(self):
        for d in [STATE_DIR, REPORT_DIR, CMD_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    def _load_state(self):
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                self.state["restart_history"] = defaultdict(list, {
                    k: v for k, v in data.get("restart_history", {}).items()})
                self.state["consecutive_down"] = defaultdict(int,
                    data.get("consecutive_down", {}))
                self.state["death_log"] = data.get("death_log", [])
                self.state["last_report_time"] = data.get("last_report_time", 0)
                self.state["last_reconcile_time"] = data.get("last_reconcile_time", 0)
                self.state["last_sync_time"] = data.get("last_sync_time", 0)
                self.state["memory_high_counts"] = defaultdict(int,
                    data.get("memory_high_counts", {}))
                self.dry_run = data.get("dry_run", True)
                log("INFO", f"State loaded. dry_run={self.dry_run}")
            except Exception as e:
                log("WARN", f"Failed to load state: {e}")

    def _save_state(self):
        self.state["dry_run"] = self.dry_run
        serializable = {
            "restart_history": dict(self.state["restart_history"]),
            "consecutive_down": dict(self.state["consecutive_down"]),
            "death_log": self.state["death_log"],
            "last_report_time": self.state["last_report_time"],
            "last_reconcile_time": self.state["last_reconcile_time"],
            "last_sync_time": self.state["last_sync_time"],
            "dry_run": self.dry_run,
            "memory_high_counts": dict(self.state["memory_high_counts"]),
        }
        STATE_FILE.write_text(json.dumps(serializable, indent=2))

    def audit(self, action: str, details: Dict[str, Any]):
        entry = {"ts": now_ts(), "epoch": now_epoch(), "action": action,
                 "dry_run": self.dry_run, **details}
        with open(AUDIT_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def alert(self, severity: str, message: str, node: str = ""):
        entry = {"ts": now_ts(), "severity": severity, "node": node, "message": message}
        log("CRIT" if severity == "CRITICAL" else "WARN", f"ALERT [{severity}] {node}: {message}")
        with open(ALERTS_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self.audit("alert", entry)
        # Also write critical alerts to comms bridge
        if severity == "CRITICAL":
            self._comms_write(f"ORCHESTRATOR ALERT [{severity}]: {node} — {message}")

    def _comms_write(self, msg: str):
        try:
            with open(COMMS_BRIDGE, "a") as f:
                f.write(f"\n[{now_ts()}] [ORCHESTRATOR] {msg}\n")
        except Exception:
            pass

    # ────────────── TOOL 1: cluster_status ──────────────

    def cluster_status(self) -> Dict[str, Dict[str, Any]]:
        """HTTP health check all 12 nodes. Returns dict of node -> status."""
        results = {}

        def check_node(name, info):
            ip, port = info["ip"], info["port"]
            url = f"http://{ip}:{port}"
            entry = {"ip": ip, "port": port, "type": info["type"],
                     "reachable": False, "latency_ms": None, "error": None}
            t0 = time.time()
            try:
                # Try /health first, fall back to /v1/models
                try:
                    resp = requests.get(f"{url}/health", timeout=HEALTH_TIMEOUT)
                    entry["reachable"] = resp.status_code < 500
                    entry["endpoint"] = "/health"
                    entry["status_code"] = resp.status_code
                except Exception:
                    resp = requests.get(f"{url}/v1/models", timeout=HEALTH_TIMEOUT)
                    entry["reachable"] = resp.status_code < 500
                    entry["endpoint"] = "/v1/models"
                    entry["status_code"] = resp.status_code
                entry["latency_ms"] = round((time.time() - t0) * 1000)
            except Exception as e:
                entry["error"] = str(e)[:120]
                entry["latency_ms"] = round((time.time() - t0) * 1000)
            results[name] = entry

        threads = []
        for name, info in NODES.items():
            t = threading.Thread(target=check_node, args=(name, info), daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=HEALTH_TIMEOUT + 2)

        self._fleet_cache = results
        return results

    # ────────────── TOOL 2: agent_activity ──────────────

    def agent_activity(self) -> Dict[str, Dict[str, Any]]:
        """SSH to sys2-sys7, check agent process status and recent log lines."""
        results = {}

        def check(node):
            # Check for claude-code agent or similar process
            rc, ps_out = ssh_cmd(node, "ps aux | grep -E 'claude|agent|python.*serve' | grep -v grep | head -5")
            rc2, uptime = ssh_cmd(node, "uptime")
            rc3, disk = ssh_cmd(node, "df -h / | tail -1 | awk '{print $5}'")
            rc4, mem = ssh_cmd(node, "vm_stat 2>/dev/null | head -5 || free -m 2>/dev/null | head -3")
            # Try to tail agent log
            rc5, log_tail = ssh_cmd(node,
                "tail -5 ~/AGENT/agent.log 2>/dev/null || tail -5 ~/AGENT/logs/agent.log 2>/dev/null || echo 'no log found'")
            results[node] = {
                "agent_running": rc == 0 and len(ps_out.strip()) > 0,
                "processes": ps_out[:500] if ps_out else "",
                "uptime": uptime[:120] if uptime else "",
                "disk_usage": disk.strip() if disk else "unknown",
                "memory_info": mem[:300] if mem else "",
                "recent_log": log_tail[:500] if log_tail else "",
            }

        threads = []
        for node in MLX_NODES:
            t = threading.Thread(target=check, args=(node,), daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=SSH_TIMEOUT + 2)
        return results

    # ────────────── TOOL 3: restart_agent ──────────────

    def restart_agent(self, node: str) -> Dict[str, Any]:
        """Restart agent on an MLX node. Enforces all guardrails."""
        result = {"node": node, "action": "restart", "executed": False, "reason": ""}

        # --- Guardrail checks ---
        if node not in RESTARTABLE_NODES:
            result["reason"] = f"{node} is not restartable (only sys2-sys7)"
            log("WARN", f"Restart blocked: {result['reason']}")
            return result

        if self.read_only:
            result["reason"] = "Orchestrator in READ-ONLY mode"
            log("WARN", f"Restart blocked: {result['reason']}")
            return result

        now = now_epoch()

        # Rate limit per node
        history = self.state["restart_history"][node]
        recent = [t for t in history if now - t < 3600]
        self.state["restart_history"][node] = recent
        if len(recent) >= MAX_RESTARTS_PER_NODE_PER_HOUR:
            result["reason"] = f"Rate limit: {node} already restarted {len(recent)}x this hour"
            log("WARN", f"Restart blocked: {result['reason']}")
            self.alert("WARNING", result["reason"], node)
            return result

        # Rate limit total
        total_recent = sum(len([t for t in ts if now - t < 3600])
                           for ts in self.state["restart_history"].values())
        if total_recent >= MAX_TOTAL_RESTARTS_PER_HOUR:
            result["reason"] = f"Total rate limit: {total_recent} restarts this hour"
            log("WARN", f"Restart blocked: {result['reason']}")
            return result

        # Cooldown check — was this node restarted recently?
        if recent and (now - recent[-1]) < COOLDOWN_AFTER_RESTART:
            remaining = int(COOLDOWN_AFTER_RESTART - (now - recent[-1]))
            result["reason"] = f"Cooldown: {remaining}s remaining for {node}"
            log("WARN", f"Restart blocked: {result['reason']}")
            return result

        # Circuit breaker
        deaths = [t for t in self.state["death_log"] if now - t < CIRCUIT_BREAKER_WINDOW]
        self.state["death_log"] = deaths
        if len(deaths) >= CIRCUIT_BREAKER_DEATHS:
            self.read_only = True
            result["reason"] = f"CIRCUIT BREAKER: {len(deaths)} deaths in {CIRCUIT_BREAKER_WINDOW}s. Entering READ-ONLY."
            log("CRIT", result["reason"])
            self.alert("CRITICAL", result["reason"])
            return result

        # --- Execute (or dry-run) ---
        if self.dry_run:
            result["reason"] = "DRY_RUN: would restart agent"
            result["dry_run"] = True
            log("DRY", f"Would restart agent on {node}")
            self.audit("restart_dry", {"node": node})
            return result

        log("WARN", f"RESTARTING agent on {node}...")
        self.audit("restart_begin", {"node": node})

        # Kill existing agent
        rc, out = ssh_cmd(node, "pkill -f 'claude.*agent' 2>/dev/null; sleep 2; "
                                "pkill -9 -f 'claude.*agent' 2>/dev/null; echo 'killed'")
        log("INFO", f"Kill on {node}: rc={rc} out={out[:100]}")

        # Relaunch — run the agent start script in background
        rc2, out2 = ssh_cmd(node,
            "cd ~/AGENT && nohup python3 start_agent.py > agent.log 2>&1 &"
            " disown; sleep 2; pgrep -f 'agent' | head -1",
            timeout=20)

        if rc2 == 0 and out2.strip():
            result["executed"] = True
            result["pid"] = out2.strip()
            result["reason"] = "Restart successful"
            log("OK", f"Agent restarted on {node}, PID={out2.strip()}")
        else:
            result["reason"] = f"Restart may have failed: {out2[:200]}"
            log("ERROR", f"Restart issue on {node}: {out2[:200]}")

        self.state["restart_history"][node].append(now)
        self.state["death_log"].append(now)
        self.state["consecutive_down"][node] = 0
        self.audit("restart_complete", {"node": node, "success": result["executed"]})
        self._save_state()
        return result

    # ────────────── TOOL 4: sync_files ──────────────

    def sync_files(self, nodes: Optional[List[str]] = None, files: Optional[List[str]] = None) -> Dict:
        """SCP updated files to nodes. 5min cooldown enforced."""
        nodes = nodes or MLX_NODES
        files = files or []
        now = now_epoch()
        result = {"synced": [], "skipped": [], "errors": []}

        if now - self.state["last_sync_time"] < SYNC_COOLDOWN:
            remaining = int(SYNC_COOLDOWN - (now - self.state["last_sync_time"]))
            result["skipped"].append(f"Cooldown: {remaining}s remaining")
            return result

        if self.dry_run:
            log("DRY", f"Would sync {len(files)} files to {len(nodes)} nodes")
            self.audit("sync_dry", {"files": files, "nodes": nodes})
            return {"dry_run": True, "files": files, "nodes": nodes}

        for node in nodes:
            info = NODES.get(node)
            if not info or info["type"] not in ("mlx",):
                result["skipped"].append(f"{node}: not an MLX node")
                continue
            for fpath in files:
                src = str(BASE / fpath)
                if not os.path.exists(src):
                    result["errors"].append(f"{fpath}: not found")
                    continue
                if info["ssh"] == "passfile":
                    cmd = ["sshpass", "-f", str(PASSFILE), "scp", "-o", "StrictHostKeyChecking=no",
                           src, f"z@{info['ip']}:~/AGENT/{fpath}"]
                else:
                    continue
                try:
                    subprocess.run(cmd, capture_output=True, timeout=30, check=True)
                    result["synced"].append(f"{fpath} -> {node}")
                except Exception as e:
                    result["errors"].append(f"{fpath} -> {node}: {e}")

        self.state["last_sync_time"] = now
        self.audit("sync", result)
        self._save_state()
        return result

    # ────────────── TOOL 5: task_summary ──────────────

    def task_summary(self) -> Dict[str, Any]:
        """Parse TASK_QUEUE_v5.md and summarize progress."""
        summary = {"done": 0, "in_progress": 0, "blocked": 0,
                   "todo": 0, "total": 0, "tasks": []}
        if not TASK_QUEUE.exists():
            return {"error": "TASK_QUEUE_v5.md not found"}
        try:
            text = TASK_QUEUE.read_text()
            for line in text.split("\n"):
                m = re.match(r"###\s+(T\d+):\s*\[([^\]]+)\]", line)
                if m:
                    tid, status_raw = m.group(1), m.group(2).strip().upper()
                    summary["total"] += 1
                    task_entry = {"id": tid, "status": status_raw, "line": line.strip()[:120]}
                    if "DONE" in status_raw:
                        summary["done"] += 1
                    elif "IN_PROGRESS" in status_raw:
                        summary["in_progress"] += 1
                    elif "BLOCK" in status_raw:
                        summary["blocked"] += 1
                    else:
                        summary["todo"] += 1
                    summary["tasks"].append(task_entry)
            summary["completion_pct"] = round(summary["done"] / max(summary["total"], 1) * 100, 1)
        except Exception as e:
            summary["error"] = str(e)
        return summary

    # ────────────── TOOL 6: council_query ──────────────

    def council_query(self, question: str) -> str:
        """Ask the local LLM (Ollama) for reasoning about a fleet issue."""
        url = "http://localhost:11434/v1/chat/completions"
        payload = {
            "model": "llama3",  # whatever is loaded locally
            "messages": [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "max_tokens": 512,
            "temperature": 0.3,
        }
        try:
            resp = requests.post(url, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                answer = data["choices"][0]["message"]["content"]
                self.audit("council_query", {"question": question[:200], "answer": answer[:300]})
                return answer
            return f"LLM returned status {resp.status_code}"
        except Exception as e:
            return f"LLM unreachable: {e}"

    # ────────────── TOOL 7: tail_log ──────────────

    def tail_log(self, node: str, lines: int = 30) -> str:
        """Read recent agent log lines from a node."""
        if node == "z4090":
            try:
                log_path = BASE / "agent.log"
                if log_path.exists():
                    all_lines = log_path.read_text().strip().split("\n")
                    return "\n".join(all_lines[-lines:])
                return "No local agent.log found"
            except Exception as e:
                return f"Error reading local log: {e}"
        if NODES.get(node, {}).get("ssh"):
            rc, out = ssh_cmd(node,
                f"tail -{lines} ~/AGENT/agent.log 2>/dev/null || "
                f"tail -{lines} ~/AGENT/logs/agent.log 2>/dev/null || echo 'no log'")
            return out
        return f"Cannot read logs from {node}"

    # ────────────── TOOL 8: fleet_metrics ──────────────

    def fleet_metrics(self) -> Dict[str, Any]:
        """Aggregate metrics: nodes up, agents running, disk, etc."""
        status = self._fleet_cache or self.cluster_status()
        activity = self.agent_activity()

        nodes_up = sum(1 for s in status.values() if s.get("reachable"))
        agents_running = sum(1 for a in activity.values() if a.get("agent_running"))

        disk_warnings = []
        for node, info in activity.items():
            usage_str = info.get("disk_usage", "").replace("%", "").strip()
            try:
                usage = int(usage_str)
                if usage > 80:
                    disk_warnings.append(f"{node}: {usage}%")
            except (ValueError, TypeError):
                pass

        return {
            "timestamp": now_ts(),
            "nodes_total": len(NODES),
            "nodes_up": nodes_up,
            "nodes_down": len(NODES) - nodes_up,
            "agents_running": agents_running,
            "agents_expected": len(MLX_NODES),
            "disk_warnings": disk_warnings,
            "avg_latency_ms": round(
                sum(s.get("latency_ms", 0) or 0 for s in status.values()) / max(len(status), 1)),
            "node_details": {n: {"reachable": s.get("reachable", False),
                                  "latency_ms": s.get("latency_ms")}
                             for n, s in status.items()},
        }

    # ────────────── AUTONOMOUS BEHAVIORS ──────────────

    def check_kill_switch(self):
        if KILL_SWITCH.exists():
            if not self.read_only:
                log("CRIT", "KILL SWITCH detected — entering READ-ONLY mode")
                self.audit("kill_switch", {"action": "enter_read_only"})
                self.read_only = True
        else:
            if self.read_only and not any(
                now_epoch() - t < CIRCUIT_BREAKER_WINDOW
                for t in self.state["death_log"][-CIRCUIT_BREAKER_DEATHS:]
            ):
                # Only exit read-only if circuit breaker is not tripped
                self.read_only = False

    def check_command_inbox(self):
        """Process command files from orchestrator_commands/."""
        if not CMD_DIR.exists():
            return
        for f in sorted(CMD_DIR.glob("cmd_*.json")):
            try:
                cmd = json.loads(f.read_text())
                action = cmd.get("action", "")
                log("INFO", f"Processing command: {action} from {f.name}")
                self.audit("command_received", {"file": f.name, "action": action})

                if action == "status":
                    self._write_report(self.fleet_metrics(), "cmd_status")
                elif action == "restart" and "node" in cmd:
                    self.restart_agent(cmd["node"])
                elif action == "sync" and "files" in cmd:
                    self.sync_files(cmd.get("nodes"), cmd["files"])
                elif action == "set_dry_run":
                    self.dry_run = cmd.get("value", True)
                    log("WARN", f"Dry run set to {self.dry_run}")
                elif action == "council":
                    answer = self.council_query(cmd.get("question", "What is fleet status?"))
                    self._write_report({"answer": answer}, "council_response")
                elif action == "task_summary":
                    self._write_report(self.task_summary(), "task_summary")
                else:
                    log("WARN", f"Unknown command: {action}")

                # Archive the command
                archive = CMD_DIR / "processed"
                archive.mkdir(exist_ok=True)
                f.rename(archive / f.name)
            except Exception as e:
                log("ERROR", f"Failed to process {f.name}: {e}")
                f.rename(CMD_DIR / f"error_{f.name}")

    def poll_fleet(self):
        """Core polling loop — check health of all nodes."""
        log("INFO", "Polling fleet health...")
        status = self.cluster_status()
        now = now_epoch()
        dead_agents = []

        for node in MLX_NODES:
            s = status.get(node, {})
            if not s.get("reachable"):
                self.state["consecutive_down"][node] = \
                    self.state["consecutive_down"].get(node, 0) + 1
                count = self.state["consecutive_down"][node]
                log("WARN", f"{node} DOWN (consecutive: {count})")

                if count >= 2:
                    dead_agents.append(node)
                    self.alert("WARNING", f"Node unreachable for {count} consecutive polls", node)
            else:
                if self.state["consecutive_down"].get(node, 0) > 0:
                    log("OK", f"{node} recovered after {self.state['consecutive_down'][node]} down polls")
                self.state["consecutive_down"][node] = 0

        # Check CUDA brain availability
        for node in CUDA_NODES:
            s = status.get(node, {})
            if not s.get("reachable"):
                self.alert("CRITICAL", "CUDA brain unreachable", node)

        # Auto-restart dead MLX agents
        for node in dead_agents:
            # Verify the server is actually down but SSH is up (node alive, agent dead)
            rc, _ = ssh_cmd(node, "echo ok")
            if rc == 0:
                log("INFO", f"{node}: SSH ok but HTTP down — agent likely dead")
                result = self.restart_agent(node)
                log("INFO", f"Restart result for {node}: {result.get('reason', '')}")
            else:
                log("WARN", f"{node}: SSH also failed — node may be fully down")
                self.alert("CRITICAL", "Node unreachable via SSH and HTTP", node)

        self._save_state()

    def check_disk_and_memory(self):
        """Check disk and memory on MLX nodes, raise alerts."""
        activity = self.agent_activity()
        for node, info in activity.items():
            # Disk check
            usage_str = info.get("disk_usage", "").replace("%", "").strip()
            try:
                usage = int(usage_str)
                if usage > 80:
                    free_pct = 100 - usage
                    if free_pct < 20:
                        self.alert("WARNING", f"Disk space low: {free_pct}% free", node)
            except (ValueError, TypeError):
                pass

            # Memory check (simplified — look for high pressure)
            mem_info = info.get("memory_info", "")
            if "Pages free" in mem_info:
                # macOS vm_stat output — crude check
                try:
                    free_match = re.search(r"Pages free:\s+(\d+)", mem_info)
                    if free_match and int(free_match.group(1)) < 50000:
                        self.state["memory_high_counts"][node] += 1
                    else:
                        self.state["memory_high_counts"][node] = 0
                except Exception:
                    pass
            if self.state["memory_high_counts"].get(node, 0) >= 3:
                self.alert("WARNING", "Memory pressure high for 3+ polls", node)

    def run_autonomous_behaviors(self):
        """Time-gated autonomous tasks."""
        now = now_epoch()

        # Disk/memory checks (every other poll)
        if int(now) % 120 < POLL_INTERVAL:
            try:
                self.check_disk_and_memory()
            except Exception as e:
                log("ERROR", f"Disk/memory check failed: {e}")

        # Reconcile task queue every 30 min
        if now - self.state["last_reconcile_time"] > RECONCILE_INTERVAL:
            try:
                summary = self.task_summary()
                log("INFO", f"Task reconcile: {summary.get('done', 0)}/{summary.get('total', 0)} done "
                    f"({summary.get('completion_pct', 0)}%), {summary.get('in_progress', 0)} in progress, "
                    f"{summary.get('blocked', 0)} blocked")
                self.state["last_reconcile_time"] = now
            except Exception as e:
                log("ERROR", f"Task reconcile failed: {e}")

        # Hourly report
        if now - self.state["last_report_time"] > REPORT_INTERVAL:
            try:
                metrics = self.fleet_metrics()
                summary = self.task_summary()
                report = {"metrics": metrics, "tasks": summary, "state": {
                    "dry_run": self.dry_run, "read_only": self.read_only,
                    "total_restarts_1h": sum(
                        len([t for t in ts if now - t < 3600])
                        for ts in self.state["restart_history"].values()),
                }}
                self._write_report(report, "hourly")
                self._comms_write(
                    f"Hourly report: {metrics['nodes_up']}/{metrics['nodes_total']} nodes up, "
                    f"{metrics['agents_running']}/{metrics['agents_expected']} agents running, "
                    f"tasks {summary.get('done', '?')}/{summary.get('total', '?')} done")
                self.state["last_report_time"] = now
                log("OK", "Hourly report generated")
            except Exception as e:
                log("ERROR", f"Report generation failed: {e}")

        self._save_state()

    def _write_report(self, data: Dict, label: str):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPORT_DIR / f"{label}_{ts}.json"
        path.write_text(json.dumps(data, indent=2, default=str))
        log("INFO", f"Report written: {path.name}")

    # ────────────── ONE-SHOT COMMANDS ──────────────

    def cmd_status(self):
        """One-shot: print cluster status and exit."""
        print("=" * 72)
        print(f"  CLUSTER ORCHESTRATOR STATUS — {now_ts()}")
        print(f"  Mode: {'DRY RUN' if self.dry_run else 'LIVE'}  |  "
              f"Read-only: {self.read_only}  |  Kill switch: {KILL_SWITCH.exists()}")
        print("=" * 72)

        status = self.cluster_status()
        print(f"\n{'Node':<8} {'IP':<18} {'Type':<12} {'Status':<10} {'Latency':>8}")
        print("-" * 60)
        for name in sorted(status.keys()):
            s = status[name]
            st = "\033[32mUP\033[0m" if s["reachable"] else "\033[31mDOWN\033[0m"
            lat = f"{s['latency_ms']}ms" if s.get("latency_ms") else "-"
            print(f"{name:<8} {s.get('ip',''):<18} {NODES[name]['type']:<12} {st:<19} {lat:>8}")

        print(f"\n{'='*72}")
        metrics = self.fleet_metrics()
        print(f"  Nodes up: {metrics['nodes_up']}/{metrics['nodes_total']}  |  "
              f"Agents: {metrics['agents_running']}/{metrics['agents_expected']}  |  "
              f"Avg latency: {metrics['avg_latency_ms']}ms")
        if metrics["disk_warnings"]:
            print(f"  Disk warnings: {', '.join(metrics['disk_warnings'])}")

        summary = self.task_summary()
        if "error" not in summary:
            print(f"  Tasks: {summary['done']}/{summary['total']} done "
                  f"({summary['completion_pct']}%), {summary['in_progress']} in progress, "
                  f"{summary['blocked']} blocked")
        print("=" * 72)

    # ────────────── MAIN LOOP ──────────────

    def run(self):
        """Main monitoring loop."""
        mode = "DRY RUN" if self.dry_run else "LIVE"
        log("INFO", f"Cluster Orchestrator starting — mode: {mode}")
        log("INFO", f"Monitoring {len(NODES)} nodes, poll interval {POLL_INTERVAL}s")
        log("INFO", f"Restartable nodes: {', '.join(RESTARTABLE_NODES)}")
        log("INFO", f"State dir: {STATE_DIR}")
        log("INFO", f"Reports dir: {REPORT_DIR}")
        log("INFO", f"Command inbox: {CMD_DIR}")
        self.audit("startup", {"mode": mode, "nodes": list(NODES.keys())})
        self._comms_write(f"Orchestrator online — mode: {mode}, monitoring {len(NODES)} nodes")

        while self.running:
            try:
                self.check_kill_switch()
                self.check_command_inbox()
                self.poll_fleet()
                self.run_autonomous_behaviors()
            except KeyboardInterrupt:
                log("INFO", "Shutting down (keyboard interrupt)")
                break
            except Exception as e:
                log("ERROR", f"Main loop error: {e}")
                log("ERROR", traceback.format_exc())
                self.audit("error", {"error": str(e), "traceback": traceback.format_exc()[:500]})

            try:
                time.sleep(POLL_INTERVAL)
            except KeyboardInterrupt:
                log("INFO", "Shutting down (keyboard interrupt)")
                break

        self._save_state()
        self.audit("shutdown", {})
        self._comms_write("Orchestrator shutting down")
        log("INFO", "Orchestrator stopped.")


# ═══════════════════════ CLI ENTRY POINT ═════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Cluster Orchestrator — 12-node AI fleet monitor")
    parser.add_argument("--command", choices=["status", "metrics", "tasks", "activity"],
                        help="Run a one-shot command and exit")
    parser.add_argument("--live", action="store_true", help="Promote to LIVE mode (executes restarts)")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run_flag",
                        help="Set DRY_RUN mode (default)")
    args = parser.parse_args()

    # Handle mode switching via CLI — update persisted state and exit
    if args.live or args.dry_run_flag:
        orch = ClusterOrchestrator()
        if args.live:
            orch.dry_run = False
            log("WARN", "Mode set to LIVE — restarts will be executed")
        else:
            orch.dry_run = True
            log("INFO", "Mode set to DRY_RUN")
        orch._save_state()
        orch.audit("mode_change", {"dry_run": orch.dry_run})
        return

    # One-shot commands
    if args.command:
        orch = ClusterOrchestrator()
        if args.command == "status":
            orch.cmd_status()
        elif args.command == "metrics":
            print(json.dumps(orch.fleet_metrics(), indent=2, default=str))
        elif args.command == "tasks":
            print(json.dumps(orch.task_summary(), indent=2))
        elif args.command == "activity":
            print(json.dumps(orch.agent_activity(), indent=2, default=str))
        return

    # Default: run the main monitoring loop
    orch = ClusterOrchestrator()
    orch.run()


if __name__ == "__main__":
    main()
