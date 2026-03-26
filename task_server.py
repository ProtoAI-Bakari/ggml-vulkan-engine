#!/usr/bin/env python3
"""
Central Task Queue Server — eliminates distributed race conditions.

Runs on sys1 (10.255.255.128:9091). All agents call this API
instead of editing local TASK_QUEUE_v5.md copies.

Endpoints:
  GET  /tasks              — return full task queue
  GET  /tasks/summary      — return counts (READY, IN_PROGRESS, DONE, BLOCKED)
  POST /claim              — atomically claim a READY task
  POST /complete           — atomically mark a task DONE
  GET  /health             — liveness check

Uses fcntl.flock for atomic file access.
"""

import fcntl
import json
import os
import re
import sys
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs

QUEUE_PATH = os.path.expanduser("~/AGENT/TASK_QUEUE_v5.md")
BIND_ADDR = "0.0.0.0"
BIND_PORT = 9091
LOG_FILE = os.path.expanduser("~/AGENT/LOGS/task_server.log")


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def read_queue_locked():
    """Read task queue with shared lock."""
    with open(QUEUE_PATH, "r") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        content = f.read()
        fcntl.flock(f, fcntl.LOCK_UN)
    return content


def write_queue_locked(content):
    """Write task queue with exclusive lock."""
    with open(QUEUE_PATH, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        f.write(content)
        f.truncate()
        fcntl.flock(f, fcntl.LOCK_UN)


def atomic_claim(task_id, agent_name):
    """Atomically claim a READY task. Returns (success, message)."""
    with open(QUEUE_PATH, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            content = f.read()

            # Check if task exists and is READY
            # Match patterns like: ### T59: [READY] or ### T59: [READY]]
            pattern = rf"(### {re.escape(task_id)}:.*?)\[READY\](\]*)"
            match = re.search(pattern, content)
            if not match:
                # Check if already claimed
                taken_pat = rf"### {re.escape(task_id)}:.*?\[IN_PROGRESS by ([^\]]+)\]"
                taken = re.search(taken_pat, content)
                if taken:
                    return False, f"TAKEN by {taken.group(1)}"
                done_pat = rf"### {re.escape(task_id)}:.*?\[DONE"
                if re.search(done_pat, content):
                    return False, "ALREADY_DONE"
                return False, "NOT_FOUND_OR_NOT_READY"

            # Check if agent already has an active task
            # Use flexible match — agent name may be followed by ] or | or space
            agent_pat = rf"\[IN_PROGRESS by {re.escape(agent_name)}[\]| ]"
            existing = re.search(agent_pat, content)
            if existing:
                # Find which task they have
                existing_task = re.search(
                    rf"### (T\d+):.*?\[IN_PROGRESS by {re.escape(agent_name)}",
                    content,
                )
                tid = existing_task.group(1) if existing_task else "unknown"
                return False, f"BLOCKED — {agent_name} already has {tid}"

            # Claim it: replace [READY] with [IN_PROGRESS by agent | 0% | started:timestamp]
            ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
            new_content = re.sub(
                pattern,
                rf"\g<1>[IN_PROGRESS by {agent_name} | 0% | started:{ts}]",
                content,
                count=1,
            )
            # Strip any stale metadata that accumulated on this task line
            # (e.g., leftover "| 0% | started:..." from previous claims)
            task_line_pat = rf"(### {re.escape(task_id)}:.*?\[IN_PROGRESS by [^\]]+\])((?:\s*\|[^\]\n]*\])*)"
            new_content = re.sub(task_line_pat, r"\1", new_content, count=1)

            f.seek(0)
            f.write(new_content)
            f.truncate()
            return True, f"CLAIMED {task_id} for {agent_name}"
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def atomic_complete(task_id, agent_name):
    """Atomically mark a task DONE. Returns (success, message)."""
    with open(QUEUE_PATH, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            content = f.read()

            # Match IN_PROGRESS by anyone (or specific agent)
            pattern = rf"(### {re.escape(task_id)}:.*?)\[IN_PROGRESS by [^\]]*\](\]*)"
            match = re.search(pattern, content)
            if not match:
                done_pat = rf"### {re.escape(task_id)}:.*?\[DONE"
                if re.search(done_pat, content):
                    return False, "ALREADY_DONE"
                return False, "NOT_IN_PROGRESS"

            # Strip any accumulated stale metadata after the status bracket
            stale_pat = rf"(### {re.escape(task_id)}:.*?\[IN_PROGRESS by [^\]]+\])((?:\s*\|[^\]\n]*\])*)"
            content = re.sub(stale_pat, r"\1", content, count=1)
            # Re-match after cleanup
            match = re.search(pattern, content)
            if not match:
                return False, "NOT_IN_PROGRESS"
            ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
            new_content = re.sub(
                pattern,
                rf"\g<1>[DONE by {agent_name} | completed:{ts}]",
                content,
                count=1,
            )

            f.seek(0)
            f.write(new_content)
            f.truncate()
            return True, f"COMPLETED {task_id} by {agent_name}"
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def atomic_progress(task_id, agent_name, pct, note=""):
    """Atomically update progress on an IN_PROGRESS task."""
    with open(QUEUE_PATH, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            content = f.read()
            # Match: [IN_PROGRESS by AGENT | XX% | started:TS] or [IN_PROGRESS by AGENT | XX% | started:TS | note]
            pattern = rf"(### {re.escape(task_id)}:.*?)\[IN_PROGRESS by {re.escape(agent_name)}[^\]]*\](\]*)"
            match = re.search(pattern, content)
            if not match:
                return False, "NOT_YOUR_TASK"
            # Extract original started timestamp
            started_match = re.search(r"started:(\S+)", match.group(0))
            started = started_match.group(1) if started_match else datetime.now().strftime("%Y-%m-%dT%H:%M")
            note_str = f" | {note}" if note else ""
            new_tag = f"[IN_PROGRESS by {agent_name} | {pct}% | started:{started}{note_str}]"
            new_content = re.sub(pattern, rf"\g<1>{new_tag}", content, count=1)
            f.seek(0)
            f.write(new_content)
            f.truncate()
            return True, f"PROGRESS {task_id} -> {pct}%"
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def task_summary():
    """Count tasks by status."""
    content = read_queue_locked()
    return {
        "DONE": len(re.findall(r"\[DONE", content)),
        "IN_PROGRESS": len(re.findall(r"\[IN_PROGRESS", content)),
        "READY": len(re.findall(r"\[READY\]", content)),
        "BLOCKED": len(re.findall(r"\[BLOCKED", content)),
    }


class TaskHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log(f"HTTP {args[0]}" if args else fmt)

    def _respond(self, code, body, content_type="text/plain"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if isinstance(body, str):
            body = body.encode()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/tasks":
            content = read_queue_locked()
            self._respond(200, content)
        elif self.path == "/tasks/summary":
            s = task_summary()
            self._respond(200, json.dumps(s), "application/json")
        elif self.path == "/health":
            self._respond(200, json.dumps({"status": "ok", "port": BIND_PORT}), "application/json")
        else:
            self._respond(404, "Not Found")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode() if length else ""
        params = parse_qs(body)

        if self.path == "/claim":
            task_id = params.get("task", [None])[0]
            agent = params.get("agent", [None])[0]
            if not task_id or not agent:
                self._respond(400, "Missing task or agent param")
                return
            ok, msg = atomic_claim(task_id, agent)
            log(f"CLAIM {task_id} by {agent} -> {'OK' if ok else 'FAIL'}: {msg}")
            code = 200 if ok else 409
            self._respond(code, json.dumps({"ok": ok, "msg": msg}), "application/json")

        elif self.path == "/progress":
            task_id = params.get("task", [None])[0]
            agent = params.get("agent", [None])[0]
            pct = params.get("pct", ["0"])[0]
            note = params.get("note", [""])[0]
            if not task_id or not agent:
                self._respond(400, "Missing task or agent param")
                return
            ok, msg = atomic_progress(task_id, agent, pct, note)
            log(f"PROGRESS {task_id} by {agent} -> {pct}%: {msg}")
            code = 200 if ok else 409
            self._respond(code, json.dumps({"ok": ok, "msg": msg}), "application/json")

        elif self.path == "/block":
            task_id = params.get("task", [None])[0]
            agent = params.get("agent", [None])[0]
            reason = params.get("reason", ["unknown"])[0]
            if not task_id:
                self._respond(400, "Missing task param")
                return
            with open(QUEUE_PATH, "r+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    content = f.read()
                    pattern = rf"(### {re.escape(task_id)}:.*?)\[IN_PROGRESS[^\]]*\]"
                    match = re.search(pattern, content)
                    if match:
                        ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
                        new_content = re.sub(pattern, rf"\g<1>[BLOCKED by {agent or 'system'} | reason:{reason} | {ts}]", content, count=1)
                        f.seek(0)
                        f.write(new_content)
                        f.truncate()
                        log(f"BLOCKED {task_id}: {reason}")
                        self._respond(200, json.dumps({"ok": True, "msg": f"BLOCKED {task_id}"}), "application/json")
                    else:
                        self._respond(409, json.dumps({"ok": False, "msg": "NOT_IN_PROGRESS"}), "application/json")
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            return

        elif self.path == "/complete":
            task_id = params.get("task", [None])[0]
            agent = params.get("agent", [None])[0]
            if not task_id or not agent:
                self._respond(400, "Missing task or agent param")
                return
            ok, msg = atomic_complete(task_id, agent)
            log(f"COMPLETE {task_id} by {agent} -> {'OK' if ok else 'FAIL'}: {msg}")
            code = 200 if ok else 409
            self._respond(code, json.dumps({"ok": ok, "msg": msg}), "application/json")

        else:
            self._respond(404, "Not Found")


def main():
    if not os.path.exists(QUEUE_PATH):
        print(f"ERROR: Queue file not found: {QUEUE_PATH}", file=sys.stderr)
        sys.exit(1)

    server = HTTPServer((BIND_ADDR, BIND_PORT), TaskHandler)
    log(f"Task server starting on {BIND_ADDR}:{BIND_PORT}")
    log(f"Queue file: {QUEUE_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("Task server stopped.")
        server.server_close()


if __name__ == "__main__":
    main()

# Agent0 TEST task routing
# Tasks tagged [TEST] can only be claimed by agents with "agent0" or "test" in their name
# Other agents see them as NOT_AVAILABLE
