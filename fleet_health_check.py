#!/usr/bin/env python3
"""Fleet Health Checker v2 — detects looping, errors, stalls, stagnation, frozen heartbeats,
inefficient token rates, task server health, per-node task history, and continuous watch mode."""
import subprocess, re, time, sys, json, argparse, signal
from collections import Counter
from datetime import datetime, timedelta

VERSION = "2.0"
PASSFILE = "/home/z/DEV/authpass"
SSH_OPTS = "-o StrictHostKeyChecking=no -o ConnectTimeout=5 -o LogLevel=ERROR"

NODES = {
    "sys1": {"ip": "127.0.0.1", "log": "~/AGENT/LOGS/main_trace.log", "auth": "local"},
    "sys2": {"ip": "10.255.255.2", "log": "~/AGENT/LOGS/agent_trace.log", "auth": "passfile"},
    "sys3": {"ip": "10.255.255.3", "log": "~/AGENT/LOGS/agent_trace.log", "auth": "passfile"},
    "sys4": {"ip": "10.255.255.4", "log": "~/AGENT/LOGS/agent_trace.log", "auth": "passfile"},
    "sys5": {"ip": "10.255.255.5", "log": "~/AGENT/LOGS/agent_trace.log", "auth": "passfile"},
    "sys6": {"ip": "10.255.255.6", "log": "~/AGENT/LOGS/agent_trace.log", "auth": "passfile"},
    "sys7": {"ip": "10.255.255.7", "log": "~/AGENT/LOGS/agent_trace.log", "auth": "passfile"},
}

# Stagnation threshold: agent running >2 hours without completing a task
STAGNANT_HOURS = 2
# Heartbeat staleness threshold in seconds
HEARTBEAT_STALE_SECS = 60
# Minimum acceptable tokens/minute rate
MIN_TOKENS_PER_MIN = 10
# Task server endpoint
TASK_SERVER_URL = "http://localhost:9091/health"
# Watch mode interval
WATCH_INTERVAL_SECS = 60


def ssh(ip, cmd, timeout=10):
    """Run command locally or via SSH. All remote SSH uses sshpass -f (never exposes password)."""
    if ip in ("127.0.0.1", "localhost"):
        try:
            r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
            return r.stdout.strip()
        except Exception:
            return ""
    try:
        args = ["sshpass", "-f", PASSFILE, "ssh"] + SSH_OPTS.split() + [f"z@{ip}", cmd]
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def check_heartbeat(ip):
    """Check heartbeat file age. Returns (age_secs, content) or (None, None) if missing."""
    # Agents write heartbeat to ~/AGENT/LOGS/heartbeat.txt
    raw = ssh(ip, "stat -c %Y ~/AGENT/LOGS/heartbeat.txt 2>/dev/null")
    if not raw:
        return None, None
    try:
        mtime = int(raw.strip())
        age = int(time.time()) - mtime
    except ValueError:
        return None, None
    content = ssh(ip, "cat ~/AGENT/LOGS/heartbeat.txt 2>/dev/null")
    return age, content


def check_token_stats(ip):
    """Read .token_stats file. Returns dict with agent, tokens, turns, elapsed_secs, tok_per_min."""
    raw = ssh(ip, "cat ~/AGENT/.token_stats 2>/dev/null")
    if not raw:
        return None
    # Format: AgentName|total_tokens|total_turns|elapsed_secs
    # e.g. "OmniAgent [Main]|24189|563|1063s"
    parts = raw.strip().split("|")
    if len(parts) < 4:
        return None
    try:
        agent_name = parts[0]
        total_tokens = int(parts[1])
        total_turns = int(parts[2])
        elapsed_str = parts[3].rstrip("s")
        elapsed_secs = float(elapsed_str) if elapsed_str else 0
        tok_per_min = (total_tokens / (elapsed_secs / 60.0)) if elapsed_secs > 0 else 0
        return {
            "agent": agent_name,
            "tokens": total_tokens,
            "turns": total_turns,
            "elapsed_secs": elapsed_secs,
            "tok_per_min": round(tok_per_min, 1),
        }
    except (ValueError, ZeroDivisionError):
        return None


def check_task_server():
    """Check if the task server is healthy on localhost:9091/health."""
    try:
        r = subprocess.run(
            ["curl", "-s", "-m", "3", TASK_SERVER_URL],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0 and r.stdout.strip():
            data = json.loads(r.stdout.strip())
            return data.get("status") == "ok"
    except Exception:
        pass
    return False


def get_task_history():
    """Read task_history.jsonl and return per-node completed tasks for this session.
    Session = tasks from the last 24 hours."""
    history_path = "/home/z/AGENT/LOGS/task_history.jsonl"
    node_tasks = {}  # node_name -> list of completed task IDs
    cutoff = datetime.now() - timedelta(hours=24)
    try:
        with open(history_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("event") != "COMPLETE":
                    continue
                ts_str = entry.get("ts", "")
                try:
                    ts = datetime.fromisoformat(ts_str)
                except ValueError:
                    continue
                if ts < cutoff:
                    continue
                agent = entry.get("agent", "")
                task = entry.get("task", "")
                # Map agent name to node name
                node = _agent_to_node(agent)
                if node not in node_tasks:
                    node_tasks[node] = []
                node_tasks[node].append(task)
    except FileNotFoundError:
        pass
    return node_tasks


def _agent_to_node(agent_name):
    """Map agent name strings to node names. Best-effort matching."""
    name = agent_name.lower()
    for node_name in NODES:
        if node_name in name:
            return node_name
    # Fallback patterns
    if "main" in name and "sys" not in name:
        return "sys1"
    if "vulkan" in name:
        return "sys1"
    return agent_name  # Unknown agent, return raw name


def check_stagnation(ip):
    """Detect agents running >2 hours without completing a task.
    Uses the CLAIM timestamp from log to estimate task start time."""
    raw = ssh(ip, "strings ~/AGENT/LOGS/agent_trace.log 2>/dev/null | grep -i 'CLAIMED\\|COMPLETE' | tail -20")
    if not raw:
        raw = ssh(ip, "strings ~/AGENT/LOGS/main_trace.log 2>/dev/null | grep -i 'CLAIMED\\|COMPLETE' | tail -20")
    if not raw:
        return None

    # Check .token_stats elapsed time vs completions
    stats = check_token_stats(ip)
    if stats and stats["elapsed_secs"] > STAGNANT_HOURS * 3600:
        # Agent has been running a long time -- check if there are recent completions
        # via task_history from the task server side
        return stats["elapsed_secs"]
    return None


def check_node(name, node, task_history_map):
    """Full health check for a single node."""
    ip, log = node["ip"], node["log"]
    issues = []

    # Get last 8KB of log
    raw = ssh(ip, f"strings {log} 2>/dev/null | tail -200")
    if not raw:
        return {
            "node": name, "status": "DEAD", "issues": ["No log output"],
            "turns": 0, "tps": 0, "ttft": 0, "task": "-",
            "heartbeat_age": None, "tok_per_min": 0,
            "completed_tasks": [], "token_stats": None,
        }

    # Count turns
    turns = len(re.findall(r'Turn \d+', raw))

    # Get latest TPS
    tps_matches = re.findall(r'(\d+\.?\d*) t/s', raw)
    tps = float(tps_matches[-1]) if tps_matches else 0

    # Get latest TTFT
    ttft_matches = re.findall(r'TTFT (\d+)ms', raw)
    ttft = int(ttft_matches[-1]) if ttft_matches else 0

    # Get claimed task
    task_match = re.findall(r'CLAIMED (T\d+)', raw)
    task = task_match[-1] if task_match else "-"

    # ---- EXISTING DETECTIONS ----

    # DETECT: Context overflow (400 error looping)
    overflow_count = raw.count("context length") + raw.count("input_tokens")
    if overflow_count >= 2:
        issues.append(f"CONTEXT OVERFLOW x{overflow_count} -- agent stuck retrying 400 error")

    # DETECT: Parse failure loop
    parse_fails = raw.count("SELF-CORRECTION")
    if parse_fails >= 3:
        issues.append(f"PARSE FAILURES x{parse_fails} -- model can't format tool calls")

    # DETECT: Claim loop (trying to claim DONE tasks)
    already_done = raw.count("ALREADY_DONE")
    if already_done >= 3:
        issues.append(f"CLAIM LOOP x{already_done} -- agent keeps trying DONE tasks")

    # DETECT: Blocked loop (trying to claim when already has task)
    blocked = raw.count("BLOCKED")
    if blocked >= 3:
        issues.append(f"BLOCKED LOOP x{blocked} -- agent has task but keeps trying to claim")

    # DETECT: Many turns but no write_file or push_changes (unproductive)
    writes = raw.count('write_file') + raw.count('push_changes') + raw.count('Created')
    if turns > 20 and writes == 0:
        issues.append(f'NO PRODUCTIVE OUTPUT: {turns} turns but no file writes or pushes')

    # DETECT: Same tool called repeatedly (stuck)
    tool_calls = re.findall(r'EXECUTING\]: (\w+)', raw)
    if len(tool_calls) >= 5:
        last5 = tool_calls[-5:]
        if len(set(last5)) == 1 and last5[0] not in ("execute_bash", "read_file", "write_file"):
            issues.append(f"STUCK: calling {last5[0]} repeatedly")

    # DETECT: Reading task queue repeatedly
    queue_reads = raw.count("TASK_QUEUE_v5.md")
    if queue_reads >= 5:
        issues.append(f"QUEUE LOOP x{queue_reads} -- keeps re-reading task queue instead of working")

    # DETECT: Same file read repeatedly
    file_reads = re.findall(r'read_file.*?path.*?([^\s"]+\.(?:py|c|md))', raw)
    if file_reads:
        freq = Counter(file_reads).most_common(1)
        if freq and freq[0][1] >= 4:
            issues.append(f"FILE LOOP: reading {freq[0][0]} x{freq[0][1]}")

    # DETECT: Stream error loop
    stream_errors = raw.count("STREAM ERROR")
    if stream_errors >= 2:
        issues.append(f"STREAM ERRORS x{stream_errors}")

    # DETECT: Stale (no new turns in last 200 lines)
    if turns == 0:
        issues.append("STALE -- no turns in recent log")

    # DETECT: Timeout loop
    timeouts = raw.count("TIMEOUT after")
    if timeouts >= 2:
        issues.append(f"TIMEOUT LOOP x{timeouts}")

    # DETECT: GO_PROMPT reload loop
    go_reloads = raw.count("re-loading GO_PROMPT")
    if go_reloads >= 3:
        issues.append(f"GO_PROMPT LOOP x{go_reloads}")

    # DETECT: Wrong file paths
    wrong_paths = raw.count("/home/z/AGENT/TASK_QUEUE_v4") + raw.count("T_TASK_QUEUE")
    if wrong_paths >= 1:
        issues.append(f"WRONG PATHS: v4 or T_TASK references ({wrong_paths}x)")

    # DETECT: Low disk space
    disk_line = ssh(ip, "df -h / 2>/dev/null | tail -1")
    if disk_line:
        pct_match = re.search(r"(\d+)%", disk_line)
        if pct_match and int(pct_match.group(1)) > 85:
            issues.append(f"LOW DISK: {pct_match.group(1)}% used")

    # ---- NEW v2 DETECTIONS ----

    # [v2] DETECT: Stagnant agent (running >2h without completing a task)
    token_stats = check_token_stats(ip)
    tok_per_min = 0
    if token_stats:
        elapsed = token_stats["elapsed_secs"]
        tok_per_min = token_stats["tok_per_min"]
        completed = task_history_map.get(name, [])
        if elapsed > STAGNANT_HOURS * 3600 and len(completed) == 0:
            hours = elapsed / 3600
            issues.append(f"STAGNANT: running {hours:.1f}h with 0 completed tasks this session")

    # [v2] DETECT: Heartbeat staleness (>60s = possibly frozen)
    hb_age, hb_content = check_heartbeat(ip)
    if hb_age is not None and hb_age > HEARTBEAT_STALE_SECS:
        issues.append(f"HEARTBEAT STALE: last beat {hb_age}s ago (>{HEARTBEAT_STALE_SECS}s)")
    elif hb_age is None and ip not in ("127.0.0.1", "localhost"):
        # Remote node with no heartbeat file at all
        issues.append("NO HEARTBEAT FILE: agent may not be running")

    # [v2] DETECT: Low token rate (<10 tok/min = inefficient)
    if token_stats and token_stats["elapsed_secs"] > 120:
        # Only flag after 2 min warmup
        if tok_per_min < MIN_TOKENS_PER_MIN:
            issues.append(f"LOW TOKEN RATE: {tok_per_min} tok/min (<{MIN_TOKENS_PER_MIN})")

    # Completed tasks for this node
    completed_tasks = task_history_map.get(name, [])

    status = "SICK" if issues else "OK"
    return {
        "node": name, "status": status, "task": task,
        "turns": turns, "tps": round(tps, 1), "ttft": ttft,
        "issues": issues,
        "heartbeat_age": hb_age,
        "tok_per_min": tok_per_min,
        "completed_tasks": completed_tasks,
        "token_stats": token_stats,
    }


def run_check(json_mode=False):
    """Execute one full fleet health check cycle."""
    print(f"{'='*74}")
    print(f"  FLEET HEALTH CHECK v{VERSION} -- {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*74}")

    # [v2] Check task server health
    ts_healthy = check_task_server()
    ts_icon = "[OK]" if ts_healthy else "[DOWN]"
    print(f"\n  Task Server (localhost:9091): {ts_icon}")
    if not ts_healthy:
        print("   !! Task server unreachable or unhealthy")

    # [v2] Load task history for session
    task_history_map = get_task_history()

    results = []
    for name, node in NODES.items():
        r = check_node(name, node, task_history_map)
        results.append(r)

        icon = "[OK]" if r["status"] == "OK" else "[!!]" if r["status"] == "SICK" else "[XX]"
        tok_info = f"{r['tok_per_min']} tok/m" if r["tok_per_min"] else "-"
        hb_info = f"hb:{r['heartbeat_age']}s" if r["heartbeat_age"] is not None else "hb:?"
        completed_count = len(r["completed_tasks"])

        print(f"\n{icon} {name}: {r['status']} | task={r['task']} | "
              f"{r['turns']} turns | {r['tps']} t/s | TTFT {r['ttft']}ms | "
              f"{tok_info} | {hb_info} | done:{completed_count}")
        for issue in r["issues"]:
            print(f"   !! {issue}")
        if r["completed_tasks"]:
            print(f"   Completed this session: {', '.join(r['completed_tasks'])}")

    sick = sum(1 for r in results if r["status"] != "OK")
    print(f"\n{'='*74}")

    # Productivity ranking
    ranked = sorted([r for r in results if r['turns'] > 0], key=lambda x: x['tps'], reverse=True)
    if ranked:
        print(f"\n  Productivity ranking (by TPS):")
        for r in ranked[:5]:
            done = len(r["completed_tasks"])
            print(f"    {r['node']}: {r['tps']} t/s, {r['turns']} turns, "
                  f"task={r['task']}, tok/m={r['tok_per_min']}, done={done}")

    # [v2] Task completion summary
    all_completed = []
    for r in results:
        all_completed.extend(r["completed_tasks"])
    if all_completed:
        print(f"\n  Session completions (24h): {len(all_completed)} tasks -- {', '.join(sorted(set(all_completed)))}")

    total_tokens = sum(
        r["token_stats"]["tokens"] for r in results
        if r.get("token_stats") and isinstance(r["token_stats"].get("tokens"), (int, float))
    )

    print(f"\n  {len(results) - sick}/{len(results)} healthy | {sick} need attention")
    if total_tokens > 0:
        print(f"  Total tokens generated: {total_tokens:,}")
    print(f"  Task server: {'UP' if ts_healthy else 'DOWN'}")
    print(f"{'='*74}")

    if json_mode:
        # Make output JSON-serializable
        for r in results:
            r["task_server_healthy"] = ts_healthy
        print(json.dumps(results, indent=2, default=str))

    return results


def main():
    parser = argparse.ArgumentParser(description=f"Fleet Health Checker v{VERSION}")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--watch", action="store_true",
                        help=f"Continuous mode: re-run every {WATCH_INTERVAL_SECS}s")
    parser.add_argument("--interval", type=int, default=WATCH_INTERVAL_SECS,
                        help=f"Watch interval in seconds (default: {WATCH_INTERVAL_SECS})")
    args = parser.parse_args()

    # Handle Ctrl+C gracefully in watch mode
    def _sigint(sig, frame):
        print("\n\nWatch mode stopped.")
        sys.exit(0)
    signal.signal(signal.SIGINT, _sigint)

    if args.watch:
        interval = args.interval
        print(f"[watch] Running every {interval}s. Press Ctrl+C to stop.\n")
        while True:
            run_check(json_mode=args.json)
            print(f"\n[watch] Next check in {interval}s ...\n")
            time.sleep(interval)
    else:
        run_check(json_mode=args.json)


if __name__ == "__main__":
    main()
