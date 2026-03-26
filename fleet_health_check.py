#!/usr/bin/env python3
"""Fleet Health Checker — detects looping, errors, stalls, and context overflow across all agents."""
import subprocess, re, time, sys, json

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

def ssh(ip, cmd, auth="passfile", timeout=10):
    if ip in ("127.0.0.1", "localhost"):
        try:
            r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
            return r.stdout.strip()
        except: return ""
    try:
        if auth == "passfile":
            args = ["sshpass", "-f", PASSFILE, "ssh"] + SSH_OPTS.split() + [f"z@{ip}", cmd]
        else:
            args = ["sshpass", "-p", "z", "ssh"] + SSH_OPTS.split() + [f"z@{ip}", cmd]
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except: return ""

def check_node(name, node):
    ip, log, auth = node["ip"], node["log"], node["auth"]
    issues = []
    
    # Get last 8KB of log
    raw = ssh(ip, f"strings {log} 2>/dev/null | tail -200", auth)
    if not raw:
        return {"node": name, "status": "DEAD", "issues": ["No log output"], "turns": 0, "tps": 0}
    
    # Count turns
    turns = len(re.findall(r'Turn \d+', raw))
    
    # Get latest TPS
    tps_matches = re.findall(r'(\d+\.?\d*) t/s', raw)
    tps = float(tps_matches[-1]) if tps_matches else 0
    
    # Get latest TTFT
    ttft_matches = re.findall(r'TTFT (\d+)ms', raw)
    ttft = int(ttft_matches[-1]) if ttft_matches else 0
    
    # DETECT: Context overflow (400 error looping)
    overflow_count = raw.count("context length") + raw.count("input_tokens")
    if overflow_count >= 2:
        issues.append(f"CONTEXT OVERFLOW x{overflow_count} — agent stuck retrying 400 error")
    
    # DETECT: Parse failure loop
    parse_fails = raw.count("SELF-CORRECTION")
    if parse_fails >= 3:
        issues.append(f"PARSE FAILURES x{parse_fails} — model can't format tool calls")
    
    # DETECT: Claim loop (trying to claim DONE tasks)
    already_done = raw.count("ALREADY_DONE")
    if already_done >= 3:
        issues.append(f"CLAIM LOOP x{already_done} — agent keeps trying DONE tasks")
    
    # DETECT: Blocked loop (trying to claim when already has task)
    blocked = raw.count("BLOCKED")
    if blocked >= 3:
        issues.append(f"BLOCKED LOOP x{blocked} — agent has task but keeps trying to claim")
    
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
        issues.append(f"QUEUE LOOP x{queue_reads} — keeps re-reading task queue instead of working")
    
    # DETECT: Same file read repeatedly
    file_reads = re.findall(r'read_file.*?path.*?([^\s"]+\.(?:py|c|md))', raw)
    if file_reads:
        from collections import Counter
        freq = Counter(file_reads).most_common(1)
        if freq and freq[0][1] >= 4:
            issues.append(f"FILE LOOP: reading {freq[0][0]} x{freq[0][1]}")
    
    # DETECT: Stream error loop
    stream_errors = raw.count("STREAM ERROR")
    if stream_errors >= 2:
        issues.append(f"STREAM ERRORS x{stream_errors}")
    
    # DETECT: Stale (no new turns in last 200 lines)
    if turns == 0:
        issues.append("STALE — no turns in recent log")
    
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
    
    # Get claimed task
    task_match = re.findall(r'CLAIMED (T\d+)', raw)
    task = task_match[-1] if task_match else "-"
    
    # DETECT: Low disk space
    disk_line = ssh(ip, "df -h / 2>/dev/null | tail -1", auth)
    if disk_line:
        import re as _r2
        pct_match = _r2.search(r"(\d+)%", disk_line)
        if pct_match and int(pct_match.group(1)) > 85:
            issues.append(f"LOW DISK: {pct_match.group(1)}% used")
    
    status = "SICK" if issues else "OK"
    return {
        "node": name, "status": status, "task": task,
        "turns": turns, "tps": round(tps, 1), "ttft": ttft,
        "issues": issues
    }

def main():
    print(f"{'='*70}")
    print(f"  FLEET HEALTH CHECK — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    results = []
    for name, node in NODES.items():
        r = check_node(name, node)
        results.append(r)
        
        icon = "✅" if r["status"] == "OK" else "❌" if r["status"] == "SICK" else "💀"
        print(f"\n{icon} {name}: {r['status']} | task={r['task']} | {r['turns']} turns | {r['tps']} t/s | TTFT {r['ttft']}ms")
        for issue in r["issues"]:
            print(f"   ⚠️  {issue}")
    
    sick = sum(1 for r in results if r["status"] != "OK")
    print(f"\n{'='*70}")
    print(f"  {len(results) - sick}/{len(results)} healthy | {sick} need attention")
    print(f"{'='*70}")
    
    if "--json" in sys.argv:
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
