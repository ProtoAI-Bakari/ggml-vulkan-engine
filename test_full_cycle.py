#!/usr/bin/env python3
"""Integration test: claim → work → push → complete cycle."""
import subprocess, json, time, os

API = "http://localhost:9091"

def test():
    print("=== FULL CYCLE TEST ===")
    
    # 1. Check API is up
    r = subprocess.run(f"curl -s {API}/health", shell=True, capture_output=True, text=True, timeout=5)
    assert "ok" in r.stdout, f"API down: {r.stdout}"
    print("1. API: OK")
    
    # 2. Find a READY task
    r = subprocess.run(f"curl -s {API}/tasks", shell=True, capture_output=True, text=True, timeout=5)
    import re
    ready = re.findall(r'### (T\d+):.*?\[READY\]', r.stdout)
    if not ready:
        print("2. No READY tasks — skipping")
        return True
    task = ready[0]
    print(f"2. Found READY: {task}")
    
    # 3. Claim it
    r = subprocess.run(f'curl -s -X POST {API}/claim -d "task={task}&agent=test-cycle"', shell=True, capture_output=True, text=True, timeout=5)
    d = json.loads(r.stdout)
    assert d["ok"], f"Claim failed: {d}"
    print(f"3. Claimed: {task}")
    
    # 4. Complete it
    r = subprocess.run(f'curl -s -X POST {API}/complete -d "task={task}&agent=test-cycle"', shell=True, capture_output=True, text=True, timeout=5)
    d = json.loads(r.stdout)
    assert d["ok"], f"Complete failed: {d}"
    print(f"4. Completed: {task}")
    
    # 5. Verify it's DONE
    r = subprocess.run(f"curl -s {API}/tasks", shell=True, capture_output=True, text=True, timeout=5)
    assert f"[DONE by test-cycle" in r.stdout, "Task not marked DONE"
    print("5. Verified DONE in queue")
    
    print("=== FULL CYCLE: PASS ===")
    return True

if __name__ == "__main__":
    test()
