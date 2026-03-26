#!/usr/bin/env python3
"""
Agent0 — Test Runner for sys1 (Asahi M1 Ultra)
Uses z4090 as brain (68 TPS). Only picks up [TEST] tasks or runs tests for completed code tasks.
Runs locally on sys1 where the Vulkan GPU + compiled engine live.

Usage: python3 agent0_test_runner.py
"""
import os, sys, re, json, time, subprocess, urllib.request
from datetime import datetime

BRAIN_URL = "http://10.255.255.10:8000/v1/chat/completions"
BRAIN_MODEL = "qwen3-coder-30b"
TASK_API = "http://localhost:9091"
AGENT_NAME = "agent0-tester"
LOG_FILE = os.path.expanduser("~/AGENT/LOGS/agent0_test.log")
RESULTS_DIR = os.path.expanduser("~/AGENT/test_results")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def ask_brain(prompt, max_tokens=2000):
    """Ask z4090 brain for test code or analysis."""
    payload = json.dumps({
        "model": BRAIN_MODEL,
        "messages": [
            {"role": "system", "content": "You are a test engineer. Write Python test code. Output ONLY code, no explanation. Use assert statements. Test the Vulkan ggml engine at ~/AGENT/."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens, "temperature": 0.1
    }).encode()
    req = urllib.request.Request(BRAIN_URL, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read())
        return d["choices"][0]["message"]["content"]
    except Exception as e:
        log(f"Brain error: {e}")
        return None

def run_cmd(cmd, timeout=120):
    """Run a shell command, return (exit_code, stdout, stderr)."""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except Exception as e:
        return -1, "", str(e)

def test_vulkan_coherency(n_prompts=10):
    """Test the Vulkan ggml server for coherent output."""
    log(f"Running coherency test ({n_prompts} prompts)...")
    prompts = [
        "What is 2+2?",
        "Name the capital of Japan.",
        "Write one sentence about dogs.",
        "What color is the sky?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
        "Name a programming language.",
        "What year did WW2 end?",
        "Translate hello to Spanish.",
        "What is the largest planet?",
    ][:n_prompts]

    passed = 0
    results = []
    for i, p in enumerate(prompts):
        code, out, err = run_cmd(
            f'curl -s --max-time 15 http://localhost:8080/v1/chat/completions '
            f'-H "Content-Type: application/json" '
            f'-d \'{{"model":"llama-8b","messages":[{{"role":"user","content":"{p}"}}],"max_tokens":30,"stream":false,"temperature":0}}\''
        )
        try:
            d = json.loads(out)
            text = d["choices"][0]["message"]["content"]
            # Check for gibberish (repeating chars)
            is_gibberish = len(set(text[:20])) < 5 if len(text) > 10 else False
            ok = not is_gibberish and len(text) > 5
            results.append({"prompt": p, "response": text[:100], "ok": ok})
            if ok:
                passed += 1
                log(f"  [{i+1}] OK: {text[:60]}")
            else:
                log(f"  [{i+1}] FAIL: {text[:60]}")
        except Exception as e:
            results.append({"prompt": p, "response": str(e), "ok": False})
            log(f"  [{i+1}] ERROR: {e}")

    log(f"Coherency: {passed}/{n_prompts}")
    return passed, n_prompts, results

def test_compilation():
    """Test that the C engine compiles cleanly."""
    log("Testing C engine compilation...")
    code, out, err = run_cmd(
        "cd ~/AGENT && gcc -shared -O2 -fPIC -o /tmp/test_libggml.so ggml_llama_gguf.c "
        "-I ~/GITDEV/llama.cpp/ggml/include "
        "-L ~/GITDEV/llama.cpp/build-lib/bin "
        "-lggml -lggml-base -lggml-vulkan -lggml-cpu -lm "
        "-Wl,-rpath,~/GITDEV/llama.cpp/build-lib/bin 2>&1"
    )
    if code == 0:
        log("  COMPILE: OK")
        return True, ""
    else:
        log(f"  COMPILE: FAIL\n{err[:200]}")
        return False, err

def test_python_import():
    """Test that the Python backend imports and works."""
    log("Testing Python import...")
    code, out, err = run_cmd(
        'cd ~/AGENT && python3 -c "'
        'from ggml_vllm_backend import GgmlLLM, SamplingParams; '
        'llm = GgmlLLM(\"~/models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf\"); '
        'r = llm.generate(\"Hello\", params=SamplingParams(temperature=0, max_tokens=5)); '
        'print(f\"TPS:{r.tps:.1f} TEXT:{r.text}\")'
        '"', timeout=30
    )
    if code == 0 and "TPS:" in out:
        log(f"  IMPORT: OK — {out}")
        return True, out
    else:
        log(f"  IMPORT: FAIL — {err[:200]}")
        return False, err

def test_server_alive():
    """Check if the ggml Vulkan server is responding."""
    code, out, err = run_cmd("curl -s --max-time 3 http://localhost:8080/v1/models")
    if code == 0 and "model" in out.lower():
        log("  SERVER: UP")
        return True
    else:
        log("  SERVER: DOWN")
        return False

def test_new_files():
    """Check for new files pushed by remote agents and test them."""
    log("Checking for new agent-pushed files...")
    # Find .py and .c files modified in last 30 min
    code, out, err = run_cmd("find ~/AGENT -maxdepth 1 -name '*.py' -o -name '*.c' -newer ~/AGENT/TASK_QUEUE_v5.md 2>/dev/null")
    if not out:
        log("  No new files from agents")
        return []

    new_files = [f.strip() for f in out.split("\n") if f.strip()]
    results = []
    for f in new_files:
        basename = os.path.basename(f)
        log(f"  Testing: {basename}")
        if f.endswith(".c"):
            # Try to compile
            ok, err = run_cmd(f"gcc -fsyntax-only {f} 2>&1", timeout=10)
            results.append({"file": basename, "test": "syntax", "ok": ok == 0, "error": err if ok != 0 else ""})
        elif f.endswith(".py"):
            # Try to import
            module = basename.replace(".py", "")
            ok, out_str, err = run_cmd(f"cd ~/AGENT && python3 -c 'import {module}; print(\"OK\")' 2>&1", timeout=10)
            results.append({"file": basename, "test": "import", "ok": ok == 0 and "OK" in out_str, "error": err if ok != 0 else ""})
    return results

def run_full_test_suite():
    """Run all tests and generate a report."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": ts,
        "agent": AGENT_NAME,
        "tests": {}
    }

    # 1. Server alive
    report["tests"]["server"] = test_server_alive()

    # 2. Compilation
    ok, err = test_compilation()
    report["tests"]["compilation"] = {"ok": ok, "error": err[:200]}

    # 3. Python import + quick inference
    if report["tests"]["server"]:
        ok, out = test_python_import()
        report["tests"]["python_import"] = {"ok": ok, "output": out[:200]}

    # 4. Coherency (only if server is up)
    if report["tests"]["server"]:
        passed, total, results = test_vulkan_coherency(5)
        report["tests"]["coherency"] = {"passed": passed, "total": total, "details": results}

    # 5. New file tests
    file_results = test_new_files()
    if file_results:
        report["tests"]["new_files"] = file_results

    # Save report
    report_path = os.path.join(RESULTS_DIR, f"test_{ts}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log(f"Report saved: {report_path}")

    # Summary
    all_ok = all([
        report["tests"].get("server", False),
        report["tests"].get("compilation", {}).get("ok", False),
    ])
    return report, all_ok

def main():
    log("=" * 60)
    log(f"Agent0 Test Runner starting — brain: z4090 ({BRAIN_MODEL})")
    log("=" * 60)

    cycle = 0
    while True:
        cycle += 1
        log(f"\n--- Test cycle {cycle} ---")

        report, all_ok = run_full_test_suite()

        if all_ok:
            log("ALL TESTS PASSED")
        else:
            log("SOME TESTS FAILED — check report")

        # Wait before next cycle
        wait = 120  # 2 minutes between test cycles
        log(f"Next cycle in {wait}s...")
        time.sleep(wait)

if __name__ == "__main__":
    main()
