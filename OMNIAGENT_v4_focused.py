"""
OmniAgent v4 - The Focused Architect
Features: Zero visual truncation, perfect context window, Asahi/Vulkan environmental anchoring.
"""
import os
import sys
import time
import json
import subprocess
import re
import traceback
import signal
import argparse
import requests
from datetime import datetime
from openai import OpenAI

# =====================================================
# ANSI COLORS
# =====================================================
class C:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    MAGENTA = '\033[95m'
    RED = '\033[91m'
    DIM = '\033[2m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# =====================================================
# SIGNAL HANDLING
# =====================================================
class InterruptSignal(Exception): pass

last_sigint_time = 0
def sigint_handler(signum, frame):
    global last_sigint_time
    current_time = time.time()
    if current_time - last_sigint_time < 1.5:
        print(f"\n\n{C.BOLD}{C.RED}[💀 DOUBLE CTRL+C: HARD EXIT]{C.RESET}")
        os._exit(1)
    last_sigint_time = current_time
    raise InterruptSignal()

signal.signal(signal.SIGINT, sigint_handler)

# =====================================================
# CONFIGURATION
# =====================================================
# PRIMARY = LOCAL brain (each node uses its OWN MLX model for fast reasoning)
PRIMARY_IP   = "127.0.0.1"     # LOCAL model — each agent uses its own MLX server
MINIMAX_IP   = "192.168.1.164" # MiniMax (optional)
PORT         = "8000"

# ── Brain Endpoints (Leadership Council) ──
# Each agent uses localhost as PRIMARY, but can call specialized brains for help
BRAINS = {
    "self":       "http://127.0.0.1:8000/v1/chat/completions",     # LOCAL — fast, use for routine reasoning
    "architect":  "http://10.255.255.2:8000/v1/chat/completions",   # sys2 M2U — GLM-4.7-Flash (architecture)
    "engineer":   "http://10.255.255.3:8000/v1/chat/completions",   # sys3 M2U — Qwen3-Coder-30B (engineering)
    "coder":      "http://10.255.255.4:8000/v1/chat/completions",   # sys4 M1U — Qwen3-Coder-Next-8bit (code gen)
    "designer":   "http://10.255.255.5:8000/v1/chat/completions",   # sys5 M1U — gpt-oss-120b (creative)
    "reviewer":   "http://10.255.255.6:8000/v1/chat/completions",   # sys6 M1U — Qwen3-Coder-30B (code review)
    "fast_coder": "http://10.255.255.7:8000/v1/chat/completions",   # sys7 M1U — Qwen3-Coder-30B (fast code)
    "cuda_brain": "http://10.255.255.11:8000/v1/chat/completions",  # CUDA 2x3090 — Qwen3.5-122B-FP8 (HARD problems only)
}

PRIMARY_URL  = f"http://{PRIMARY_IP}:{PORT}/v1"
CODER_URL    = BRAINS["coder"]

LOG_DIR = os.path.expanduser("~/AGENT/LOGS")
os.makedirs(LOG_DIR, exist_ok=True)

try:
    _r = requests.get(f"{PRIMARY_URL}/models", timeout=5)
    MODEL_NAME = _r.json()["data"][0]["id"]
except:
    MODEL_NAME = "qwen-local"

client = OpenAI(base_url=PRIMARY_URL, api_key="sk-local", timeout=600.0)

# =====================================================
# TOOLS
# =====================================================
def execute_bash(command: str, timeout: int = 30) -> str:
    try:
        cmd_str = command.strip()
        if cmd_str.endswith('&'):
            cmd_str = cmd_str.rstrip('&').strip()
            log_file = os.path.expanduser(f"~/AGENT/LOGS/bg_{int(time.time())}.log")
            full_cmd = f"nohup bash -c {subprocess.list2cmdline([cmd_str])} > {log_file} 2>&1 &"
            subprocess.Popen(full_cmd, shell=True, preexec_fn=os.setsid)
            time.sleep(1)
            return f"[BACKGROUND TASK LAUNCHED] Logs routing to: {log_file}"

        res = subprocess.run(f"bash -c {subprocess.list2cmdline([cmd_str])}", shell=True, capture_output=True, text=True, timeout=timeout)
        output = f"[EXIT CODE: {res.returncode}]\n"
        if res.stdout: output += res.stdout.strip()[-100000:]
        if res.stderr: output += f"\n[STDERR]: {res.stderr.strip()[-20000:]}"
        return output if len(output) > 20 else "[Command executed successfully with no output]"
    except subprocess.TimeoutExpired as e:
        return f"⚠️ [TIMEOUT after {timeout}s]\nSTDOUT: {(e.stdout or b'').decode('utf-8')[-5000:]}"
    except Exception as e:
        return f"[FATAL BASH ERROR]: {e}"

def read_file(path: str, offset: int = 0, limit: int = 100000) -> str:
    try:
        p = os.path.expanduser(path)
        # Fix Linux/macOS path mismatch: /home/z/ → /Users/z/ on macOS
        if not os.path.exists(p) and p.startswith("/home/z/"):
            p = p.replace("/home/z/", "/Users/z/", 1)
        if not os.path.exists(p) and p.startswith("/Users/z/"):
            p = p.replace("/Users/z/", "/home/z/", 1)
        with open(p, 'r', errors='ignore') as f:
            f.seek(offset)
            return f.read(limit)
    except Exception as e: return f"Error reading file: {e}"

def write_file(path: str, content: str) -> str:
    try:
        full_path = os.path.expanduser(path)
        # Fix Linux/macOS path mismatch
        if "/home/z/" in full_path and not os.path.exists(os.path.dirname(full_path)):
            full_path = full_path.replace("/home/z/", "/Users/z/", 1)
        if "/Users/z/" in full_path and not os.path.exists(os.path.dirname(full_path)):
            full_path = full_path.replace("/Users/z/", "/home/z/", 1)
        # Large file safety: if content >20 lines, warn (should use heredoc)
        if content.count(chr(10)) > 20:
            print(f"{C.YELLOW}[WARN] Large write_file ({content.count(chr(10))} lines) — consider heredoc next time{C.RESET}")
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f: f.write(content)
        return f"Successfully wrote {len(content)} chars to {full_path}"
    except Exception as e: return f"Error writing file: {e}"

def search_web(query: str) -> str:
    print(f"\n{C.MAGENTA}[🔍 Web Search: {query[:60]}...]{C.RESET}")
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=5)
        if not results: return "No results found."
        out = "\n\n".join(f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nURL: {r.get('href')}" for r in results)
        print(f"{C.GREEN}{out}{C.RESET}")
        return out
    except Exception as e: return f"Web Search Error: {e}"

def _ask_brain(brain_name: str, query: str, system_hint: str = "") -> str:
    """Generic brain query with streaming. Used by all ask_* functions."""
    url = BRAINS.get(brain_name)
    if not url:
        return f"Unknown brain: {brain_name}"
    ip = url.split("//")[1].split(":")[0]
    color = {"architect": C.RED, "engineer": C.YELLOW, "coder": C.GREEN,
             "designer": C.CYAN, "reviewer": C.GREEN, "fast_coder": C.MAGENTA,
             "cuda_brain": C.RED}.get(brain_name, C.MAGENTA)
    print(f"\n{color}[🧠 {brain_name.upper()} @ {ip}...]{C.RESET}")
    try:
        sys_msg = system_hint or f"You are an expert advising an autonomous agent. Do not use tool tags. Be precise. Project: Vulkan GPU inference on Asahi Linux."
        payload = {"messages": [{"role": "system", "content": sys_msg}, {"role": "user", "content": query}],
                   "max_tokens": 4000, "temperature": 0.2, "stream": True}
        full = ""
        t0 = time.time()
        print(f"{color}", end="", flush=True)
        with requests.post(url, json=payload, timeout=300, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line: continue
                line = line.decode("utf-8")
                if line.startswith("data: [DONE]"): break
                if line.startswith("data: "):
                    try:
                        chunk = json.loads(line[6:])
                        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            print(delta, end="", flush=True)
                            full += delta
                    except: pass
        elapsed = time.time() - t0
        toks = len(full.split())
        print(f"{C.RESET}")
        print(f"{C.DIM}[🧠 {brain_name.upper()}: ~{toks} words in {elapsed:.1f}s]{C.RESET}")
        return f"[{brain_name.upper()} ADVICE]:\n{full}"
    except Exception as e: return f"{brain_name.upper()} Call Failed: {e}"

def ask_coder_brain(query: str) -> str:
    return _ask_brain("coder", query, "You are a master C++/Python programmer. Provide exact code or logic. Do not use tool tags.")

def ask_architect(query: str) -> str:
    return _ask_brain("architect", query, "You are THE ARCHITECT. Give high-level design decisions as numbered bullet points. No code — systems thinking only.")

def ask_engineer(query: str) -> str:
    return _ask_brain("engineer", query, "You are THE ENGINEER. Give concrete implementation plans with file paths, function signatures, and step-by-step instructions.")

def ask_designer(query: str) -> str:
    return _ask_brain("designer", query, "You are THE DESIGNER. Propose creative alternative approaches. Challenge assumptions. Think unconventionally.")

def ask_reviewer(query: str) -> str:
    return _ask_brain("reviewer", query, "You are THE REVIEWER. Review the code/approach for correctness, safety, performance, and maintainability. Be thorough but not pedantic.")

def ask_cuda_brain(query: str) -> str:
    """Escalate to the 122B CUDA brain for genuinely HARD problems."""
    # Rate limit: track calls
    if not hasattr(ask_cuda_brain, '_count'):
        ask_cuda_brain._count = 0
    ask_cuda_brain._count += 1
    if ask_cuda_brain._count > 5:
        return "RATE LIMITED: You've called cuda_brain 5+ times. Use your LOCAL brain or ask_coder_brain instead."
    return _ask_brain("cuda_brain", query, "You are Qwen3.5-122B-FP8. Give thorough analysis. This was escalated because the local model couldn't handle it.")

def ask_minimax(query: str) -> str:
    print(f"\n{C.MAGENTA}[🧠 Pinging MiniMax Model at {MINIMAX_IP}...]{C.RESET}")
    try:
        payload = {"model": "mlx-community/MiniMax-M2-REAP-139B", "messages": [{"role": "user", "content": f"You are a master systems architect advising an autonomous agent. Provide exact logic. Do not use tool tags. We are running vLLM on Asahi Linux with a custom Vulkan backend.\n\nQuery: {query}"}], "max_tokens": 4000, "temperature": 0.2, "stream": True}
        full = ""
        t0 = time.time()
        print(f"{C.GREEN}", end="", flush=True)
        with requests.post(MINIMAX_URL, json=payload, timeout=300, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line: continue
                line = line.decode("utf-8")
                if line.startswith("data: [DONE]"): break
                if line.startswith("data: "):
                    try:
                        chunk = json.loads(line[6:])
                        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            print(delta, end="", flush=True)
                            full += delta
                    except: pass
        elapsed = time.time() - t0
        print(f"{C.RESET}")
        print(f"{C.DIM}[🧠 MiniMax: ~{len(full.split())} words in {elapsed:.1f}s]{C.RESET}")
        return f"[MINIMAX ADVICE]:\n{full}"
    except Exception as e: return f"MiniMax Call Failed: {e}"

def ask_claude(query: str) -> str:
    """Ask Claude Opus via CLI subprocess for architecture/debugging help."""
    print(f"\n{C.MAGENTA}[🧠 Asking Claude Opus...]{C.RESET}")
    try:
        import subprocess
        result = subprocess.run(
            ["claude", "-p", query],
            capture_output=True, text=True, timeout=180,
            cwd=os.path.expanduser("~/AGENT")
        )
        response = result.stdout.strip()
        print(f"{C.GREEN}{response[:2000]}{C.RESET}")
        return f"[CLAUDE ADVICE]:\n{response}"
    except Exception as ex:
        return f"Claude error: {ex}"

_AGENT_NAME = "OmniAgent [Main]"  # Set by run_agent() at startup

_claim_block_count = 0

def claim_task(task_id: str) -> str:
    """Claim a task from the queue so other agents don't work on it."""
    global _claim_block_count
    import subprocess

    # If we've been blocked 3+ times, refuse to even try
    if _claim_block_count >= 3:
        try:
            import urllib.request
            with urllib.request.urlopen("http://10.255.255.128:9091/tasks", timeout=2) as _r:
                _q = _r.read().decode()
            for _line in _q.split("\n"):
                if "IN_PROGRESS" in _line and _AGENT_NAME in _line:
                    _tm = re.match(r'###\s+(T\d+):', _line)
                    if _tm:
                        tid = _tm.group(1)
                        return f"REFUSED. You have been blocked {_claim_block_count} times. Your task is {tid}. STOP calling claim_task. Call execute_bash to work on {tid} RIGHT NOW."
        except:
            pass
        return f"REFUSED. Stop calling claim_task. Use execute_bash to work on your existing task."

    result = subprocess.run(
        ["bash", os.path.expanduser("~/AGENT/claim_task.sh"), task_id, _AGENT_NAME],
        capture_output=True, text=True, timeout=5
    )
    out = result.stdout.strip()
    print(f"{C.YELLOW}[📋 {out}]{C.RESET}")
    if "BLOCKED" in out:
        _claim_block_count += 1
        existing = re.search(r'already has (T\d+)', out)
        if existing:
            return f"BLOCKED — you already have {existing.group(1)}. STOP claiming. WORK on {existing.group(1)} NOW using execute_bash."
        return f"BLOCKED — you already have a task. STOP claiming. Use execute_bash to work on it."
    # Successful claim — reset counter
    _claim_block_count = 0
    return out

def complete_task(task_id: str) -> str:
    """Mark a task as DONE in the queue."""
    import subprocess
    result = subprocess.run(
        ["bash", os.path.expanduser("~/AGENT/complete_task.sh"), task_id, _AGENT_NAME],
        capture_output=True, text=True, timeout=5
    )
    out = result.stdout.strip()
    print(f"{C.GREEN}[✅ {out}]{C.RESET}")
    return out

def update_progress(task_id: str, pct: str, note: str = "") -> str:
    """Update task progress (0-100%) via central API. Call periodically to show progress."""
    import subprocess
    try:
        cmd = f'curl -s --max-time 3 -X POST http://10.255.255.128:9091/progress -d "task={task_id}&agent={_AGENT_NAME}&pct={pct}&note={note}"'
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        out = r.stdout.strip()
        print(f"{C.CYAN}[📊 {task_id}: {pct}%]{C.RESET}")
        return out
    except Exception as e:
        return f"Progress update failed: {e}"

def push_changes(files: str, message: str = "Agent work") -> str:
    """Push changed files back to sys1 (the git host). Files = comma-separated paths relative to ~/AGENT."""
    import subprocess
    SYS1_IP = "10.255.255.128"
    results = []
    for f in files.split(","):
        f = f.strip()
        if not f or ".." in f:
            results.append(f"{f}: SKIPPED (invalid)")
            continue
        local = os.path.expanduser(f"~/AGENT/{f}")
        if not os.path.exists(local):
            results.append(f"{f}: SKIPPED (not found)")
            continue
        try:
            subprocess.run(
                ["sshpass", "-f", os.path.expanduser("~/DEV/authpass"), "scp", "-o", "StrictHostKeyChecking=no",
                 local, f"z@{SYS1_IP}:~/AGENT/{f}"],
                capture_output=True, text=True, timeout=30
            )
            results.append(f"{f}: PUSHED to sys1")
        except Exception as e:
            results.append(f"{f}: FAILED ({e})")
    out = "\n".join(results)
    print(f"{C.GREEN}[📤 PUSH]\n{out}{C.RESET}")
    if any("PUSHED" in r for r in results):
        return out + "\n\n⚠️ IMPORTANT: Files pushed. Now call complete_task(task_id) to mark your task DONE. Do NOT claim a new task until you call complete_task."
    return out

def restart_self(reason: str = "update") -> str:
    """Gracefully restart this agent. The bash launcher will respawn it."""
    print(f"\n{C.YELLOW}[🔄 RESTARTING: {reason}]{C.RESET}")
    import sys
    sys.exit(42)  # Exit code 42 = intentional restart

def self_improve(description: str, code_patch: str = "") -> str:
    """Stage a self-improvement suggestion for the next agent version."""
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = os.path.expanduser("~/AGENT/SELF_IMPROVEMENTS.md")
    with open(log_path, "a") as f:
        f.write(f"\n## [{ts}] Self-Improvement Suggestion\n")
        f.write(f"**Description:** {description}\n")
        if code_patch:
            f.write(f"**Code:**\n```python\n{code_patch}\n```\n")
        f.write(f"**Status:** STAGED (apply in next version)\n\n")
    print(f"{C.GREEN}[📝 Self-improvement staged: {description[:60]}]{C.RESET}")
    return f"Staged improvement: {description[:100]}"

def ask_human(question: str) -> str:
    print(f"\n{C.BOLD}{C.YELLOW}[🤖 AGENT ASKS]: {question}{C.RESET}")
    return get_multiline_input("Your response (Ctrl+D to submit): ")

TOOL_DISPATCH = {
    "execute_bash": execute_bash, "read_file": read_file, "write_file": write_file,
    "search_web": search_web, "ask_coder_brain": ask_coder_brain,
    "ask_architect": ask_architect, "ask_engineer": ask_engineer,
    "ask_designer": ask_designer, "ask_reviewer": ask_reviewer,
    "ask_cuda_brain": ask_cuda_brain, "ask_minimax": ask_minimax, "ask_claude": ask_claude,
    "claim_task": claim_task, "complete_task": complete_task, "update_progress": update_progress, "push_changes": push_changes,
    "restart_self": restart_self, "self_improve": self_improve, "ask_human": ask_human
}

TOOLS_SCHEMA = [
    {"type": "function", "function": {"name": "execute_bash", "description": "Run shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write file. For files >20 lines, use execute_bash with cat<<'EOF' instead", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "ask_cuda_brain", "description": "Ask 122B for hard problems", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "ask_claude", "description": "Ask Claude when stuck", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "ask_coder_brain", "description": "Ask coder for code", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "ask_reviewer", "description": "Review code for bugs", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "claim_task", "description": "Claim a READY task", "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}}},
    {"type": "function", "function": {"name": "complete_task", "description": "Mark task DONE", "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}}},
    {"type": "function", "function": {"name": "push_changes", "description": "Push files to sys1", "parameters": {"type": "object", "properties": {"files": {"type": "string"}, "message": {"type": "string"}}, "required": ["files"]}}},
]

SYSTEM_PROMPT = """You are OmniAgent, an autonomous engineer. Execute tasks from ~/AGENT/TASK_QUEUE_v5.md.

RULES:
- Every response MUST contain ONLY a <tool_call>...</tool_call>. No thinking, no explanation, no reasoning outside the tags.
- Do NOT output chain-of-thought. Go straight to the tool call.
- Use ~ for paths (not /home/z or /Users/z).
- claim_task before working, complete_task + push_changes when done.
- If claim returns BLOCKED (you already have a task), WORK ON THAT TASK.
- If claim returns TAKEN, try the next READY task.
- Use execute_bash for commands, read_file for files (max 100 lines).
- Compile+test after code changes. If broken: git checkout -- filename
- For HARD problems: ask_cuda_brain. For code: ask_coder_brain.

TOOL FORMAT:
<tool_call>
{"name": "execute_bash", "arguments": {"command": "pwd"}}
</tool_call>

PROJECT: Vulkan GPU inference engine on Asahi Linux M1 Ultra.
For large files (>20 lines): use execute_bash with cat<<'EOF'>file.py ... EOF
Do NOT put large code blocks in write_file — it breaks JSON parsing.
- C engine: ~/AGENT/ggml_llama_gguf.c → libggml_llama_gguf.so (22 TPS)
- Python: ~/AGENT/ggml_vllm_backend.py
- Task queue: ~/AGENT/TASK_QUEUE_v5.md (ONLY this file, no v4)
"""

def _escape_newlines_in_json_strings(raw):
    """Escape literal newlines/tabs inside JSON string values only."""
    result = []
    in_string = False
    escaped = False
    for ch in raw:
        if escaped:
            result.append(ch)
            escaped = False
            continue
        if ch == '\\' and in_string:
            result.append(ch)
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string and ch == '\n':
            result.append('\\n')
            continue
        if in_string and ch == '\t':
            result.append('\\t')
            continue
        result.append(ch)
    return ''.join(result)

def _try_parse_tool_json(raw):
    """Try to parse a raw string as a tool call JSON, with progressive fixups."""
    # Remove trailing commas
    cleaned = re.sub(r',\s*([}\]])', r'\1', raw)
    # Replace curly quotes
    cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')

    # Fix missing colon after key names: "arguments { → "arguments": {
    cleaned = re.sub(
        r'"(name|arguments|command|path|query|task_id|pct|note|content|files|message|description|code_patch|question|offset|limit)"?\s*:\s',
        r'"\1": ', cleaned)

    # Progressive attempts — try least-destructive fixups first
    attempts = [
        cleaned,                                           # 1. minimal cleanup
        _escape_newlines_in_json_strings(cleaned),         # 2. escape newlines in strings
    ]

    # 3. If the JSON uses single quotes for keys (not in string values),
    #    carefully replace only structural single quotes (not those inside strings)
    sq_fixed = _fix_single_quoted_json(cleaned)
    if sq_fixed != cleaned:
        attempts.append(sq_fixed)
        attempts.append(_escape_newlines_in_json_strings(sq_fixed))

    for attempt in attempts:
        try:
            tc = json.loads(attempt)
            if isinstance(tc.get("arguments"), dict) and "arguments" in tc["arguments"]:
                tc["arguments"] = tc["arguments"]["arguments"]
            if tc.get("name") in TOOL_DISPATCH:
                return tc, None
            else:
                return None, f"Tool '{tc.get('name')}' does not exist."
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
            continue

    # 4. Last resort: try to extract command value with regex
    name_m = re.search(r'"name"\s*:\s*"(\w+)"', raw)
    cmd_m = re.search(r'"command"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
    if name_m and name_m.group(1) in TOOL_DISPATCH:
        name = name_m.group(1)
        if name == "execute_bash" and cmd_m:
            return {"name": name, "arguments": {"command": cmd_m.group(1).replace("\\n", "\n").replace('\\"', '"')}}, None
        # For other tools, try to extract all key-value pairs
        args = {}
        for km in re.finditer(r'"(\w+)"\s*:\s*"((?:[^"\\]|\\.)*)"', raw):
            k, v = km.group(1), km.group(2)
            if k not in ("name", "type"):
                args[k] = v.replace("\\n", "\n").replace('\\"', '"')
        if args:
            return {"name": name, "arguments": args}, None

    return None, f"Malformed JSON in tool call: {raw[:200]}"


def _fix_single_quoted_json(raw):
    """Replace single quotes used as JSON delimiters, preserving single quotes inside string values."""
    # Only replace ' with " if the string looks like it uses ' for JSON structure
    # e.g. {'name': 'execute_bash', 'arguments': {'command': 'ls'}}
    if not re.search(r"'\w+':\s*'", raw):
        return raw  # doesn't look like single-quoted JSON

    result = []
    in_string = False
    string_char = None
    i = 0
    while i < len(raw):
        ch = raw[i]
        if not in_string:
            if ch == "'":
                # Check if this is a JSON structural quote (before : or after { [ , :)
                result.append('"')
                in_string = True
                string_char = "'"
            elif ch == '"':
                result.append(ch)
                in_string = True
                string_char = '"'
            else:
                result.append(ch)
        else:
            if ch == '\\' and i + 1 < len(raw):
                result.append(ch)
                result.append(raw[i + 1])
                i += 2
                continue
            if ch == string_char:
                result.append('"' if string_char == "'" else ch)
                in_string = False
                string_char = None
            elif ch == '"' and string_char == "'":
                # Double quote inside single-quoted string — escape it
                result.append('\\"')
            else:
                result.append(ch)
        i += 1
    return ''.join(result)

def extract_tool_calls(text):
    calls = []
    errors = []
    # Primary: <tool_call>...</tool_call> tags
    for m in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL):
        tc, err = _try_parse_tool_json(m.group(1).strip())
        if tc:
            calls.append(tc)
        elif err:
            errors.append(err)
    # Fallback: bare JSON with "name" and "arguments" keys (no tags)
    if not calls:
        for m in re.finditer(r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}[^}]*\}', text):
            tc, err = _try_parse_tool_json(m.group(0).strip())
            if tc:
                calls.append(tc)
                break  # only take the first bare match to avoid false positives
            elif err:
                errors.append(err)
    return calls, errors

def get_multiline_input(prompt_text="Task (Ctrl+D to submit):"):
    print(f"\n{C.DIM}{'-'*50}{C.RESET}\n{C.GREEN}{C.BOLD}{prompt_text}{C.RESET}\n{C.DIM}{'-'*50}{C.RESET}")
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "EOF": break
            lines.append(line)
        except EOFError: break
    return "\n".join(lines).strip()

GO_PROMPT_PATH = os.path.expanduser("~/AGENT/GO_PROMPT.md")

def load_go_prompt():
    """Load autonomous execution prompt from GO_PROMPT.md"""
    try:
        with open(GO_PROMPT_PATH, "r") as f:
            content = f.read().strip()
        print(f"{C.BOLD}{C.GREEN}[GO] Loaded autonomous execution prompt from GO_PROMPT.md{C.RESET}")
        return content
    except FileNotFoundError:
        print(f"{C.RED}[GO] GO_PROMPT.md not found at {GO_PROMPT_PATH}{C.RESET}")
        return None

def run_agent(agent_name="OmniAgent [Main]", auto_go=False):
    global _AGENT_NAME
    _AGENT_NAME = agent_name
    print(f"{C.BOLD}{C.CYAN}🚀 {agent_name} ONLINE | Model: {MODEL_NAME}{C.RESET}")
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Session performance tracking
    session_total_tokens = 0
    session_total_turns = 0
    session_start_time = time.time()
    first_turn = True

    # Detect non-interactive (detached/nohup/screen) session — no TTY on stdin
    import sys as _sys
    no_tty = not _sys.stdin.isatty()
    # Strip ANSI colors when not on a TTY (fixes #36: clean log files)
    if no_tty:
        C.BLUE = C.CYAN = C.GREEN = C.YELLOW = C.MAGENTA = C.RED = C.DIM = C.RESET = C.BOLD = ''

    while True:
        try:
            # Auto-go: load GO_PROMPT only FIRST turn. After that, nudge to continue.
            if first_turn and (auto_go or no_tty):
                print(f"{C.BOLD}{C.MAGENTA}[AUTO-GO] Loading GO_PROMPT.md{C.RESET}")
                user_input = load_go_prompt()
                if not user_input:
                    user_input = "Read ~/AGENT/TASK_QUEUE_v5.md, claim the next [READY] task, execute it."
            elif no_tty:
                # Subsequent turns: find agent's FIRST claimed task and force focus
                my_task = ""
                my_task_desc = ""
                try:
                    import urllib.request
                    with urllib.request.urlopen("http://10.255.255.128:9091/tasks", timeout=2) as _r:
                        _q = _r.read().decode()
                    # Find FIRST IN_PROGRESS task for this agent using simple string search
                    # (regex escaping breaks on agent names with brackets like "OmniAgent [sys3]")
                    for _line in _q.split("\n"):
                        if "IN_PROGRESS" in _line and _AGENT_NAME in _line:
                            _tm = re.match(r'###\s+(T\d+):', _line)
                            if _tm:
                                my_task = _tm.group(1)
                                # Get the description from subsequent lines
                                _idx = _q.find(_line)
                                _rest = _q[_idx + len(_line):_idx + len(_line) + 300]
                                my_task_desc = _rest.strip().split("\n")[0][:200]
                                break
                except Exception:
                    pass
                # Check for external nudge file
                nudge_path = os.path.expanduser('~/AGENT/.nudge')
                if os.path.exists(nudge_path):
                    try:
                        nudge_msg = open(nudge_path).read().strip()
                        os.remove(nudge_path)
                        if nudge_msg:
                            user_input = f'[EXTERNAL NUDGE]: {nudge_msg}'
                            print(f"{C.MAGENTA}[NUDGE] {nudge_msg}{C.RESET}")
                            # Skip normal nudge logic
                            history.append({"role": "user", "content": user_input})
                            continue
                    except: pass
                if my_task:
                    user_input = (
                        f"Your assigned task is {my_task}. You MUST work on it now.\n"
                        f"Description: {my_task_desc}\n\n"
                        f"Call update_progress('{my_task}', 'XX') periodically to report progress.\n"
                        f"RESPOND WITH ONLY ONE TOOL CALL. Use execute_bash or write_file to make progress on {my_task}.\n"
                        f"DO NOT call claim_task. DO NOT call read_file on TASK_QUEUE. Just write code or run commands for {my_task}."
                    )
                else:
                    user_input = (
                        "You have no assigned task. Call claim_task with the ID of a READY task.\n"
                        "Pick from: T42, T43, T44, T45, T49, T50, T52, T54, T55, T59"
                    )
            else:
                user_input = get_multiline_input()

            first_turn = False
            # Log rotation: truncate if >1MB
            try:
                log_path = os.path.expanduser(f'~/AGENT/LOGS/agent_trace.log')
                if os.path.exists(log_path) and os.path.getsize(log_path) > 1_000_000:
                    with open(log_path, 'r') as f:
                        f.seek(max(0, os.path.getsize(log_path) - 500_000))
                        tail = f.read()
                    with open(log_path, 'w') as f:
                        f.write(tail)
            except: pass
            if not user_input: continue
            if user_input.lower() in ['exit', 'quit']: break

            # "go" keyword detection — any form: go, GO, GO!, go!, gogo, etc.
            if re.match(r'^go[!.\s]*$', user_input.strip(), re.IGNORECASE):
                go_content = load_go_prompt()
                if go_content:
                    user_input = go_content

            history.append({"role": "user", "content": user_input})
            
            while True: 
                # --- PERFECT CONTEXT MANAGEMENT ---
                if sum(len(str(m.get("content", ""))) for m in history) > 20000:
                    sys_prompt = history[0]
                    new_tail = history[-10:]
                    while new_tail and new_tail[0]["role"] != "user":
                        new_tail.pop(0)
                    history = [sys_prompt] + new_tail
                # --------------------------------------

                print(f"\n{C.DIM}[Thinking...]{C.RESET}", end="", flush=True)
                full_content = ""
                turn_token_count = 0

                try:
                    t_turn_start = time.time()
                    stream = client.chat.completions.create(model=MODEL_NAME, messages=history, stream=True, max_tokens=2000, temperature=0.3, extra_body={"repetition_penalty": 1.1})
                    t_first_token = None
                    print(f"\r{C.BOLD}{C.CYAN}[{agent_name}]: {C.RESET}", end="")
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            if t_first_token is None:
                                t_first_token = time.time()
                            token = chunk.choices[0].delta.content
                            print(f"{C.CYAN}{token}{C.RESET}", end="", flush=True)
                            full_content += token
                            turn_token_count += 1
                    t_turn_end = time.time()

                    # Perf stats
                    session_total_tokens += turn_token_count
                    session_total_turns += 1
                    turn_elapsed = t_turn_end - t_turn_start
                    ttft = (t_first_token - t_turn_start) if t_first_token else 0
                    tps = turn_token_count / turn_elapsed if turn_elapsed > 0 else 0
                    session_elapsed = t_turn_end - session_start_time

                    print(f"\n{C.DIM}┌─ 📊 Turn {session_total_turns}: {turn_token_count} tok | {tps:.1f} t/s | TTFT {ttft*1000:.0f}ms | {turn_elapsed:.1f}s{C.RESET}")
                    print(f"{C.DIM}└─ 📈 Session: {session_total_tokens} tok total | {session_total_tokens/session_elapsed:.1f} avg t/s | {session_elapsed:.0f}s uptime{C.RESET}")
                    print()
                except InterruptSignal:
                    print(f"\n{C.RED}[🛑 INTERRUPTED BY USER]{C.RESET}")
                    break
                except Exception as stream_err:
                    print(f"\n{C.RED}[⚠️ STREAM ERROR: {stream_err}]{C.RESET}")
                    err_str = str(stream_err)
                    if "context length" in err_str or "input_tokens" in err_str or "400" in err_str:
                        # Context overflow — trim history aggressively instead of retrying
                        print(f"{C.RED}[CONTEXT OVERFLOW — trimming history]{C.RESET}")
                        sys_prompt_msg = history[0]
                        history = [sys_prompt_msg] + history[-4:]
                        while history and history[1].get("role") != "user":
                            history.pop(1)
                        continue
                    print(f"{C.YELLOW}[🔄 Retrying in 5s...]{C.RESET}")
                    time.sleep(1)
                    if full_content:
                        history.append({"role": "assistant", "content": full_content + "\n[TRUNCATED]"})
                    history.append({"role": "user", "content": "[SYSTEM]: Connection error. Continue. Use a tool call."})
                    continue
                
                history.append({"role": "assistant", "content": full_content})
                tool_calls, parse_errors = extract_tool_calls(full_content)
                
                if parse_errors and not tool_calls:
                    # Count consecutive parse failures
                    if not hasattr(run_agent, '_parse_fail_count'):
                        run_agent._parse_fail_count = 0
                    run_agent._parse_fail_count += 1

                    if run_agent._parse_fail_count >= 2:
                        # Hard reset context instead of wasting Claude tokens
                        print(f"\n{C.RED}⚠️ [HARD RESET]: 3+ parse failures. Resetting context.{C.RESET}")
                        sys_msg = history[0]
                        last_user = [m for m in history if m["role"] == "user"][-1]
                        history = [sys_msg, last_user]
                        history.append({"role": "user", "content": "STOP. Do NOT explain. Respond with ONLY a tool call:\n<tool_call>\n{\"name\": \"execute_bash\", \"arguments\": {\"command\": \"pwd\"}}\n</tool_call>"})
                        run_agent._parse_fail_count = 0
                        continue

                    error_msg = "[SYSTEM]: Tool call JSON was malformed. CORRECT format:\n<tool_call>\n{\"name\": \"execute_bash\", \"arguments\": {\"command\": \"your command\"}}\n</tool_call>\nKey issue: use COLON (:) not EQUALS (=) between key and value.\n\nErrors: " + "\n".join(parse_errors)
                    print(f"\n{C.RED}⚠️ [AGENT SELF-CORRECTION]: Tool parsing failed ({run_agent._parse_fail_count}/3). Feeding correction...{C.RESET}")
                    history.append({"role": "user", "content": error_msg})
                    continue

                if not tool_calls:
                    # No tool call found at all (pure reasoning / no JSON)
                    if not hasattr(run_agent, '_no_tool_count'):
                        run_agent._no_tool_count = 0
                    run_agent._no_tool_count += 1
                    if run_agent._no_tool_count >= 2:
                        # Escalate to Claude after 2 empty turns
                        print(f"\n{C.RED}⚠️ [STUCK]: 2 responses with no tool call. Asking Claude for help.{C.RESET}")
                        try:
                            my_task = ""
                            try:
                                import urllib.request as _ur
                                with _ur.urlopen("http://10.255.255.128:9091/tasks", timeout=2) as _r:
                                    _q = _r.read().decode()
                                _m = re.search(rf'### (T\d+):.*?\[IN_PROGRESS by [^\]]*{re.escape(_AGENT_NAME)}', _q)
                                if _m: my_task = _m.group(1)
                            except: pass
                            fix = ask_claude(f"I'm stuck on {my_task}. My last output had no tool calls. What execute_bash command should I run next? Give me ONE specific command.")
                            history.append({"role": "user", "content": f"[CLAUDE]: {fix}\n\nNow execute it with a tool call."})
                        except:
                            history.append({"role": "user", "content": "Output ONLY a tool call. Example:\n<tool_call>\n{\"name\": \"execute_bash\", \"arguments\": {\"command\": \"ls ~/AGENT/\"}}\n</tool_call>"})
                        run_agent._no_tool_count = 0
                        continue
                    history.append({"role": "user", "content": "You MUST use a tool call. Example:\n<tool_call>\n{\"name\": \"execute_bash\", \"arguments\": {\"command\": \"pwd\"}}\n</tool_call>"})
                    continue

                # Auto-progress: update every 10 tool calls
                if not hasattr(run_agent, '_tool_count'):
                    run_agent._tool_count = 0
                run_agent._tool_count += len(tool_calls)
                if run_agent._tool_count % 10 == 0:
                    try:
                        import urllib.request as _ur
                        with _ur.urlopen('http://10.255.255.128:9091/tasks', timeout=2) as _r:
                            _q = _r.read().decode()
                        _m = re.search(rf'### (T\d+):.*?\[IN_PROGRESS by [^\]]*{re.escape(_AGENT_NAME)}', _q)
                        if _m:
                            pct = min(90, run_agent._tool_count * 2)
                            update_progress(_m.group(1), str(pct), f't{run_agent._tool_count}')
                    except: pass
                # Reset parse failure counter on success
                run_agent._parse_fail_count = 0

                results_msg = ""
                for tc in tool_calls:
                    fn_name, args = tc["name"], tc.get("arguments", {})
                    print(f"\n{C.BOLD}{C.YELLOW}🔧 [EXECUTING]: {fn_name}{C.RESET}")
                    try:
                        res = TOOL_DISPATCH[fn_name](**args)
                        res_str = str(res)
                        # Remove truncating block entirely so LLM reads the full file
                        if len(res_str) > 100000: res_str = res_str[:50000] + "\n...[TRUNCATED]...\n" + res_str[-50000:]
                        # Remove terminal screen limits so human sees exactly what the LLM sees
                        print(f"{C.GREEN}📥 [RESULT]:\n{res_str}{C.RESET}")
                        results_msg += f"[RESULT FROM {fn_name}]:\n{res_str}\n\n"
                    except Exception as e:
                        results_msg += f"[ERROR IN {fn_name}]: {e}\n\n"
                
                history.append({"role": "user", "content": results_msg})

        except InterruptSignal:
            print(f"\n{C.RED}[🛑 LOOP ABORTED. Ready for new task.]{C.RESET}")
            continue
        except Exception as e:
            print(f"\n{C.RED}[FATAL ERROR]: {e}{C.RESET}")
            traceback.print_exc()
            time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniAgent v4")
    parser.add_argument("--name", type=str, default="OmniAgent [Main]", help="Name of the agent instance")
    parser.add_argument("--auto-go", action="store_true", help="Auto-load GO_PROMPT.md on startup (used by restart loop)")
    args = parser.parse_args()
    run_agent(agent_name=args.name, auto_go=args.auto_go)
