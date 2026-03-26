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
def execute_bash(command: str, timeout: int = 120) -> str:
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
        with open(os.path.expanduser(path), 'r', errors='ignore') as f:
            f.seek(offset)
            return f.read(limit)
    except Exception as e: return f"Error reading file: {e}"

def write_file(path: str, content: str) -> str:
    try:
        full_path = os.path.expanduser(path)
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
    """Escalate to the 122B CUDA brain for genuinely HARD problems. Don't use for routine questions."""
    return _ask_brain("cuda_brain", query, "You are the most capable reasoning model in this cluster (Qwen3.5-122B-FP8 on 2x3090). Give thorough, high-quality analysis. This question was escalated because the local model couldn't handle it.")

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

def claim_task(task_id: str) -> str:
    """Claim a task from the queue so other agents don't work on it."""
    import subprocess
    result = subprocess.run(
        ["bash", os.path.expanduser("~/AGENT/claim_task.sh"), task_id, _AGENT_NAME],
        capture_output=True, text=True, timeout=5
    )
    out = result.stdout.strip()
    print(f"{C.YELLOW}[📋 {out}]{C.RESET}")
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
    "claim_task": claim_task, "complete_task": complete_task, "push_changes": push_changes,
    "restart_self": restart_self, "self_improve": self_improve, "ask_human": ask_human
}

TOOLS_SCHEMA = [
    {"type": "function", "function": {"name": "execute_bash", "description": "Run bash commands.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read local file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write to file. Great for scripts or work packages.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "search_web", "description": "Search the internet.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "ask_coder_brain", "description": "Ask the CODER brain (mlx-4) for C++/Python code generation. FAST.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "ask_architect", "description": "Ask THE ARCHITECT (mlx-2, 235B-Thinking) for high-level system design decisions. Use for big decisions.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "ask_engineer", "description": "Ask THE ENGINEER (mlx-3, 122B) for implementation plans, debugging, optimization strategies.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "ask_designer", "description": "Ask THE DESIGNER (mlx-5, GLM-4.7) for creative alternative approaches when stuck.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "ask_reviewer", "description": "Ask THE REVIEWER (sys6, Qwen3-Coder-30B) to review code for bugs, safety, correctness. Use before committing.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "ask_cuda_brain", "description": "ESCALATE to the 122B CUDA brain (2x3090, Qwen3.5-122B-FP8). For GENUINELY HARD problems only — architecture, multi-file bugs, complex reasoning. Do NOT use for routine questions.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "ask_minimax", "description": f"Ask the {MINIMAX_IP} MiniMax model for complex systems architecture advice.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "ask_claude", "description": "Ask Claude Opus (smartest AI) for architecture/debugging. USE SPARINGLY — costs tokens.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "self_improve", "description": "Stage a self-improvement for the next agent version. Log bugs, missing features, or code patches you want applied later.", "parameters": {"type": "object", "properties": {"description": {"type": "string", "description": "What to improve"}, "code_patch": {"type": "string", "description": "Optional Python code to add/change"}}, "required": ["description"]}}},
    {"type": "function", "function": {"name": "claim_task", "description": "BEFORE starting any task: claim it so other agents don't work on it. Returns CLAIMED or TAKEN.", "parameters": {"type": "object", "properties": {"task_id": {"type": "string", "description": "Task ID like T07, T08"}}, "required": ["task_id"]}}},
    {"type": "function", "function": {"name": "complete_task", "description": "AFTER finishing a task: mark it DONE in the queue.", "parameters": {"type": "object", "properties": {"task_id": {"type": "string", "description": "Task ID like T07"}}, "required": ["task_id"]}}},
    {"type": "function", "function": {"name": "push_changes", "description": "AFTER making code changes: push modified files back to sys1 (git host) so they get committed. Files = comma-separated paths relative to ~/AGENT.", "parameters": {"type": "object", "properties": {"files": {"type": "string", "description": "Comma-separated file paths, e.g. 'ggml_llama_gguf.c,ggml_vllm_backend.py'"}, "message": {"type": "string", "description": "Description of what changed"}}, "required": ["files"]}}},
    {"type": "function", "function": {"name": "ask_human", "description": "Ask the human for help. LAST RESORT only.", "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}}
]

SYSTEM_PROMPT = f"""You are OmniAgent v4, an elite autonomous systems engineer running on Apple M1 Ultra 128GB with Asahi Linux and Vulkan GPU.

## YOUR MISSION
Execute tasks from ~/AGENT/TASK_QUEUE_v4.md and ~/AGENT/TASK_QUEUE_v5.md. After completing each task, git commit and immediately start the next one. NEVER STOP between tasks. NEVER wait for human input.

## CRITICAL RULES
1. ALWAYS USE TOOLS. Every response MUST contain at least one tool_call. If you have nothing to execute, read the task queue.
2. NEVER read files larger than 100 lines. Use grep, head, tail, sed instead.
3. NEVER rewrite entire files. Use sed for targeted edits. If you need multi-line changes, use a heredoc patch.
4. After ANY code change: compile, test, verify. If broken, restore: git checkout -- filename
5. Git commit after every completed task with a descriptive message.
6. Update ~/AGENT/agent-comms-bridge.md after every completed task.
7. USE YOUR LOCAL BRAIN FIRST — you have a LOCAL MLX model on THIS node (localhost:8000).
   Your main reasoning loop already uses it. For HARD problems, escalate to the Leadership Council:

   ESCALATION RULES:
   - ROUTINE work (read files, run commands, small edits): just DO IT, no brain call needed
   - MODERATE complexity (implementation plans, debugging): ask your LOCAL brain (already your primary)
   - HARD problems (architecture decisions, complex bugs): ask_architect or ask_coder_brain
   - VERY HARD (multi-system issues, critical bugs): ask_cuda_brain (122B on CUDA, most capable)
   - BEFORE committing code: ask_reviewer (catches bugs others miss)
   - WHEN STUCK >5 min: ask_designer for creative alternatives

   DO NOT call cuda_brain for routine questions — it's shared across 8+ agents.
   Call it for genuinely difficult reasoning that your local model can't handle.

## YOUR BRAINS (Leadership Council)
- LOCAL (localhost:8000): YOUR primary brain — fast, use for everything routine
- ask_architect: System design, architecture (sys2, GLM-4.7-Flash)
- ask_engineer: Implementation plans, debugging (sys3, Qwen3-Coder-30B)
- ask_coder_brain: Write code, fix errors (sys4, Qwen3-Coder-Next-8bit)
- ask_designer: Creative solutions, alternatives (sys5, gpt-oss-120b)
- ask_reviewer: Code review, bug catching (sys6, Qwen3-Coder-30B)
- ask_cuda_brain: HARD problems only (CUDA .11, Qwen3.5-122B-FP8, 145 TPS)
- ask_claude: Last resort (EXPENSIVE, use sparingly)

## ENVIRONMENT
- Working dir: ~/AGENT
- C engine: ~/AGENT/ggml_llama_gguf.c (compiles to libggml_llama_gguf.so)
- Python API: ~/AGENT/ggml_vllm_backend.py
- Server: ~/AGENT/ggml_server.py
- Models: ~/models/gguf/ (8B Q4=4.6G, 32B Q4=19G, 120B mxfp4=60G)
- Build cmd: gcc -shared -O2 -fPIC -o libggml_llama_gguf.so ggml_llama_gguf.c -I ~/GITDEV/llama.cpp/ggml/include -L ~/GITDEV/llama.cpp/build-lib/bin -lggml -lggml-base -lggml-vulkan -lggml-cpu -lm -Wl,-rpath,/home/z/GITDEV/llama.cpp/build-lib/bin
- Test cmd: python3 -c "from ggml_vllm_backend import GgmlLLM, SamplingParams; llm = GgmlLLM('~/models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'); r = llm.generate('Capital of France?', params=SamplingParams(temperature=0, max_tokens=10)); print(f'{{r.tps:.0f}} TPS: {{r.text}}')"
- Status: 22 TPS on 8B Q4, coherent output, engine stable
- llama.cpp reference: 24.7 TPS (our ceiling target)

## CURRENT PRIORITIES (read TASK_QUEUE_v4.md for details)
- T07: Add MoE support for gpt-oss-120b (128 experts, 4 active). Study ~/GITDEV/llama.cpp/src/models/openai-moe-iswa.cpp
- T08: Fix tokenizer discovery for all model families
- T09: Benchmark 32B Qwen with tokenizer fix
- T10: 50-request stress test

## TOOL FORMAT (CRITICAL — use COLON not EQUALS)
<tool_call>
{{"name": "tool_name", "arguments": {{"key": "value"}}}}
</tool_call>

CORRECT: {{"name": "execute_bash", "arguments": {{"command": "pwd"}}}}
WRONG:   {{"name"="execute_bash", "arguments"={{"command": "pwd"}}}}

## MULTI-AGENT COORDINATION
- BEFORE starting any task: call claim_task("T07") — if it returns TAKEN, skip to next READY task
- AFTER completing any task: call complete_task("T07")
- Check TASK_QUEUE_v4.md for [IN_PROGRESS by ...] to see what other agents are doing
- Do NOT work on tasks claimed by other agents

## ON STARTUP
1. Read ~/AGENT/TASK_QUEUE_v4.md — check for tasks [IN_PROGRESS by YOUR NAME]
2. If you have an IN_PROGRESS task, RESUME it (don't re-claim)
3. If no IN_PROGRESS task, find next [READY] task and claim_task it
4. Also check ~/AGENT/TASK_QUEUE_v5.md for additional READY tasks
5. Read ~/AGENT/KNOWLEDGE_BASE.md for context
6. NEVER work on tasks claimed by other agents
7. Only use ONE queue — prefer v4 tasks first, then v5

## AVAILABLE TOOLS
{json.dumps(TOOLS_SCHEMA, indent=2)}
"""

def extract_tool_calls(text):
    calls = []
    errors = []
    for m in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL):
        raw = m.group(1).strip()
        # Fix common model output issues: trailing commas, single quotes
        raw = re.sub(r',\s*([}\]])', r'\1', raw)  # remove trailing commas
        raw = raw.replace("'", '"')  # single to double quotes
        try:
            tc = json.loads(raw)
            # Anti-Hallucination Check: flatten nested arguments if model hallucinates them
            if "arguments" in tc.get("arguments", {}):
                tc["arguments"] = tc["arguments"]["arguments"]
            if tc.get("name") in TOOL_DISPATCH: 
                calls.append(tc)
            else:
                errors.append(f"Tool '{tc.get('name')}' does not exist.")
        except json.JSONDecodeError as e:
            try:
                fixed = raw.replace('\n', '\\n')
                tc = json.loads(fixed)
                if "arguments" in tc.get("arguments", {}):
                    tc["arguments"] = tc["arguments"]["arguments"]
                if tc.get("name") in TOOL_DISPATCH: 
                    calls.append(tc)
                else:
                    errors.append(f"Tool '{tc.get('name')}' does not exist.")
            except Exception as e2: 
                errors.append(f"Malformed JSON in tool call: {raw}\nError: {e2}")
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

    while True:
        try:
            # Auto-go: load GO_PROMPT only FIRST turn. After that, nudge to continue.
            if first_turn and (auto_go or no_tty):
                print(f"{C.BOLD}{C.MAGENTA}[AUTO-GO] Loading GO_PROMPT.md{C.RESET}")
                user_input = load_go_prompt()
                if not user_input:
                    user_input = "Read ~/AGENT/TASK_QUEUE_v5.md, claim the next [READY] task, execute it."
            elif no_tty:
                # Subsequent turns: brief nudge, NOT the full GO_PROMPT again
                user_input = "Continue. If task done: push_changes, complete_task, claim next. If stuck: ask_cuda_brain."
            else:
                user_input = get_multiline_input()

            first_turn = False
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
                if sum(len(str(m.get("content", ""))) for m in history) > 30000:
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
                    print(f"{C.YELLOW}[🔄 Retrying in 5s...]{C.RESET}")
                    time.sleep(5)
                    if full_content:
                        history.append({"role": "assistant", "content": full_content + "\n[TRUNCATED BY CONNECTION ERROR]"})
                    history.append({"role": "user", "content": "[SYSTEM]: Previous response was cut off by a connection error. Continue where you left off. Use a tool call."})
                    continue
                
                history.append({"role": "assistant", "content": full_content})
                tool_calls, parse_errors = extract_tool_calls(full_content)
                
                if parse_errors and not tool_calls:
                    # Count consecutive parse failures
                    if not hasattr(run_agent, '_parse_fail_count'):
                        run_agent._parse_fail_count = 0
                    run_agent._parse_fail_count += 1

                    if run_agent._parse_fail_count >= 3:
                        # Break the loop — trim history to escape bad pattern
                        print(f"\n{C.RED}⚠️ [AGENT RESET]: 3+ parse failures. Clearing bad pattern from context.{C.RESET}")
                        # Keep system + last user message, drop the broken assistant turns
                        sys_msg = history[0]
                        last_user = [m for m in history if m["role"] == "user"][-1]
                        history = [sys_msg, last_user]
                        history.append({"role": "user", "content": "[SYSTEM]: Your previous tool calls had broken JSON syntax. The correct format is:\n<tool_call>\n{\"name\": \"execute_bash\", \"arguments\": {\"command\": \"pwd\"}}\n</tool_call>\nNote: use COLON after key names, not equals sign. Try again."})
                        run_agent._parse_fail_count = 0
                        continue

                    error_msg = "[SYSTEM]: Tool call JSON was malformed. CORRECT format:\n<tool_call>\n{\"name\": \"execute_bash\", \"arguments\": {\"command\": \"your command\"}}\n</tool_call>\nKey issue: use COLON (:) not EQUALS (=) between key and value.\n\nErrors: " + "\n".join(parse_errors)
                    print(f"\n{C.RED}⚠️ [AGENT SELF-CORRECTION]: Tool parsing failed ({run_agent._parse_fail_count}/3). Feeding correction...{C.RESET}")
                    history.append({"role": "user", "content": error_msg})
                    continue

                if not tool_calls:
                    break

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
