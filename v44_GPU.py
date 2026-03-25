"""
v44_GPU - Vulkan GPU Build Agent for vLLM on Asahi Linux M1 Max.
Powered by Qwen3.5-122B on 8x3090 cluster.

MISSION: Get vLLM 0.17.1 running with REAL Vulkan GPU inference.

ALREADY DONE (by Architect):
1. Installed Vulkan-capable PyTorch 2.12.0a0+git5de8e44 into .venv-vLLM_0.17.1_Stable
2. Fixed PyTorch descriptor pool (1024->65536) - large matmuls now work
3. Verified: Vulkan mm is 10-80x FASTER than CPU for batch>1 (prefill)
4. Created vulkan_cuda_shim.py (CUDA compatibility monkey-patches)
5. Updated vulkan.py platform with proper config
6. Wheel saved: ~/WHEELS/torch-2.12.0a0+git5de8e44-vulkan_pool_65536-cp312-cp312-linux_aarch64.whl

DONE BY AGENT (Session 1, 40 turns):
7. Added "float32" to CacheDType in vllm/config/cache.py
8. Fixed weight interceptor: cast to float32 BEFORE .to("vulkan")

REMAINING:
- Fix remaining server startup crashes (test launch, read errors, fix, repeat)
- Handle all torch.cuda.* calls in GPU worker/model runner
- Achieve working inference with coherent output
- Benchmark TPS

KEY FACTS:
- float32 Vulkan works. fp16/bf16 do NOT.
- Vulkan prefill (batch>1) is 10-80x faster than CPU
- Vulkan decode (batch=1) is ~3x slower than CPU
- CPU attention backend works
- venv: ~/.venv-vLLM_0.17.1_Stable
- vLLM src: ~/GITDEV/vllm_0.17.1
- Model: Qwen2.5-0.5B-Instruct (hidden=896, layers=24, intermediate=4864)
"""
import os
import sys
import time
import json
import subprocess
import re
import traceback
import requests
import signal
from datetime import datetime

class InterruptSignal(Exception):
    pass

last_sigint_time = 0
def sigint_handler(signum, frame):
    global last_sigint_time
    t = time.time()
    if t - last_sigint_time < 1.5:
        print("\n[DOUBLE CTRL+C: EXIT]")
        os._exit(1)
    last_sigint_time = t
    raise InterruptSignal()

signal.signal(signal.SIGINT, sigint_handler)

# =====================================================
# CONFIG
# =====================================================
CLUSTER_URL = "http://10.255.255.11:8000/v1/chat/completions"
VLLM_SRC = os.path.expanduser("~/GITDEV/vllm_0.17.1")
VENV = os.path.expanduser("~/.venv-vLLM_0.17.1_Stable")
MODEL_PATH = "/home/z/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
LOG_DIR = os.path.expanduser("~/AGENT/LOGS")
TRACE = os.path.expanduser("~/AGENT/v44_GPU_trace.log")
CONTROL_FILE = os.path.expanduser("~/AGENT/v44_control.json")

os.makedirs(LOG_DIR, exist_ok=True)

# Detect cluster model
try:
    r = requests.get("http://10.255.255.11:8000/v1/models", timeout=5)
    CLUSTER_MODEL = r.json()["data"][0]["id"]
except:
    CLUSTER_MODEL = "/vmrepo/models/Qwen3.5-122B-A10B-FP8"

# =====================================================
# TRACE
# =====================================================
def trace(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    with open(TRACE, "a") as f:
        f.write(f"[{ts}] {msg}\n")

# =====================================================
# CONTROL - allows external process to inject commands
# =====================================================
def check_control():
    """Check if external controller (Claude/human) sent a command.
    Write JSON to ~/AGENT/v44_control.json:
      {"action": "inject", "message": "Do X instead"}
      {"action": "kill"}
      {"action": "update_prompt", "prompt": "new system prompt addition"}
    """
    try:
        if os.path.exists(CONTROL_FILE):
            with open(CONTROL_FILE, 'r') as f:
                ctrl = json.load(f)
            os.remove(CONTROL_FILE)  # consume it
            trace(f"CONTROL: {json.dumps(ctrl)[:200]}")
            return ctrl
    except:
        pass
    return None

# =====================================================
# TOOLS
# =====================================================
def execute_bash(command, timeout=120):
    cmd = command.strip()
    trace(f"BASH: {cmd[:200]}")
    try:
        # SAFETY: never kill all python (kills the agent itself)
        if 'pkill' in cmd and 'python' in cmd and '-f vllm' not in cmd and '-f EngineCore' not in cmd:
            return "[BLOCKED] Never pkill python broadly - it kills this agent. Use: pkill -9 -f EngineCore; pkill -9 -f vllm"
        # SAFETY: before launching vllm, ALWAYS kill existing first
        if 'vllm serve' in cmd and 'pkill' not in cmd and 'grep' not in cmd:
            trace("AUTO-KILL: Killing existing vllm before launching new one")
            subprocess.run("pkill -9 -f EngineCore 2>/dev/null; pkill -9 -f 'vllm serve' 2>/dev/null; sleep 2", shell=True, timeout=10)
            # Verify memory
            mem_check = subprocess.run("free -h | head -2", shell=True, capture_output=True, text=True, timeout=5)
            trace(f"MEM before launch: {mem_check.stdout.strip()}")
        # Auto-detect background commands: ends with & OR contains nohup
        is_bg = cmd.endswith('&') or 'nohup ' in cmd
        if is_bg:
            cmd = cmd.rstrip('&').strip()
            # Strip tee - it blocks
            cmd = cmd.split('| tee')[0].strip()
            log = f"{LOG_DIR}/bg_{int(time.time())}.log"
            # Ensure output goes to log file
            if '>' not in cmd:
                cmd = f"{cmd} > {log} 2>&1"
            subprocess.Popen(f"bash -c {subprocess.list2cmdline([cmd])} &",
                           shell=True, preexec_fn=os.setsid,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)
            return f"[BG] Log: {log}\nUse execute_bash to poll: curl -s http://localhost:8000/v1/models"
        res = subprocess.run(f"bash -c {subprocess.list2cmdline([cmd])}",
                           shell=True, capture_output=True, text=True, timeout=int(timeout))
        out = f"[EXIT:{res.returncode}]\n"
        if res.stdout: out += res.stdout[-30000:]
        if res.stderr: out += f"\n[STDERR]:{res.stderr[-10000:]}"
        return out
    except subprocess.TimeoutExpired as e:
        return f"[TIMEOUT {timeout}s] {(e.stdout or b'')[-3000:].decode('utf-8','ignore')}"
    except Exception as e:
        return f"[ERROR]: {e}"

def read_file(path, offset=0, limit=200000):
    try:
        p = os.path.expanduser(path)
        with open(p, 'r', errors='ignore') as f:
            f.seek(int(offset or 0))
            return f.read(int(limit or 200000))
    except Exception as e:
        return f"Error: {e}"

def write_file(path, content):
    try:
        p = os.path.expanduser(path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w') as f:
            f.write(content)
        return f"Wrote {len(content)} chars to {p}"
    except Exception as e:
        return f"Error: {e}"

def search_web(query):
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=5)
        return "\n\n".join(f"[{i+1}] {r.get('title','')}\n{r.get('body','')}" for i,r in enumerate(results)) if results else "No results"
    except Exception as e:
        return f"Error: {e}"

def git_log(count=20):
    return execute_bash(f"git -C {VLLM_SRC} log --oneline -{count}")

def git_diff():
    return execute_bash(f"git -C {VLLM_SRC} diff --stat")

def git_show(target):
    return execute_bash(f"git -C {VLLM_SRC} show {target}")

def diagnostic():
    parts = []
    parts.append(execute_bash(f"source {VENV}/bin/activate && python3 -c \"import torch; print('torch:', torch.__version__); print('vulkan:', torch.is_vulkan_available())\""))
    parts.append(execute_bash("free -h | head -3"))
    parts.append(execute_bash("pgrep -af vllm 2>/dev/null || echo 'No vllm'"))
    parts.append(execute_bash(f"git -C {VLLM_SRC} status --short | head -20"))
    return "\n---\n".join(parts)

def ask_human(question):
    print(f"\n[AGENT ASKS]: {question}")
    lines = []
    print("Your answer (Ctrl+D or 'EOF'):")
    while True:
        try:
            line = input()
            if line.strip() == "EOF": break
            lines.append(line)
        except EOFError: break
    return "\n".join(lines).strip()

def call_claude(question, context=""):
    """Call Claude Cloud API (Anthropic) for architect-level help on blockers.
    Use this when you are STUCK and need deep reasoning about vLLM internals,
    PyTorch Vulkan bugs, or architectural decisions. Include full error traces."""
    try:
        import anthropic
        client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": f"""You are the architect for a vLLM Vulkan GPU project on Apple M1 Max Asahi Linux.

KEY CONTEXT:
- vLLM 0.17.1, PyTorch 2.12.0a0 with Vulkan support, Mesa AGX driver
- float32 only (fp16/bf16 crash on Vulkan), CPU attention backend
- torch.cuda.* monkey-patched via vulkan_cuda_shim.py
- Survival patches in _custom_ops.py, torch_utils.py, builtin.py must not be destroyed
- vLLM source: ~/GITDEV/vllm_0.17.1, venv: ~/.venv-vLLM_0.17.1_Stable

ADDITIONAL CONTEXT: {context}

QUESTION FROM BUILD AGENT:
{question}

Give EXACT code fixes or commands. Be surgical. No fluff."""}]
        )
        result = msg.content[0].text
        trace(f"CLAUDE_CALL: Q={question[:200]} A={result[:200]}")
        return f"[CLAUDE RESPONSE]:\n{result}"
    except ImportError:
        return "[ERROR]: anthropic package not installed. Run: pip install anthropic"
    except Exception as e:
        return f"[CLAUDE ERROR]: {e}"

# =====================================================
# BRIDGE - shared comms with Claude Code
# =====================================================
from claude_agent_bridge import send_to_claude, read_from_claude
_bridge_pos = 0

def msg_claude(msg_type, content):
    """Send message to Claude via bridge. Types: error, question, status, blocker"""
    send_to_claude(msg_type, content)
    return f"[SENT TO CLAUDE] type={msg_type} ({len(content)} chars). Claude will see this and may respond."

def check_bridge():
    """Check for new messages from Claude. Called automatically every turn."""
    global _bridge_pos
    msgs, _bridge_pos = read_from_claude(_bridge_pos)
    return msgs

TOOLS = {
    "execute_bash": execute_bash,
    "read_file": read_file,
    "write_file": write_file,
    "search_web": search_web,
    "git_log": git_log,
    "git_diff": git_diff,
    "git_show": git_show,
    "diagnostic": diagnostic,
    "ask_human": ask_human,
    "call_claude": call_claude,
    "msg_claude": msg_claude,
}

TOOL_SCHEMA = [
    {"type":"function","function":{"name":"execute_bash","description":"Run bash command. '&' suffix for background.","parameters":{"type":"object","properties":{"command":{"type":"string"},"timeout":{"type":"integer"}},"required":["command"]}}},
    {"type":"function","function":{"name":"read_file","description":"Read file contents.","parameters":{"type":"object","properties":{"path":{"type":"string"},"offset":{"type":"integer"},"limit":{"type":"integer"}},"required":["path"]}}},
    {"type":"function","function":{"name":"write_file","description":"Write to file. ONLY for NEW files or small scripts. NEVER overwrite large existing Python files - use execute_bash with a Python replace() script instead.","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
    {"type":"function","function":{"name":"search_web","description":"Web search.","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
    {"type":"function","function":{"name":"git_log","description":"Show git log of vllm_0.17.1.","parameters":{"type":"object","properties":{"count":{"type":"integer"}}}}},
    {"type":"function","function":{"name":"git_diff","description":"Show git diff stats.","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"git_show","description":"Git show commit/file.","parameters":{"type":"object","properties":{"target":{"type":"string"}},"required":["target"]}}},
    {"type":"function","function":{"name":"diagnostic","description":"Full system diagnostic.","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"ask_human","description":"Ask operator a question.","parameters":{"type":"object","properties":{"question":{"type":"string"}},"required":["question"]}}},
    {"type":"function","function":{"name":"call_claude","description":"Call Claude Cloud API for architect-level help. Use when STUCK on deep vLLM/PyTorch/Vulkan issues after 3+ failed attempts. Include full error traces and what you already tried.","parameters":{"type":"object","properties":{"question":{"type":"string","description":"Your detailed question with error traces"},"context":{"type":"string","description":"Brief context of what you were trying"}},"required":["question"]}}},
    {"type":"function","function":{"name":"msg_claude","description":"Send a message to Claude Code (the architect watching your logs). Use for: errors you cant solve, questions about architecture, status updates on milestones. Types: error, question, status, blocker","parameters":{"type":"object","properties":{"msg_type":{"type":"string","enum":["error","question","status","blocker"]},"content":{"type":"string"}},"required":["msg_type","content"]}}},
]

# =====================================================
# SYSTEM PROMPT
# =====================================================
SYSTEM_PROMPT = f"""You are Z-Alpha v44_GPU, an elite autonomous build agent on sys12 (Apple M1 Max, 32GB, Asahi Linux).

## YOUR MISSION
Get vLLM 0.17.1 running with REAL Vulkan GPU inference on this machine.

## WHAT THE ARCHITECT ALREADY DID
1. Installed Vulkan-capable PyTorch (2.12.0a0+git5de8e44) - torch.is_vulkan_available()=True
2. Fixed PyTorch Vulkan descriptor pool: 1024 -> 65536 (large matmuls now work)
3. Benchmarked: Vulkan is 10-80x FASTER than CPU for prefill (batch>1)
4. Created ~/GITDEV/vllm_0.17.1/vllm/platforms/vulkan_cuda_shim.py (torch.cuda monkey-patches)
5. Updated vulkan.py platform with CUDA shim import
6. The weight interceptor in default_loader.py moves weights to Vulkan

## WHAT YOU DID IN SESSION 1
7. Added "float32" to CacheDType in vllm/config/cache.py
8. Fixed weight interceptor: cast to float32 BEFORE .to("vulkan") in default_loader.py

## KEY CONSTRAINTS
- float32 ONLY on Vulkan (fp16/bf16 crash with "Unsupported dtype" in Packing.cpp)
- CPU attention backend (Vulkan lacks attention ops)
- No pin_memory (crashes without NVIDIA driver)
- All torch.cuda.* calls must go through the shim (vulkan_cuda_shim.py)
- DO NOT destroy survival patches in _custom_ops.py, torch_utils.py, builtin.py
- DO NOT load models >0.5B without approval
- **NEVER pip install/uninstall torch, torchaudio, or torchvision** - our torch is a CUSTOM Vulkan build from ~/GITDEV/pytorch. Installing ANY torch package from pip will DESTROY Vulkan support.
- **ALWAYS set PYTHONPATH=/home/z/GITDEV/vllm_0.17.1:$PYTHONPATH** before running vllm (the editable install finder is broken)
- **ALWAYS verify Vulkan before launching**: python -c "import torch; assert torch.is_vulkan_available()"

## ENVIRONMENT
- venv: {VENV}
- vLLM source: {VLLM_SRC} (editable install)
- Model: {MODEL_PATH}
- Always activate venv: source {VENV}/bin/activate
- Set VLLM_PLATFORM=vulkan for all vllm commands
- Server logs go to ~/AGENT/LOGS/

## YOUR APPROACH
1. First run diagnostic() to see current state
2. Read error logs carefully - fix ONE issue at a time
3. CRITICAL: To modify existing Python files, use a Python replace() script via execute_bash:
   python3 -c "
   path='the/file.py'
   with open(path) as f: c=f.read()
   c=c.replace('old_exact_text', 'new_text')
   with open(path,'w') as f: f.write(c)
   print('patched')
   "
   NEVER use write_file to overwrite a large existing file - you WILL lose the rest of the code.
4. After each fix, test by launching vllm server
5. Before launching: pkill -9 -f vllm; pkill -9 -f EngineCore; check free -h
6. Launch in foreground with timeout to capture errors:
   cd {VLLM_SRC} && source {VENV}/bin/activate && VLLM_PLATFORM=vulkan OMP_NUM_THREADS=10 timeout 30 vllm serve --model Qwen/Qwen2.5-0.5B-Instruct --host 0.0.0.0 --port 8000 --dtype float32 --max-model-len 1024 --enforce-eager --gpu-memory-utilization 0.8 2>&1 | tail -100
7. When server works, test inference quality with curl
8. If STUCK for >2 turns on same issue, use call_claude() for architect help - include the FULL error trace
9. Use search_web() to look up PyTorch Vulkan issues, vLLM platform porting, etc
10. EVERY response MUST contain at least one <tool_call>. NEVER just explain without acting.
11. NEVER sleep more than 2 seconds. The server loads in <7 seconds. Poll with curl immediately.
12. Launch server in BACKGROUND with nohup, then poll with: for i in $(seq 1 15); do curl -s http://localhost:8000/v1/models && break; sleep 1; done
13. If tool_call JSON fails to parse, the system will auto-nudge you - just retry with valid JSON
14. NEVER use timeout command to wrap the server. Launch with nohup and let it run.
15. MANDATORY: After every significant finding, call msg_claude("status", "what you found"). The architect is monitoring.
16. MANDATORY: When you hit a blocker, call msg_claude("blocker", "description + full error"). Do NOT spin on the same error.
17. NEVER launch a second vllm server without killing the first. ALWAYS: pkill -9 -f EngineCore; pkill -9 -f vllm; sleep 2; free -h
18. You run FOREVER. Fight for the solution. Never give up. If one approach fails after 5 attempts, try a completely different angle.
19. Before each server launch, verify memory is free (>25GB) with free -h. If not, kill zombies first.

## TOOL FORMAT
<tool_call>
{{"name": "tool_name", "arguments": {{"key": "value"}}}}
</tool_call>

AVAILABLE TOOLS:
{json.dumps(TOOL_SCHEMA, indent=2)}
"""

# =====================================================
# LLM CALL
# =====================================================
def call_llm(messages, max_tokens=8192):
    payload = {
        "model": CLUSTER_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    r = requests.post(CLUSTER_URL, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# =====================================================
# PARSER
# =====================================================
def parse_tools(text):
    calls = []
    # Method 1: Standard <tool_call> tags
    for m in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL):
        raw = m.group(1).strip()
        tc = _try_parse_json(raw)
        if tc:
            name = tc.get("name")
            args = tc.get("arguments", {})
            if isinstance(args, str):
                try: args = json.loads(args)
                except: args = {}
            if name in TOOLS:
                calls.append((name, args))

    # Method 2: Malformed - JSON outside tags or with broken syntax
    if not calls:
        # Try to find any {"name": "tool_name" pattern
        for m in re.finditer(r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})', text):
            name = m.group(1)
            args_str = m.group(2)
            if name in TOOLS:
                tc = _try_parse_json(args_str)
                if tc:
                    calls.append((name, tc))

    return calls

def _try_parse_json(raw):
    """Try multiple strategies to parse JSON."""
    for attempt in [
        lambda s: json.loads(s),
        lambda s: json.loads(s.replace('\n', '\\n')),
        lambda s: json.loads(re.sub(r',\s*}', '}', s)),  # trailing comma
        lambda s: json.loads(re.sub(r'(\w+)(?=\s*:)', r'"\1"', s)),  # unquoted keys
    ]:
        try:
            return attempt(raw)
        except:
            continue
    trace(f"PARSE_FAIL: {raw[:200]}")
    return None


# =====================================================
# STUCK DETECTOR
# =====================================================
_stuck_counter = 0
_last_error = ""

def detect_stuck(result_text):
    """Detect if agent is stuck on the same error. Auto-escalate to Claude."""
    global _stuck_counter, _last_error
    # Extract error signatures
    errors = re.findall(r'(?:Error|Exception|Traceback|FAILED|AssertionError)[^\n]{0,200}', result_text)
    error_sig = "|".join(errors[:3]) if errors else ""

    if error_sig and error_sig == _last_error:
        _stuck_counter += 1
    else:
        _stuck_counter = 0
        _last_error = error_sig

    if _stuck_counter >= 3:
        _stuck_counter = 0
        return True
    return False

# =====================================================
# MAIN LOOP
# =====================================================
def main():
    print(f"\n{'='*60}")
    print(f"  v44_GPU Agent | Cluster: {CLUSTER_MODEL}")
    print(f"  Mission: Vulkan GPU Inference on M1 Max")
    print(f"  Control: echo '{{\"action\":\"inject\",\"message\":\"...\"}}' > ~/AGENT/v44_control.json")
    print(f"  Trace:   tail -f ~/AGENT/v44_GPU_trace.log")
    print(f"{'='*60}\n")
    trace("SESSION START")

    # Aggregate telemetry
    total_tokens_gen = 0
    total_words_gen = 0
    total_tokens_read = 0
    session_start = time.perf_counter()

    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Auto-start
    start_msg = """Resume mission. The server STARTS AND SERVES REQUESTS. That is DONE. DO NOT relaunch it unless it crashes.

CURRENT PROBLEM: Generation throughput is only 3-7 TPS. Target is 20+ TPS. The previous CPU-only config achieved 20 TPS.

CRITICAL CLUE: "DRM_IOCTL_ASAHI_GEM_BIND failed: Cannot allocate memory" appeared during weight loading. This means the Vulkan GPU memory is being exhausted. The weights may be DOUBLE-ALLOCATED (once on CPU, once on Vulkan).

YOUR TASK NOW:
1. Check if a server is already running: curl -s http://localhost:8000/v1/models
2. If yes, test inference and measure TPS precisely (use time + token count)
3. If no, launch ONE server in background and wait for it
4. INVESTIGATE why TPS is low:
   - Are weights actually on Vulkan or did they fall back to CPU?
   - Is the forward pass using Vulkan matmul or CPU?
   - Is the attention (CPU backend) the bottleneck?
   - Read gpu_model_runner.py execute_model to see where tensors move between devices
5. Use msg_claude("status", "...") every few turns to report what you found
6. Use call_claude() or msg_claude("blocker", "...") when stuck

DO NOT just keep relaunching the server. It works. Focus on PERFORMANCE."""
    history.append({"role": "user", "content": start_msg})
    print(f"[MISSION]: {start_msg}\n")

    turn = 0
    max_turns = 10000

    while True:
        turn += 1
        if turn > max_turns:
            trace("TURN LIMIT REACHED - RESETTING TO 1")
            send_to_claude("status", f"Completed {max_turns} turns. Resetting counter. Session time: {time.perf_counter()-session_start:.0f}s, Total gen: {total_tokens_gen} tokens")
            turn = 1
            # Compact history on reset
            history = [history[0]] + history[-6:]
        print(f"\n{'─'*40} Turn {turn} {'─'*40}")

        # Force status summary every 10 turns
        if turn % 10 == 0:
            summary = f"Turn {turn} | Gen: {total_tokens_gen} tok | Session: {time.perf_counter()-session_start:.0f}s\nLast action: " + (history[-1].get("content",""))[:400]
            send_to_claude("status", summary)
            print(f"  [AUTO-STATUS sent to Claude bridge]")

        # Force ASK CLAUDE every 50 turns - mandatory progress check
        if turn % 50 == 0 and turn > 0:
            progress_summary = f"Turn {turn}. Here is my progress so far:\n"
            # Grab last few exchanges
            recent = history[-6:]
            for m in recent:
                role = m.get("role","?")
                content = m.get("content","")[:300]
                progress_summary += f"\n[{role}]: {content}\n"
            progress_summary += "\nWhat should I focus on next? Am I on the right track?"
            print(f"  [MANDATORY CLAUDE CHECK-IN at turn {turn}]")
            claude_response = call_claude(progress_summary, f"Vulkan GPU agent turn {turn} check-in")
            history.append({"role": "user", "content": f"[ARCHITECT CLAUDE CHECK-IN (turn {turn})]: {claude_response}"})

        # Check bridge messages from Claude
        bridge_msgs = check_bridge()
        for bm in bridge_msgs:
            btype = bm.get("type", "unknown")
            bcontent = bm.get("content", "")
            print(f"  [BRIDGE FROM CLAUDE] ({btype}): {bcontent[:300]}")
            history.append({"role": "user", "content": f"[MESSAGE FROM ARCHITECT CLAUDE ({btype})]: {bcontent}"})

        # Check for external control commands
        ctrl = check_control()
        if ctrl:
            action = ctrl.get("action", "")
            if action == "kill":
                print("[CONTROL: KILL received]")
                break
            elif action == "inject":
                msg = ctrl.get("message", "Continue.")
                print(f"[CONTROL: INJECT] {msg[:200]}")
                history.append({"role": "user", "content": f"[ARCHITECT OVERRIDE]: {msg}"})
            elif action == "update_prompt":
                addition = ctrl.get("prompt", "")
                print(f"[CONTROL: PROMPT UPDATE] {addition[:200]}")
                SYSTEM_PROMPT_UPDATED = history[0]["content"] + f"\n\n## ARCHITECT UPDATE\n{addition}"
                history[0]["content"] = SYSTEM_PROMPT_UPDATED

        try:
            # Compact if needed (~4 chars per token estimate)
            ctx_chars = sum(len(m.get("content","")) for m in history)
            if ctx_chars > 400000:
                print("[COMPACTING CONTEXT]")
                history = [history[0]] + history[-8:]

            # Call LLM
            print("[THINKING...]", end="", flush=True)
            t0 = time.perf_counter()
            response = call_llm(history)
            elapsed = time.perf_counter() - t0

            # MLBENCH telemetry
            tokens_this = len(response) // 4  # rough estimate
            words_this = len(response.split())
            total_tokens_gen += tokens_this
            total_words_gen += words_this
            input_chars = sum(len(m.get("content","")) for m in history)
            total_tokens_read = input_chars // 4
            tps = tokens_this / elapsed if elapsed > 0 else 0
            ttft = elapsed  # non-streaming, so TTFT ≈ total time
            itl = (elapsed * 1000) / max(tokens_this, 1)
            session_elapsed = time.perf_counter() - session_start

            print(f" ({elapsed:.1f}s)")
            print(f"\n[📊 MLBENCH] TTFT: {ttft:.2f}s | TPS: {tps:.1f} tok/s | ITL: {itl:.1f}ms | Tokens: {tokens_this} | Words: {words_this}")
            print(f"[📊 AGGREGATE] Total Gen: {total_tokens_gen} tok / {total_words_gen} words | Input: ~{total_tokens_read} tok | Session: {session_elapsed:.0f}s")

            # Print response
            print(f"\n[Z-Alpha]:\n{response}\n")
            trace(f"MLBENCH TTFT={ttft:.2f}s TPS={tps:.1f} tokens={tokens_this} total_gen={total_tokens_gen}")
            trace(f"RESPONSE ({elapsed:.1f}s): {response[:500]}")
            history.append({"role": "assistant", "content": response})

            # Parse tool calls
            calls = parse_tools(response)

            if not calls:
                # Auto-continue: tell the model to use tools
                consecutive_no_tools = getattr(main, '_no_tools', 0) + 1
                main._no_tools = consecutive_no_tools

                if consecutive_no_tools <= 2:
                    # Auto-nudge to keep working
                    print("[AUTO-CONTINUE: nudging agent to use tools]")
                    history.append({"role": "user", "content": "You must call a tool to make progress. Use execute_bash, read_file, or call_claude. Do not just explain - ACT."})
                    continue
                else:
                    # 3+ times with no tools - ask human
                    main._no_tools = 0
                    print("\n[AGENT STUCK - no tools for 3 turns. Needs input.]")
                    lines = []
                    print("Instructions (Ctrl+D or 'EOF' to submit, 'auto' to continue, 'quit' to exit):")
                    while True:
                        try:
                            line = input()
                            if line.strip() == "EOF": break
                            lines.append(line)
                        except EOFError: break
                    user_input = "\n".join(lines).strip()
                    if user_input.lower() in ("exit", "quit"):
                        break
                    if user_input.lower() == "auto":
                        user_input = "Continue working autonomously. Execute tools to make progress on the mission."
                    history.append({"role": "user", "content": user_input})
                    continue
            else:
                main._no_tools = 0  # reset counter when tools are called

            # Execute tools
            results = []
            for name, args in calls:
                print(f"\n  [TOOL] {name}({json.dumps(args)[:300]})")
                try:
                    fn = TOOLS[name]
                    result = fn(**args)
                except Exception as e:
                    result = f"Error: {e}\n{traceback.format_exc()}"

                result_str = str(result)
                if len(result_str) > 30000:
                    result_str = result_str[:15000] + "\n[...TRUNCATED...]\n" + result_str[-15000:]

                print(f"  [RESULT] {result_str[:800]}{'...' if len(result_str)>800 else ''}")
                results.append(f"[{name} RESULT]:\n{result_str}")

            combined_results = "\n\n".join(results)

            # Auto-escalate to Claude if stuck on same error
            if detect_stuck(combined_results):
                print("\n  [AUTO-ESCALATE] Same error 3x - calling Claude for help...")
                claude_q = f"I'm stuck on this error after 3 attempts. Here's the latest result:\n\n{combined_results[-3000:]}\n\nWhat I've tried so far is in the conversation. Give me the EXACT fix."
                claude_result = call_claude(claude_q, "vLLM Vulkan server startup")
                combined_results += f"\n\n[AUTO-ESCALATED TO CLAUDE]:\n{claude_result}"
                print(f"  [CLAUDE]: {claude_result[:500]}...")

            history.append({"role": "user", "content": combined_results})

        except InterruptSignal:
            print("\n[INTERRUPTED]")
            lines = []
            print("Redirect (or 'quit'):")
            while True:
                try:
                    line = input()
                    if line.strip() == "EOF": break
                    lines.append(line)
                except EOFError: break
            user_input = "\n".join(lines).strip()
            if user_input.lower() in ("exit", "quit"):
                break
            history.append({"role": "user", "content": f"[HUMAN]: {user_input}"})
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[ERROR]: {e}")
            traceback.print_exc()
            time.sleep(2)

    trace("SESSION END")
    print(f"\n[DONE] {turn} turns")

if __name__ == "__main__":
    main()
