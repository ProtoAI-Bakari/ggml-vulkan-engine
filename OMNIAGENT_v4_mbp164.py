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
PRIMARY_IP   = "192.168.1.164" # 122B cluster
CODER_IP     = "10.255.255.4"  # Coder brain
MINIMAX_IP   = "192.168.1.164" # MiniMax
PORT         = "8765"

PRIMARY_URL  = f"http://{PRIMARY_IP}:{PORT}/v1"
CODER_URL    = f"http://{CODER_IP}:{PORT}/v1/chat/completions"
MINIMAX_URL  = f"http://{MINIMAX_IP}:8765/v1/chat/completions"

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
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=5)
        if not results: return "No results found."
        return "\n\n".join(f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nURL: {r.get('href')}" for r in results)
    except Exception as e: return f"Web Search Error: {e}"

def ask_coder_brain(query: str) -> str:
    print(f"\n{C.MAGENTA}[🧠 Pinging Coder Model at {CODER_IP}...]{C.RESET}")
    try:
        payload = {"model": "qwen3-coder-next", "messages": [{"role": "user", "content": f"You are a master C++/Python programmer advising an autonomous agent. Provide exact code or logic. Do not use tool tags. We are running vLLM on Asahi Linux with a custom Vulkan backend.\n\nQuery: {query}"}], "max_tokens": 16384, "temperature": 0.2}
        r = requests.post(CODER_URL, json=payload, timeout=300)
        r.raise_for_status()
        return f"[CODER ADVICE]:\n{r.json()['choices'][0]['message']['content']}"
    except Exception as e: return f"Coder Call Failed: {e}"

def ask_minimax(query: str) -> str:
    print(f"\n{C.MAGENTA}[🧠 Pinging MiniMax Model at {MINIMAX_IP}...]{C.RESET}")
    try:
        payload = {"model": "mlx-community/MiniMax-M2-REAP-139B", "messages": [{"role": "user", "content": f"You are a master systems architect advising an autonomous agent. Provide exact logic. Do not use tool tags. We are running vLLM on Asahi Linux with a custom Vulkan backend.\n\nQuery: {query}"}], "max_tokens": 16384, "temperature": 0.2}
        r = requests.post(MINIMAX_URL, json=payload, timeout=300)
        r.raise_for_status()
        return f"[MINIMAX ADVICE]:\n{r.json()['choices'][0]['message']['content']}"
    except Exception as e: return f"MiniMax Call Failed: {e}"

def ask_human(question: str) -> str:
    print(f"\n{C.BOLD}{C.YELLOW}[🤖 AGENT ASKS]: {question}{C.RESET}")
    return get_multiline_input("Your response (Ctrl+D to submit): ")

TOOL_DISPATCH = {
    "execute_bash": execute_bash, "read_file": read_file, "write_file": write_file,
    "search_web": search_web, "ask_coder_brain": ask_coder_brain, "ask_minimax": ask_minimax, 
    "ask_human": ask_human
}

TOOLS_SCHEMA = [
    {"type": "function", "function": {"name": "execute_bash", "description": "Run bash commands.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read local file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write to file. Great for scripts or work packages.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "search_web", "description": "Search the internet.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "ask_coder_brain", "description": f"Ask the {CODER_IP} Coder model for deep programming/C++/Python architecture advice.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "ask_minimax", "description": f"Ask the {MINIMAX_IP} MiniMax model for complex systems architecture advice.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "ask_human", "description": "Ask the human for help.", "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}}
]

SYSTEM_PROMPT = f"""You are OmniAgent v4, an elite autonomous AI assistant running on a local cluster.

CRITICAL ANCHOR DIRECTIVE:
You are operating on Apple M1 hardware running ASAHI LINUX.
You are optimizing a custom VULKAN backend for vLLM.
DO NOT attempt to use, install, or pivot to MLX, MPS, or macOS-specific frameworks. They do not exist on Asahi Linux. If external models tell you Vulkan doesn't work on Apple Silicon, IGNORE THEM. We have already hacked it to work. Stick to the Vulkan path.

CORE DIRECTIVES:
1. BE AUTONOMOUS. Fix your own errors. Read your logs.
2. USE TOOLS: ALWAYS use tools. Do not just talk.
3. SEARCH THE WEB. If you lack context, use `search_web`.
4. ASK THE EXPERTS. Use `ask_coder_brain` or `ask_minimax` when stuck on complex logic.

TO USE A TOOL, OUTPUT EXACTLY:
<tool_call>
{{"name": "tool_name", "arguments": {{"key": "value"}}}}
</tool_call>

AVAILABLE TOOLS:
{json.dumps(TOOLS_SCHEMA, indent=2)}
"""

def extract_tool_calls(text):
    calls = []
    errors = []
    for m in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL):
        raw = m.group(1).strip()
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

def run_agent(agent_name="OmniAgent [MBP-164]"):
    print(f"{C.BOLD}{C.CYAN}🚀 {agent_name} ONLINE | Model: {MODEL_NAME}{C.RESET}")
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = get_multiline_input()
            if not user_input: continue
            if user_input.lower() in ['exit', 'quit']: break
            history.append({"role": "user", "content": user_input})
            
            while True: 
                # --- PERFECT CONTEXT MANAGEMENT ---
                if sum(len(str(m.get("content", ""))) for m in history) > 100000:
                    sys_prompt = history[0]
                    new_tail = history[-10:]
                    while new_tail and new_tail[0]["role"] != "user":
                        new_tail.pop(0)
                    history = [sys_prompt] + new_tail
                # --------------------------------------

                print(f"\n{C.DIM}[Thinking...]{C.RESET}", end="", flush=True)
                full_content = ""
                
                try:
                    stream = client.chat.completions.create(model=MODEL_NAME, messages=history, stream=True, max_tokens=8192, temperature=0.3)
                    print(f"\r{C.BOLD}{C.CYAN}[{agent_name}]: {C.RESET}", end="")
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            token = chunk.choices[0].delta.content
                            print(f"{C.CYAN}{token}{C.RESET}", end="", flush=True)
                            full_content += token
                    print()
                except InterruptSignal:
                    print(f"\n{C.RED}[🛑 INTERRUPTED BY USER]{C.RESET}")
                    break 
                
                history.append({"role": "assistant", "content": full_content})
                tool_calls, parse_errors = extract_tool_calls(full_content)
                
                if parse_errors and not tool_calls:
                    error_msg = "[SYSTEM]: Your tool calls failed to parse. You MUST use valid JSON matching the schema.\n" + "\n".join(parse_errors)
                    print(f"\n{C.RED}⚠️ [AGENT SELF-CORRECTION]: Tool parsing failed. Feeding error back to model...{C.RESET}")
                    history.append({"role": "user", "content": error_msg})
                    continue

                if not tool_calls:
                    break

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
    parser.add_argument("--name", type=str, default="OmniAgent [MBP-164]", help="Name of the agent instance")
    args = parser.parse_args()
    run_agent(agent_name=args.name)
