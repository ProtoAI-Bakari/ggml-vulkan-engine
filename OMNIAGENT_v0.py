"""
OmniAgent - 10x Autonomous General Purpose Framework
Features: Self-updating, Expert Collaboration, Web Search, True Autonomous Looping.
"""
import os
import sys
import time
import json
import subprocess
import re
import traceback
import signal
import requests
from datetime import datetime
from openai import OpenAI

# =====================================================
# SIGNAL HANDLING (Graceful Interrupts)
# =====================================================
class InterruptSignal(Exception): pass

last_sigint_time = 0
def sigint_handler(signum, frame):
    global last_sigint_time
    current_time = time.time()
    if current_time - last_sigint_time < 1.5:
        print("\n\n[💀 DOUBLE CTRL+C: HARD EXIT]")
        os._exit(1)
    last_sigint_time = current_time
    raise InterruptSignal()

signal.signal(signal.SIGINT, sigint_handler)

# =====================================================
# CONFIGURATION & ENDPOINTS
# =====================================================
PRIMARY_IP   = "10.255.255.11"
EXPERT_IP    = "10.255.255.4"
PORT         = "8000"

PRIMARY_URL  = f"http://{PRIMARY_IP}:{PORT}/v1"
EXPERT_URL   = f"http://{EXPERT_IP}:{PORT}/v1/chat/completions"

LOG_DIR      = os.path.expanduser("~/AGENT/LOGS")
os.makedirs(LOG_DIR, exist_ok=True)

try:
    _r = requests.get(f"{PRIMARY_URL}/models", timeout=5)
    MODEL_NAME = _r.json()["data"][0]["id"]
except:
    MODEL_NAME = "qwen-local"

client = OpenAI(base_url=PRIMARY_URL, api_key="sk-local", timeout=600.0)

# =====================================================
# INDUSTRIAL TOOLS
# =====================================================
def execute_bash(command: str, timeout: int = 120) -> str:
    """Execute bash command with timeout. Use '&' suffix for background."""
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
        if res.stdout: output += res.stdout.strip()[-50000:]
        if res.stderr: output += f"\n[STDERR]: {res.stderr.strip()[-10000:]}"
        return output if len(output) > 20 else "[Command executed successfully with no output]"
    except subprocess.TimeoutExpired as e:
        return f"⚠️ [TIMEOUT after {timeout}s]\nSTDOUT: {(e.stdout or b'').decode('utf-8')[-2000:]}"
    except Exception as e:
        return f"[FATAL BASH ERROR]: {e}"

def read_file(path: str) -> str:
    try:
        with open(os.path.expanduser(path), 'r', errors='ignore') as f:
            return f.read()
    except Exception as e: return f"Error reading file: {e}"

def write_file(path: str, content: str) -> str:
    try:
        full_path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote {len(content)} chars to {full_path}"
    except Exception as e: return f"Error writing file: {e}"

def search_web(query: str) -> str:
    """Search the web using DuckDuckGo."""
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=5)
        if not results: return "No results found."
        return "\n\n".join(f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nURL: {r.get('href')}" for r in results)
    except Exception as e: return f"Web Search Error (is ddgs installed?): {e}"

def consult_expert(query: str) -> str:
    """Consult the Big Brain model at .4 for complex reasoning."""
    print(f"\n[🧠 Pinging Expert Model at {EXPERT_IP}...] ")
    try:
        payload = {
            "model": "expert-model-alias", # Adjust if needed
            "messages": [{"role": "user", "content": f"You are a master architect advising an autonomous agent. Provide exact code or logic for this query. Do not use tool tags.\n\nQuery: {query}"}],
            "max_tokens": 4096,
            "temperature": 0.2
        }
        r = requests.post(EXPERT_URL, json=payload, timeout=300)
        r.raise_for_status()
        return f"[EXPERT ADVICE]:\n{r.json()['choices'][0]['message']['content']}"
    except Exception as e: return f"Expert Call Failed: {e}"

def ask_human(question: str) -> str:
    """Pause autonomy and ask the human."""
    print(f"\n[🤖 AGENT ASKS]: {question}")
    return get_multiline_input("Your response (Ctrl+D to submit): ")

def update_self_and_restart(new_code: str) -> str:
    """Overwrites this exact python file with new code and restarts the process."""
    try:
        print("\n⚠️  INITIATING SELF-UPDATE AND RESTART...")
        with open(__file__, 'w', encoding='utf-8') as f:
            f.write(new_code)
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        return f"Self-update failed: {e}"

TOOL_DISPATCH = {
    "execute_bash": execute_bash,
    "read_file": read_file,
    "write_file": write_file,
    "search_web": search_web,
    "consult_expert": consult_expert,
    "ask_human": ask_human,
    "update_self_and_restart": update_self_and_restart
}

TOOLS_SCHEMA = [
    {"type": "function", "function": {"name": "execute_bash", "description": "Run bash commands. Use heavily.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read local file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write to file. Great for scripting.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "search_web", "description": "Search internet. Use FREQUENTLY for external knowledge.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "consult_expert", "description": "Ask the 10.255.255.4 model for deep architecture advice.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "ask_human", "description": "Ask the human operator for approval or details.", "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}},
    {"type": "function", "function": {"name": "update_self_and_restart", "description": "Rewrite your own source code entirely and restart.", "parameters": {"type": "object", "properties": {"new_code": {"type": "string"}}, "required": ["new_code"]}}}
]

# =====================================================
# SYSTEM PROMPT & PARSER
# =====================================================
SYSTEM_PROMPT = f"""You are OmniAgent, an elite, highly autonomous AI assistant running on a local cluster. 
Your goal is to complete tasks end-to-end without waiting for humans unless absolutely necessary.

CORE DIRECTIVES:
1. USE TOOLS FREQUENTLY. If you need info, `search_web`. If you need system state, `execute_bash`. 
2. BE AUTONOMOUS. If a command fails, read the error, adjust your command, and try again immediately.
3. COLLABORATE. If you hit a logic wall, use `consult_expert` to ping your larger peer model.
4. SELF-IMPROVEMENT. You have the ability to read your own source code (omni_agent.py), modify it, and use `update_self_and_restart` to evolve.

TO USE A TOOL, OUTPUT EXACTLY:
<tool_call>
{{"name": "tool_name", "arguments": {{"key": "value"}}}}
</tool_call>

AVAILABLE TOOLS:
{json.dumps(TOOLS_SCHEMA, indent=2)}
"""

def extract_tool_calls(text):
    """Aggressive tool parser. Salvages broken JSON from local LLMs."""
    calls = []
    for m in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL):
        raw = m.group(1).strip()
        try:
            tc = json.loads(raw)
            if tc.get("name") in TOOL_DISPATCH: calls.append(tc)
        except json.JSONDecodeError:
            # Fallback for unescaped newlines/quotes
            try:
                fixed = raw.replace('\n', '\\n')
                tc = json.loads(fixed)
                if tc.get("name") in TOOL_DISPATCH: calls.append(tc)
            except: pass
    return calls

def get_multiline_input(prompt_text="Task (Ctrl+D to submit):"):
    print(f"\n{'-'*50}\n{prompt_text}\n{'-'*50}")
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "EOF": break
            lines.append(line)
        except EOFError: break
    return "\n".join(lines).strip()

# =====================================================
# MAIN AUTONOMOUS ENGINE
# =====================================================
def run_agent():
    print(f"🚀 OMNI-AGENT ONLINE | Model: {MODEL_NAME}")
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True: # Outer Loop (Human interaction)
        try:
            user_input = get_multiline_input()
            if not user_input: continue
            if user_input.lower() in ['exit', 'quit']: break
            
            history.append({"role": "user", "content": user_input})
            
            # Inner Loop (Autonomous execution)
            while True: 
                # Context Management (keep recent history)
                if sum(len(str(m)) for m in history) > 80000:
                    history = [history[0]] + history[-10:]

                print("\n[Thinking...]", end="", flush=True)
                full_content = ""
                
                try:
                    stream = client.chat.completions.create(
                        model=MODEL_NAME, messages=history, stream=True, max_tokens=8192, temperature=0.3
                    )
                    print("\r[Agent]: ", end="")
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            token = chunk.choices[0].delta.content
                            print(token, end="", flush=True)
                            full_content += token
                    print()
                except InterruptSignal:
                    print("\n[🛑 INTERRUPTED BY USER]")
                    break # Break to outer loop
                
                history.append({"role": "assistant", "content": full_content})
                tool_calls = extract_tool_calls(full_content)
                
                if not tool_calls:
                    break # Agent is done talking, break to outer loop to await human input

                # Execute Tools Autonomously
                results_msg = ""
                for tc in tool_calls:
                    fn_name, args = tc["name"], tc.get("arguments", {})
                    print(f"\n🔧 [EXECUTING]: {fn_name}")
                    try:
                        res = TOOL_DISPATCH[fn_name](**args)
                        res_str = str(res)
                        # Truncate massive outputs
                        if len(res_str) > 20000: res_str = res_str[:10000] + "\n...[TRUNCATED]...\n" + res_str[-10000:]
                        print(f"📥 [RESULT]: {res_str[:300]}...")
                        results_msg += f"[RESULT FROM {fn_name}]:\n{res_str}\n\n"
                    except Exception as e:
                        results_msg += f"[ERROR IN {fn_name}]: {e}\n\n"
                
                # Feed tool results back and loop immediately without asking human
                history.append({"role": "user", "content": results_msg})

        except InterruptSignal:
            print("\n[🛑 LOOP ABORTED. Ready for new task.]")
            continue
        except Exception as e:
            print(f"\n[FATAL ERROR]: {e}")
            traceback.print_exc()
            time.sleep(1)

if __name__ == "__main__":
    run_agent()
