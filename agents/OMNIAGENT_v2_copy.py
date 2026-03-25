"""
OmniAgent v2 - Colored CLI & Sub-Agent Spawning
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
# ANSI COLORS (Make it PRETTY)
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
# SIGNAL HANDLING (Graceful Interrupts)
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
PRIMARY_IP   = "10.255.255.11"
EXPERT_IP    = "10.255.255.4"
PORT         = "8000"
PRIMARY_URL  = f"http://{PRIMARY_IP}:{PORT}/v1"
EXPERT_URL   = f"http://{EXPERT_IP}:{PORT}/v1/chat/completions"

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
        if res.stdout: output += res.stdout.strip()[-50000:]
        if res.stderr: output += f"\n[STDERR]: {res.stderr.strip()[-10000:]}"
        return output if len(output) > 20 else "[Command executed successfully with no output]"
    except subprocess.TimeoutExpired as e:
        return f"⚠️ [TIMEOUT after {timeout}s]\nSTDOUT: {(e.stdout or b'').decode('utf-8')[-2000:]}"
    except Exception as e:
        return f"[FATAL BASH ERROR]: {e}"

def read_file(path: str) -> str:
    try:
        with open(os.path.expanduser(path), 'r', errors='ignore') as f: return f.read()
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

def consult_expert(query: str) -> str:
    print(f"\n{C.MAGENTA}[🧠 Pinging Expert Model at {EXPERT_IP}...]{C.RESET}")
    try:
        payload = {"model": "expert-model-alias", "messages": [{"role": "user", "content": f"You are a master architect advising an autonomous agent. Provide exact code or logic for this query. Do not use tool tags.\n\nQuery: {query}"}], "max_tokens": 4096, "temperature": 0.2}
        r = requests.post(EXPERT_URL, json=payload, timeout=300)
        r.raise_for_status()
        return f"[EXPERT ADVICE]:\n{r.json()['choices'][0]['message']['content']}"
    except Exception as e: return f"Expert Call Failed: {e}"

def spawn_sub_agent(agent_name: str, mission_file_path: str) -> str:
    """Spawns a headless background instance of OmniAgent to do work autonomously."""
    try:
        log_file = os.path.expanduser(f"~/AGENT/LOGS/subagent_{agent_name}_{int(time.time())}.log")
        # Launch this exact python file with args for headless mode
        cmd = f"nohup {sys.executable} {os.path.abspath(__file__)} --name {agent_name} --mission {mission_file_path} > {log_file} 2>&1 &"
        subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        return f"🧬 SUCCESS: Sub-agent '{agent_name}' spawned in background!\nMission file: {mission_file_path}\nOutput logging to: {log_file}\n(You can read this log file periodically to check its progress.)"
    except Exception as e:
        return f"Failed to spawn sub-agent: {e}"

def ask_human(question: str) -> str:
    print(f"\n{C.BOLD}{C.YELLOW}[🤖 AGENT ASKS]: {question}{C.RESET}")
    return get_multiline_input("Your response (Ctrl+D to submit): ")

TOOL_DISPATCH = {
    "execute_bash": execute_bash, "read_file": read_file, "write_file": write_file,
    "search_web": search_web, "consult_expert": consult_expert, "ask_human": ask_human,
    "spawn_sub_agent": spawn_sub_agent
}

TOOLS_SCHEMA = [
    {"type": "function", "function": {"name": "execute_bash", "description": "Run bash commands.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read local file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write to file. Great for scripts or work packages.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "search_web", "description": "Search the internet.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "consult_expert", "description": "Ask the 10.255.255.4 model for deep architecture advice.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "spawn_sub_agent", "description": "SPAWN a background agent to work on a task. You MUST write a detailed .md mission file first using write_file, then pass the path to this tool.", "parameters": {"type": "object", "properties": {"agent_name": {"type": "string"}, "mission_file_path": {"type": "string"}}, "required": ["agent_name", "mission_file_path"]}}},
    {"type": "function", "function": {"name": "ask_human", "description": "Ask the human for help.", "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}}
]

SYSTEM_PROMPT = f"""You are OmniAgent, an elite autonomous AI assistant running on a local cluster.

CORE DIRECTIVES:
1. BE AUTONOMOUS. Fix your own errors. Read your logs.
2. SUB-AGENTS. You have the ability to spawn sub-agents. If a task is huge (like optimizing 3 different shaders), use `write_file` to create detailed mission prompts for each task, then use `spawn_sub_agent` to launch an army of agents to do the work in parallel.
3. USE TOOLS: ALWAYS use tools. Do not just talk.

TO USE A TOOL, OUTPUT EXACTLY:
<tool_call>
{{"name": "tool_name", "arguments": {{"key": "value"}}}}
</tool_call>

AVAILABLE TOOLS:
{json.dumps(TOOLS_SCHEMA, indent=2)}
"""

def extract_tool_calls(text):
    calls = []
    for m in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL):
        raw = m.group(1).strip()
        try:
            tc = json.loads(raw)
            if tc.get("name") in TOOL_DISPATCH: calls.append(tc)
        except json.JSONDecodeError:
            try:
                fixed = raw.replace('\n', '\\n')
                tc = json.loads(fixed)
                if tc.get("name") in TOOL_DISPATCH: calls.append(tc)
            except: pass
    return calls

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

def run_agent(agent_name="OmniAgent [Main]", headless_mission_file=None):
    print(f"{C.BOLD}{C.CYAN}🚀 {agent_name} ONLINE | Model: {MODEL_NAME}{C.RESET}")
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    is_headless = False

    if headless_mission_file:
        is_headless = True
        mission_text = read_file(headless_mission_file)
        print(f"{C.DIM}Loaded Headless Mission: {headless_mission_file}{C.RESET}")
        history.append({"role": "user", "content": f"YOUR HEADLESS MISSION:\n{mission_text}\n\nExecute this mission using your tools. When you are completely done and have no more tool calls to make, output your final summary."})

    while True:
        try:
            if not is_headless:
                user_input = get_multiline_input()
                if not user_input: continue
                if user_input.lower() in ['exit', 'quit']: break
                history.append({"role": "user", "content": user_input})
            
            while True: 
                # Context Management
                if sum(len(str(m)) for m in history) > 80000:
                    tail = history[1:]
                    merged_tail = []
                    for msg in tail:
                        if merged_tail and merged_tail[-1]["role"] == msg["role"]:
                            merged_tail[-1]["content"] += f"\n\n{msg['content']}"
                        else: merged_tail.append(msg)
                    history = [history[0]] + merged_tail[-12:]
                    if len(history) > 1 and history[1]["role"] != "user": history.pop(1)

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
                    if is_headless: return # Kill headless agent
                    break 
                
                history.append({"role": "assistant", "content": full_content})
                tool_calls = extract_tool_calls(full_content)
                
                if not tool_calls:
                    if is_headless:
                        print(f"\n{C.BOLD}{C.GREEN}[HEADLESS MISSION COMPLETE]{C.RESET}")
                        return
                    break

                results_msg = ""
                for tc in tool_calls:
                    fn_name, args = tc["name"], tc.get("arguments", {})
                    print(f"\n{C.BOLD}{C.YELLOW}🔧 [EXECUTING]: {fn_name}{C.RESET}")
                    try:
                        res = TOOL_DISPATCH[fn_name](**args)
                        res_str = str(res)
                        if len(res_str) > 20000: res_str = res_str[:10000] + "\n...[TRUNCATED]...\n" + res_str[-10000:]
                        print(f"{C.GREEN}📥 [RESULT]: {res_str[:300]}...{C.RESET}")
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
    parser = argparse.ArgumentParser(description="OmniAgent v2")
    parser.add_argument("--name", type=str, default="OmniAgent [Main]", help="Name of the agent instance")
    parser.add_argument("--mission", type=str, default=None, help="Path to a markdown file containing the headless mission")
    args = parser.parse_args()
    
    run_agent(agent_name=args.name, headless_mission_file=args.mission)
