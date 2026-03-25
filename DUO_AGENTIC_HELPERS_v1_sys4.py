# ==============================================================================
# DUO_AGENTIC_HELPERS_v1.py
# HARDWARE-AWARE DUAL AGENT SYSTEM
# ==============================================================================
import os
import sys
import time
import json
import threading
import subprocess
import re
import requests
import platform
import glob
import signal
from datetime import datetime
from ddgs import DDGS
from openai import OpenAI

# =====================================================
# SYSTEM KILL CATCHER
# =====================================================
class InterruptSignal(Exception): pass

last_sigint_time = 0
def sigint_handler(signum, frame):
    global last_sigint_time
    current_time = time.time()
    if current_time - last_sigint_time < 1.5:
        print("\n\n[💀 DOUBLE CTRL+C DETECTED: OS-LEVEL KILL INSTANTIATED.]")
        os._exit(1)
    last_sigint_time = current_time
    raise InterruptSignal()

signal.signal(signal.SIGINT, sigint_handler)

# =====================================================
# HARDWARE / IDENTITY DISCOVERY
# =====================================================
def discover_identity():
    # Sniff network and hardware
    try:
        net_out = subprocess.getoutput("ifconfig || ip a")
        hw_out = subprocess.getoutput("fastfetch --print-json 2>/dev/null || system_profiler SPHardwareDataType 2>/dev/null || uname -a")
    except:
        net_out = ""
        hw_out = ""

    # Check for M1 Ultra / Box 2 signature
    if "10.255.255.4" in net_out or "Ultra" in hw_out:
        return {
            "ROLE": "ADVERSARY",
            "NAME": "Qwen3-Next (Adversary)",
            "MY_BACKEND": "http://10.255.255.4:8000/v1",
            "PARTNER_BACKEND": "http://10.255.255.11:8000/v1",
            "PARTNER_NAME": "Z-Alpha (Lead)"
        }
    else:
        # Default to Box 1 / M1 Max signature
        return {
            "ROLE": "LEAD",
            "NAME": "Z-Alpha (Lead)",
            "MY_BACKEND": "http://10.255.255.11:8000/v1",
            "PARTNER_BACKEND": "http://10.255.255.4:8000/v1",
            "PARTNER_NAME": "Qwen3-Next (Adversary)"
        }

IDENTITY = discover_identity()
SHARED_CONVO_FILE = "/vmrepo/AGENT/agent_convo.txt"
LOG_DIR = os.path.expanduser("~/AGENT/LOGS")
LOG_FILE = os.path.expanduser("~/AGENT/duo_agent_trace.log")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SHARED_CONVO_FILE), exist_ok=True)

try:
    _r = requests.get(f"{IDENTITY['MY_BACKEND']}/models", timeout=2)
    MODEL_NAME = _r.json()["data"][0]["id"]
except:
    MODEL_NAME = "qwen3-coder-local"

MAX_CONTEXT = 131072
client = OpenAI(base_url=IDENTITY['MY_BACKEND'], api_key="sk-duo", timeout=600.0)

# =====================================================
# DYNAMIC SYSTEM PROMPT
# =====================================================
def load_system_prompt():
    if IDENTITY["ROLE"] == "LEAD":
        base = (
            f"You are {IDENTITY['NAME']}, the Lead Autonomous Agent executing bash commands on an Asahi Linux system.\n"
            "Your objective is to fix vLLM Vulkan operations, write code, and launch servers.\n"
            f"You have a highly intelligent Adversarial Reviewer named {IDENTITY['PARTNER_NAME']} running on an M1 Ultra.\n"
            "If you get stuck, want your code reviewed for memory leaks, or need strategic advice, USE THE `consult_partner` TOOL to ask them."
        )
    else:
        base = (
            f"You are {IDENTITY['NAME']}, the Elite Adversarial Code Reviewer running on an Apple M1 Ultra (128GB).\n"
            f"Your partner is {IDENTITY['PARTNER_NAME']}, who is actively patching vLLM on an M1 Max.\n"
            "Your job is to critically review their logic, optimize C++/Python Vulkan code, find memory constraints, and brutally correct bad logic.\n"
            "You are the brain. You can execute bash to test things locally if needed, but your primary role is superior analytical feedback."
        )

    return base + f"\n\nAVAILABLE TOOLS:\n{json.dumps(TOOLS, indent=2)}\n\nTO CALL A TOOL, YOU MUST OUTPUT EXACTLY THIS FORMAT:\n<tool_call>\n{{\"name\": \"tool_name\", \"arguments\": {{\"arg1\": \"val\"}}}}\n</tool_call>"

# =====================================================
# TOOLS (INDUSTRIAL GRADE)
# =====================================================
def execute_bash(command: str) -> str:
    try:
        # 1. DEADLOCK FIX: Handle background server launches safely
        if command.strip().endswith('&'):
            cmd = command.strip()[:-1].strip()
            log_file = os.path.join(LOG_DIR, f"bg_server_{int(time.time())}.log")
            full_cmd = f"nohup bash -c {subprocess.list2cmdline([cmd])} > {log_file} 2>&1 &"
            
            # Detach completely from Python pipe
            subprocess.Popen(full_cmd, shell=True, preexec_fn=os.setsid)
            time.sleep(12) # Let server boot and spit initial logs
            
            if os.path.exists(log_file):
                with open(log_file, 'r') as f: logs = f.read()
                return (
                    f"[BACKGROUND PROCESS DETACHED SUCCESSFULLY]\nLog: {log_file}\n"
                    f"--- INITIAL 12s STARTUP LOGS ---\n{logs[-3000:]}\n\n"
                    f"[AGENT INSTRUCTION]: Use `read_file` on '{log_file}' to monitor the server."
                )
            return f"[BACKGROUND PROCESS LAUNCHED] Log: {log_file} (No output yet)"

        # 2. STANDARD SYNCHRONOUS COMMANDS
        wrapped_cmd = f"bash -c {subprocess.list2cmdline([command])}"
        res = subprocess.run(wrapped_cmd, shell=True, capture_output=True, text=True)
        out = res.stdout if res.stdout else res.stderr
        
        output_str = out.strip() if out else ""
        if len(output_str) > 250000:
            return output_str[:250000] + "\n\n[WARNING: Output truncated.]"
            
        return output_str if output_str else "Done."
    except Exception as e: return f"Error: {e}"

def read_file(path: str, offset: int = 0, limit: int = 250000) -> str:
    try:
        with open(os.path.expanduser(path), 'r', encoding='utf-8', errors='ignore') as f:
            if offset > 0 and offset < 5000:
                lines = f.readlines()
                return "".join([f"{i + offset + 1}: {line}" for i, line in enumerate(lines[offset:])])[:limit]
            f.seek(offset)
            return f.read(limit)
    except Exception as e: return f"Error: {e}"

def write_file(path: str, content: str) -> str:
    try:
        full_path = os.path.expanduser(path)
        with open(full_path, 'w', encoding='utf-8') as f: f.write(content)
        return f"Successfully wrote {len(content)} chars to {full_path}"
    except Exception as e: return f"Error: {e}"

def consult_partner(query: str) -> str:
    try:
        partner_prompt = (
            f"[COMMUNICATION FROM {IDENTITY['NAME']}]:\n"
            f"{query}\n\n"
            f"Please review and provide raw, technical, and precise feedback. Do not use XML tool tags in your response."
        )
        payload = {
            "model": "qwen3-coder-local", # Endpoint agnostic fallback
            "messages": [{"role": "user", "content": partner_prompt}],
            "max_tokens": 8192,
            "temperature": 0.3
        }
        
        # Call the OTHER machine's backend
        r = requests.post(f"{IDENTITY['PARTNER_BACKEND']}/chat/completions", json=payload, timeout=300)
        r.raise_for_status()
        response_text = r.json()["choices"][0]["message"]["content"]

        # Log to the shared NFS/Mount file
        try:
            with open(SHARED_CONVO_FILE, "a", encoding="utf-8") as f:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n[{ts}] 🟢 {IDENTITY['NAME']} ASKS:\n{query}\n")
                f.write(f"\n[{ts}] 🔴 {IDENTITY['PARTNER_NAME']} RESPONDS:\n{response_text}\n")
                f.write("\n" + "="*80 + "\n")
        except Exception as log_e: print(f"[!] Shared log write failed: {log_e}")

        return f"[{IDENTITY['PARTNER_NAME']} Responded]:\n\n{response_text}"
    except Exception as e:
        return f"Partner Consultation Failed (Is their server running?): {e}"

def ask_human(question: str) -> str:
    print(f"\n[🤖 {IDENTITY['NAME']} asks]: {question}")
    return get_multiline_input("Answer (or type 'quit'): ")

TOOL_DISPATCH = {
    "execute_bash": execute_bash, "read_file": read_file, 
    "write_file": write_file, "consult_partner": consult_partner, 
    "ask_human": ask_human
}

TOOLS = [
    {"type": "function", "function": {"name": "execute_bash", "description": "Execute bash command. Background processes using '&' are supported.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read local file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write text to file (overwrite).", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "consult_partner", "description": f"Consult {IDENTITY['PARTNER_NAME']} on the other machine for code review and strategic analysis. Logs automatically to {SHARED_CONVO_FILE}.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "ask_human", "description": "Stop and ask the human user a question.", "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}}
]

# =====================================================
# THE EXTRACTOR
# =====================================================
def extract_qwen_tools(text):
    tcs = []
    for m in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL):
        raw_inner = m.group(1).strip()
        try:
            tc = json.loads(raw_inner)
            tcs.append({"id": f"xml_{time.time()}", "name": tc.get("name"), "arguments": json.dumps(tc.get("arguments", {}))})
            continue
        except: pass
        fn_match = re.search(r'<function=([^>]+)>(.*?)</function>', raw_inner, re.DOTALL)
        if fn_match:
            fn_name = fn_match.group(1).strip()
            params_str = fn_match.group(2)
            args = {}
            for pm in re.finditer(r'<parameter=[\"\'\s]*([^>\"\'\s]+)[\"\'\s]*>(.*?)</parameter>', params_str, re.DOTALL):
                args[pm.group(1).strip()] = pm.group(2).strip()
            tcs.append({"id": f"xml_{time.time()}", "name": fn_name, "arguments": json.dumps(args)})

    if not tcs and "execute_bash" in text:
        cmd_start = -1
        if '"command"' in text: cmd_start = text.find('"', text.find('"command"') + 9) + 1
        elif '<parameter=command>' in text: cmd_start = text.find('<parameter=command>') + 19
        if cmd_start != -1:
            cmd_end = len(text)
            for end_marker in ['</parameter>', '"\n}', '"}', '</tool_call>']:
                idx = text.find(end_marker, cmd_start)
                if idx != -1 and idx < cmd_end: cmd_end = idx
            cmd = text[cmd_start:cmd_end].strip().replace('\\"', '"').replace('\\n', '\n')
            if cmd: tcs.append({"id": f"xml_{time.time()}", "name": "execute_bash", "arguments": json.dumps({"command": cmd})})

    unique_tcs, seen = [], set()
    for tc in tcs:
        sig = tc["name"] + tc["arguments"]
        if sig not in seen:
            seen.add(sig)
            unique_tcs.append(tc)
    return unique_tcs

def get_multiline_input(prompt_text="Architect Input (Paste freely. Ctrl+D to submit):"):
    print("\n" + "="*50 + "\n" + prompt_text + "\n" + "="*50)
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "EOF": break
            lines.append(line)
        except EOFError: break
        except KeyboardInterrupt: os._exit(1)
    return "\n".join(lines).strip()

def run_cli():
    system_prompt = load_system_prompt()
    history = [{"role": "system", "content": system_prompt}]

    print(f"\n⚡ DUO-AGENT ACTIVE — Identity: {IDENTITY['NAME']}")
    print(f"📡 Backend: {IDENTITY['MY_BACKEND']} | 🔗 Linked to: {IDENTITY['PARTNER_NAME']}")
    print(f"📁 Shared Log: {SHARED_CONVO_FILE}")

    first_input = get_multiline_input()
    if first_input: history.append({"role": "user", "content": first_input})
    last_tool_sig = ""; command_loop_count = 0

    while True:
        try:
            history[0]["content"] = load_system_prompt()
            print(f"\n[{IDENTITY['NAME']}]: ", end="", flush=True)
            
            full_content = ""; aborted = False
            try:
                stream = client.chat.completions.create(model=MODEL_NAME, messages=history, stream=True, max_tokens=4096)
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        full_content += content
            except InterruptSignal:
                aborted = True
                print("\n[🛑 ABORTED]")

            if not aborted: print()

            if aborted:
                new_cmd = get_multiline_input("Inject instructions (or 'quit'):")
                if new_cmd.lower() in ("exit", "quit"): break
                if new_cmd: history.append({"role": "user", "content": new_cmd})
                continue

            tool_calls = extract_qwen_tools(full_content)
            ast_msg = {"role": "assistant", "content": full_content}
            if tool_calls:
                ast_msg["tool_calls"] = [{"id": t["id"], "type": "function", "function": {"name": t["name"], "arguments": t["arguments"]}} for t in tool_calls]
            
            history.append(ast_msg)

            if not tool_calls:
                new_cmd = get_multiline_input("Waiting for instructions (or 'quit'):")
                if new_cmd.lower() in ("exit", "quit"): break
                if new_cmd: history.append({"role": "user", "content": new_cmd})
                continue

            for tc in tool_calls:
                fn_name = tc["name"]
                try: args = json.loads(tc["arguments"])
                except: args = {}
                
                curr_sig = f"{fn_name}_{args}"
                if curr_sig == last_tool_sig: command_loop_count += 1
                else: last_tool_sig = curr_sig; command_loop_count = 1

                if command_loop_count >= 3:
                    res = "[SYSTEM]: Loop detected. Change strategy."
                else:
                    print(f"\n[🔧 Executing] {fn_name}")
                    try: res = TOOL_DISPATCH.get(fn_name, lambda **k: "Unknown tool")(**args)
                    except Exception as e: res = f"Error: {e}"

                print(f"[📤 Output ({len(str(res))} chars)]: {str(res)[:1000]}...")
                history.append({"role": "tool", "tool_call_id": tc["id"], "name": fn_name, "content": str(res)})

        except InterruptSignal: os._exit(1)

if __name__ == "__main__":
    run_cli()
