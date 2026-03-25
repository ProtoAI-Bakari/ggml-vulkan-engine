#!/usr/bin/env python3
import os, sys, time, json, threading, subprocess, re, requests, glob, signal
from datetime import datetime
from ddgs import DDGS
from openai import OpenAI

# =====================================================
# CONFIGURATION & GUARDRAILS
# =====================================================
MODEL_NAME  = "Qwen3.5-122B-A10B-FP8"
SERVER_IP   = "10.255.255.11"
SERVER_PORT = "8000"
BASE_URL    = f"http://{SERVER_IP}:{SERVER_PORT}/v1"
LOG_DIR     = os.path.expanduser("~/AGENT/LOGS")
LOG_FILE    = os.path.expanduser("~/AGENT/trace_v33_SAgent.log")
KB_FILE     = os.path.expanduser("~/AGENT/KNOWLEDGE_BASE.md")
PROMPT_FILE = os.path.expanduser("~/AGENT/SYSTEM_PROMPT.txt")
GIT_REPO    = os.path.expanduser("~/GITDEV/vllm_0.17.1")

MAX_CONTEXT = 131072
CONTEXT_RESET_THRESHOLD = 110000

os.makedirs(LOG_DIR, exist_ok=True)

# =====================================================
# CORE UTILITIES & SIGNALS
# =====================================================
class InterruptSignal(Exception): pass

def sigint_handler(signum, frame):
    global last_sigint_time
    current_time = time.time()
    if current_time - last_sigint_time < 1.5:
        print("\n\n[💀 DOUBLE CTRL+C: EMERGENCY EXIT]")
        os._exit(1)
    last_sigint_time = time.time()
    raise InterruptSignal()

last_sigint_time = 0

def get_multiline_input(prompt_text="Directive (Paste freely. 'EOF' on new line to submit):"):
    print(f"\n{'='*60}\n{prompt_text}\n{'='*60}")
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "EOF": break
            lines.append(line)
        except EOFError: break
    return "\n".join(lines).strip()

# =====================================================
# PERSISTENCE & GIT FORENSICS
# =====================================================
def update_knowledge_base(finding_type, content):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n### [{timestamp}] {finding_type}\n{content}\n"
    with open(KB_FILE, "a") as f: f.write(entry)
    return f"KB Updated: {finding_type}"

def git_forensics(action, target=""):
    try:
        if action == "log":
            cmd = ["git", "-C", GIT_REPO, "log", "--oneline", "-n", "20", "vllm/platforms/"]
        elif action == "diff":
            cmd = ["git", "-C", GIT_REPO, "diff", target]
        elif action == "show":
            cmd = ["git", "-C", GIT_REPO, "show", target]
        return f"[GIT {action.upper()}]\n{subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)}"
    except Exception as e: return f"Git Error: {e}"

def warm_up_recon():
    print("⚡ [WAKEUP] Z-ALPHA v33: INITIATING RECONNAISSANCE...")
    time.sleep(2)
    forensics = git_forensics("log")
    kb_summary = subprocess.getoutput(f"tail -n 30 {KB_FILE}") if os.path.exists(KB_FILE) else "No KB found."
    mem_state = subprocess.getoutput('free -h')
    return f"RECON COMPLETE.\n\nGIT HISTORY:\n{forensics}\n\nKB FINDINGS:\n{kb_summary}\n\nMEM STATE:\n{mem_state}"

# =====================================================
# INDUSTRIAL TOOLS
# =====================================================
def execute_bash(command: str) -> str:
    try:
        cmd_str = command.strip()
        if any(x in cmd_str for x in ["vllm serve", "vrun.sh"]):
            subprocess.run("pkill -9 -fi vllm; pkill -9 -f EngineCore", shell=True)
        
        if cmd_str.endswith('&'):
            cmd_str = cmd_str[:-1].strip().split('| tee')[0].strip()
            log_path = os.path.join(LOG_DIR, f"bg_{int(time.time())}.log")
            subprocess.Popen(f"nohup bash -c {subprocess.list2cmdline([cmd_str])} > {log_path} 2>&1 &", shell=True, preexec_fn=os.setsid)
            return f"[LAUNCHED] Log: {log_path}"
        
        res = subprocess.run(cmd_str, shell=True, capture_output=True, text=True, timeout=300)
        return f"STDOUT: {res.stdout}\nSTDERR: {res.stderr}"[:500000]
    except Exception as e: return f"Error: {e}"

def read_file(path: str, offset: int = 0, limit: int = 250000) -> str:
    try:
        with open(os.path.expanduser(path), 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(offset or 0)
            return f.read(limit or 250000)
    except Exception as e: return str(e)

def write_file(path: str, content: str) -> str:
    try:
        with open(os.path.expanduser(path), 'w', encoding='utf-8') as f: f.write(content)
        return f"Wrote to {path}"
    except Exception as e: return str(e)

# =====================================================
# AGENT INTERFACE
# =====================================================
TOOLS = [
    {"type": "function", "function": {"name": "execute_bash", "description": "Run bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "git_forensics", "description": "Git forensics", "parameters": {"type": "object", "properties": {"action": {"type": "string"}, "target": {"type": "string"}}, "required": ["action"]}}},
    {"type": "function", "function": {"name": "update_knowledge_base", "description": "Update KB", "parameters": {"type": "object", "properties": {"finding_type": {"type": "string"}, "content": {"type": "string"}}, "required": ["finding_type", "content"]}}},
    {"type": "function", "function": {"name": "ask_human", "description": "Ask human", "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}}
]

TOOL_DISPATCH = {
    "execute_bash": execute_bash, "read_file": read_file, "write_file": write_file,
    "git_forensics": git_forensics, "update_knowledge_base": update_knowledge_base,
    "ask_human": lambda question: get_multiline_input(question)
}

def extract_qwen_tools(text):
    tcs = []
    for m in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL):
        try:
            tc = json.loads(m.group(1).strip())
            tcs.append({"id": f"tc_{int(time.time())}_{len(tcs)}", "name": tc.get("name"), "arguments": json.dumps(tc.get("arguments", {}))})
        except: pass
    if not tcs:
        fn_match = re.search(r'<function=([^>]+)>(.*?)</function>', text, re.DOTALL)
        if fn_match:
            args = {}
            for pm in re.finditer(r'<parameter=([^>]+)>(.*?)</parameter>', fn_match.group(2), re.DOTALL):
                args[pm.group(1).strip()] = pm.group(2).strip()
            tcs.append({"id": f"xml_{int(time.time())}", "name": fn_match.group(1).strip(), "arguments": json.dumps(args)})
    return tcs

def load_system_prompt():
    prompt = open(PROMPT_FILE, "r").read() if os.path.exists(PROMPT_FILE) else "Z-Alpha sys12 Engineer."
    return prompt + f"\n\nTOOLS:\n{json.dumps(TOOLS, indent=2)}\n\nFormat: <tool_call>{{\"name\": \"...\", \"arguments\": {{...}}}}</tool_call>"

def run_cli():
    recon = warm_up_recon()
    client = OpenAI(base_url=BASE_URL, api_key="sk-4090", timeout=600.0)
    
    # Verify Model
    try:
        r = requests.get(f"{BASE_URL}/models", timeout=2)
        active_model = r.json()["data"][0]["id"]
    except: active_model = MODEL_NAME

    history = [{"role": "system", "content": load_system_prompt()}, {"role": "user", "content": f"START RECON:\n{recon}"}]
    print(f"\n⚡ [READY] sys12 Operational. Cluster Model: {active_model}")
    
    directive = get_multiline_input("ARCHITECTURE DIRECTIVE (Ctrl+D or 'EOF' to start):")
    if directive: history.append({"role": "user", "content": directive})

    while True:
        try:
            history[0]["content"] = load_system_prompt()
            if len(str(history)) > 400000:
                history = [history[0], history[1]] + history[-6:]
            
            full_res = ""
            print(f"\n[Z-Alpha]: ", end="", flush=True)
            start_time = time.perf_counter()
            first_token_time = None
            token_count = 0
            stream = client.chat.completions.create(model=active_model, messages=history, stream=True)
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    if not first_token_time: first_token_time = time.perf_counter()
                    c = chunk.choices[0].delta.content
                    print(c, end="", flush=True)
                    full_res += c
                    token_count += 1
            end_t = time.perf_counter()
            if first_token_time and token_count > 1:
                ttft = first_token_time - start_time
                tps = (token_count - 1) / (end_t - first_token_time)
                itl = ((end_t - first_token_time) / (token_count - 1)) * 1000
                print(f"\n\n[📊 MLBENCH] TTFT: {ttft:.3f}s | TPS: {tps:.2f} | ITL: {itl:.2f}ms | Tokens: {token_count}")
            
            history.append({"role": "assistant", "content": full_res})
            calls = extract_qwen_tools(full_res)
            if not calls:
                history.append({"role": "user", "content": get_multiline_input("Instructions:")})
                continue
            
            for tc in calls:
                print(f"\n[🔧 {tc['name']}]")
                try:
                    res = TOOL_DISPATCH[tc['name']](**json.loads(tc['arguments']))
                except Exception as e: res = f"Tool Error: {e}"
                history.append({"role": "tool", "tool_call_id": tc['id'], "name": tc['name'], "content": str(res)})
        except InterruptSignal:
            print("\n[!] User Interrupted. Awaiting instructions...")
            history.append({"role": "user", "content": get_multiline_input("Instructions:")})
        except Exception as e:
            print(f"\n[ERROR] {e}")
            time.sleep(2)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, sigint_handler)
    run_cli()
