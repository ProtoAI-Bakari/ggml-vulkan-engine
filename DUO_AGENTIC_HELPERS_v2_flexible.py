import os, sys, time, json, subprocess, re, requests, signal
from datetime import datetime
from openai import OpenAI

# =====================================================
# 1. IDENTITY SETUP (Directly from Argument)
# =====================================================
if len(sys.argv) < 2:
    print("Usage: python script.py <BACKEND_IP>")
    sys.exit(1)

TARGET_IP = sys.argv[1]
PARTNER_IP = "10.255.255.11" if TARGET_IP == "10.255.255.4" else "10.255.255.4"

# Define Identity globally so run_bridge() can see it
if TARGET_IP == "10.255.255.4":
    IDENTITY = {
        "ROLE": "ADVERSARY",
        "NAME": "Qwen3-Next (Adversary)",
        "PARTNER_NAME": "Z-Alpha (Lead)",
        "MY_IP": "10.255.255.4",
        "PARTNER_IP": "10.255.255.11"
    }
else:
    IDENTITY = {
        "ROLE": "LEAD",
        "NAME": "Z-Alpha (Lead)",
        "PARTNER_NAME": "Qwen3-Next (Adversary)",
        "MY_IP": "10.255.255.11",
        "PARTNER_IP": "10.255.255.4"
    }

MY_BACKEND = f"http://{IDENTITY['MY_IP']}:8000/v1"
PARTNER_BACKEND = f"http://{IDENTITY['PARTNER_IP']}:8000/v1"
SHARED_CONVO_FILE = "/vmrepo/AGENT/agent_convo.txt"
LOG_DIR = os.path.expanduser("~/AGENT/LOGS")
os.makedirs(LOG_DIR, exist_ok=True)

# =====================================================
# 2. BACKEND & MODEL DETECTION
# =====================================================
client = OpenAI(base_url=MY_BACKEND, api_key="sk-duo", timeout=600.0)

try:
    # "Detect what model it is as usual"
    models = client.models.list()
    MODEL_NAME = models.data[0].id
    print(f"✅ Connected to {IDENTITY['MY_IP']} | Model: {MODEL_NAME}")
    print(f"🎭 Role: {IDENTITY['NAME']} | Partner: {IDENTITY['PARTNER_NAME']}")
except Exception as e:
    print(f"❌ Connection Failed to {MY_BACKEND}: {e}")
    sys.exit(1)

# =====================================================
# 3. TOOLS
# =====================================================
def execute_bash(command: str) -> str:
    try:
        # Surgical Pipe Deadlock Fix for background processes
        if command.strip().endswith('&'):
            cmd = command.strip()[:-1].strip()
            log_f = os.path.join(LOG_DIR, f"bg_{int(time.time())}.log")
            full_cmd = f"nohup {cmd} > {log_f} 2>&1 &"
            subprocess.Popen(full_cmd, shell=True, preexec_fn=os.setsid)
            return f"[LAUNCHED] {cmd}\nLogs: {log_f}"

        res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)
        return (res.stdout + res.stderr).strip() or "Done."
    except Exception as e: return f"Error: {e}"

def consult_partner(query: str) -> str:
    try:
        payload = {
            "model": "qwen3-coder-local", # Logic-only fallback
            "messages": [{"role": "user", "content": f"Review REQ from {IDENTITY['NAME']}: {query}"}],
            "temperature": 0.2
        }
        r = requests.post(f"{PARTNER_BACKEND}/chat/completions", json=payload, timeout=300)
        ans = r.json()["choices"][0]["message"]["content"]
        
        # Log to shared file
        with open(SHARED_CONVO_FILE, "a") as f:
            ts = datetime.now().strftime("%H:%M:%S")
            f.write(f"\n[{ts}] {IDENTITY['NAME']} -> {IDENTITY['PARTNER_NAME']}:\n{query}\n\nRESPONSE:\n{ans}\n{'-'*40}\n")
        return f"[{IDENTITY['PARTNER_NAME']}]: {ans}"
    except Exception as e: return f"Partner at {IDENTITY['PARTNER_IP']} unreachable."

TOOL_DISPATCH = {"execute_bash": execute_bash, "consult_partner": consult_partner}

# =====================================================
# 4. EXECUTION BRIDGE
# =====================================================
def run_bridge():
    sys_prompt = (
        f"You are {IDENTITY['NAME']}. Your partner is {IDENTITY['PARTNER_NAME']} at {IDENTITY['PARTNER_IP']}.\n"
        "Use <tool_call>{\"name\": \"execute_bash\", \"arguments\": {\"command\": \"...\"}}</tool_call> to act.\n"
        "Use consult_partner to get strategic advice from the other machine."
    )
    history = [{"role": "system", "content": sys_prompt}]

    while True:
        prompt = input(f"\n({IDENTITY['ROLE']}) > ")
        if prompt.lower() in ['exit', 'quit']: break
        history.append({"role": "user", "content": prompt})

        full_reply = ""
        print(f"[{IDENTITY['NAME']}]: ", end="", flush=True)
        
        stream = client.chat.completions.create(model=MODEL_NAME, messages=history, stream=True)
        for chunk in stream:
            txt = chunk.choices[0].delta.content or ""
            print(txt, end="", flush=True)
            full_reply += txt
        
        # Tool Extraction
        match = re.search(r'<tool_call>(.*?)</tool_call>', full_reply, re.DOTALL)
        if match:
            try:
                tc = json.loads(match.group(1).strip())
                fn, args = tc['name'], tc['arguments']
                print(f"\n\n[🔧 {fn}]: {args}")
                result = TOOL_DISPATCH[fn](**args)
                print(f"[📤]: {result[:500]}...")
                
                history.append({"role": "assistant", "content": full_reply})
                history.append({"role": "tool", "content": str(result)})
            except Exception as e: print(f"\nTool Error: {e}")
        else:
            history.append({"role": "assistant", "content": full_reply})

if __name__ == "__main__":
    run_bridge()
