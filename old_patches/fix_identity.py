import os, requests

path = os.path.expanduser("~/AGENT/v33agent.py")
with open(path, "r") as f:
    code = f.read()

# 1. BRAIN SURGERY: Auto-detect the model name from the server instead of hardcoding a 404
auto_detect_logic = """
# --- BLOCK: CONFIGURATION (AUTO-DETECT) ---
SERVER_IP = "10.255.255.11"
BASE_URL = f"http://{SERVER_IP}:8000/v1"
QWEN3_URL = "http://10.255.255.4:8000/v1/chat/completions"

try:
    models_resp = requests.get(f"{BASE_URL}/models", timeout=5).json()
    MODEL_NAME = models_resp["data"][0]["id"]
    print(f"✅ IDENTITY FOUND: Serving as '{MODEL_NAME}'")
except Exception as e:
    MODEL_NAME = "qwen25"  # Fallback to the name in vrun.sh
    print(f"⚠️ IDENTITY GUESSED: Defaulting to '{MODEL_NAME}'")
"""

# Replace the old hardcoded config block
start_marker = "# --- BLOCK: CONFIGURATION ---"
end_marker = "client = OpenAI"
if start_marker in code:
    before = code.split(start_marker)[0]
    after = code.split(end_marker)[1]
    new_code = before + auto_detect_logic + "\nclient = OpenAI" + after
    with open(path, "w") as f:
        f.write(new_code)
    print("✅ IDENTITY PATCH APPLIED: v33agent.py will now auto-detect the model.")
else:
    print("❌ ERROR: Could not find configuration block to patch.")
