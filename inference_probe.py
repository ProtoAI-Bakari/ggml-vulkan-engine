import requests
import time
import sys

print("🚀 PROBE: Waiting for vLLM to reach 'Healthy' state...")
for i in range(60):
    try:
        r = requests.get("http://localhost:8000/health", timeout=2)
        if r.status_code == 200:
            print("✅ PROBE: Server is Healthy!")
            break
    except:
        pass
    time.sleep(5)
else:
    print("❌ PROBE: Server never became healthy.")
    sys.exit(1)

print("💬 PROBE: Sending first inference request (First token will take time due to shader init)...")
payload = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 5,
    "temperature": 0
}
try:
    start = time.time()
    res = requests.post("http://localhost:8000/v1/chat/completions", json=payload, timeout=120)
    end = time.time()
    print(f"✨ PROBE: SUCCESS in {end-start:.2f}s!")
    print(f"FIRST TOKEN RESPONSE: {res.json()['choices'][0]['message']['content']}")
except Exception as e:
    print(f"❌ PROBE: Inference failed: {e}")