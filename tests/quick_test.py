#!/usr/bin/env python3
"""Quick test: Start vLLM server and test inference in one command."""
import subprocess
import time
import requests
import os
import sys

# Set environment for Vulkan
os.environ["VLLM_PLATFORM"] = "vulkan"

print("🚀 Starting vLLM Vulkan server...")
proc = subprocess.Popen(
    [
        "vllm", "serve", "Qwen/Qwen2.5-0.5B-Instruct",
        "--dtype", "float32",
        "--max-model-len", "256",
        "--enforce-eager",
        "--block-size", "16",
        "--gpu-memory-utilization", "0.05",
        "--port", "8000"
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    env=os.environ
)

# Wait for server to be ready (poll /health endpoint)
print("⏳ Waiting for server to start...")
for i in range(30):
    try:
        resp = requests.get("http://localhost:8000/health", timeout=2)
        if resp.status_code == 200:
            print("✅ Server is ready!")
            break
    except:
        time.sleep(1)
else:
    print("❌ Server failed to start in time")
    proc.terminate()
    sys.exit(1)

# Test inference
print("🧪 Testing inference...")
try:
    resp = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5,
            "temperature": 0
        },
        timeout=30
    )
    print(f"📊 Response status: {resp.status_code}")
    print(f"📝 Response body: {resp.text[:1000]}")
except Exception as e:
    print(f"❌ Inference test failed: {e}")

# Cleanup
print("🛑 Shutting down server...")
proc.terminate()
proc.wait()
print("✅ Test complete")