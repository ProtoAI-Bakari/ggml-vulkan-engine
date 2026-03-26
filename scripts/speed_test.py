#!/usr/bin/env python3
"""Quick TPS speed test — all endpoints, 1 prompt, max 50 tokens. Finishes in <60s."""
import time, json, requests, sys

ENDPOINTS = {
    "ARCH-2":  ("http://10.255.255.2:8000",  "GLM-4.7-Flash"),
    "ENGR-3":  ("http://10.255.255.3:8000",  "Qwen3.5-35B"),
    "CODE-4":  ("http://10.255.255.4:8000",  "Coder-Next-8b"),
    "OSS-5":   ("http://10.255.255.5:8000",  "gpt-oss-120b"),
    "FAST-6":  ("http://10.255.255.6:8000",  "Coder-30B-4b"),
    "FAST-7":  ("http://10.255.255.7:8000",  "Coder-30B-4b"),
    "CUDA":    ("http://10.255.255.11:8000", "122B-FP8"),
}

PROMPT = "Write a Python fibonacci function. Code only, no explanation."
MAX_TOKENS = 50

print(f"{'EP':<10} {'MODEL':<20} {'TTFT':>6} {'TPS':>6} {'TOKS':>5}")
print("─" * 52)

for name, (url, model) in ENDPOINTS.items():
    try:
        t0 = time.perf_counter()
        first = None
        tokens = 0
        r = requests.post(f"{url}/v1/chat/completions", json={
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAX_TOKENS, "stream": True, "temperature": 0.2,
        }, stream=True, timeout=30)
        r.raise_for_status()
        for line in r.iter_lines():
            if not line: continue
            line = line.decode()
            if line.startswith("data: [DONE]"): break
            if line.startswith("data: "):
                try:
                    d = json.loads(line[6:])
                    c = d.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if c:
                        if first is None: first = time.perf_counter()
                        tokens += 1
                except: pass
        elapsed = time.perf_counter() - (first or t0)
        ttft = (first - t0) if first else 0
        tps = tokens / elapsed if elapsed > 0 else 0
        print(f"{name:<10} {model:<20} {ttft:>5.2f}s {tps:>5.1f} {tokens:>5}")
    except Exception as e:
        err = str(e)[:30]
        print(f"{name:<10} {model:<20} {'ERR':>6} {'---':>6} {'---':>5}  {err}")
