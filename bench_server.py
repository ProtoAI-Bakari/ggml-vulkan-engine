#!/usr/bin/env python3
"""T42-T43: Sustained and concurrent server benchmarks."""
import requests
import time
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

SERVER = "http://localhost:8080"

def single_request(prompt="Write about AI:", max_tokens=50):
    t0 = time.perf_counter()
    r = requests.post(f"{SERVER}/v1/chat/completions", json={
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0
    })
    elapsed = time.perf_counter() - t0
    data = r.json()
    tps = data.get("_tps", 0)
    tokens = data.get("usage", {}).get("completion_tokens", 0)
    return {"tps": tps, "tokens": tokens, "elapsed": elapsed, "status": r.status_code}

# T42: Sustained benchmark
print("=" * 60)
print("T42: Sustained TPS over 20 requests (sequential)")
print("=" * 60)
tps_list = []
total_tokens = 0
t_start = time.time()
for i in range(20):
    r = single_request(f"Topic {i}: explain briefly", max_tokens=30)
    tps_list.append(r["tps"])
    total_tokens += r["tokens"]
    if (i + 1) % 5 == 0:
        print(f"  {i+1}/20: avg TPS={sum(tps_list)/len(tps_list):.1f}, last={r['tps']:.1f}")

total_time = time.time() - t_start
print(f"  RESULT: {total_tokens} tokens in {total_time:.1f}s = {total_tokens/total_time:.1f} effective TPS")
print(f"  Average per-request TPS: {sum(tps_list)/len(tps_list):.1f}")
print(f"  Min/Max TPS: {min(tps_list):.1f} / {max(tps_list):.1f}")

# T43: Concurrent users
for n_concurrent in [2, 4, 8]:
    print(f"\n{'='*60}")
    print(f"T43: {n_concurrent} concurrent users")
    print(f"{'='*60}")

    prompts = [f"Topic {i}: explain in one sentence" for i in range(n_concurrent * 3)]
    results = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=n_concurrent) as pool:
        futures = [pool.submit(single_request, p, 30) for p in prompts]
        for f in as_completed(futures):
            results.append(f.result())

    elapsed = time.time() - t0
    total_tok = sum(r["tokens"] for r in results)
    avg_tps = sum(r["tps"] for r in results) / len(results)
    errors = sum(1 for r in results if r["status"] != 200)

    print(f"  Requests: {len(results)}, Errors: {errors}")
    print(f"  Total tokens: {total_tok} in {elapsed:.1f}s")
    print(f"  Aggregate TPS: {total_tok/elapsed:.1f}")
    print(f"  Per-request TPS: {avg_tps:.1f}")
    print(f"  Throughput: {len(results)/elapsed:.1f} req/s")

# Metrics
print(f"\n{'='*60}")
print("Server metrics:")
print(json.dumps(requests.get(f"{SERVER}/metrics").json(), indent=2))
