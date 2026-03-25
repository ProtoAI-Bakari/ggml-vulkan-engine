#!/usr/bin/env python3
"""TESTER AGENT: Comprehensive Vulkan MLP test suite for 0.5B, 1.5B, 3B models.
Direct Python — no API server. Reports TPS, coherence, errors."""

import os, sys, time, json, gc, traceback

os.environ['VLLM_PLATFORM'] = 'vulkan'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['VLLM_USE_V1'] = '1'

import torch
print(f"Vulkan available: {torch.is_vulkan_available()}")
print(f"PyTorch: {torch.__version__}")

from vllm import LLM, SamplingParams

RESULTS = {}

MODELS = [
    {
        "name": "Qwen2.5-0.5B",
        "path": "Qwen/Qwen2.5-0.5B-Instruct",
        "vk_layers": "24",   # all 24 layers fit (24 * 52MB = 1.25GB)
    },
    {
        "name": "Qwen2.5-1.5B",
        "path": "/home/z/models/Qwen2.5-1.5B-Instruct",
        "vk_layers": "14",   # 14 * 158MB = 2.2GB < 2.6GB limit
    },
    {
        "name": "Qwen2.5-3B",
        "path": "/home/z/models/Qwen2.5-3B-Instruct",
        "vk_layers": "8",    # 8 * 300MB = 2.4GB < 2.6GB limit
    },
]

COHERENCE_PROMPTS = [
    ("Capital", "The capital of France is", 0),
    ("Math", "2 + 2 =", 0),
    ("Knowledge", "The speed of light is approximately", 0),
]

TPS_PROMPTS = [
    ("Short-t0", "Explain quantum computing in simple terms:", SamplingParams(temperature=0, max_tokens=50)),
    ("Long-t07", "Write a story about a robot exploring Mars:", SamplingParams(temperature=0.7, max_tokens=100)),
]

def test_model(model_cfg):
    name = model_cfg["name"]
    print(f"\n{'='*60}")
    print(f"  TESTING: {name}")
    print(f"  VK MLP layers: {model_cfg['vk_layers']}")
    print(f"{'='*60}")

    os.environ['VLLM_VK_MLP_LAYERS'] = model_cfg['vk_layers']

    # Reset the per-class counter so each model starts fresh
    # (the counter is on the class, not instance)
    try:
        from vllm.model_executor.models.qwen2 import Qwen2MLP
        Qwen2MLP._vk_count = 0
    except:
        pass

    result = {"name": name, "coherence": {}, "tps": {}, "errors": []}

    try:
        t_load_start = time.time()
        llm = LLM(
            model=model_cfg["path"],
            dtype="float16",
            enforce_eager=True,
            max_model_len=256,
            enable_chunked_prefill=False,
            gpu_memory_utilization=0.01,
        )
        t_load = time.time() - t_load_start
        result["load_time"] = f"{t_load:.1f}s"
        print(f"  Loaded in {t_load:.1f}s")
    except Exception as e:
        result["errors"].append(f"LOAD FAILED: {e}")
        print(f"  LOAD FAILED: {e}")
        traceback.print_exc()
        return result

    # Coherence tests (temp=0)
    print(f"\n  --- Coherence Tests (temp=0) ---")
    params0 = SamplingParams(temperature=0, max_tokens=30)
    for label, prompt, _ in COHERENCE_PROMPTS:
        try:
            t0 = time.time()
            out = llm.generate([prompt], params0)
            t1 = time.time()
            text = out[0].outputs[0].text.strip()
            toks = len(out[0].outputs[0].token_ids)
            tps = toks / (t1 - t0) if (t1 - t0) > 0 else 0
            # Check for garbage
            garbage = text.startswith("!!!") or text.startswith("???") or len(set(text[:20])) < 3
            status = "GARBAGE" if garbage else "OK"
            result["coherence"][label] = {
                "text": text[:100],
                "tps": round(tps, 1),
                "status": status,
            }
            print(f"  [{status}] {label}: \"{text[:60]}\" ({tps:.1f} TPS)")
        except Exception as e:
            result["coherence"][label] = {"text": "", "tps": 0, "status": f"ERROR: {e}"}
            result["errors"].append(f"Coherence/{label}: {e}")
            print(f"  [ERROR] {label}: {e}")

    # TPS tests
    print(f"\n  --- TPS Tests ---")
    for label, prompt, params in TPS_PROMPTS:
        try:
            t0 = time.time()
            out = llm.generate([prompt], params)
            t1 = time.time()
            text = out[0].outputs[0].text.strip()
            toks = len(out[0].outputs[0].token_ids)
            tps = toks / (t1 - t0) if (t1 - t0) > 0 else 0
            result["tps"][label] = {
                "tokens": toks,
                "seconds": round(t1 - t0, 2),
                "tps": round(tps, 1),
                "text_preview": text[:80],
            }
            print(f"  {label}: {toks} tok / {t1-t0:.2f}s = {tps:.1f} TPS")
            print(f"    \"{text[:80]}\"")
        except Exception as e:
            result["tps"][label] = {"tokens": 0, "seconds": 0, "tps": 0}
            result["errors"].append(f"TPS/{label}: {e}")
            print(f"  [ERROR] {label}: {e}")

    # Batch test (4 prompts at once)
    print(f"\n  --- Batch Test (4 prompts) ---")
    try:
        batch_prompts = [
            "Hello, my name is",
            "The largest ocean is",
            "Python is a programming language that",
            "In the year 2050,",
        ]
        params_batch = SamplingParams(temperature=0.5, max_tokens=30)
        t0 = time.time()
        outs = llm.generate(batch_prompts, params_batch)
        t1 = time.time()
        total_toks = sum(len(o.outputs[0].token_ids) for o in outs)
        batch_tps = total_toks / (t1 - t0)
        result["tps"]["Batch4"] = {
            "tokens": total_toks,
            "seconds": round(t1 - t0, 2),
            "tps": round(batch_tps, 1),
        }
        print(f"  Batch4: {total_toks} tok / {t1-t0:.2f}s = {batch_tps:.1f} TPS")
        for i, o in enumerate(outs):
            print(f"    [{i}] \"{o.outputs[0].text.strip()[:60]}\"")
    except Exception as e:
        result["errors"].append(f"Batch: {e}")
        print(f"  [ERROR] Batch: {e}")

    # Cleanup
    del llm
    gc.collect()
    time.sleep(2)

    return result


def main():
    all_results = []
    for model_cfg in MODELS:
        r = test_model(model_cfg)
        all_results.append(r)
        RESULTS[r["name"]] = r

    # Summary table
    print(f"\n{'='*70}")
    print(f"  TESTER AGENT — FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<16} {'Load':<8} {'Coh(t=0)':<12} {'TPS(t=0)':<10} {'TPS(t=0.7)':<12} {'Batch4':<10} {'Errors'}")
    print(f"{'-'*16} {'-'*8} {'-'*12} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")
    for r in all_results:
        name = r["name"]
        load = r.get("load_time", "FAIL")
        coh_status = "ALL OK" if all(v.get("status") == "OK" for v in r["coherence"].values()) else "FAIL"
        tps_t0 = r["tps"].get("Short-t0", {}).get("tps", "-")
        tps_t07 = r["tps"].get("Long-t07", {}).get("tps", "-")
        batch = r["tps"].get("Batch4", {}).get("tps", "-")
        errs = len(r.get("errors", []))
        print(f"{name:<16} {load:<8} {coh_status:<12} {tps_t0:<10} {tps_t07:<12} {batch:<10} {errs}")

    # Dump JSON for bridge update
    with open("/tmp/tester_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results: /tmp/tester_results.json")


if __name__ == "__main__":
    main()
