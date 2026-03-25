#!/usr/bin/env python3
"""
T30: Comprehensive benchmark suite — one command tests all models + quants.
Usage: python benchmark_all.py [--models 8b,0.5b] [--tokens 100]
"""
import ctypes, numpy as np, time, os, sys, json
from datetime import datetime

LIB = ctypes.CDLL(os.path.expanduser("~/AGENT/libggml_llama_gguf.so"))
LIB.engine_load_gguf.argtypes = [ctypes.c_char_p, ctypes.c_int]
LIB.engine_load_gguf.restype = ctypes.c_void_p
LIB.engine_forward.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
LIB.engine_forward.restype = ctypes.c_int
LIB.engine_reset_kv.argtypes = [ctypes.c_void_p]
LIB.engine_free.argtypes = [ctypes.c_void_p]

MODELS = {
    "llama-8b-q4": {"path": "~/models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "vocab": 128256, "bos": 128000},
    "llama-8b-q8": {"path": "~/models/gguf/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf", "vocab": 128256, "bos": 128000},
    "llama-8b-f16": {"path": "~/models/gguf/llama-3.1-8b-instruct-f16.gguf", "vocab": 128256, "bos": 128000},
    "qwen-3b-q4": {"path": "~/models/gguf/qwen2.5-3b-instruct-q4_k_m.gguf", "vocab": 151936, "bos": 151643},
    "qwen-1.5b-q4": {"path": "~/models/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf", "vocab": 151936, "bos": 151643},
    "qwen-0.5b-q4": {"path": "~/models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf", "vocab": 151936, "bos": 151643},
}

LLAMA_CPP_REF = {
    "llama-8b-q4": {"decode": 24.7, "prefill": 137.0},
    "llama-8b-q8": {"decode": 22.8, "prefill": 169.3},
    "llama-8b-f16": {"decode": 13.9, "prefill": 171.1},
    "qwen-3b-q4": {"decode": 0, "prefill": 0},    # will measure
    "qwen-1.5b-q4": {"decode": 0, "prefill": 0},  # will measure
    "qwen-0.5b-q4": {"decode": 67.1, "prefill": 1014.5},
}

# Add warmup support
LIB.engine_warmup.argtypes = [ctypes.c_void_p]
LIB.engine_warmup.restype = ctypes.c_int


def bench_model(name, cfg, n_decode=100, n_ctx=512):
    path = os.path.expanduser(cfg["path"])
    if not os.path.exists(path):
        return {"model": name, "error": "file not found"}

    engine = LIB.engine_load_gguf(path.encode(), n_ctx)
    if not engine:
        return {"model": name, "error": "load failed"}

    LIB.engine_warmup(engine)
    V = cfg["vocab"]
    BOS = cfg["bos"]
    LIB.engine_reset_kv(engine)

    # Prefill 6 tokens
    prompt = np.array([BOS, 791, 6864, 315, 9822, 374], dtype=np.int32)
    prompt_pos = np.arange(6, dtype=np.int32)
    logits_p = np.empty((6, V), dtype=np.float32)

    t0 = time.perf_counter()
    ret = LIB.engine_forward(engine, 6, prompt.ctypes.data, prompt_pos.ctypes.data, logits_p.ctypes.data)
    prefill_ms = (time.perf_counter() - t0) * 1000
    if ret != 0:
        LIB.engine_free(engine)
        return {"model": name, "error": f"prefill failed: {ret}"}

    prefill_tps = 6 * 1000 / prefill_ms

    # Decode
    tok_id = int(np.argmax(logits_p[-1]))
    logits_1 = np.empty((1, V), dtype=np.float32)
    times = []

    for i in range(n_decode):
        tok = np.array([tok_id], dtype=np.int32)
        pos = np.array([6 + i], dtype=np.int32)
        t0 = time.perf_counter()
        ret = LIB.engine_forward(engine, 1, tok.ctypes.data, pos.ctypes.data, logits_1.ctypes.data)
        times.append(time.perf_counter() - t0)
        if ret != 0:
            break
        tok_id = int(np.argmax(logits_1[0]))

    LIB.engine_free(engine)

    if not times:
        return {"model": name, "error": "no decode times"}

    decode_ms = np.median(times) * 1000
    decode_tps = 1000 / decode_ms
    size_gb = os.path.getsize(path) / (1024**3)

    ref = LLAMA_CPP_REF.get(name, {})
    ref_decode = ref.get("decode", 0)
    ratio = decode_tps / ref_decode * 100 if ref_decode else 0

    return {
        "model": name,
        "decode_tps": round(decode_tps, 1),
        "decode_ms": round(decode_ms, 1),
        "prefill_tps": round(prefill_tps, 1),
        "llama_cpp_tps": ref_decode,
        "ratio_pct": round(ratio, 0),
        "size_gb": round(size_gb, 2),
        "n_decode": len(times),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all", help="Comma-separated model names or 'all'")
    parser.add_argument("--tokens", type=int, default=100, help="Number of decode tokens")
    args = parser.parse_args()

    models = MODELS if args.models == "all" else {k: MODELS[k] for k in args.models.split(",") if k in MODELS}

    print(f"\n{'='*75}")
    print(f"ggml Vulkan Engine — Comprehensive Benchmark")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Decode tokens: {args.tokens}")
    print(f"{'='*75}")
    print(f"{'Model':>20} | {'Decode TPS':>10} | {'ms/tok':>7} | {'Prefill':>8} | {'llama.cpp':>9} | {'Ratio':>6} | {'VRAM':>6}")
    print(f"{'-'*20}-+-{'-'*10}-+-{'-'*7}-+-{'-'*8}-+-{'-'*9}-+-{'-'*6}-+-{'-'*6}")

    results = []
    for name, cfg in models.items():
        r = bench_model(name, cfg, n_decode=args.tokens)
        results.append(r)
        if "error" in r:
            print(f"{name:>20} | {'ERROR':>10} | {r['error']}")
        else:
            print(f"{name:>20} | {r['decode_tps']:>10.1f} | {r['decode_ms']:>7.1f} | {r['prefill_tps']:>8.1f} | "
                  f"{r['llama_cpp_tps']:>9.1f} | {r['ratio_pct']:>5.0f}% | {r['size_gb']:>5.1f}G")

    # Save JSON results
    out_path = os.path.expanduser("~/AGENT/benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump({"date": datetime.now().isoformat(), "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
