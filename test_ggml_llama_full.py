#!/usr/bin/env python3
"""Test the full Llama transformer engine with ggml Vulkan."""
import ctypes, numpy as np, time, os, sys
from safetensors import safe_open

LIB = ctypes.CDLL(os.path.expanduser('~/AGENT/libggml_llama_full.so'))
LIB.llama_engine_init.argtypes = [ctypes.c_int]*6 + [ctypes.c_int, ctypes.c_float, ctypes.c_float]
LIB.llama_engine_init.restype = ctypes.c_void_p
LIB.llama_engine_alloc_weights.argtypes = [ctypes.c_void_p]
LIB.llama_engine_alloc_weights.restype = ctypes.c_int
LIB.llama_engine_set_weight.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
LIB.llama_engine_set_weight.restype = ctypes.c_int
LIB.llama_engine_forward.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
LIB.llama_engine_forward.restype = ctypes.c_int
LIB.llama_engine_free.argtypes = [ctypes.c_void_p]
LIB.llama_engine_reset_kv.argtypes = [ctypes.c_void_p]
LIB.llama_engine_reset_kv.restype = None

MODEL_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)

# Llama-3.1-8B config
N_LAYERS = 32
HIDDEN = 4096
INTERMEDIATE = 14336
N_HEADS = 32
N_KV_HEADS = 8
VOCAB = 128256
N_CTX = 256
RMS_EPS = 1e-5
ROPE_THETA = 500000.0

print("Initializing engine...")
engine = LIB.llama_engine_init(N_LAYERS, HIDDEN, INTERMEDIATE, N_HEADS, N_KV_HEADS, VOCAB, N_CTX, RMS_EPS, ROPE_THETA)
assert engine, "Engine init failed"

print("Allocating weight buffers on Vulkan...")
ret = LIB.llama_engine_alloc_weights(engine)
if ret != 0:
    print(f"Weight alloc FAILED: {ret}")
    sys.exit(1)

# Load weights from safetensors
print("Loading weights from safetensors...")
import glob
shard_files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.safetensors")))

# Weight name mapping: HF → our engine names
def map_weight_name(hf_name):
    """Map HuggingFace weight name to our engine name."""
    if hf_name == "model.embed_tokens.weight":
        return "tok_embd"
    if hf_name == "model.norm.weight":
        return "output_norm"
    if hf_name == "lm_head.weight":
        return "output"
    # Layer weights: model.layers.N.xxx
    if hf_name.startswith("model.layers."):
        parts = hf_name.split(".")
        layer = int(parts[2])
        rest = ".".join(parts[3:])
        mapping = {
            "self_attn.q_proj.weight": "wq",
            "self_attn.k_proj.weight": "wk",
            "self_attn.v_proj.weight": "wv",
            "self_attn.o_proj.weight": "wo",
            "mlp.gate_proj.weight": "gate",
            "mlp.up_proj.weight": "up",
            "mlp.down_proj.weight": "down",
            "input_layernorm.weight": "attn_norm",
            "post_attention_layernorm.weight": "ffn_norm",
        }
        if rest in mapping:
            return f"l{layer}.{mapping[rest]}"
    return None

import torch as _torch
loaded = 0
t0 = time.time()
for shard in shard_files:
    with safe_open(shard, framework="pt") as f:
        for key in f.keys():
            ename = map_weight_name(key)
            if ename is None:
                continue
            tensor = f.get_tensor(key).float()  # bf16 → f32 via torch
            data = np.ascontiguousarray(tensor.numpy())
            ret = LIB.llama_engine_set_weight(engine, ename.encode(), data.ctypes.data, data.nbytes)
            if ret != 0:
                print(f"  FAILED: {key} → {ename}")
            else:
                loaded += 1
                if loaded % 50 == 0:
                    print(f"  Loaded {loaded} weights...")

print(f"Loaded {loaded} weights in {time.time()-t0:.1f}s")

# Test forward pass
print("\nRunning forward pass (1 token)...")
tokens = np.array([128000], dtype=np.int32)  # BOS token
positions = np.array([0], dtype=np.int32)
logits = np.empty((1, VOCAB), dtype=np.float32)

t0 = time.time()
ret = LIB.llama_engine_forward(engine, 1, tokens.ctypes.data, positions.ctypes.data, logits.ctypes.data)
elapsed = time.time() - t0

if ret == 0:
    top5 = np.argsort(logits[0])[-5:][::-1]
    print(f"Forward pass: {elapsed*1000:.1f}ms")
    print(f"Top-5 token IDs: {top5}")
    print(f"Top-5 logits: {logits[0][top5]}")
    print(f"Single-token TPS: {1/elapsed:.1f}")

    # Benchmark sequential token generation WITH KV cache
    print("\nBenchmarking sequential generation with KV cache...")
    LIB.llama_engine_reset_kv(engine)

    # Prefill: process a short prompt
    prompt_tokens = np.array([128000, 791, 6864, 315, 9822, 374], dtype=np.int32)  # "The capital of France is"
    prompt_pos = np.arange(len(prompt_tokens), dtype=np.int32)
    logits_prompt = np.empty((len(prompt_tokens), VOCAB), dtype=np.float32)

    t0 = time.perf_counter()
    ret = LIB.llama_engine_forward(engine, len(prompt_tokens), prompt_tokens.ctypes.data, prompt_pos.ctypes.data, logits_prompt.ctypes.data)
    prefill_ms = (time.perf_counter() - t0) * 1000
    print(f"Prefill ({len(prompt_tokens)} tokens): {prefill_ms:.1f}ms = {len(prompt_tokens)*1000/prefill_ms:.1f} TPS")

    if ret == 0:
        # Decode: generate tokens one at a time using KV cache
        next_token = np.argmax(logits_prompt[-1])
        decode_times = []
        generated = [next_token]

        for i in range(20):
            tok = np.array([next_token], dtype=np.int32)
            pos = np.array([len(prompt_tokens) + i], dtype=np.int32)
            logits_1 = np.empty((1, VOCAB), dtype=np.float32)

            t0 = time.perf_counter()
            ret = LIB.llama_engine_forward(engine, 1, tok.ctypes.data, pos.ctypes.data, logits_1.ctypes.data)
            decode_times.append(time.perf_counter() - t0)

            if ret != 0:
                print(f"Decode failed at step {i}: {ret}")
                break

            next_token = np.argmax(logits_1[0])
            generated.append(next_token)

        if decode_times:
            avg_decode = np.median(decode_times) * 1000
            print(f"Decode (20 tokens): median {avg_decode:.1f}ms per token = {1000/avg_decode:.1f} TPS")
            print(f"Generated token IDs: {generated[:10]}...")
    else:
        print(f"Prefill failed: {ret}")
else:
    print(f"Forward FAILED: {ret}")

LIB.llama_engine_free(engine)
