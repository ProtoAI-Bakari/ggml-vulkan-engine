#!/usr/bin/env python3
"""Quick Vulkan smoke test for Sys0 (M1 Ultra 128GB)"""
import os
import sys
import time

os.environ['VLLM_PLATFORM'] = 'vulkan'
os.environ['VLLM_VK_MLP_LAYERS'] = os.environ.get('VLLM_VK_MLP_LAYERS', '32')
os.environ['VLLM_VK_BATCH_THRESHOLD'] = '4'

import torch
print(f"PyTorch: {torch.__version__}, Vulkan: {torch.is_vulkan_available()}")

from vllm import LLM, SamplingParams

model_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/models/Qwen2.5-1.5B-Instruct")
max_layers = int(os.environ.get('VLLM_VK_MLP_LAYERS', '32'))

print(f"\nLoading {model_path} with {max_layers} Vulkan MLP layers...")
t0 = time.time()

llm = LLM(
    model=model_path,
    dtype="float32",
    enforce_eager=True,
    max_model_len=256,
    gpu_memory_utilization=0.9,
)
print(f"Model loaded in {time.time()-t0:.1f}s")

# Test prompts
prompts = [
    "The capital of France is",
    "2 + 2 =",
    "Explain gravity in one sentence:",
    "Write a haiku about silicon:",
]

params = SamplingParams(temperature=0, max_tokens=50)

print("\n--- Single request tests (temp=0) ---")
for prompt in prompts:
    t0 = time.time()
    out = llm.generate([prompt], params)
    elapsed = time.time() - t0
    text = out[0].outputs[0].text.strip()
    ntok = len(out[0].outputs[0].token_ids)
    tps = ntok / elapsed if elapsed > 0 else 0
    print(f"  [{tps:.1f} TPS] {prompt} -> {text[:80]}")

# Batch test
print("\n--- Batch test (4 prompts) ---")
t0 = time.time()
params_batch = SamplingParams(temperature=0.7, max_tokens=50)
outs = llm.generate(prompts, params_batch)
elapsed = time.time() - t0
total_tok = sum(len(o.outputs[0].token_ids) for o in outs)
print(f"  Batch TPS: {total_tok/elapsed:.1f} ({total_tok} tokens in {elapsed:.1f}s)")

print("\nDone!")
