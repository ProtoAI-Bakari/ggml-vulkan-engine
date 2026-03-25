#!/usr/bin/env python3
"""Direct Vulkan MLP test for Qwen2.5 models (0.5B, 1.5B, 3B).
No API server - uses vllm.LLM directly.

Usage:
  VLLM_VK_MLP_LAYERS=16 python test_vulkan_models.py --model Qwen2.5-1.5B-Instruct
  VLLM_VK_MLP_LAYERS=8  python test_vulkan_models.py --model Qwen2.5-3B-Instruct
  VLLM_VK_MLP_LAYERS=24 python test_vulkan_models.py --model Qwen2.5-0.5B-Instruct
"""
import os
import sys
import time
import argparse

# Force Vulkan platform BEFORE any vllm imports
os.environ['VLLM_PLATFORM'] = 'vulkan'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['VLLM_USE_V1'] = '1'

def main():
    parser = argparse.ArgumentParser(description='Vulkan MLP model tester')
    parser.add_argument('--model', default='Qwen2.5-0.5B-Instruct',
                       help='Model name (looked up in ~/models/)')
    parser.add_argument('--layers', type=int, default=None,
                       help='Number of MLP layers on Vulkan (overrides VLLM_VK_MLP_LAYERS)')
    parser.add_argument('--max-tokens', type=int, default=100,
                       help='Max tokens to generate')
    parser.add_argument('--max-model-len', type=int, default=256,
                       help='Maximum context length')
    parser.add_argument('--prompt', default='The capital of France is',
                       help='Prompt to test')
    parser.add_argument('--no-chunked-prefill', action='store_true', default=True,
                       help='Disable chunked prefill')
    args = parser.parse_args()

    # Resolve model path
    model_path = args.model
    if not os.path.exists(model_path):
        model_path = os.path.expanduser(f'~/models/{args.model}')
    if not os.path.exists(model_path):
        # Check HF cache
        model_path = None
        hf_cache = os.path.expanduser('~/.cache/huggingface/hub')
        for d in os.listdir(hf_cache):
            if args.model.replace('/', '--') in d or args.model.replace('/', '--').lower() in d.lower():
                snapshots = os.path.join(hf_cache, d, 'snapshots')
                if os.path.exists(snapshots):
                    subs = os.listdir(snapshots)
                    if subs:
                        model_path = os.path.join(snapshots, subs[0])
                        break
    if not model_path or not os.path.exists(model_path):
        print(f"Model not found: {args.model}")
        sys.exit(1)

    # Set layer count
    if args.layers is not None:
        os.environ['VLLM_VK_MLP_LAYERS'] = str(args.layers)

    vk_layers = os.environ.get('VLLM_VK_MLP_LAYERS', 'all (default 24)')
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Vulkan MLP layers: {vk_layers}")
    print(f"Max model len: {args.max_model_len}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"{'='*60}")

    # Import vllm AFTER env setup
    from vllm import LLM, SamplingParams

    print("\nLoading model...")
    t0 = time.time()
    llm = LLM(
        model=model_path,
        dtype='float16',
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.01,
        enable_chunked_prefill=not args.no_chunked_prefill,
    )
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Test 1: Coherence (temperature=0, deterministic)
    print(f"\n--- Test 1: Coherence (temp=0) ---")
    print(f"Prompt: '{args.prompt}'")
    sp = SamplingParams(temperature=0, max_tokens=30)
    t1 = time.time()
    outputs = llm.generate([args.prompt], sp)
    t2 = time.time()
    text = outputs[0].outputs[0].text
    tokens = len(outputs[0].outputs[0].token_ids)
    tps = tokens / (t2 - t1) if (t2 - t1) > 0 else 0
    print(f"Output: {text}")
    print(f"Tokens: {tokens} in {t2-t1:.2f}s = {tps:.1f} TPS")

    # Test 2: Generation quality (temp=0.7)
    print(f"\n--- Test 2: Generation ({args.max_tokens} tokens, temp=0.7) ---")
    sp2 = SamplingParams(temperature=0.7, max_tokens=args.max_tokens)
    t3 = time.time()
    outputs2 = llm.generate(["Write a haiku about computing:"], sp2)
    t4 = time.time()
    text2 = outputs2[0].outputs[0].text
    tokens2 = len(outputs2[0].outputs[0].token_ids)
    tps2 = tokens2 / (t4 - t3) if (t4 - t3) > 0 else 0
    print(f"Output: {text2}")
    print(f"Tokens: {tokens2} in {t4-t3:.2f}s = {tps2:.1f} TPS")

    # Test 3: Batch throughput
    print(f"\n--- Test 3: Batch of 4 prompts ---")
    prompts = [
        "2 + 2 =",
        "The meaning of life is",
        "Python is a programming language that",
        "In 2025, artificial intelligence",
    ]
    sp3 = SamplingParams(temperature=0, max_tokens=30)
    t5 = time.time()
    outputs3 = llm.generate(prompts, sp3)
    t6 = time.time()
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs3)
    batch_tps = total_tokens / (t6 - t5) if (t6 - t5) > 0 else 0
    for i, o in enumerate(outputs3):
        print(f"  [{i}] {prompts[i]}{o.outputs[0].text}")
    print(f"Total: {total_tokens} tokens in {t6-t5:.2f}s = {batch_tps:.1f} TPS")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.model}")
    print(f"  Load time:    {load_time:.1f}s")
    print(f"  Single TPS:   {tps:.1f} (temp=0)")
    print(f"  Gen TPS:      {tps2:.1f} (temp=0.7)")
    print(f"  Batch TPS:    {batch_tps:.1f} (4 prompts)")
    print(f"  VK MLP layers: {vk_layers}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
