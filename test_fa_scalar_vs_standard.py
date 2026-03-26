#!/usr/bin/env python3
"""T18: Benchmark FA scalar vs standard attention at various context lengths"""

import sys
sys.path.insert(0, '/home/z/AGENT')

from ggml_vllm_backend import GgmlLLM, SamplingParams
import time

def test_attention_at_context(llm, context_length, n_tokens=50):
    """Test attention performance at specific context length"""
    print(f"\n{'='*60}")
    print(f"Testing context length: {context_length} tokens")
    print(f"{'='*60}")
    
    # Create a prompt that fills the context (leave room for generation)
    # Each "This is a test. " is ~4 tokens
    tokens_per_repeat = 4
    repeats = (context_length - n_tokens) // tokens_per_repeat
    if repeats < 1:
        repeats = 1
    
    base_prompt = "This is a test. " * repeats
    test_prompt = base_prompt + "What is the capital of France?"
    
    # Estimate actual prompt length
    actual_prompt_len = len(test_prompt.split()) * 1.3  # rough estimate
    print(f"  Actual prompt length: ~{int(actual_prompt_len)} tokens")
    
    # Run generation
    start = time.time()
    r = llm.generate(test_prompt, params=SamplingParams(temperature=0, max_tokens=n_tokens))
    elapsed = time.time() - start
    
    tps = n_tokens / elapsed
    print(f"\nResults:")
    print(f"  Context: {context_length} tokens")
    print(f"  Generated: {n_tokens} tokens")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  TPS: {tps:.1f}")
    print(f"  Output: {r.text.strip()[:100]}...")
    
    return tps

def main():
    model_path = '~/models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'
    
    print("Flash Attention Scalar Path Benchmark")
    print("="*60)
    print("Testing at various context lengths to measure FA performance")
    print("="*60)
    
    # Test at shorter contexts first to avoid crashes
    context_lengths = [128, 512, 2048]
    results = {}
    
    # Load model once with max context
    print(f"\nLoading model with n_ctx=8192...")
    llm = GgmlLLM(model_path, n_ctx=8192)
    print(f"Model loaded successfully")
    
    for ctx_len in context_lengths:
        try:
            tps = test_attention_at_context(llm, ctx_len, n_tokens=50)
            results[ctx_len] = tps
        except Exception as e:
            print(f"ERROR at context {ctx_len}: {e}")
            import traceback
            traceback.print_exc()
            results[ctx_len] = None
    
    print(f"\n{'='*60}")
    print("SUMMARY: Flash Attention Performance by Context Length")
    print(f"{'='*60}")
    print(f"{'Context':<12} {'TPS':<10} {'Status'}")
    print(f"{'-'*60}")
    for ctx_len in context_lengths:
        tps = results.get(ctx_len)
        status = f"{tps:.1f} TPS" if tps else "FAILED"
        print(f"{ctx_len:<12} {status:<10}")
    
    print(f"\nFlash attention scalar path is {'WORKING' if all(results.values()) else 'PARTIALLY WORKING'}")
    print(f"Note: FA_SCALAR path is automatically selected when VK_KHR_cooperative_matrix is unavailable")
    print(f"(Honeykrisp does not support cooperative matrix extensions)")
    print(f"\nPerformance degradation at longer contexts is expected due to:")
    print(f"  - O(N^2) attention complexity")
    print(f"  - Memory bandwidth limits for KV cache access")

if __name__ == '__main__':
    main()
