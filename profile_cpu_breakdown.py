#!/usr/bin/env python3
"""T05: Profile CPU time breakdown in ggml Vulkan inference engine.

Measures per-token timing with a single continuous generation.
"""

import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path.home()))

from ggml_vllm_backend import GgmlLLM, SamplingParams

def profile_inference(model_path: str, num_tokens: int = 50):
    """Profile CPU time breakdown for token generation."""
    
    print(f"\n{'='*70}")
    print(f"T05: CPU TIME BREAKDOWN PROFILE")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Tokens to generate: {num_tokens}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"{'='*70}\n")
    
    # Load model
    print("Loading model...")
    start_load = time.perf_counter()
    llm = GgmlLLM(model_path)
    load_time = time.perf_counter() - start_load
    print(f"Model loaded in {load_time*1000:.1f}ms\n")
    
    # Warmup
    print("Warmup (5 tokens)...")
    warmup_prompt = "The capital of France is Paris and it is"
    r = llm.generate(warmup_prompt, params=SamplingParams(temperature=0, max_tokens=5))
    print(f"Warmup TPS: {r.tps:.1f}\n")
    
    # Profile with detailed timing - generate ALL tokens in ONE request
    print("Profiling generation (all tokens in single request)...")
    print("-" * 70)
    
    prompt = "Count from 1 to 100: 1, 2, 3,"
    
    start_gen = time.perf_counter()
    r = llm.generate(prompt, params=SamplingParams(temperature=0, max_tokens=num_tokens))
    total_time = time.perf_counter() - start_gen
    
    actual_tokens = len(r.token_ids)
    avg_tps = r.tps
    
    print(f"\nGenerated {actual_tokens} tokens in {total_time*1000:.1f}ms")
    print(f"Average TPS: {avg_tps:.1f}")
    print(f"Per-token average: {total_time/actual_tokens*1000:.1f}ms")
    print(f"\nOutput: {r.text[:200]}...")
    
    # Save results
    output_file = f"T05_profile_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    profile_data = {
        'timestamp': datetime.now().isoformat(),
        'model': model_path,
        'num_tokens_requested': num_tokens,
        'num_tokens_generated': actual_tokens,
        'load_time_ms': load_time * 1000,
        'warmup_tps': r.tps,
        'total_time_ms': total_time * 1000,
        'avg_tps': avg_tps,
        'per_token_ms': total_time / actual_tokens * 1000 if actual_tokens > 0 else 0,
        'output_text': r.text
    }
    
    with open(output_file, 'w') as f:
        json.dump(profile_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return profile_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile CPU time breakdown")
    parser.add_argument("--model", default="~/models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                       help="Path to GGUF model")
    parser.add_argument("--tokens", type=int, default=50,
                       help="Number of tokens to generate")
    args = parser.parse_args()
    
    profile_inference(args.model, args.tokens)
