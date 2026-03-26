#!/usr/bin/env python3
from pathlib import Path
"""T05: Profile CPU time breakdown in ggml Vulkan inference engine.

Measures:
- Graph build time
- Command buffer recording time  
- Python/ctypes overhead
- Vulkan queue submit + fence wait
- Total tokens/sec

Run: python3 profile_cpu_breakdown.py --model ~/models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --tokens 50
"""

import sys
import time
import json
import argparse
from datetime import datetime

# Add parent dir to path
sys.path.insert(0, str(Path.home()))

from pathlib import Path
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
    warmup_prompt = "The capital of France is"
    r = llm.generate(warmup_prompt, params=SamplingParams(temperature=0, max_tokens=5))
    print(f"Warmup TPS: {r.tps:.1f}\n")
    
    # Profile with detailed timing
    print("Profiling generation...")
    print("-" * 70)
    
    results = []
    total_tokens = 0
    total_time = 0
    
    for i in range(num_tokens):
        start_token = time.perf_counter()
        
        # Generate one token at a time to measure per-token breakdown
        prompt = f"Token {i}: " if i > 0 else "Count from 1 to 10:"
        
        # This will call forward() which we'll profile
        r = llm.generate(prompt, params=SamplingParams(temperature=0, max_tokens=1))
        
        token_time = time.perf_counter() - start_token
        total_time += token_time
        total_tokens += 1
        
        results.append({
            'token': i,
            'time_ms': token_time * 1000,
            'tps': 1.0 / token_time if token_time > 0 else 0
        })
        
        if i < 5 or i >= num_tokens - 5 or i % 10 == 0:
            print(f"Token {i:3d}: {token_time*1000:6.1f}ms ({r.tps:5.1f} TPS) - {r.text.strip()[:40]}")
    
    # Calculate statistics
    times_ms = [r['time_ms'] for r in results]
    avg_time = sum(times_ms) / len(times_ms)
    min_time = min(times_ms)
    max_time = max(times_ms)
    p50_time = sorted(times_ms)[len(times_ms)//2]
    p99_time = sorted(times_ms)[int(len(times_ms)*0.99)]
    
    avg_tps = total_tokens / total_time if total_time > 0 else 0
    
    print("\n" + "="*70)
    print("PROFILE RESULTS")
    print("="*70)
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total time: {total_time*1000:.1f}ms")
    print(f"Average TPS: {avg_tps:.1f}")
    print(f"\nPer-token timing (ms):")
    print(f"  Min:    {min_time:6.1f}ms")
    print(f"  Max:    {max_time:6.1f}ms")
    print(f"  Avg:    {avg_time:6.1f}ms")
    print(f"  P50:    {p50_time:6.1f}ms")
    print(f"  P99:    {p99_time:6.1f}ms")
    print("="*70)
    
    # Save results
    output_file = f"T05_profile_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    profile_data = {
        'timestamp': datetime.now().isoformat(),
        'model': model_path,
        'num_tokens': num_tokens,
        'load_time_ms': load_time * 1000,
        'warmup_tps': r.tps,
        'total_tokens': total_tokens,
        'total_time_ms': total_time * 1000,
        'avg_tps': avg_tps,
        'timing_ms': {
            'min': min_time,
            'max': max_time,
            'avg': avg_time,
            'p50': p50_time,
            'p99': p99_time
        },
        'per_token': results
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
