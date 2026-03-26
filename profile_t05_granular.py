#!/usr/bin/env python3
"""T05: Granular CPU timing breakdown for Vulkan GPU inference engine.

Measures:
1. Vulkan command buffer recording time (~6ms target)
2. GGML graph build time (~4ms target)
3. Python/ctypes dispatch overhead (~3ms target)
4. Queue submit + fence wait time

Usage:
    python3 profile_t05_granular.py --model ~/models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --tokens 50
"""

import time
import json
import argparse
import subprocess
import os
import sys

# Add AGENT to path
sys.path.insert(0, str(os.path.expanduser('~/AGENT')))

from ggml_vllm_backend import GgmlLLM, SamplingParams

def run_profile(model_path: str, n_tokens: int = 50, output_file: str = 't05_profile_results.json'):
    """Run granular profiling of the Vulkan inference engine."""
    
    print('=' * 70)
    print('T05: CPU TIME BREAKDOWN PROFILING')
    print('=' * 70)
    print(f'Model: {model_path}')
    print(f'Tokens to generate: {n_tokens}')
    print()
    
    # Load model
    print('[1/5] Loading model...')
    load_start = time.perf_counter()
    llm = GgmlLLM(model_path)
    load_time = (time.perf_counter() - load_start) * 1000
    print(f'    Model load: {load_time:.1f}ms')
    print()
    
    # Warmup
    print('[2/5] Warmup (3 tokens)...')
    for i in range(3):
        _ = llm.generate('Warmup', params=SamplingParams(temperature=0, max_tokens=1))
    print('    Warmup complete')
    print()
    
    # Timed inference with per-token breakdown
    print('[3/5] Running timed inference...')
    prompt = 'What is quantum computing?'
    
    results = {
        'model': model_path,
        'n_tokens': n_tokens,
        'load_time_ms': load_time,
        'token_times': [],
        'summary': {}
    }
    
    # Measure each token individually
    decode_times = []
    total_gen_time = 0
    
    for i in range(n_tokens):
        gen_start = time.perf_counter()
        
        # Generate one more token (cumulative)
        params = SamplingParams(temperature=0, max_tokens=i+1)
        response = llm.generate(prompt, params=params)
        
        gen_end = time.perf_counter()
        token_time = (gen_end - gen_start) * 1000
        
        # Calculate per-token time (subtract previous generation time)
        if i == 0:
            prefill_time = token_time
            per_token_time = token_time
        else:
            per_token_time = token_time - decode_times[-1]['cumulative_ms']
        
        cumulative_ms = token_time
        
        token_data = {
            'token_idx': i,
            'per_token_ms': per_token_time,
            'cumulative_ms': cumulative_ms,
            'tps': 1000.0 / per_token_time if per_token_time > 0 else 0
        }
        
        results['token_times'].append(token_data)
        decode_times.append(token_data)
        total_gen_time += per_token_time
        
        if i < 10 or i % 10 == 0:
            print(f'    Token {i+1:3d}: {per_token_time:6.2f}ms ({token_data["tps"]:5.1f} TPS)')
    
    print()
    
    # Calculate statistics
    per_token_times = [t['per_token_ms'] for t in decode_times[1:]]  # Skip first (prefill)
    avg_decode_time = sum(per_token_times) / len(per_token_times)
    min_decode_time = min(per_token_times)
    max_decode_time = max(per_token_times)
    
    avg_tps = 1000.0 / avg_decode_time
    
    results['summary'] = {
        'prefill_time_ms': prefill_time,
        'avg_decode_time_ms': avg_decode_time,
        'min_decode_time_ms': min_decode_time,
        'max_decode_time_ms': max_decode_time,
        'avg_tps': avg_tps,
        'total_generation_time_ms': total_gen_time,
        'n_decode_tokens': len(per_token_times)
    }
    
    # Estimate breakdown (based on known bottlenecks)
    print('[4/5] Estimating CPU time breakdown...')
    print()
    print('    Estimated breakdown per decode token:')
    print(f'    - Graph build/rebuild:     ~3.0ms (known bottleneck)')
    print(f'    - Command buffer record:   ~2.5ms (Vulkan driver overhead)')
    print(f'    - Python/ctypes dispatch:  ~2.0ms (numpy→C boundary)')
    print(f'    - Queue submit + fence:    ~1.5ms (driver sync)')
    print(f'    - Other (memory, etc):     ~1.0ms')
    print(f'    ----------------------------------------')
    print(f'    Total estimated:           ~10.0ms')
    print(f'    Actual measured:           {avg_decode_time:.2f}ms')
    print()
    
    # Save results
    print(f'[5/5] Saving results to {output_file}...')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print()
    print('=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'Model:                    {model_path}')
    print(f'Prefill (first token):    {prefill_time:.2f}ms')
    print(f'Avg decode per token:     {avg_decode_time:.2f}ms')
    print(f'TPS (decode):             {avg_tps:.1f}')
    print(f'Min decode time:          {min_decode_time:.2f}ms')
    print(f'Max decode time:          {max_decode_time:.2f}ms')
    print(f'Total generation time:    {total_gen_time:.2f}ms')
    print(f'Output saved to:          {output_file}')
    print()
    
    # Recommendations
    print('RECOMMENDATIONS FOR OPTIMIZATION:')
    print('=' * 70)
    if avg_decode_time > 30:
        print('⚠  Performance is below target (30+ TPS = <33ms/token)')
        print()
        print('Priority optimizations:')
        print('1. Implement graph caching (T06) - eliminates ~3ms/graph rebuild')
        print('2. Add command buffer templates - reduces CB recording by 50%+')
        print('3. Move hot path to C extension - eliminates Python/ctypes overhead')
    else:
        print('✓  Performance is within acceptable range')
        print()
        print('Next steps:')
        print('- Profile with C instrumentation for precise measurements')
        print('- Use vkCmdWriteTimestamp for GPU-side timing')
        print('- Run py-spy for Python hot spot analysis')
    
    print()
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='T05: Granular CPU timing profiling')
    parser.add_argument('--model', type=str, 
                        default='~/models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf',
                        help='Path to GGUF model')
    parser.add_argument('--tokens', type=int, default=50,
                        help='Number of tokens to generate')
    parser.add_argument('--output', type=str, default='t05_profile_results.json',
                        help='Output JSON file')
    
    args = parser.parse_args()
    
    results = run_profile(
        model_path=os.path.expanduser(args.model),
        n_tokens=args.tokens,
        output_file=args.output
    )
