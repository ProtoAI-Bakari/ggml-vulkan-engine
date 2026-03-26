#!/usr/bin/env python3
"""Benchmark Phase 1 optimizations on 8B Q4 model."""
import time
from ggml_vllm_backend import GgmlLLM, SamplingParams

MODEL = '/home/z/models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'

def benchmark(model_path, n_tokens=100):
    print(f"\n=== Benchmark: {n_tokens} tokens ===")
    llm = GgmlLLM(model_path)
    
    # Warmup
    print("Warmup...")
    for i in range(3):
        r = llm.generate(f"Test {i}", params=SamplingParams(temperature=0, max_tokens=10))
    
    # Benchmark
    print(f"Generating {n_tokens} tokens...")
    start = time.time()
    r = llm.generate("What is quantum computing? Explain in detail.", 
                     params=SamplingParams(temperature=0, max_tokens=n_tokens))
    elapsed = time.time() - start
    
    print(f"\n=== RESULTS ===")
    print(f"Tokens generated: {r.token_ids}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"TPS: {r.tps:.2f}")
    print(f"Text: {r.text[:200]}...")
    
    return r.tps

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    benchmark(MODEL, n)
