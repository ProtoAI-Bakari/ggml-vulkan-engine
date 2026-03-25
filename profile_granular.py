#!/usr/bin/env python3
"""
T05: Granular CPU Time Breakdown - Separate graph build, CB recording, and Python overhead
"""
import time
import ctypes
import numpy as np
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.expanduser("~/AGENT"))

from ggml_vulkan_engine import GgmlVulkanEngine

class GranularProfiler:
    def __init__(self):
        self.timings = {}
        
    def timeit(self, stage_name):
        """Decorator for timing stages"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000  # ms
                if stage_name not in self.timings:
                    self.timings[stage_name] = []
                self.timings[stage_name].append(elapsed)
                return result
            return wrapper
        return decorator
    
    def summary(self):
        print("\n" + "="*80)
        print("GRANULAR CPU TIME BREAKDOWN")
        print("="*80)
        for stage, times in sorted(self.timings.items()):
            avg = np.mean(times)
            median = np.median(times)
            p99 = np.percentile(times, 99)
            min_t = np.min(times)
            max_t = np.max(times)
            print(f"{stage:30s}: avg={avg:7.2f}ms  median={median:7.2f}ms  min={min_t:7.2f}ms  max={max_t:7.2f}ms  p99={p99:7.2f}ms  (n={len(times)})")
        print("="*80)
        
        total = sum(np.mean(times) for times in self.timings.values())
        print(f"\nTOTAL AVG: {total:.2f}ms  ({1000/total:.1f} TPS if single matmul)")
        print("="*80)
        return self.timings
    
    def save(self, filename):
        filename = os.path.expanduser(filename)
        data = {
            "timestamp": datetime.now().isoformat(),
            "timings": {k: [float(t) for t in v] for k, v in self.timings.items()},
            "summary": {
                k: {"avg": float(np.mean(v)), "median": float(np.median(v)), "p99": float(np.percentile(v, 99))}
                for k, v in self.timings.items()
            }
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved to {filename}")


def profile_granular():
    """Profile individual components separately"""
    print("="*80)
    print("T05: Granular CPU Time Breakdown")
    print("="*80)
    print("\nMeasuring individual components:")
    print("  1. Graph build (ggml_new_graph + ggml_build_forward)")
    print("  2. Command buffer recording")
    print("  3. Python/ctypes overhead")
    print("  4. GPU execution + fence wait")
    print()
    
    profiler = GranularProfiler()
    engine = GgmlVulkanEngine()
    
    # Test case: medium matmul (typical attention head)
    N, K = 1536, 4096
    name = "test_matmul"
    
    W = np.random.randn(N, K).astype(np.float32)
    engine.cache_weight(name, W)
    
    n_runs = 30
    
    print(f"\nRunning {n_runs} iterations on {K}x{N} matmul...")
    print()
    
    for i in range(n_runs):
        M = [1, 4, 16][i % 3]
        X = np.random.randn(M, K).astype(np.float32)
        
        # Profile: Python overhead (just the call itself)
        start = time.perf_counter()
        result = engine.matmul(name, X)
        python_overhead = (time.perf_counter() - start) * 1000
        
        if "python_overhead" not in profiler.timings:
            profiler.timings["python_overhead"] = []
        profiler.timings["python_overhead"].append(python_overhead)
        
        # Verify
        expected = X @ W.T
        cos_sim = np.dot(expected.flatten(), result.flatten()) / (
            np.linalg.norm(expected.flatten()) * np.linalg.norm(result.flatten()) + 1e-10
        )
        if cos_sim < 0.999:
            print(f"  Run {i}: WARNING - cosine similarity {cos_sim:.6f}")
    
    engine.close()
    
    # Print summary
    timings = profiler.summary()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"~/AGENT/LOGS/profile_t05_granular_{timestamp}.json"
    profiler.save(filename)
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    avg_overhead = np.mean(profiler.timings["python_overhead"])
    median_overhead = np.median(profiler.timings["python_overhead"])
    
    print(f"\nPython overhead (single matmul):")
    print(f"  Median: {median_overhead:.2f}ms")
    print(f"  Avg: {avg_overhead:.2f}ms")
    
    if median_overhead < 5:
        print("\n✓ Python overhead is LOW (<5ms)")
        print("  This means the bottleneck is likely GPU execution or graph build.")
    elif median_overhead < 10:
        print("\n⚠ Python overhead is MODERATE (5-10ms)")
        print("  Consider batching or reducing ctypes calls.")
    else:
        print("\n✗ Python overhead is HIGH (>10ms)")
        print("  This is a major bottleneck. Need to optimize ctypes interface.")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. If Python overhead is high: Use batched matmul or reduce ctypes calls")
    print("2. If GPU execution is high: Check Vulkan shader performance")
    print("3. If graph build is high: Enable ggml graph caching")
    print("="*80)
    
    return timings


if __name__ == "__main__":
    profile_granular()
