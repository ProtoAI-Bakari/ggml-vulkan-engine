#!/usr/bin/env python3
"""
Benchmark custom tiled matmul shader vs ggml's built-in matmul.
Uses raw Vulkan compute via the kompute library (if available)
or falls back to a C harness.
"""
import ctypes, numpy as np, time, os, sys
sys.path.insert(0, os.path.expanduser('~/AGENT'))
from ggml_vulkan_engine import GgmlVulkanEngine

# Reference: our current ggml matmul speed
def bench_ggml(M, K, N, n_iter=20):
    engine = GgmlVulkanEngine()
    W = np.random.randn(N, K).astype(np.float32)
    engine.cache_weight("w", W)
    X = np.random.randn(M, K).astype(np.float32)
    # Warmup
    engine.matmul("w", X)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        engine.matmul("w", X)
        times.append(time.perf_counter() - t0)
    engine.close()
    return np.median(times)

# Run benchmarks at key sizes
print(f"{'Config':>30} | {'ggml ms':>8} | {'GFLOPS':>8}")
print("-" * 55)

configs = [
    ("8B gate batch=1", 1, 4096, 14336),
    ("8B gate batch=4", 4, 4096, 14336),
    ("8B gate batch=16", 16, 4096, 14336),
    ("8B gate batch=64", 64, 4096, 14336),
    ("8B gate batch=256", 256, 4096, 14336),
    ("QKV batch=1", 1, 4096, 4096),
    ("QKV batch=64", 64, 4096, 4096),
]

for name, M, K, N in configs:
    t = bench_ggml(M, K, N)
    flops = 2 * M * K * N
    gflops = flops / (t * 1e9)
    print(f"{name:>30} | {t*1000:>8.2f} | {gflops:>8.1f}")

print()
print("Custom shader SPIR-V compiled at: ~/AGENT/shaders/tiled_matmul.spv")
print("Next step: Wire into Vulkan pipeline and benchmark against ggml")
print("Expected improvement: 2-5x at large batch sizes (currently at 11% GPU utilization)")
