#!/usr/bin/env python3
"""W2: Batch Scaling Benchmark on M1 Ultra 128GB
Reproduces Sys12 benchmarks to get Ultra-specific numbers.
Flushed output + adaptive iterations for large batch sizes.
"""
import torch
import time
import os
import sys
import gc

os.environ.setdefault("HK_SYSMEM", "112000000000")

assert torch.is_vulkan_available(), "Vulkan not available!"

BATCH_SIZES = [1, 4, 8, 16, 64, 256, 1024, 4096, 8192, 16384]
MATRIX_CONFIGS = [
    ("QKV 1536x2048", 1536, 2048),
    ("MLP gate 1536x8960", 1536, 8960),
    ("MLP down 8960x1536", 8960, 1536),
    ("8B gate 4096x14336", 4096, 14336),
    ("8B down 14336x4096", 14336, 4096),
]

WARMUP = 2


def get_iters(batch, in_dim, out_dim):
    """Fewer iterations for huge matmuls."""
    flops = 2 * batch * in_dim * out_dim
    if flops > 1e12:
        return 2
    elif flops > 1e11:
        return 3
    elif flops > 1e10:
        return 5
    return 8


def p(msg):
    print(msg, flush=True)


def benchmark_cpu(x_cpu, w_cpu, iters):
    for _ in range(WARMUP):
        _ = torch.mm(x_cpu, w_cpu.t())
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = torch.mm(x_cpu, w_cpu.t())
    return (time.perf_counter() - t0) / iters


def benchmark_vulkan(x_cpu, w_vk, iters):
    for _ in range(WARMUP):
        x_vk = x_cpu.to('vulkan')
        r_vk = torch.mm(x_vk, w_vk.t())
        _ = r_vk.cpu()
    t0 = time.perf_counter()
    for _ in range(iters):
        x_vk = x_cpu.to('vulkan')
        r_vk = torch.mm(x_vk, w_vk.t())
        _ = r_vk.cpu()
    return (time.perf_counter() - t0) / iters


def compute_gflops(M, N, K, time_s):
    flops = 2 * M * N * K
    return flops / time_s / 1e9 if time_s > 0 else 0


p("=" * 90)
p("W2: BATCH SCALING BENCHMARK — M1 Ultra 128GB")
p(f"HK_SYSMEM={os.environ.get('HK_SYSMEM', 'not set')}")
p("=" * 90)

results = []

for config_name, in_dim, out_dim in MATRIX_CONFIGS:
    p(f"\n{'─' * 90}")
    p(f"MATRIX: {config_name} (in={in_dim}, out={out_dim})")
    p(f"{'─' * 90}")
    p(f"{'Batch':>8} | {'CPU (ms)':>10} | {'Vulkan (ms)':>12} | {'Speedup':>8} | {'CPU GFLOPS':>11} | {'VK GFLOPS':>10}")
    p(f"{'-'*8}-+-{'-'*10}-+-{'-'*12}-+-{'-'*8}-+-{'-'*11}-+-{'-'*10}")

    w_cpu = torch.randn(out_dim, in_dim, dtype=torch.float32)
    try:
        w_vk = w_cpu.to('vulkan')
    except Exception as e:
        p(f"  FAILED to load weight to Vulkan: {e}")
        continue

    for batch in BATCH_SIZES:
        x_cpu = torch.randn(batch, in_dim, dtype=torch.float32)
        iters = get_iters(batch, in_dim, out_dim)

        try:
            cpu_time = benchmark_cpu(x_cpu, w_cpu, iters)
        except Exception as e:
            p(f"{batch:>8} | CPU FAILED: {e}")
            continue

        try:
            vk_time = benchmark_vulkan(x_cpu, w_vk, iters)
        except Exception as e:
            p(f"{batch:>8} | {cpu_time*1000:>10.2f} | VK FAILED: {e}")
            continue

        speedup = cpu_time / vk_time if vk_time > 0 else 0
        cpu_gf = compute_gflops(batch, out_dim, in_dim, cpu_time)
        vk_gf = compute_gflops(batch, out_dim, in_dim, vk_time)

        marker = "<<<GPU" if speedup > 1.0 else "   CPU"
        p(f"{batch:>8} | {cpu_time*1000:>10.2f} | {vk_time*1000:>12.2f} | {speedup:>7.1f}x | {cpu_gf:>10.1f} | {vk_gf:>10.1f} {marker}")

        results.append({
            'config': config_name, 'batch': batch,
            'cpu_ms': cpu_time * 1000, 'vk_ms': vk_time * 1000,
            'speedup': speedup, 'cpu_gflops': cpu_gf, 'vk_gflops': vk_gf,
        })

        del x_cpu
        gc.collect()

    del w_cpu, w_vk
    gc.collect()
    time.sleep(0.3)

# Summary
p(f"\n{'=' * 90}")
p("SUMMARY: Optimal batch threshold per matrix config")
p(f"{'=' * 90}")
for config_name, _, _ in MATRIX_CONFIGS:
    config_results = [r for r in results if r['config'] == config_name]
    if not config_results:
        continue
    crossover = None
    for r in config_results:
        if r['speedup'] > 1.0:
            crossover = r['batch']
            break
    peak = max(config_results, key=lambda r: r['vk_gflops'])
    p(f"  {config_name}:")
    p(f"    GPU wins at batch >= {crossover if crossover else 'NEVER'}")
    p(f"    Peak Vulkan: {peak['vk_gflops']:.1f} GFLOPS at batch={peak['batch']}")

p(f"\n{'=' * 90}")
p("W2 COMPLETE")
p(f"{'=' * 90}")
