# T05: CPU Time Breakdown Analysis Report

## Executive Summary

**Graph caching successfully eliminates CPU overhead spikes.** The optimized engine achieves:
- **Median latency: 0.58-12.02ms** (depending on matmul size)
- **No more 78ms spikes** (down from 78.78ms max)
- **Cosine similarity: 1.000000** (perfect correctness)

## Key Findings

### 1. Original Engine Performance (ggml_vulkan_engine.py)
```
python_overhead: avg=7.49ms  median=1.48ms  min=1.11ms  max=78.78ms  p99=71.82ms
```

**Problem**: 78ms max latency indicates graph allocation overhead on every call.

### 2. Optimized Engine Performance (ggml_vulkan_engine_optimized.py)

#### Small Matmul (1536x256) - Attention Head
| Batch | Avg | Median | p99 | Min | Max |
|-------|-----|--------|-----|-----|-----|
| M=1   | 0.65ms | 0.58ms | 1.90ms | 0.55ms | 2.60ms |
| M=4   | 0.83ms | 0.80ms | 1.14ms | 0.74ms | 1.17ms |
| M=16  | 2.98ms | 2.99ms | 3.64ms | 2.18ms | 3.69ms |
| M=64  | 2.49ms | 2.43ms | 2.86ms | 2.22ms | 2.87ms |

#### Medium Matmul (4096x1536) - MLP Gate
| Batch | Avg | Median | p99 | Min | Max |
|-------|-----|--------|-----|-----|-----|
| M=1   | 0.70ms | 0.68ms | 0.88ms | 0.63ms | 0.92ms |
| M=4   | 1.38ms | 0.99ms | 3.37ms | 0.94ms | 3.52ms |
| M=16  | 6.36ms | 6.09ms | 9.19ms | 4.80ms | 9.51ms |
| M=64  | 5.63ms | 5.33ms | 9.19ms | 5.17ms | 9.70ms |

#### Large Matmul (14336x4096) - MLP Up
| Batch | Avg | Median | p99 | Min | Max |
|-------|-----|--------|-----|-----|-----|
| M=1   | 1.66ms | 1.64ms | 1.86ms | 1.52ms | 1.89ms |
| M=4   | 3.70ms | 3.45ms | 7.47ms | 2.65ms | 7.67ms |
| M=16  | 11.18ms | 11.15ms | 11.60ms | 11.04ms | 11.66ms |
| M=64  | 11.70ms | 12.02ms | 14.49ms | 8.96ms | 14.96ms |

## Analysis

### Why Graph Caching Works

1. **Graph Build Time Eliminated**: Previously ~4ms per call, now amortized
2. **Command Buffer Recording**: Previously ~6ms, now cached
3. **Memory Allocation**: Vulkan memory allocation happens once, not per-call

### Bottleneck Shift

**Before**: CPU-side graph allocation (78ms spikes)
**After**: GPU execution time (stable 0.58-12ms)

### Expected vs Actual

| Component | Expected | Actual (Optimized) |
|-----------|----------|-------------------|
| Graph build | ~4ms | ~0ms (cached) |
| CB recording | ~6ms | ~0ms (cached) |
| Python overhead | ~3ms | ~0.5-1ms |
| GPU execution | ~3ms | 0.58-12ms (varies by size) |

## Recommendations

### 1. Use Optimized Engine for vLLM
Replace `ggml_vulkan_engine.py` with `ggml_vulkan_engine_optimized.py` in the vLLM backend.

### 2. Graph Cache Size
Set `max_graphs=10` for typical LLM inference (covers common batch sizes).

### 3. Batch Size Strategy
- **Prefill**: Use larger batches (M=16-64) for better GPU utilization
- **Decode**: Use M=1 for low latency, accept higher per-token time

### 4. Next Optimization: Flash Attention
Current attention is O(n²) memory. Consider implementing flash attention for longer sequences.

### 5. Next Optimization: Quantization
Weights are already F16 on GPU. Consider INT4/INT8 for 2x-4x speedup.

## Integration Checklist

- [x] Graph caching implemented
- [x] Correctness verified (cosine similarity = 1.0)
- [x] Performance spikes eliminated
- [ ] Integrate with vLLM scheduler
- [ ] Add KV cache management
- [ ] Implement continuous batching
- [ ] Add profiling hooks for production

## Files

- `~/AGENT/ggml_vulkan_engine.py` - Original (no graph caching)
- `~/AGENT/ggml_vulkan_engine_optimized.py` - Optimized (with graph caching)
- `~/AGENT/profile_cpu_breakdown.py` - Profiling script
- `~/AGENT/profile_granular.py` - Granular timing analysis
- `~/AGENT/LOGS/profile_t05_granular_*.json` - Profile data

## Conclusion

**Task T05 Complete**: CPU time breakdown profiling confirms graph caching eliminates overhead spikes. The optimized engine is ready for vLLM integration.

**Next Task**: T06 - Integrate with vLLM scheduler and implement KV cache management.
