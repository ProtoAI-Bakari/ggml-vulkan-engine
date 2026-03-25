# Batch=1 Optimization: THE Critical Problem

## Sources
- NVIDIA Deep Learning Performance Guide
- ScienceDirect: Highly parallel GEMV with register blocking
- bealto.com: GPU matrix-vector product
- arxiv.org: Mind the Memory Gap (large batch LLM inference)
- llama.cpp vulkan-shaders/mul_mat_vec.comp and mul_mat_vec_q4_k.comp
- Apple M1 Ultra specifications

---

## Why Batch=1 Is Fundamentally Different

### The Arithmetic Intensity Gap

| Operation | Data Loaded | Compute | Arithmetic Intensity |
|---|---|---|---|
| GEMM (M=N=K=4096) | 2 * 4096^2 = 33M elements | 2 * 4096^3 = 137B FLOPs | ~4096 FLOP/element |
| GEMV (M=4096, N=1, K=4096) | 4096^2 + 4096 = 16.8M elements | 2 * 4096^2 = 33.5M FLOPs | ~2 FLOP/element |

**GEMM is compute-bound** (limited by ALU throughput).
**GEMV is ALWAYS memory-bandwidth-bound** (limited by DRAM bandwidth).

### The Implication

For batch=1 token generation:
- Every weight in the model is read from memory ONCE per token
- Each weight participates in exactly ONE multiply-add
- The GPU ALUs sit idle >90% of the time waiting for memory
- **No amount of ALU optimization will help** -- bandwidth is the bottleneck

---

## Memory Bandwidth Analysis for 8B Q4_K_M at Batch=1

### Model Size
- 8B parameters at Q4_K_M: ~4.5 GB of weights
- Plus KV cache, activations, embeddings: ~0.5 GB overhead
- **Total memory read per token**: ~5 GB

### Available Bandwidth

| Chip | DRAM BW | Achievable BW (80%) | Min Time/Token | Max TPS |
|---|---|---|---|---|
| M1 | 68 GB/s | 54 GB/s | 93 ms | 10.7 |
| M1 Pro | 200 GB/s | 160 GB/s | 31 ms | 32 |
| M1 Max | 400 GB/s | 320 GB/s | 15.6 ms | 64 |
| M1 Ultra | 800 GB/s | 640 GB/s | 7.8 ms | **128** |

**NOTE**: 80% achievable bandwidth accounts for:
- Non-contiguous access patterns
- Cache miss overhead
- SLC contention with other accelerators
- Vulkan/driver overhead between dispatches

### Our Current Performance

**24.8 TPS on M1 Ultra** = 40.3 ms per token

**Theoretical minimum**: 7.8 ms per token (128 TPS)
**Efficiency**: 24.8 / 128 = **19.4% of theoretical bandwidth utilization**

### Where Is The Other 80%?

Likely breakdown:
1. **Pipeline overhead** (~30%): Barriers, descriptor binding, command buffer overhead between layers
2. **Memory access inefficiency** (~20%): Non-coalesced accesses, cache conflicts
3. **Compute overhead** (~10%): Dequantization ALU work, reduction overhead
4. **Underutilization** (~20%): Not all GPU cores active for all operations (small tensors, reduction phases)

---

## Best Known Techniques for Fast GEMV on GPU

### 1. Maximize Memory Bandwidth Utilization

**Coalesced reads**: Every thread in a warp/SIMD-group must read adjacent memory addresses.

```glsl
// GOOD: Coalesced - threads 0-31 read bytes 0-127
float val = weights[row * K + tid];

// BAD: Strided - thread 0 reads byte 0, thread 1 reads byte 4096
float val = weights[tid * K + col];
```

**Vectorized loads**: Read 4 or 8 elements at once per thread.

```glsl
// Load 4 f16 values (8 bytes) per thread per cycle
vec4 w = data_b_v4[base / 4];
```

### 2. Register Blocking for GEMV

Process multiple output rows per thread to amortize the cost of loading the input vector:

```glsl
// Each thread computes NUM_ROWS partial sums
// The input vector B is loaded ONCE and reused across rows
float temp[NUM_ROWS];
for (uint k = 0; k < K; k += BLOCK_SIZE) {
    float b_val = data_b[b_offset + k + tid];
    for (uint r = 0; r < NUM_ROWS; r++) {
        temp[r] += weight[r][k + tid] * b_val;
    }
}
```

**llama.cpp uses NUM_ROWS = 1-4** depending on the quantization type.

### 3. Warp-Level Reduction

Use subgroup operations instead of shared memory for the final reduction:

```glsl
// Fast: Single warp reduces 32 partial sums
float sum = subgroupAdd(partial);

// Slow: Shared memory tree reduction with barriers
shared float tmp[256];
tmp[tid] = partial;
barrier();
for (s = 128; s > 0; s >>= 1) {
    if (tid < s) tmp[tid] += tmp[tid + s];
    barrier();
}
```

The subgroup path eliminates **all barriers** for workgroups with a single subgroup.

### 4. Dequantize In-Place (No Staging Buffer)

For quantized weights, dequantize directly during the dot product computation:

```glsl
// Instead of: dequant weights -> temp buffer -> matmul
// Do: load packed weights -> dequant in registers -> fma -> accumulate
uint packed = data_a_packed32[ib].qs[iqs];
vec4 unpacked = vec4(unpack8(packed & 0x0F0F0F0F));
float result = dot(unpacked * scale, b_values);
```

This avoids writing dequantized values to shared memory (saves bandwidth and barriers).

### 5. Minimize Dispatch Overhead

For a transformer model with ~60 layers, each with ~6 matmul operations:
- **360 dispatches per token** (minimum)
- Each dispatch has: descriptor binding, push constant update, pipeline barrier, command buffer submission
- At 40 ms per token, that's ~111 microseconds per dispatch

**Optimization**: Fuse operations where possible:
- Fuse bias addition into the GEMV shader (llama.cpp does this via fusion_flags)
- Fuse residual addition
- Fuse RMSNorm + GEMV

### 6. Overlap Computation with Memory

**Prefetching**: Start loading the next layer's weights while computing the current layer:
```
Time: |---- Layer N compute ----|---- Layer N+1 compute ----|
      |---- Layer N+1 prefetch ----|
```

In Vulkan, this requires careful use of pipeline stages and semaphores to allow overlap.

---

## What llama.cpp Does Differently for Batch=1

### Separate Shader (mul_mat_vec vs mul_mm)

llama.cpp has **completely separate shaders** for batch=1 (GEMV) vs batch>1 (GEMM):

| Feature | mul_mm.comp (GEMM) | mul_mat_vec.comp (GEMV) |
|---|---|---|
| Shared memory | BM*BK + BN*BK tiles | Only for reduction |
| Tiling | 2D tiling (BM x BN) | 1D over rows |
| Threads | 128-256 per workgroup | 32-256 per workgroup |
| K processing | BK elements at a time | K_PER_ITER (2 or 8) per thread per iter |
| Data reuse | High (BM*BN reuses from shared) | Low (each weight used once) |
| Bottleneck | Compute (ALU) | Memory bandwidth |

### Q4_K Specialized GEMV

The `mul_mat_vec_q4_k.comp` shader is hand-optimized for Q4_K_M format:

1. **16 threads per quant superblock** (256 elements in Q4_K_M)
2. **Each thread processes 16 quant values** in a specific pattern
3. **Scale extraction is fully optimized**: Bit manipulation extracts 6-bit scales and 4-bit quants using `unpack8()` and bitfield operations
4. **No shared memory for weights**: Loads directly from global buffer, dequants in registers
5. **vec4 loads for activations**: `data_b_v4[(offset) / 4]` loads 4 floats at once
6. **FMA chains**: `fma(a, b, fma(c, d, fma(e, f, g)))` -- 4 FMAs deep per scale group

### Reduction Path Choice

llama.cpp selects the reduction strategy based on hardware:
1. **Single subgroup** (BLOCK_SIZE <= 32): `subgroupAdd` only, no shared memory
2. **Multiple subgroups with subgroupAdd**: Reduce within subgroups, then shared memory across subgroups
3. **Fallback**: Full shared memory tree reduction

---

## Theoretical Minimum Latency Calculation

### For 8B Q4_K_M on M1 Ultra

**Per-token work**:
- 32 transformer layers
- Per layer: q_proj (4096x4096), k_proj (4096x1024), v_proj (4096x1024), o_proj (4096x4096), gate (4096x14336), up (4096x14336), down (14336x4096)
- Total weights per layer: ~185M parameters at Q4_K_M = ~100 MB
- Total weights: ~3.2 GB (model-specific, this is approximate for 8B)
- Plus KV cache reads/writes, embeddings, norms: ~300 MB

**Total memory traffic**: ~3.5 GB per token

**M1 Ultra bandwidth**: 800 GB/s (theoretical), ~640 GB/s (achievable)

**Minimum time**: 3.5 GB / 640 GB/s = **5.5 ms per token**

**Maximum theoretical TPS**: **~182 TPS**

**Realistic target with overhead**: 60-80% of theoretical = **109-146 TPS**

### Achieving 50 TPS (Our Target)

50 TPS requires 20 ms per token.

At 640 GB/s achievable bandwidth:
- 20 ms * 640 GB/s = 12.8 GB of data budget
- We need ~3.5 GB of data -> **27.3% bandwidth utilization required**

This is achievable by:
1. Eliminating pipeline overhead between dispatches
2. Ensuring coalesced memory access
3. Using subgroup reductions
4. Minimizing unnecessary memory reads

**From our current 24.8 TPS to 50 TPS requires a 2x improvement**, which is within reach with the optimizations identified.

---

## Priority Action Items for Batch=1 Optimization

### Highest Impact (estimated 1.5-2x improvement)

1. **Reduce dispatch overhead**: Profile and minimize time between GPU dispatches. Aim for <20 microseconds per dispatch barrier.
2. **Tune workgroup sizes for AGX**: Profile different BLOCK_SIZE values (32, 64, 128, 256) for mul_mat_vec on M1 Ultra.
3. **Enable subgroup reduction**: Ensure USE_SUBGROUP_ADD is active on Honeykrisp (it should be, with VK_KHR_shader_subgroup).

### Medium Impact (estimated 1.2-1.5x improvement)

4. **Vectorize activation loads**: Ensure all B vector loads use vec4 (4 elements per load).
5. **Increase NUM_ROWS**: Process 2-4 output rows per workgroup to amortize vector loading.
6. **Operation fusion**: Fuse bias addition and residual into the GEMV dispatch.

### Lower Impact (estimated 1.1-1.2x improvement)

7. **Memory layout optimization**: Ensure weights are laid out for coalesced access per the dequant pattern.
8. **Double buffering**: Overlap next-dispatch prefetch with current computation (complex).
9. **SLC utilization**: Profile and minimize SLC thrashing from CPU-side work during inference.
