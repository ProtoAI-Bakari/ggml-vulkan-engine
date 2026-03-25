# SYNTHESIS: Vulkan Compute for LLM Inference on Apple Silicon

## Current State
- **Hardware**: M1 Ultra, 64 GPU cores, 800 GB/s bandwidth, 96 MB SLC
- **Software**: llama.cpp ggml Vulkan backend, Honeykrisp driver (Mesa)
- **Achievement**: 24.8 TPS on 8B Llama Q4_K_M
- **Target**: 50+ TPS (2x improvement required)
- **Theoretical maximum**: ~128-182 TPS (bandwidth-limited)

---

## What We Know About AGX Hardware

### Architecture
- 32-thread SIMD-groups (matches Vulkan subgroup size)
- 128 ALUs per core (16 EUs x 8 ALUs)
- ~208 KB register file per core
- ~60 KB threadgroup memory per core
- 8 KB L1 data cache (very small -- use threadgroup memory instead)
- 48 MB SLC on M1 Max, 96 MB on M1 Ultra
- Scalar at all precisions, but 16-bit may have 2x throughput (superscalar)

### Memory System
- 800 GB/s DRAM bandwidth (M1 Ultra)
- ~12.5 GB/s per GPU core
- Unified memory: device-local IS host-visible (no staging copies needed)
- SLC shared with CPU/Neural Engine (contention risk)

### Key Constraints
- No VK_KHR_cooperative_matrix in Honeykrisp
- simdgroup_matrix (hardware 8x8 MMA) only accessible via Metal
- Bank count for threadgroup memory is unknown (must use empirical padding)

---

## What MLX Does That We Can't (and Workarounds)

| MLX Feature | Available via Vulkan? | Workaround |
|---|---|---|
| `simdgroup_multiply_accumulate` | **NO** | Manual FMA loops (3-6x slower for GEMM) |
| 8x8 hardware MMA tiles | **NO** | Use register-blocked FMA with TM=4, TN=2 |
| `threadgroup(0)` / `threadgroup(1)` separate allocations | **NO** | Single shared memory, manually partitioned |
| Function constants | **YES** (specialization constants) | Direct equivalent |
| Metal `half` type | **YES** (f16 via extension) | `GL_EXT_shader_explicit_arithmetic_types_float16` |
| Serpentine MMA traversal | **YES** | Same FMA ordering in GLSL |
| Swizzle for L2 locality | **YES** | Same math in workgroup ID computation |
| Split-K with reduction | **YES** | Two-pass dispatch pattern |
| Vectorized wide loads | **YES** | vec4 / uvec4 buffer loads |

### The Critical Missing Piece

The ~6x gap in prompt processing and ~2.8x gap in token generation come from:
1. **Missing cooperative matrix** (~3x for GEMM, ~1.3x for GEMV)
2. **Pipeline barrier overhead** (~1.5-2x for both)
3. **Compiler optimization gap** (~1.1-1.2x)

For **batch=1 token generation** (our target), the gap is primarily #2 (barriers) since GEMV is bandwidth-bound and cooperative matrix matters less.

---

## What llama.cpp Does That We Already Match

- Tiled GEMM with shared memory (mul_mm.comp)
- Specialized GEMV per quantization type (mul_mat_vec_q4_k.comp)
- Dequantization during shared memory load
- Vectorized loads (vec4)
- Pipeline variant selection (l/m/s, aligned/unaligned)
- Split-K for small M
- Subgroup operations for reduction
- Push constants for per-dispatch parameters
- Pre-compiled SPIR-V shaders with specialization constants

---

## The 3 Most Impactful Optimizations Still Possible

### 1. REDUCE PIPELINE BARRIER OVERHEAD (Estimated: 1.5-2x improvement)

**Current state**: `ggml_vk_sync_buffers()` inserts a full pipeline barrier between every dependent dispatch. For ~360 dispatches per token at 24.8 TPS (40 ms/token), barriers may consume 20-30 ms.

**The fix**:
- **Profile barrier cost**: Use VK_EXT_pipeline_statistics_query or timestamps to measure actual barrier overhead
- **Batch independent dispatches**: Group operations that don't depend on each other before inserting barriers
- **Use events instead of barriers**: `vkCmdSetEvent` + `vkCmdWaitEvents` can be more granular
- **Reduce barrier scope**: Use per-buffer memory barriers instead of global pipeline barriers
- **Command buffer reuse**: Record command buffers once, replay per token

**Complexity**: Medium. Requires modifying `ggml-vulkan.cpp` sync logic.

### 2. TUNE WORKGROUP AND TILE SIZES FOR AGX (Estimated: 1.2-1.5x improvement)

**Current state**: Tile sizes are tuned for NVIDIA/AMD, not AGX specifically.

**The fix**:
- For GEMV (mul_mat_vec): Test BLOCK_SIZE = {32, 64, 128, 256} on AGX
- For GEMM (mul_mm): Test BM/BN = {32, 64, 128}, BK = {16, 32}
- Use specialization constants -- no shader rewrite needed
- Profile on actual M1 Ultra hardware
- Consider AGX-specific pipeline variant selection thresholds (the l/m/s crossover points)

**Key hypothesis**: AGX may perform better with smaller tiles (64x64) than NVIDIA (128x256) because:
- Smaller L1 cache (8 KB vs 48 KB)
- Different shared memory bank structure
- Different register pressure characteristics

**Complexity**: Low. Mostly parameter tuning in ggml-vulkan.cpp.

### 3. OPERATION FUSION AND DISPATCH REDUCTION (Estimated: 1.3-1.5x improvement)

**Current state**: Each operation (norm, matmul, activation, residual add) is a separate dispatch with its own barrier.

**The fix**:
- **Fuse RMSNorm + GEMV**: Single dispatch that normalizes input then multiplies
- **Fuse GEMV + bias + activation**: Already partially done (fusion_flags), extend coverage
- **Fuse residual add**: Accumulate residual during the GEMV store phase
- **Fuse SiLU(x) * y**: The gate+up multiplication in MLP can be one dispatch

Each fusion eliminates:
- 1 barrier (~50-100 us)
- 1 descriptor set update
- 1 pipeline bind
- 1 dispatch command
- 1 global memory read + write (intermediate tensor)

For a 32-layer model, fusing just the basic patterns could eliminate ~64-128 dispatches (of ~360 total), reducing per-token overhead by 6-13 ms.

**Complexity**: High. Requires writing new fused shader variants.

---

## Estimated TPS Improvement for Each

| Optimization | Current | After | TPS Gain | Difficulty |
|---|---|---|---|---|
| Barrier reduction | 24.8 | 37-50 | +49-100% | Medium |
| Tile size tuning | 24.8 | 30-37 | +20-50% | Low |
| Operation fusion | 24.8 | 32-37 | +30-50% | High |
| **All combined** | **24.8** | **50-75** | **+100-200%** | N/A |

These improvements are multiplicative to some degree, so the combined effect should exceed 50 TPS.

---

## Priority-Ordered Action Plan

### Phase 1: Profile and Measure (1-2 days)
1. Add Vulkan timestamp queries around every dispatch to measure per-operation time
2. Measure barrier cost specifically (dispatch-barrier-dispatch pattern)
3. Identify the top 10 most expensive operations per token
4. Measure actual DRAM bandwidth utilization during inference
5. Compare different BLOCK_SIZE values for mul_mat_vec on AGX

### Phase 2: Quick Wins (3-5 days)
6. Tune specialization constants for AGX: {BLOCK_SIZE, BM, BN, BK, NUM_ROWS}
7. Reduce barrier granularity (per-buffer vs global)
8. Ensure USE_SUBGROUP_ADD is active for all reduction paths
9. Verify unified memory path is used (no unnecessary staging copies)
10. Profile and optimize the top 3 most expensive operations

### Phase 3: Barrier Optimization (1-2 weeks)
11. Implement dependency tracking to skip unnecessary barriers
12. Batch independent dispatches (identify which ops are truly independent)
13. Consider command buffer pre-recording for the forward pass
14. Experiment with vkCmdSetEvent/WaitEvents vs pipelineBarrier

### Phase 4: Kernel Optimization (2-4 weeks)
15. Write AGX-optimized mul_mat_vec_q4_k variant
16. Implement fused RMSNorm + GEMV shader
17. Implement fused GEMV + bias + residual shader
18. Implement fused SiLU gate+up shader
19. Test and benchmark each fusion

### Phase 5: Advanced (1-2 months, optional)
20. Implement VK_KHR_cooperative_matrix in Honeykrisp (Mesa contribution)
21. Implement custom AGX-aware memory allocator
22. Investigate async compute overlap between layers
23. Explore split-K optimization for specific layer sizes

---

## Links to All Research Documents

- [MLX Steel GEMM Analysis](mlx_steel_analysis.md) - How Apple's ML framework achieves peak GPU performance
- [llama.cpp Vulkan Shaders](llama_cpp_shaders.md) - Deep dive into mul_mm.comp and mul_mat_vec.comp
- [AGX Microarchitecture](agx_microarch.md) - Hardware details for M1/M1 Max/M1 Ultra GPU
- [Mesa Honeykrisp Driver](mesa_honeykrisp.md) - Vulkan driver status and limitations
- [Vulkan Compute Techniques](vulkan_compute_techniques.md) - Best practices for Vulkan matmul
- [Vulkan ML Frameworks](vulkan_ml_frameworks.md) - Survey of existing frameworks
- [Batch=1 Optimization](batch1_optimization.md) - GEMV analysis and techniques
- [ggml Vulkan Internals](ggml_vulkan_internals.md) - Buffer management, pipeline caching, dispatch
- [Code Snippets](code_snippets/) - Annotated code and templates

## External References
- [Apple GPU Microarchitecture (metal-benchmarks)](https://github.com/philipturner/metal-benchmarks)
- [Apple G13 ISA Reference](https://dougallj.github.io/applegpu/docs.html)
- [Asahi GPU Blog (Rosenzweig)](https://alyssarosenzweig.ca/blog/asahi-gpu-part-3.html)
- [Vulkan 1.3 on M1 (Asahi)](https://asahilinux.org/2024/06/vk13-on-the-m1-in-1-month/)
- [Metal vs Vulkan Performance Issue](https://github.com/ggml-org/llama.cpp/issues/10982)
- [VK_KHR_cooperative_matrix Spec](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_cooperative_matrix.html)
- [NVIDIA Cooperative Matrix Blog](https://developer.nvidia.com/blog/machine-learning-acceleration-vulkan-cooperative-matrices/)
- [Kompute Framework](https://github.com/KomputeProject/kompute)
- [GPGPU ML Inference & Vulkan](https://www.lei.chat/posts/gpgpu-ml-inference-and-vulkan-compute/)
- [Mesa Honeykrisp Sparse Support](https://www.phoronix.com/news/Honeykrisp-Sparse-Mesa-25.1)
- [Mesa 26.0 Release Notes](https://docs.mesa3d.org/relnotes/26.0.0.html)
