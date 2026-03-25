# Vulkan Custom Shader Research — Sources & Strategy
## For achieving TFLOPS on Apple Silicon via Vulkan
## 2026-03-25

---

## 1. HARDWARE TRUTH (from metal-benchmarks)

The M1 Max/Ultra GPU is CAPABLE of TFLOPS. Proven numbers:

| Chip | GPU Cores | FP32 TFLOPS | Clock | ALUs per core | SIMD width |
|------|-----------|-------------|-------|---------------|------------|
| M1 Max | 32 | **10.6** | 1.296 GHz | 128 | 32 |
| M1 Ultra | 64 | **21.2** | 1.296 GHz | 128 | 32 |

Per-core: 256 FP32 ops/cycle, 4 independent schedulers, each dispatching 32-wide SIMD.
**We're hitting 541 GFLOPS = 5% of M1 Max.** 95% of the GPU is idle.

Shared memory per core: ~60 KB. Register file: ~208 KB. This is PLENTY for tiled matmul.

Source: [philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks)

---

## 2. MLX SOURCE CODE — WHERE THE MAGIC LIVES

MLX's GEMM implementation is called "Steel" and lives at:
```
github.com/ml-explore/mlx/tree/main/mlx/backend/metal/kernels/steel/gemm/
```

### Key files:
| File | Purpose |
|------|---------|
| `gemm.h` | Main GEMM dispatch and kernel selection |
| `mma.h` | Matrix Multiply-Accumulate — the core operation |
| `loader.h` | Tiled data loading from global → shared memory |
| `params.h` | Tile sizes, workgroup configs, matrix shapes |
| `transforms.h` | Data format transformations (fp16↔fp32, quantized) |
| `kernels/` | Actual Metal shader source files |

### How MLX achieves TFLOPS:
1. **simdgroup_matrix** — Apple's "tensor core" equivalent. Hardware matrix multiply.
   - 8x8 matrix multiply-accumulate in a single instruction
   - Bypasses scalar ALU path entirely
   - Available on M1+ GPUs
2. **Tiled blocking** — BM×BN×BK tiles fit in shared memory
3. **Double buffering** — Load next tile while computing current tile
4. **Register-level accumulation** — Partial results stay in registers, never hit memory
5. **Vectorized loads** — 128-bit loads from unified memory

### Translation path to Vulkan:
```
simdgroup_matrix (Metal)  →  GL_KHR_cooperative_matrix (Vulkan)
threadgroup memory (Metal) →  shared memory (GLSL)
simdgroup (Metal)          →  subgroup (Vulkan/GLSL)
float4 loads (Metal)       →  vec4 loads (GLSL)
```

**CRITICAL**: The Asahi Vulkan driver does NOT yet expose `GL_KHR_cooperative_matrix`.
llama.cpp's pp512 is "3x faster with coopmat than without on other platforms."
This is the single biggest missing piece. Without it, we can still tile + shared memory
but lose the hardware matrix multiply acceleration.

Sources:
- [MLX GitHub](https://github.com/ml-explore/mlx)
- [MLX Steel GEMM kernels](https://github.com/ml-explore/mlx/tree/main/mlx/backend/metal/kernels/steel/gemm)
- [MLX Custom Metal Kernels docs](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
- [SIMD vector matrix multiplication discussion](https://github.com/ml-explore/mlx/issues/171)

---

## 3. LLAMA.CPP VULKAN SHADERS — THE REFERENCE IMPLEMENTATION

llama.cpp has production Vulkan compute shaders at:
```
github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-vulkan/vulkan-shaders/
```

### Key shader: `mul_mm.comp` (matrix multiply)
- **Tiling**: BM×BN blocks (default 64×64)
- **Workgroup**: Runtime-configurable via specialization constants
- **Shared memory**: Double-buffered with stride padding to avoid bank conflicts
  - `SHMEM_STRIDE = (BK / 2 + 1)` — prevents bank conflicts
- **Thread-level tiles**: TM×TN per thread (default 4×2)
- **Warp organization**: WM×WN per warp (default 32×32)
- **Cooperative matrix**: Optional via `GL_KHR_cooperative_matrix` extension
- **Split-K**: Large matmuls split K dimension across workgroups, reduce separately
- **Pipeline variants**: 200+ compiled variants for different dtypes and tile sizes
  - l/m/s (large/medium/small) based on matrix dimensions
  - Aligned variants with vectorized 16-byte loads

### Performance on Apple Silicon (M2 Max, 7B Q4_K):
| Backend | Prompt (pp512) | Decode (tg128) |
|---------|---------------|----------------|
| Metal | 580 TPS | 61 TPS |
| Vulkan | 92 TPS | 22 TPS |
| **Gap** | **6.3x** | **2.8x** |

Vulkan improved from 35→92 TPS pp512 in 2 months of optimization (2024).
The gap is primarily from missing cooperative matrix support on Asahi.

Sources:
- [Metal vs Vulkan performance research](https://github.com/ggml-org/llama.cpp/issues/10982)
- [Vulkan shader source: mul_mm.comp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp)
- [Vulkan backend architecture (DeepWiki)](https://deepwiki.com/ggml-org/llama.cpp/4.4-vulkan-backend)
- [Vulkan performance discussion](https://github.com/ggml-org/llama.cpp/discussions/10879)

---

## 4. EXISTING VULKAN MATMUL PROJECTS

### VulkanShaderCUDA
PyTorch-like tensor ops via Vulkan compute shaders. Implements matmul with shared memory tiling.
- Languages: C++ 58%, Python 28%
- Uses SPIR-V shaders, PyBind11 for Python integration
- Could be adapted as our custom kernel backend

Source: [VulkanShaderCUDA](https://github.com/waefrebeorn/VulkanShaderCUDA)

### metalQwen3
Full Qwen3-4B transformer in pure Metal compute shaders. NO CPU fallbacks.
- **75 TPS** on Apple Silicon (2.1x over CPU)
- Implements: RMSNorm, QuantizedMatMul (INT8), Softmax, SwiGLU, RoPE, Multi-Head Attention
- All in Metal Shading Language
- Shows what's possible when you go full GPU

Source: [metalQwen3](https://github.com/BoltzmannEntropy/metalQwen3)

---

## 5. THE GAP ANALYSIS: WHY WE'RE SLOW AND HOW TO FIX IT

### Current: PyTorch Vulkan torch.mm
- Stores tensors as **Vulkan images** (texture format, not buffers)
- No tiling — one monolithic operation
- No shared memory usage
- No cooperative matrix / simdgroup
- Per-operation Vulkan command buffer submit + fence wait
- **Result: 541 GFLOPS = 5% utilization**

### Target: Custom Vulkan compute shader
| Feature | PyTorch Vulkan (now) | Custom shader (target) |
|---------|---------------------|----------------------|
| Storage | Images | **Buffers** |
| Tiling | None | **64×64 BM×BN** |
| Shared memory | None | **Double-buffered** |
| Cooperative matrix | No | **If driver supports** |
| Command batching | One per op | **Fused pipeline** |
| Data type | FP32 only | **FP16 + FP32 accumulate** |
| Expected GFLOPS | 541 | **3,000-6,000** |
| Expected utilization | 5% | **30-60%** |

---

## 6. IMPLEMENTATION STRATEGY

### Phase 1: Port llama.cpp's mul_mm.comp (20-30 hours)
1. Extract `mul_mm.comp` from llama.cpp
2. Adapt for standalone use (remove ggml dependencies)
3. Wire into PyTorch via custom Vulkan extension or direct dispatch
4. Test on Asahi — does it hit their 92 TPS pp512 on Apple Silicon?
5. If yes: we have a working fast shader. Integrate into vLLM.

### Phase 2: Study MLX Steel kernels (10-20 hours)
1. Read `mma.h`, `loader.h`, `gemm.h` — understand the tiling strategy
2. Translate the tiling/blocking pattern from MSL to GLSL
3. Focus on: tile sizes, shared memory layout, vectorized loads
4. Skip simdgroup_matrix for now (no Vulkan coopmat on Asahi yet)

### Phase 3: Write fused MLP shader (20-30 hours)
1. Fuse gate_proj + up_proj + SiLU + down_proj into ONE dispatch
2. Intermediate results stay in shared memory / registers
3. Only 1 round-trip to global memory per layer (input in, output out)
4. This alone could 3-5x MLP throughput

### Phase 4: FP16 matmul (10-15 hours)
1. Check if Asahi Vulkan supports `shaderFloat16` capability
2. If yes: FP16 inputs, FP32 accumulate = 2x less memory bandwidth
3. If no: mixed precision via manual fp16 pack/unpack in shader

### Phase 5: Push for cooperative matrix (long-term, depends on Asahi driver)
1. Check Mesa 25.3.6 for `VK_KHR_cooperative_matrix` support
2. If available: enable in shader, expect 3x matmul speedup
3. If not: file feature request with Asahi team, provide benchmark data

---

## 7. REFERENCE PROJECTS TO CLONE/STUDY

```bash
# MLX (Metal kernels to study)
git clone https://github.com/ml-explore/mlx.git
# Key: mlx/backend/metal/kernels/steel/gemm/

# llama.cpp (Vulkan shaders to port)
git clone https://github.com/ggml-org/llama.cpp.git
# Key: ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp

# VulkanShaderCUDA (Vulkan compute framework)
git clone https://github.com/waefrebeorn/VulkanShaderCUDA.git
# Key: shared memory matmul shader

# metalQwen3 (full Metal transformer — shows what's possible)
git clone https://github.com/BoltzmannEntropy/metalQwen3.git
# Key: all GPU, no CPU fallbacks, 75 TPS

# Metal benchmarks (Apple GPU microarchitecture data)
git clone https://github.com/philipturner/metal-benchmarks.git
# Key: TFLOPS numbers, ALU counts, shared memory sizes
```

---

## 8. KEY INSIGHT: GAMES PROVE THE HARDWARE CAN DO IT

AAA games run at 60 FPS on Asahi Linux via Vulkan (Honeykrisp driver).
Games do thousands of draw calls per frame, each involving matmul-like shader operations.
The GPU is NOT the bottleneck — our SHADER CODE is.

Same silicon. Same driver. Same memory bus. Different code = different performance.
MLX proves it with Metal. We need to prove it with Vulkan.

Source: [AAA Gaming on ARM64 Mac](https://boilingsteam.com/m1-arm64-gaming-progress/)
