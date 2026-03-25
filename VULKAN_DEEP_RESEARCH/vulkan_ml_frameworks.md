# Existing Vulkan ML Frameworks

## Sources
- KomputeProject/kompute (GitHub)
- waefrebeorn/VulkanShaderCUDA (GitHub)
- lei.chat GPGPU ML inference article
- NVIDIA cooperative matrix blog
- AMD ROCm/Vulkan comparison
- Khronos Vulkan ML resources

---

## Framework Inventory

### 1. Kompute (Linux Foundation)
- **URL**: https://github.com/KomputeProject/kompute
- **Status**: Active, backed by LF AI & Data Foundation
- **Language**: C++ core, Python bindings
- **Purpose**: General-purpose GPU compute via Vulkan
- **Key features**:
  - Cross-vendor (AMD, NVIDIA, Qualcomm, Apple via MoltenVK)
  - Tensor-based API similar to PyTorch
  - Async execution support
  - Automatic memory management (device, host, staging)
  - SPIR-V shader loading
- **Matmul performance**: No published benchmarks for raw matmul throughput
- **Memory management pattern**:
  ```cpp
  // Tensor-based: manages VkBuffer + VkDeviceMemory
  auto tensorA = mgr.tensor({1.0, 2.0, 3.0});
  auto tensorB = mgr.tensor({4.0, 5.0, 6.0});
  // Shader loaded from SPIR-V, bound to tensors
  mgr.sequence()->record<kp::OpAlgoDispatch>(algo, tensors);
  ```

### 2. VulkanShaderCUDA
- **URL**: https://github.com/waefrebeorn/VulkanShaderCUDA
- **Status**: Experimental/proof-of-concept
- **Purpose**: CUDA-like functionality using Vulkan + GLSL shaders
- **Operations**: Addition, matmul, ReLU, softmax, 2D convolution, pooling
- **Architecture**: Dynamic pipelines with SPIR-V shader execution
- **Key insight**: Demonstrates that CUDA-style ops can be mapped to Vulkan compute, but performance is not competitive with native CUDA

### 3. llama.cpp Vulkan Backend
- **URL**: https://github.com/ggml-org/llama.cpp
- **Status**: Production-quality, actively developed
- **Purpose**: LLM inference with Vulkan GPU acceleration
- **Matmul performance** (M2 Max):
  - Prompt processing: 92 t/s (Vulkan) vs 580 t/s (Metal)
  - Token generation: 22 t/s (Vulkan) vs 61 t/s (Metal)
- **Memory management**: Custom buffer allocator with device-local and host-visible pools
- **Key insight**: Most mature Vulkan ML inference implementation, but significant performance gap on Apple Silicon

### 4. MLC-LLM (Machine Learning Compilation)
- **URL**: https://github.com/mlc-ai/mlc-llm
- **Status**: Active
- **Purpose**: Universal LLM inference engine
- **Vulkan support**: Via TVM's Vulkan runtime
- **Key features**: Compiler-optimized kernels, model compilation
- **Note**: Primarily targets Android/mobile via Vulkan

### 5. NCNN
- **URL**: https://github.com/Tencent/ncnn
- **Status**: Active (Tencent)
- **Purpose**: Mobile neural network inference
- **Vulkan support**: Primary GPU backend
- **Key features**:
  - Compute shader based inference
  - Model optimization (quantization, pruning)
  - Mobile-first design
- **Performance**: Competitive on mobile GPUs (Adreno, Mali)

### 6. TVM (Apache TVM)
- **URL**: https://github.com/apache/tvm
- **Status**: Active
- **Purpose**: Deep learning compiler framework
- **Vulkan support**: Via Vulkan runtime target
- **Key features**: Auto-tuning, operator fusion, code generation
- **Matmul**: Auto-tuned GEMM kernels for Vulkan target

### 7. ONNX Runtime (via Vulkan EP)
- **Status**: Limited Vulkan support
- **Purpose**: Cross-platform inference engine
- **Note**: Vulkan execution provider is less mature than CUDA/DirectML

### 8. Vulkan Compute Samples (Google/Khronos)
- Various sample projects demonstrating compute patterns
- Not production frameworks, but useful reference implementations

---

## Matmul Performance Comparison

| Framework | Hardware | Precision | Performance | Notes |
|---|---|---|---|---|
| llama.cpp (Metal) | M2 Max | Q4_K_M | 580 t/s pp, 61 t/s tg | Gold standard on Apple |
| llama.cpp (Vulkan) | M2 Max | Q4_K_M | 92 t/s pp, 22 t/s tg | 6x slower pp, 2.8x tg |
| llama.cpp (Vulkan) | M1 Ultra | Q4_K_M | ~24.8 t/s tg | Our current baseline |
| MLX | M2 Max | Various | ~700+ t/s pp | Uses simdgroup_matrix |
| MLC-LLM (Vulkan) | Various | f16 | Varies | Mobile-focused |
| NCNN (Vulkan) | Mobile | Various | Varies | Not LLM-focused |

**Key observation**: No Vulkan framework achieves competitive matmul performance on Apple Silicon compared to Metal. The gap is fundamental: no cooperative matrix support.

---

## Code Patterns We Can Reuse

### From Kompute: Memory Management Pattern

```cpp
class Manager {
    // Pool-based buffer allocation
    // Automatic staging buffer management
    // Fence-based synchronization
    void sequence()->eval<kp::OpMemoryBarrier>(tensors);
};
```

**Useful for**: Clean buffer lifecycle management in our custom Vulkan pipeline.

### From llama.cpp: Shader Dispatch Pattern

```cpp
// Specialization constants for compile-time optimization
VkSpecializationMapEntry entries[] = {
    {0, 0, sizeof(uint32_t)},   // BLOCK_SIZE
    {1, 4, sizeof(uint32_t)},   // BM
    {2, 8, sizeof(uint32_t)},   // BN
    // ...
};

// Push constants for runtime parameters
vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT,
    0, sizeof(params), &params);

// Dispatch with calculated workgroup counts
vkCmdDispatch(cmd, wg_x, wg_y, wg_z);
```

**Useful for**: The pattern of specialization constants for tile sizes + push constants for dimensions is the standard approach.

### From VulkanShaderCUDA: GLSL Shader Template

```glsl
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer InputA { float a[]; };
layout(set = 0, binding = 1) buffer InputB { float b[]; };
layout(set = 0, binding = 2) buffer Output { float c[]; };

layout(push_constant) uniform Params { uint M, N, K; } p;

shared float tile_a[TILE_SIZE][TILE_SIZE];
shared float tile_b[TILE_SIZE][TILE_SIZE];

void main() {
    // Standard tiled matmul...
}
```

### From TVM: Auto-Tuning Approach

TVM searches the space of tile sizes, thread mappings, and loop orderings to find the optimal configuration for each hardware target. We could adopt a similar approach:
1. Define a search space of (BM, BN, BK, WM, WN, TM, TN) combinations
2. Benchmark each on target hardware
3. Select the best for each matrix size range

---

## Memory Management Patterns

### Pattern 1: Pre-allocated Device Pools (llama.cpp)

```
Initialization:
  1. Allocate large device-local buffer for weights (read-only)
  2. Allocate device-local scratch buffers for intermediates
  3. Allocate host-visible staging buffers for input/output

Runtime:
  1. Copy input tokens to staging buffer
  2. Transfer staging -> device (or use host-visible device memory)
  3. Dispatch compute shaders
  4. Transfer device -> staging for output
  5. Read results from staging
```

### Pattern 2: Unified Memory (Apple Silicon Advantage)

On Apple Silicon, device-local and host-visible can be the **same physical memory**:
```
Optimization: Skip staging buffers entirely
  1. Allocate host-visible + device-local buffers
  2. Map directly for CPU access
  3. Dispatch compute shaders on same buffer
  4. Read results directly from mapped pointer
```

This eliminates all copy overhead, which is significant for small tensors.

### Pattern 3: Buffer Reuse and Aliasing

For intermediate tensors in a transformer:
```
Layer 1 output -> Buffer A
Layer 2 output -> Buffer B
Layer 3 output -> Buffer A (reuse)
Layer 4 output -> Buffer B (reuse)
```

Requires careful synchronization but halves intermediate memory usage.

---

## Recommendations

### For Our Project (50+ TPS on M1 Ultra)

1. **Don't use Kompute or other frameworks** -- too much abstraction, need raw Vulkan control
2. **Reuse llama.cpp's shader architecture** -- it's already the most optimized Vulkan matmul for LLM inference
3. **Focus optimization within the ggml-vulkan pipeline** rather than building from scratch
4. **Key optimizations should target**:
   - The GEMV (mul_mat_vec) shader for batch=1
   - Pipeline barrier reduction
   - Buffer management for unified memory
   - Specialization constant tuning for AGX
