# Vulkan Compute Optimization Techniques for Matmul

## Sources
- NVIDIA cooperative matrix blog post
- Vulkan Documentation Project (VK_KHR_cooperative_matrix)
- lei.chat GPGPU ML inference article
- Parallel matmul on GPGPU with Vulkan
- dev.to advanced GPU optimization guide
- jeffbolznv/vk_cooperative_matrix_perf benchmark
- llama.cpp Vulkan backend (DeepWiki)
- Khronos GLSL_KHR_cooperative_matrix specification

---

## Best Practices for Tiled Matmul in Vulkan

### 1. Hierarchical Tiling

Three levels of tiling, matching GPU hardware hierarchy:

```
Level 1: Workgroup tile    (BM x BN) - maps to threadgroup/workgroup
Level 2: Warp/subgroup tile (WM x WN) - maps to SIMD-group
Level 3: Thread tile        (TM x TN) - maps to per-thread registers
```

**Recommended sizes for Apple AGX (32-wide SIMD)**:
- Workgroup: 64x64 to 128x128
- Subgroup: 32x32
- Thread: 4x2 to 4x4

### 2. K-Dimension Blocking

Process K in blocks of BK:
```glsl
for (uint k = 0; k < K; k += BK) {
    // Load BM x BK from A -> shared
    // Load BK x BN from B -> shared
    barrier();
    // Multiply BM x BK x BN from shared -> registers
    barrier();
}
```

**BK selection**:
- f16/f32: BK=32 (matches cache line and gives good reuse)
- Quantized (Q4_K): BK=16 (matches quant block boundaries, 256-element blocks)

### 3. Shared Memory Padding

Avoid bank conflicts by adding padding to shared memory stride:
```glsl
// Without padding: columns = BK/2 (storing vec2)
// With padding: columns = BK/2 + 1 (or +4 for coopmat alignment)
#define SHMEM_STRIDE (BK / 2 + 1)
shared FLOAT_TYPE_VEC2 buf_a[BM * SHMEM_STRIDE];
```

The +1 ensures that threads in the same SIMD-group access different banks when loading consecutive elements.

### 4. Vectorized Memory Access

Load multiple elements per thread per memory operation:
```glsl
// Load 4 f16 values as a single uint64:
layout (binding = 0) readonly buffer A_PACKED32 { A_TYPE_PACKED32 data_a_packed32[]; };

// Or use vec4 loads for f32:
layout (binding = 1) readonly buffer B_V4 { vec4 data_b_v4[]; };
```

Benefits:
- Reduces number of memory transactions
- Improves bandwidth utilization (fewer but wider loads)
- Better cache line utilization

### 5. Load Coalescing

Ensure adjacent threads access adjacent memory addresses:
```glsl
// GOOD: Thread 0 loads address 0, thread 1 loads address 4, etc.
uint idx = gl_LocalInvocationID.x;
float val = data[base + idx];

// BAD: Thread 0 loads address 0, thread 1 loads address 1024
uint idx = gl_LocalInvocationID.x * stride;
float val = data[base + idx];
```

For shared memory loading, map threads to the K dimension (contiguous in memory) rather than the M dimension.

---

## Subgroup Operations for Matmul

### Available in Vulkan 1.1+ (with extensions)

```glsl
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_shuffle : enable
#extension GL_KHR_shader_subgroup_ballot : enable
```

### Subgroup Shuffle for Data Sharing

Instead of going through shared memory, subgroup shuffle allows direct register-to-register communication:

```glsl
// Broadcast element from lane 0 to all lanes:
float val = subgroupBroadcast(my_val, 0);

// Get value from another lane:
float neighbor_val = subgroupShuffle(my_val, target_lane);

// Butterfly shuffle for reduction:
float sum = my_val;
sum += subgroupShuffleXor(sum, 1);
sum += subgroupShuffleXor(sum, 2);
sum += subgroupShuffleXor(sum, 4);
// ... etc for full reduction
```

### Subgroup Reduction

```glsl
// Fast parallel sum across all lanes:
float total = subgroupAdd(partial_sum);

// Only lane 0 has the correct result (or all lanes for inclusive)
if (gl_SubgroupInvocationID == 0) {
    output[idx] = total;
}
```

### Using Subgroups to Avoid Shared Memory Barriers

For GEMV (batch=1), the reduction can be done entirely within a subgroup:
1. Each thread computes a partial dot product
2. `subgroupAdd()` reduces within the subgroup (no barrier needed)
3. If workgroup has multiple subgroups, use shared memory only for inter-subgroup reduction

This eliminates most barrier overhead.

---

## Memory Coalescing Patterns

### Matrix A (Row-Major, typically the weights)

For quantized weights, the quant block layout determines the optimal access pattern:
- Q4_K: 256-element blocks with per-block scales
- Each thread should process elements from the same quant block
- Adjacent threads should load from adjacent blocks

### Matrix B (Column-Major or transposed for the activations)

```glsl
// Pattern: Each thread in the subgroup loads adjacent elements
const uint lane = gl_SubgroupInvocationID;
const uint base = block_start + lane;
float b_val = data_b[base];  // Coalesced: lanes 0-31 load addresses base..base+31
```

### Shared Memory Access Pattern

When reading from shared memory for the compute phase:
```glsl
// Each thread reads its own row of A and column of B
// A: Row access (no bank conflicts if stride is padded)
cache_a = buf_a[thread_row * SHMEM_STRIDE + k];
// B: Column access (potential bank conflicts without padding)
cache_b = buf_b[thread_col * SHMEM_STRIDE + k];
```

---

## Pipeline Barriers and Synchronization

### Minimizing Barrier Cost

1. **One barrier between load and compute** (required for shared memory consistency)
2. **One barrier between iterations** (prevents next load from overwriting current data)
3. **No additional barriers** within the compute phase (registers are thread-private)

```glsl
for (k ...) {
    load_to_shared();
    barrier();              // Barrier 1: Shared memory written, now safe to read
    compute_from_shared();
    barrier();              // Barrier 2: Compute done, now safe to overwrite shared
}
```

### Avoiding Barriers with Subgroup Operations

If the entire reduction fits within one subgroup (32 threads), **no barrier is needed**:
```glsl
// Subgroup operations are implicitly synchronized
float sum = subgroupAdd(partial);  // No barrier required
```

### Pipeline Barriers Between Dispatches

In Vulkan, each compute dispatch is independent. Between dispatches that share buffers:
```cpp
vkCmdPipelineBarrier(cmd,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    0, 1, &memBarrier, 0, NULL, 0, NULL);
```

**Optimization**: Group independent operations together before inserting barriers. Batch multiple matmuls that don't depend on each other.

---

## Descriptor Set vs Push Constant Performance

### Push Constants (Preferred for Small Data)

```glsl
layout (push_constant) uniform params {
    uint M, N, K;
    uint stride_a, stride_b, stride_d;
    // ... up to 128 bytes (typical Vulkan minimum)
};
```

**Advantages**:
- No descriptor allocation
- No descriptor set binding
- Updated inline with command buffer
- Near-zero latency

**Limitations**:
- 128 bytes minimum guaranteed (many GPUs support 256)
- Only for small, per-dispatch data (dimensions, strides)

### Descriptor Sets (Required for Buffers)

```glsl
layout (binding = 0) readonly buffer A { ... };
layout (binding = 1) readonly buffer B { ... };
layout (binding = 2) writeonly buffer D { ... };
```

**Optimization tips**:
- Pre-allocate descriptor pools
- Use descriptor set caching (don't recreate for each dispatch)
- Group infrequently-changing bindings in set 0, frequently-changing in set 1
- Use `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC` for buffer offset changes

llama.cpp allocates descriptor sets in pools and reuses them across dispatches.

---

## Buffer vs Image Storage for Weights

### Storage Buffers (Used by llama.cpp)

```glsl
layout (binding = 0) readonly buffer A { A_TYPE data_a[]; };
```

**Advantages**:
- Simple, direct addressing
- No format conversion overhead
- Can use packed types (uint32, uvec4) for quant data
- Random access pattern support
- Works well with Vulkan on AGX

### Image/Texture Storage

```glsl
layout (binding = 0, rgba16f) readonly uniform image2D weights;
```

**Potential advantages**:
- Hardware texture filtering
- Built-in format conversion
- 2D/3D spatial locality optimized by hardware

**Why NOT used for ML weights**:
- Quantized formats (Q4_K) don't map to standard image formats
- Random access patterns don't benefit from texture cache
- Buffer access is more predictable and cacheable for streaming loads
- Image descriptors have higher setup cost

**Recommendation**: Use storage buffers for all ML weight data. The texture path offers no advantage for quantized LLM inference.

---

## Cooperative Matrix (When Available)

### VK_KHR_cooperative_matrix

GLSL usage:
```glsl
#extension GL_KHR_cooperative_matrix : enable

// Declare matrix types
coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> matA;
coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matB;
coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> matC;

// Load from shared memory
coopMatLoad(matA, shared_a, offset, stride, gl_CooperativeMatrixLayoutRowMajor);
coopMatLoad(matB, shared_b, offset, stride, gl_CooperativeMatrixLayoutColumnMajor);

// Hardware multiply-accumulate
matC = coopMatMulAdd(matA, matB, matC);

// Store result
coopMatStore(matC, output, offset, stride, gl_CooperativeMatrixLayoutRowMajor);
```

### Supported Matrix Sizes (varies by hardware)
- NVIDIA Turing+: 16x16x16 f16 -> f32
- NVIDIA Ampere+: 16x16x16, 8x8x8, various integer types
- Apple AGX: 8x8x8 (via Metal, NOT via Vulkan currently)

### NOT Available on Honeykrisp
The entire cooperative matrix path is dead code when running on Asahi Linux. All matmul must use the scalar FMA path.

---

## Practical Optimization Checklist for AGX Vulkan

1. [ ] Use 128-thread workgroups (4 subgroups of 32)
2. [ ] Tile BM=64, BN=64, BK=32 for f16; BK=16 for quant
3. [ ] Pad shared memory stride by +1 vec2 element
4. [ ] Vectorize global loads (vec4 where possible)
5. [ ] Use subgroupAdd for reductions instead of shared memory tree
6. [ ] Minimize barriers: exactly 2 per K-iteration
7. [ ] Use push constants for dimensions/strides
8. [ ] Pre-allocate descriptor pools
9. [ ] Dequantize during shared memory load (not during compute)
10. [ ] Use split-K for small M with large K
