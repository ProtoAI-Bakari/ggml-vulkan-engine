# ggml Vulkan Backend Internals

## Source File
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (~16,783 lines)
- All Vulkan compute shaders in `vulkan-shaders/`

---

## How ggml Manages Vulkan Device Memory

### Buffer Structure

```cpp
struct vk_buffer_struct {
    vk::Buffer buffer;              // Vulkan buffer handle
    vk::DeviceMemory device_memory; // Backing memory
    vk::MemoryPropertyFlags memory_property_flags;
    void* ptr;                      // Mapped host pointer (if host-visible)
    size_t size;
    vk::DeviceAddress bda_addr;     // Buffer device address
    vk_device device;               // Owner device
};

struct vk_subbuffer {
    vk_buffer buffer;
    uint64_t offset;
    uint64_t size;
};
```

### Memory Allocation Strategy

`ggml_vk_create_buffer_device()` allocates with fallback chain:

1. **Host-visible + host-coherent** (preferred for unified memory architectures like Apple Silicon)
2. **Device-local** only (dedicated GPU memory)
3. **Device-local + host-visible + host-coherent** (unified fallback)
4. **Device-local + host-visible** (least preferred)

On Apple Silicon (unified memory), the first or third option typically succeeds, meaning **all GPU buffers are directly CPU-mappable** -- no staging copies needed.

### Pre-allocated Buffers

The backend pre-allocates several key buffers:
- `prealloc_x` - Input activations (device-local)
- `prealloc_y` - Second input / activations
- `prealloc_split_k` - Split-K intermediate results
- `sync_staging` - Host-visible staging for sync operations

### Buffer Reuse

Buffers are managed via `shared_ptr` reference counting. The backend caches buffers and reuses them across operations within a compute graph. There is no per-operation allocation -- all memory is pre-allocated during graph planning.

---

## Pipeline Creation and Caching Strategy

### Pipeline Structure

```cpp
struct vk_pipeline_struct {
    std::string name;
    vk::Pipeline pipeline;          // Compiled pipeline
    vk::PipelineLayout layout;
    vk::DescriptorSetLayout dsl;
    uint32_t parameter_count;       // Number of buffer bindings
    uint32_t push_constant_size;
    uint32_t local_size[3];         // Workgroup size
    std::array<uint32_t, 3> wg_denoms;  // Workgroup count denominators
    uint32_t align;                 // Alignment requirement
    std::vector<uint32_t> specialization_constants;
    // Linked list for variants:
    std::shared_ptr<vk_pipeline_struct> next;
    bool compiled;
};
```

### Matmul Pipeline Organization

```cpp
struct vk_matmul_pipeline_struct {
    vk_pipeline l, m, s;       // Large, medium, small (unaligned)
    vk_pipeline a_l, a_m, a_s; // Large, medium, small (aligned)
};

struct vk_matmul_pipeline2 {
    vk_matmul_pipeline f32acc;  // Float32 accumulation
    vk_matmul_pipeline f16acc;  // Float16 accumulation
};
```

### Creation Process

1. **At initialization**: All shader variants are compiled in parallel using `std::async`
2. **SPIR-V is pre-compiled** and embedded as binary blobs (compiled offline)
3. Each pipeline variant gets its own specialization constants (tile sizes, block sizes)
4. Pipelines are stored in the device struct and never recompiled

```cpp
// Parallel compilation example:
compiles.push_back(std::async(
    ggml_vk_create_pipeline_func,
    device, pipeline,
    spv_size, spv_data,
    entrypoint,
    parameter_count,
    push_constant_size,
    wg_denoms,
    specialization_constants,
    ...
));
```

### Pipeline Variant Selection (for matmul)

Three tiers based on matrix dimensions:
- **Small (s)**: M <= 32 or N <= 32, uses 64x64 tiles
- **Medium (m)**: M <= 64 or N <= 64, uses 128x128 tiles
- **Large (l)**: Everything else, uses 128x256 tiles

Each tier has aligned (a_) and unaligned variants:
```cpp
// Aligned check:
bool aligned = (M % pipeline->align == 0) && (N % pipeline->align == 0);
pipeline = aligned ? mmp->a_l : mmp->l;
```

---

## Command Buffer Submission Pattern

### Command Buffer Management

```cpp
struct vk_command_buffer {
    vk::CommandBuffer buf;
    uint64_t use_counter;
    bool in_use;
};

struct vk_command_pool {
    vk::CommandPool pool;
    std::deque<vk_command_buffer> buffers;
};
```

Command buffers are pooled and reused. The `use_counter` enables tracking which command buffers have completed.

### Submission Flow

```
1. ggml_vk_ctx_begin()          - Acquire/create context and command buffer
2. For each operation:
   a. ggml_vk_sync_buffers()    - Insert pipeline barrier (if needed)
   b. ggml_vk_dispatch_pipeline() - Record: push constants + bind pipeline + bind descriptors + dispatch
3. ggml_vk_ctx_end()            - End command buffer recording
4. ggml_vk_submit()             - Submit to queue with semaphores
```

### The Dispatch Pattern

```cpp
static void ggml_vk_dispatch_pipeline(...) {
    // 1. Calculate workgroup counts
    uint32_t wg0 = CEIL_DIV(elements[0], pipeline->wg_denoms[0]);
    uint32_t wg1 = CEIL_DIV(elements[1], pipeline->wg_denoms[1]);
    uint32_t wg2 = CEIL_DIV(elements[2], pipeline->wg_denoms[2]);

    // 2. Update descriptor set with buffer bindings
    vk::DescriptorSet& ds = ctx->descriptor_sets[ctx->descriptor_set_idx++];
    device.updateDescriptorSets({write_descriptor_set}, {});

    // 3. Record commands
    cmd.pushConstants(layout, VK_SHADER_STAGE_COMPUTE, 0, size, &push_constants);
    cmd.bindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    cmd.bindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, {ds}, {});
    cmd.dispatch(wg0, wg1, wg2);
}
```

### Synchronization: ggml_vk_sync_buffers()

This is the **most performance-critical synchronization point**. It inserts a full pipeline barrier:

```cpp
cmd.pipelineBarrier(
    stage_flags,        // srcStageMask = COMPUTE_SHADER
    stage_flags,        // dstStageMask = COMPUTE_SHADER
    {},                 // dependencyFlags
    {memory_barrier},   // ShaderRead|ShaderWrite -> ShaderRead|ShaderWrite
    {},                 // buffer memory barriers
    {}                  // image memory barriers
);
```

**This is called between EVERY pair of dependent dispatches.** For a 32-layer transformer with ~6 ops per layer, that's approximately **192+ barriers per token**.

This is the primary source of overhead and the main reason for the performance gap versus Metal.

---

## How It Handles Different Tensor Layouts

### Layout Detection

```cpp
// Check if tensor is contiguous
bool is_contiguous = ggml_is_contiguous(tensor);

// For non-contiguous tensors, may need copy/transpose
if (!is_contiguous) {
    // Copy to contiguous buffer or use special shader
}
```

### Quantized vs Non-Quantized

- **Non-quantized (f16, f32, bf16)**: Direct access via `data_a[]` buffer
- **Quantized (Q4_K, Q5_K, etc.)**: Access via typed buffers:
  - `data_a[]` - Element-level access
  - `data_a_packed16[]` - 16-bit packed access (for quant blocks)
  - `data_a_packed32[]` - 32-bit packed access

The shader includes files (`types.glsl`) that define structs matching each quantization format's memory layout.

### Batch and Broadcast Handling

```glsl
// Broadcast: A may have fewer batches than B
const uint i03 = i13 / p.broadcast3;
const uint i02 = i12 / p.broadcast2;
const uint batch_idx_a = i03 * p.ne02 + i02;
```

This allows a single weight tensor to be broadcast across multiple batch elements.

---

## What Optimizations Already Exist

### 1. Specialization Constants for Tile Sizes
All tile dimensions (BM, BN, BK, WM, WN, TM, TN) are specialization constants, allowing the driver to optimize code generation for specific sizes.

### 2. Subgroup Operations
`subgroupAdd()` is used for reductions when the device supports `VK_KHR_shader_subgroup_arithmetic`.

### 3. Vectorized Memory Loads
Data is loaded as `vec2` pairs (FLOAT_TYPE_VEC2) and `vec4` quads where possible, reducing the number of memory transactions.

### 4. Dequantize During Load
Quantized data is dequantized while loading into shared memory, avoiding a separate dequant pass.

### 5. Fused Operations
GEMV supports fused bias addition via `fusion_flags`:
```glsl
if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
    temp[j][n] += FLOAT_TYPE(data_fuse0[...]);
}
```

### 6. Pipeline Variant Selection
Three pipeline tiers (l/m/s) + aligned variants = 6 specializations per quant type per operation.

### 7. Split-K
For small M but large K, the matmul is split along K dimension with a separate reduction pass.

### 8. Async Transfer Queue
When available, data transfers use a separate queue, overlapping with compute:
```cpp
if (device->async_use_transfer_queue) {
    result->s->wait_semaphores.push_back(ctx->transfer_semaphore);
}
```

### 9. Batch Dispatch Overflow Handling
For large batch counts exceeding `maxComputeWorkGroupCount[2]`, the dispatch is split into multiple submissions with `base_work_group_z` tracking.

---

## Where the Overhead Comes From

### 1. Pipeline Barriers (~30-40% of overhead)

`ggml_vk_sync_buffers()` inserts a **full pipeline barrier** between every dependent dispatch. For ~360 dispatches per token:
- **360 barriers * ~50-100 us each** = 18-36 ms of barrier overhead alone
- At 40 ms per token (24.8 TPS), barriers could be **45-90% of total time**

**Fix**: Profile actual barrier cost. Consider:
- Using events instead of full barriers where possible
- Batching independent operations to reduce barrier count
- Using subpass dependencies or semaphores instead

### 2. Descriptor Set Updates (~5-10%)

Each dispatch requires a descriptor set update:
```cpp
device.updateDescriptorSets({write_descriptor_set}, {});
```

This is a CPU-side cost that serializes dispatch recording.

**Fix**: Pre-bind descriptor sets for common buffer configurations.

### 3. Missing Cooperative Matrix (~20-30% for GEMM, less for GEMV)

Without `VK_KHR_cooperative_matrix`, all matmul uses manual FMA loops instead of hardware MMA. This primarily affects prompt processing (GEMM) more than token generation (GEMV, which is bandwidth-bound).

### 4. Command Buffer Recording (~5%)

Each operation is recorded individually. Overhead from:
- Push constant updates
- Pipeline binding
- Descriptor set binding

### 5. Suboptimal Tile Sizes (~5-10%)

The tile sizes are tuned for NVIDIA/AMD hardware, not specifically for Apple AGX. AGX has different optimal sizes due to:
- Different shared memory size (60 KB vs 48 KB for NVIDIA)
- Different register file organization
- Different cache hierarchy (8 KB L1)

**Fix**: Profile and tune tile sizes specifically for AGX hardware.

### 6. No Operation Fusion (~10%)

Most operations are dispatched individually:
- RMSNorm -> separate dispatch
- MatMul -> separate dispatch
- Residual Add -> separate dispatch
- SiLU/GeLU -> separate dispatch

Each boundary requires a barrier. Fusing adjacent operations would reduce dispatch count.

---

## The 12% Gap to llama.cpp Native (Vulkan)

On non-Apple hardware (NVIDIA/AMD), the Vulkan backend achieves ~88% of the native CUDA/ROCm performance. The remaining 12% comes from:

1. **Pipeline barrier overhead**: ~5% (more barriers than native implementations)
2. **Descriptor set management**: ~2% (CUDA uses unified addressing)
3. **Push constant vs uniform buffer**: ~1% (minor)
4. **Shader compiler differences**: ~2-3% (NVIDIA's GLSL compiler vs native CUDA compiler)
5. **Graph optimization**: ~2% (less aggressive fusion)

On Apple Silicon, the gap is much larger (6x for pp, 2.8x for tg) due to the missing cooperative matrix support.

---

## Key Code Paths for Optimization

### For batch=1 token generation (our primary target):

1. **mul_mat_vec dispatch**: `ggml-vulkan.cpp:7400-7600` -- where GEMV operations are dispatched
2. **Pipeline selection**: `ggml_vk_guess_matmul_pipeline()` line 7030 -- may need AGX-specific logic
3. **Barrier insertion**: `ggml_vk_sync_buffers()` line 2775 -- minimize or replace
4. **Buffer allocation**: `ggml_vk_create_buffer_device()` line 2722 -- ensure unified memory path
5. **Workgroup size**: Specialization constants in pipeline creation ~line 3200-3300
