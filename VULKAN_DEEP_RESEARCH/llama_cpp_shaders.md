# llama.cpp Vulkan Shader Deep Dive

## Source Files Analyzed
- `vulkan-shaders/mul_mm.comp` - Main GEMM shader (batch > 1 matmul)
- `vulkan-shaders/mul_mat_vec.comp` - Generic GEMV (batch=1)
- `vulkan-shaders/mul_mat_vec_q4_k.comp` - Q4_K specialized GEMV
- `vulkan-shaders/mul_mat_vec_base.glsl` - GEMV base with reduction
- `vulkan-shaders/mul_mm_funcs.glsl` - Shared memory load/dequant for GEMM
- `vulkan-shaders/dequant_funcs.glsl` - Dequantization functions
- `ggml-vulkan.cpp` - Pipeline creation and dispatch logic

---

## mul_mm.comp: Full Tiling Strategy

### Tile Dimensions (Specialization Constants)

```glsl
layout (constant_id = 0) const uint BLOCK_SIZE = 64;   // threads per workgroup
layout (constant_id = 1) const uint BM = 64;            // tile rows
layout (constant_id = 2) const uint BN = 64;            // tile columns
layout (constant_id = 3) const uint BK = 16;            // tile K (quant types)
layout (constant_id = 4) const uint WM = 32;            // warp tile M
layout (constant_id = 5) const uint WN = 32;            // warp tile N
layout (constant_id = 6) const uint WMITER = 2;         // warp M iterations
layout (constant_id = 7) const uint TM = 4;             // thread tile M
layout (constant_id = 8) const uint TN = 2;             // thread tile N
layout (constant_id = 9) const uint TK = 1;             // coopmat only
layout (constant_id = 10) const uint WARP = 32;         // subgroup size
```

**For f16/f32 types**: BK is hardcoded to 32 with BK_STEP=4.
**For quantized types**: BK defaults to 16, BK_STEP=2.

### Pipeline Variant Tile Sizes (from ggml-vulkan.cpp)

**Without coopmat2 (scalar path)**:
```
l_warptile = { 256, 128, 256, 64, 1 }    -> BLOCK=256, BM=128, BN=256, BK=64
m_warptile = { 256, 128, 128, 64, 0 }    -> BLOCK=256, BM=128, BN=128, BK=64
s_warptile = { 128,  64,  64, 64, 0 }    -> BLOCK=128, BM=64,  BN=64,  BK=64
```
l_wg_denoms = {128, 256, 1}
m_wg_denoms = {128, 128, 1}
s_wg_denoms = {64, 64, 1}

**With coopmat2**:
```
l_warptile = { 128, 128, 128, 16, ... }   -> BLOCK=128, BM=128, BN=128, BK=16
m_warptile = { 128,  64,  64, 16, ... }   -> BLOCK=128, BM=64,  BN=64,  BK=16
s_warptile = { 32,   32,  32, 16, ... }   -> BLOCK=32,  BM=32,  BN=32,  BK=16
```

### Shared Memory Layout

```glsl
#ifdef COOPMAT
#define SHMEM_STRIDE (BK / 2 + 4)    // Extra padding for coopmat alignment
#else
#define SHMEM_STRIDE (BK / 2 + 1)    // +1 element to avoid bank conflicts
#endif

shared FLOAT_TYPE_VEC2 buf_a[BM * SHMEM_STRIDE];
shared FLOAT_TYPE_VEC2 buf_b[BN * SHMEM_STRIDE];
```

Data is stored as **vec2** (pairs), so SHMEM_STRIDE is in units of vec2.
For BK=32: SHMEM_STRIDE = 32/2 + 1 = 17 vec2 = 34 floats per row.

**Total shared memory** (BM=128, BN=128, BK=32):
- buf_a: 128 * 17 * 8 bytes = ~17 KB
- buf_b: 128 * 17 * 8 bytes = ~17 KB
- Total: ~34 KB (fits in AGX threadgroup memory)

### The Main Loop

```glsl
for (uint block = start_k; block < end_k; block += BK) {
    // 1. Load A and B tiles from global -> shared memory
    load_a_to_shmem(pos_a, ...);  // Includes dequantization
    load_b_to_shmem(pos_b, ...);
    barrier();

    // 2. Compute tile multiply from shared -> registers
    for (uint i = 0; i < BK / BK_STEP; i++) {
        // Load from shared into thread-local registers
        cache_a[...] = buf_a[...];
        cache_b = buf_b[...];
        // Manual FMA accumulation
        sums[idx].x = fma(cache_a.x, cache_b.x, fma(cache_a.y, cache_b.y, sums[idx].x));
    }
    barrier();
}
```

### Dequantization During Load (Q4_K_M example)

For quantized types, dequantization happens **inside load_a_to_shmem()**, converting packed quant data to f16 pairs in shared memory. This is critical -- the shared memory always holds dequantized values.

From mul_mm_funcs.glsl for Q4_K:
```glsl
// Each thread loads one block of 4 values, dequantizes, stores to shared
const uint ib = idx / 64;
const uint iqs = (idx % 64) * 2;
const vec2 loadd = vec2(data_a[ib].dm);  // scale + min
// ... complex scale extraction ...
// Final dequant: val = dm.x * scale * quant - dm.y * min
buf_a[buf_idx] = FLOAT_TYPE_VEC2(v.xy);
buf_a[buf_idx + 1] = FLOAT_TYPE_VEC2(v.zw);
```

---

## mul_mat_vec.comp: Batch=1 Handling

### Key Differences from mul_mm.comp

1. **No shared memory tiling** for the weight matrix (A) -- loaded directly from global memory
2. **Each workgroup computes NUM_ROWS output elements** (typically 1-4)
3. **Parallel reduction across K** using either:
   - Shared memory reduction tree
   - `subgroupAdd()` (faster, when available)
   - Combined subgroup + shared memory (large workgroups)

### Iteration Strategy

```glsl
#define K_PER_ITER 8    // for quantized types (process 8 K elements per iteration)
#define K_PER_ITER 2    // for f16/f32

// Each thread processes K_PER_ITER elements per iteration
uint num_iters = p.ncols / (K_PER_ITER * BLOCK_SIZE);

// Manual partial unroll (4x, then 2x, then 1x for tail)
while (i < unrolled_iters) {
    [[unroll]] for (uint k = 0; k < 4; ++k) {
        iter(temp, first_row, num_rows, tid, i*K_PER_ITER, false);
        i++;
    }
}
```

### Q4_K Specific GEMV (mul_mat_vec_q4_k.comp)

This is a **specialized high-performance GEMV for Q4_K_M quantization**:

```glsl
// 16 threads process each block (not full workgroup)
const uint it_size = gl_WorkGroupSize.x / 16;
const uint itid = tid % 16;   // 0..15 within block-processing group

// Thread mapping within a Q4_K superblock:
const uint il = itid / 4;           // 0..3
const uint ir = itid - 4*il;        // 0..3
const uint v_im = il / 2;           // 0 or 1 (which half of 256-element block)

// Each thread computes dot products with:
// - 16 quantized values from A
// - 16 corresponding B values
// Using 4 vec4 loads and manual FMA chains
```

The Q4_K shader avoids dequantizing to shared memory; instead it:
1. Loads packed quant data directly from A buffer
2. Extracts scales and quants using bit manipulation (`unpack8`, `bitfieldExtract`)
3. Multiplies against B vector elements loaded as vec4
4. Accumulates into per-thread partial sums
5. Reduces via shared memory or subgroup operations

### Reduction Strategy

Three paths for final reduction:

1. **subgroupAdd only** (USE_SUBGROUP_ADD_NO_SHMEM): Single subgroup, no shared memory needed
2. **subgroupAdd + shared** (USE_SUBGROUP_ADD): Each subgroup reduces internally, then shared memory across subgroups
3. **Shared memory tree** (fallback): Classic parallel reduction with barrier-synchronized halvings

---

## Push Constant Layouts

### mul_mm.comp Push Constants
```glsl
layout (push_constant) uniform parameter {
    uint M, N, K;           // Matrix dimensions
    uint stride_a, stride_b, stride_d;  // Row strides
    uint batch_stride_a, batch_stride_b, batch_stride_d;  // Batch strides
    uint base_work_group_z;  // For batched dispatch overflow
    uint num_batches;
    uint k_split;            // Split-K size (0 = no split)
    uint ne02, ne12;         // Broadcast dimensions
    uint broadcast2, broadcast3;  // Broadcast factors
} p;
```
Total: **17 uint32 values = 68 bytes** (within 128-byte push constant limit)

### mul_mat_vec Push Constants
```glsl
layout (push_constant) uniform parameter {
    uint ncols;              // K dimension
    uint stride_a, stride_b, stride_d;
    uint batch_stride_a, batch_stride_b, batch_stride_d;
    uint fusion_flags;       // Bias/scale fusion
    uint base_work_group_y;
    uint ne02, ne12;
    uint broadcast2, broadcast3;
} p;
```

---

## Cooperative Matrix (COOPMAT) Conditional Path

### How It's Enabled

```glsl
#ifdef COOPMAT
#extension GL_KHR_cooperative_matrix : enable
#extension GL_KHR_memory_scope_semantics : enable
```

When COOPMAT is defined, the shader uses:
```glsl
coopmat<FLOAT_TYPE, gl_ScopeSubgroup, TM, TK, gl_MatrixUseA> cache_a;
coopmat<FLOAT_TYPE, gl_ScopeSubgroup, TK, TN, gl_MatrixUseB> cache_b;
coopmat<ACC_TYPE, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> sums[...];

// Load from shared memory:
coopMatLoad(cache_a, buf_a, offset, SHMEM_STRIDE, gl_CooperativeMatrixLayoutRowMajor);
coopMatLoad(cache_b, buf_b, offset, SHMEM_STRIDE, gl_CooperativeMatrixLayoutColumnMajor);

// Hardware multiply-accumulate:
sums[i] = coopMatMulAdd(cache_a, cache_b, sums[i]);
```

### Store Path (coopmat)

Three cases for storing results:
1. **Aligned and in-bounds**: Direct `coopMatStore` to output buffer
2. **In-bounds but unaligned stride**: Store to shared memory staging, then scalar store
3. **Partial tile**: Store to shared memory staging with bounds checking

### The coopmat2 Path (mul_mm_cm2.comp)

A separate shader file exists for cooperative matrix v2 (NV extension), with significantly different tile sizes and structure.

---

## Pipeline Variant Selection Logic

### l/m/s Pipeline Selection (ggml_vk_guess_matmul_pipeline)

**Without coopmat2** (standard path -- our case on AGX):
```cpp
// Small: m <= 32 OR n <= 32
if (m <= 32 || n <= 32) return s_pipeline;    // 64x64 tiles
// Medium: m <= 64 OR n <= 64
if (m <= 64 || n <= 64) return m_pipeline;    // 128x128 tiles
// Large: everything else
return l_pipeline;                             // 128x256 tiles
```

**With coopmat2** (more sophisticated):
```cpp
uint tiles_l = ceil(m/l_denom) * ceil(n/l_denom);
uint tiles_m = ceil(m/m_denom) * ceil(n/m_denom);

// Prefer large if GPU would be underutilized with medium tiles
bool prefer_large = tiles_m > shader_core_count ||
    (tiles_l <= shader_core_count/3 && tiles_m > shader_core_count/2);
```

### Aligned vs Unaligned

A pipeline is "aligned" when M and N are multiples of the tile dimensions. The aligned version skips bounds checking in the inner loop, giving a measurable speedup.

```cpp
const uint32_t kpad = ggml_vk_align_size(ne10, align);
bool aligned = (ne01 % align == 0) && (ne11 % align == 0);
vk_pipeline pipeline = aligned ? mmp->a_l : mmp->l;
```

---

## Dispatch Strategy

### Workgroup Counts

For mul_mm.comp:
```cpp
// dispatch: { m, n, groups_z }
// m = total output rows (m dimension, handled by wg_denoms[0])
// n = total output cols (n dimension, handled by wg_denoms[1])
// groups_z = batch count
ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, {a, b, d}, pc, {m, n, groups_z});
```

The pipeline's `wg_denoms` divides the dispatch dimensions to get workgroup counts:
- workgroups_x = ceil(m / wg_denoms[0]) (and for split-K, multiplied by split_k)
- workgroups_y = ceil(n / wg_denoms[1])
- workgroups_z = batch count

### Split-K Dispatch

For small M but large K:
```cpp
uint32_t k_split = CEIL_DIV(k, split_k);
k_split = ROUNDUP_POW2(k_split, 256);  // Align to quant block boundary

// Pass 1: Compute partial results
dispatch({m * split_k, n, batch});  // Extra workgroups for split-K

// Pass 2: Reduce
dispatch({m * n * batch, 1, 1});  // Simple element-wise reduction
```

### Mul_mat_vec Dispatch

```cpp
// Each workgroup handles NUM_ROWS output rows
// X dimension: number of workgroups = ceil(nrows / NUM_ROWS)
// Y dimension: batch
first_row = NUM_ROWS * (gl_WorkGroupID.x + gl_NumWorkGroups.x * gl_WorkGroupID.z);
```

---

## Key Performance Insights

### What llama.cpp Does Well

1. **Dequant during load**: Quant values are unpacked into f16 during shared memory load, not during compute
2. **Vectorized loads**: `data_b_v4` loads 4 floats at once (vec4 alignment)
3. **Packed quant extraction**: `unpack8()` and bitfield operations extract 4 values simultaneously
4. **Fused bias**: GEMV supports fused bias addition in the reduction phase
5. **Subgroup reduction**: Uses `subgroupAdd()` when available, avoiding shared memory
6. **Partial unrolling**: The GEMV loop unrolls 4x, then 2x, then 1x, balancing ILP vs code size

### Performance Gaps on AGX

1. **No cooperative matrix**: The COOPMAT path is dead code on Honeykrisp
2. **32-wide subgroups assumed**: AGX uses 32-wide SIMD, which matches
3. **Shared memory size**: AGX has ~60KB threadgroup memory, the 34KB for large tiles fits
4. **Memory coalescing**: The load patterns assume GPU memory coalescing exists (it does on AGX)
