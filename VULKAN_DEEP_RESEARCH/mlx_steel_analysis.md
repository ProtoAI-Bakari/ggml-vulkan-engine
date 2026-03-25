# MLX Steel GEMM Kernel Analysis

## Source Files Analyzed
- `mlx/backend/metal/kernels/steel/gemm/gemm.h` - Main GEMM kernel orchestrator
- `mlx/backend/metal/kernels/steel/gemm/mma.h` - Matrix-multiply-accumulate with simdgroup_matrix
- `mlx/backend/metal/kernels/steel/gemm/loader.h` - Block loader for device->threadgroup
- `mlx/backend/metal/kernels/steel/gemm/params.h` - Parameter structs (GEMMParams)
- `mlx/backend/metal/kernels/steel/gemm/transforms.h` - Output transforms and epilogues
- `mlx/backend/metal/kernels/steel/gemm/kernels/` - Instantiated kernel variants

---

## Tile Sizes (Template Parameters)

MLX Steel uses **template parameters** for tile sizes, not runtime constants:

```metal
template <typename T, typename U, int BM, int BN, int BK, int WM, int WN, ...>
struct GEMMKernel { ... };
```

**Common instantiation sizes** (from kernel files):
- **BM** = 32, 64 (block rows)
- **BN** = 32, 64 (block columns)
- **BK** = 8, 16, 32 (block K dimension)
- **WM** = 2, 4 (simdgroup tiles in M direction)
- **WN** = 2, 4 (simdgroup tiles in N direction)

**Thread counts**: `tgp_size = WM * WN * 32` (each warp tile needs a 32-thread simdgroup)
- WM=2, WN=2 -> 128 threads
- WM=4, WN=4 -> 512 threads

**Fragment size**: Always **8x8** (kFragSize = 8), matching Metal's `simdgroup_matrix<T, 8, 8>`.

---

## Shared Memory (Threadgroup Memory) Layout and Size

```metal
// With padding to avoid bank conflicts:
STEEL_CONST short tgp_padding_a = 16 / sizeof(T);  // 8 for f16, 4 for f32
STEEL_CONST short tgp_padding_b = 16 / sizeof(T);

// Layout depends on transpose:
// Non-transposed A: BM rows x (BK + padding) columns
tgp_mem_size_a = transpose_a ? BK * (BM + padding) : BM * (BK + padding);
tgp_mem_size_b = transpose_b ? BN * (BK + padding) : BK * (BN + padding);
tgp_mem_size = tgp_mem_size_a + tgp_mem_size_b;
```

**Example for BM=64, BN=64, BK=32, f16**:
- padding = 16/2 = 8
- mem_size_a = 64 * (32 + 8) = 2560 elements = 5120 bytes
- mem_size_b = 32 * (64 + 8) = 2304 elements = 4608 bytes
- **Total = ~9.5 KB** per threadgroup

The padding strategy uses **16-byte alignment** to prevent bank conflicts. For f16, that's 8 elements of padding per row.

---

## How simdgroup_matrix Is Used

### BaseMMAFrag (the primitive 8x8 operation)

```metal
template <typename T>
struct BaseMMAFrag<T, 8, 8> {
    STEEL_CONST int kFragRows = 8;
    STEEL_CONST int kFragCols = 8;
    STEEL_CONST int kElemsPerFrag = (8 * 8) / 32;  // = 2 elements per thread

    typedef metal::simdgroup_matrix<T, 8, 8> mat_type;
    typedef metal::vec<T, 2> frag_type;  // Each thread holds 2 elements

    // The actual multiply-accumulate:
    static void mma(mat_type& D, mat_type& A, mat_type& B, mat_type& C) {
        simdgroup_multiply_accumulate(D, A, B, C);  // Hardware instruction
    }
};
```

### BlockMMA (composing tiles from 8x8 fragments)

Each simdgroup (32 threads) computes a tile of size `TM * 8` x `TN * 8`:
```metal
STEEL_CONST short TM = BM / (kFragSize * WM);  // tiles per simdgroup in M
STEEL_CONST short TN = BN / (kFragSize * WN);  // tiles per simdgroup in N
```

The MMA proceeds with a **serpentine traversal** pattern to improve data reuse:
```metal
for (short m = 0; m < M; ++m) {
    for (short n = 0; n < N; ++n) {
        short n_serp = (m % 2) ? (N - 1 - n) : n;  // Alternating direction
        for (short k = 0; k < K; ++k) {
            mma(D[m][n_serp], A[m][k], B[k][n_serp], C[m][n_serp]);
        }
    }
}
```

The serpentine pattern ensures that adjacent K-iteration tiles share either the A or B fragment from registers, reducing redundant loads from threadgroup memory.

### Key Insight: Thread Element Distribution

Each thread in a 32-thread simdgroup holds exactly **2 elements** of each 8x8 matrix fragment. The mapping is:
```metal
short qid = simd_lane_id / 4;
short fm = (qid & 4) + ((simd_lane_id / 2) % 4);  // row
short fn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;  // col
```

This is a **non-trivial scattered layout** that the hardware handles natively -- cannot be replicated in GLSL.

---

## Double Buffering Strategy

**MLX Steel does NOT use classic double buffering** in the threadgroup memory sense. Instead, it uses a simpler **load-barrier-compute-barrier** pattern:

```metal
for (int k = 0; k < gemm_k_iterations; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_a.load_unsafe();   // Load tile from device -> threadgroup
    loader_b.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);      // Compute from threadgroup -> registers
    loader_a.next();          // Advance device pointer
    loader_b.next();
}
```

This means each iteration has **two full barriers**. The latency hiding comes from:
1. High thread occupancy (many threadgroups in flight)
2. The loader uses **vectorized reads** (ReadVector) that load multiple elements per thread
3. Hardware prefetching in the memory subsystem

The BlockLoader's vectorization:
```metal
STEEL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
STEEL_CONST short vec_size = n_reads;  // elements per thread per load

struct alignas(alignment * sizeof(T)) ReadVector {
    uint8_t v[sizeof(T) * vec_size];
};

// Single wide load per row iteration:
*((threadgroup ReadVector*)(&dst[i * dst_ld])) =
    *((const device ReadVector*)(&src[i * src_ld]));
```

---

## Handling Different Matrix Shapes

### Aligned vs Unaligned Paths

MLX uses **compile-time template specialization**:
```metal
template <bool M_aligned, bool N_aligned, bool K_aligned>
struct LoopAlignment {};
```

The `run()` method dispatches to one of four variants:
1. `<true, true, K_aligned>` - Full tiles in M and N
2. `<false, true, K_aligned>` - Partial M, full N
3. `<true, false, K_aligned>` - Full M, partial N
4. `<false, false, K_aligned>` - Partial both

For unaligned edges:
- **Loader**: `load_safe()` checks bounds per element with predication
- **Store**: `store_result_safe()` with tile dimension limits
- **K tail**: Special handling for the last iteration when K is not divisible by BK

### Swizzling for L2 Cache Efficiency

```metal
const int tid_y = ((tid.y) << params->swizzle_log) +
    ((tid.x) & ((1 << params->swizzle_log) - 1));
const int tid_x = (tid.x) >> params->swizzle_log;
```

This reorders threadgroup grid execution to improve L2 cache locality by processing nearby tiles together rather than in strict row-major order.

### Split-K Support

For small M, N (but large K), MLX has `GEMMSpiltKParams` with:
- `split_k_partitions` - number of splits
- `split_k_partition_stride` - stride between partial results
- Separate reduction kernel to sum partial results

---

## What Can Be Translated to GLSL (and What Cannot)

### CAN Translate

| MLX Feature | GLSL Equivalent |
|---|---|
| Threadgroup memory (shared) | `shared` memory in GLSL |
| Tiled loading pattern | Same pattern in GLSL |
| BlockLoader vectorized reads | `uvec4` / `vec4` buffer loads |
| Serpentine MMA traversal | Manual FMA loop with same pattern |
| Bounds checking (load_safe) | Conditional loads in GLSL |
| Swizzle pattern for L2 | Same math in workgroup ID remapping |
| Split-K with reduction | Two-pass dispatch in Vulkan |
| 16-byte padding for bank avoidance | Same padding in GLSL shared memory |

### CANNOT Translate (Metal-specific)

| MLX Feature | Why It Can't Translate |
|---|---|
| `simdgroup_matrix<T, 8, 8>` | No equivalent in GLSL on AGX. Would need `VK_KHR_cooperative_matrix` which Honeykrisp doesn't support |
| `simdgroup_multiply_accumulate()` | Hardware MMA instruction. Must use manual FMA loops instead |
| `thread_index_in_simdgroup` | Available as `gl_SubgroupInvocationID` in Vulkan |
| `simdgroup_index_in_threadgroup` | Available as `gl_SubgroupID` in Vulkan |
| Metal function constants | Use Vulkan specialization constants |
| Metal `threadgroup(0)` / `threadgroup(1)` | Single shared memory in Vulkan, manually partitioned |
| Compile-time template specialization | Must use #ifdef / specialization constants in GLSL |

### The Critical Gap: simdgroup_matrix

MLX's entire performance advantage comes from `simdgroup_multiply_accumulate()`, which maps to dedicated hardware instructions on AGX. In Vulkan/GLSL without cooperative matrix support, we must emulate this with manual FMA operations.

**Estimated performance impact**: The simdgroup_matrix hardware path is approximately **3-6x faster** for prompt processing (GEMM) compared to manual FMA on the same hardware, based on Metal vs Vulkan benchmarks on M2 Max showing 580 vs 92 t/s for pp512.

### Workaround Strategy

For Vulkan on AGX without cooperative matrix:
1. **Use the same tiling structure** (BM/BN/BK) as MLX
2. **Use subgroup operations** (`subgroupShuffle`, `subgroupAdd`) for partial reductions
3. **Maximize register-level FMA** with careful unrolling
4. **Match the shared memory padding** (16-byte alignment)
5. **Implement swizzle** for L2 cache efficiency
6. **Consider split-K** for small-batch scenarios where M is small but K is large
