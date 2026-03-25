// MLX Steel GEMM Tiling Pattern - Extracted and Annotated
// Source: mlx/backend/metal/kernels/steel/gemm/
//
// This shows the core tiling structure that achieves near-peak performance
// on Apple Silicon GPUs. The key is simdgroup_matrix (hardware MMA).

using namespace metal;

// ============================================================
// TILE SIZE CONFIGURATION
// ============================================================
// BM, BN: Threadgroup tile dimensions (output block size)
// BK: K-dimension tile size (inner loop blocking)
// WM, WN: Number of simdgroups in M and N directions
// Fragment size is always 8x8 (hardware constraint)

// Typical configurations:
//   BM=64, BN=64, BK=32, WM=2, WN=2 -> 128 threads, 4 simdgroups
//   BM=64, BN=64, BK=16, WM=2, WN=2 -> 128 threads, 4 simdgroups

// ============================================================
// SHARED MEMORY LAYOUT
// ============================================================
// Padding to avoid bank conflicts (16-byte alignment):
//   padding = 16 / sizeof(T)  (8 for f16, 4 for f32)
//
// For non-transposed A:
//   Layout: BM rows x (BK + padding) columns
//   tgp_mem_size_a = BM * (BK + padding)
//
// For non-transposed B:
//   Layout: BK rows x (BN + padding) columns
//   tgp_mem_size_b = BK * (BN + padding)

// ============================================================
// BLOCK LOADER (device -> threadgroup)
// ============================================================
// Each thread loads vec_size elements per iteration
// vec_size = (BCOLS * BROWS) / tgp_size
// Uses alignas ReadVector for wide loads

template <typename T, short BROWS, short BCOLS, short dst_ld,
          short reduction_dim, short tgp_size>
struct BlockLoader {
    static constexpr short n_reads = (BCOLS * BROWS) / tgp_size;
    static constexpr short vec_size = n_reads;
    static constexpr short TCOLS = BCOLS / n_reads;
    static constexpr short TROWS = tgp_size / TCOLS;
    static constexpr short n_rows = (BROWS + TROWS - 1) / TROWS;

    const int src_ld;
    const int tile_stride;
    const short bi, bj;  // Thread's position in the tile

    threadgroup T* dst;
    const device T* src;

    // Unsafe load: no bounds checking, maximum speed
    void load_unsafe() const {
        for (short i = 0; i < BROWS; i += TROWS) {
            // Wide vectorized load: reads vec_size elements at once
            *((threadgroup ReadVector*)(&dst[i * dst_ld])) =
                *((const device ReadVector*)(&src[i * src_ld]));
        }
    }

    void next() { src += tile_stride; }
};

// ============================================================
// MMA FRAGMENT (8x8 simdgroup_matrix operations)
// ============================================================
// Each 32-thread simdgroup holds one 8x8 matrix
// Each thread holds exactly 2 elements of the 8x8 matrix

template <typename T>
struct BaseMMAFrag_8x8 {
    using mat_type = simdgroup_matrix<T, 8, 8>;
    using frag_type = vec<T, 2>;  // 2 elements per thread

    // Hardware multiply-accumulate instruction
    static void mma(mat_type& D, mat_type& A, mat_type& B, mat_type& C) {
        simdgroup_multiply_accumulate(D, A, B, C);
    }
};

// ============================================================
// BLOCK MMA (composing larger tiles from 8x8 fragments)
// ============================================================
// TM = BM / (8 * WM)  -- fragment tiles per simdgroup in M
// TN = BN / (8 * WN)  -- fragment tiles per simdgroup in N
//
// Serpentine traversal for data reuse:
template <typename T, int M, int N, int K>
void tile_matmad(MMATile<T,M,N>& D, MMATile<T,M,K>& A,
                 MMATile<T,K,N>& B, MMATile<T,M,N>& C) {
    for (short m = 0; m < M; ++m) {
        for (short n = 0; n < N; ++n) {
            // Serpentine: alternating N direction
            short n_serp = (m % 2) ? (N - 1 - n) : n;
            for (short k = 0; k < K; ++k) {
                BaseMMAFrag_8x8<T>::mma(
                    D.frag_at(m, n_serp),
                    A.frag_at(m, k),
                    B.frag_at(k, n_serp),
                    C.frag_at(m, n_serp));
            }
        }
    }
}

// ============================================================
// MAIN GEMM LOOP
// ============================================================
void gemm_main_loop(
    threadgroup T* As,       // Shared memory for A tile
    threadgroup T* Bs,       // Shared memory for B tile
    int K_iterations,        // Number of K-dimension blocks
    BlockLoader& loader_a,
    BlockLoader& loader_b,
    BlockMMA& mma_op) {

    for (int k = 0; k < K_iterations; k++) {
        // Barrier: ensure previous iteration's compute is done
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load tiles from device memory to threadgroup memory
        loader_a.load_unsafe();
        loader_b.load_unsafe();

        // Barrier: ensure all threads have finished loading
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Multiply-accumulate from threadgroup memory to registers
        mma_op.mma(As, Bs);

        // Advance pointers for next iteration
        loader_a.next();
        loader_b.next();
    }
}

// ============================================================
// SWIZZLE FOR L2 CACHE EFFICIENCY
// ============================================================
// Remaps threadgroup grid to improve L2 locality
// swizzle_log controls the swizzle granularity
int2 swizzle(uint3 tid, int swizzle_log) {
    int tid_x = (tid.x) >> swizzle_log;
    int tid_y = ((tid.y) << swizzle_log) +
                ((tid.x) & ((1 << swizzle_log) - 1));
    return int2(tid_x, tid_y);
}
