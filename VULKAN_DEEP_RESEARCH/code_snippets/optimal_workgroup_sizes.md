# Optimal Workgroup Sizes for Apple AGX GPU (Vulkan)

## Hardware Constraints
- SIMD width: 32 threads
- Max threads per threadgroup: 1024 (typical)
- Threadgroup memory: ~60 KB
- Register file: ~208 KB per core
- L1 data cache: 8 KB per core (very small)
- GPU cores: 32 (M1 Max), 64 (M1 Ultra)

## GEMM (mul_mm.comp) - Batch > 1

### Recommended Configurations

| Matrix Size | Workgroup | BM | BN | BK | Shared Mem | Notes |
|---|---|---|---|---|---|---|
| Large (>256x256) | 256 threads | 128 | 128 | 32 | ~34 KB | Maximum throughput |
| Medium (64-256) | 128 threads | 64 | 64 | 32 | ~8.5 KB | Good balance |
| Small (<64) | 64 threads | 32 | 32 | 32 | ~2 KB | Low overhead |

### llama.cpp Current Settings (without coopmat)
```
Large:  BLOCK_SIZE=256, BM=128, BN=256, BK=64  -> Likely too large for AGX
Medium: BLOCK_SIZE=256, BM=128, BN=128, BK=64  -> May be OK
Small:  BLOCK_SIZE=128, BM=64,  BN=64,  BK=64  -> Good
```

### AGX-Tuned Recommendations
```
Large:  BLOCK_SIZE=128, BM=64,  BN=64,  BK=32  -> 4 SIMD-groups, ~8.5 KB shared
Medium: BLOCK_SIZE=128, BM=64,  BN=64,  BK=16  -> For quant, smaller BK
Small:  BLOCK_SIZE=64,  BM=32,  BN=32,  BK=32  -> 2 SIMD-groups, ~2 KB shared
```

Rationale:
- 128 threads (4 SIMD-groups) is optimal occupancy on AGX
- BK=32 for f16, BK=16 for quant matches memory access patterns
- Smaller tiles than NVIDIA because AGX has smaller L1 (8 KB vs 48 KB)
- Leave shared memory room for other threadgroups on the same core

## GEMV (mul_mat_vec.comp) - Batch = 1

### Recommended Configurations

| Quant Type | BLOCK_SIZE | NUM_ROWS | K_PER_ITER | Reduction | Notes |
|---|---|---|---|---|---|
| Q4_K_M | 256 | 2 | 8 | subgroup+shared | 16 threads per quant block |
| Q4_0 | 128 | 2 | 8 | subgroup+shared | Simpler quant |
| Q8_0 | 128 | 2 | 8 | subgroup+shared | |
| F16 | 32 | 2 | 2 | subgroup only | Single subgroup, no barriers |
| F32 | 32 | 1 | 2 | subgroup only | |

### AGX-Specific Tuning

For Q4_K_M on AGX:
```
BLOCK_SIZE = 128  (4 subgroups of 32)
NUM_ROWS = 2      (2 output rows per workgroup)
K_PER_ITER = 8    (8 elements per thread per iter)
USE_SUBGROUP_ADD = 1
```

Key insight: BLOCK_SIZE should be large enough to cover the full K dimension
with reasonable iteration count, but not so large that subgroup coordination
becomes expensive. 128 threads processing 8 elements each = 1024 K elements
per iteration. For K=4096, that's just 4 iterations of the bulk loop.

## Special Cases

### Reduction Kernels (softmax, RMSNorm)
```
BLOCK_SIZE = 256    # Maximum parallelism for reduction
Reduction: subgroupAdd + shared memory tree
```

### Element-wise Operations (SiLU, GELU, add, scale)
```
BLOCK_SIZE = 256    # Simple, fully parallel
Elements per thread: 4-8 (vectorized)
```

### Split-K Reduction
```
BLOCK_SIZE = 256    # Simple element-wise sum of split results
```

## Performance Tuning Checklist

1. Profile with different BLOCK_SIZE: 32, 64, 128, 256
2. Measure shared memory pressure (are we exceeding 60 KB?)
3. Check occupancy (how many threadgroups per core?)
4. Verify coalesced memory access patterns
5. Compare subgroupAdd vs shared memory reduction times
6. Test NUM_ROWS = 1 vs 2 vs 4 for GEMV
7. Measure barrier cost impact at each BLOCK_SIZE
