# Custom Shader Optimization Plan
## Path from 11% to 50%+ GPU Utilization

### Current State
- M1 Ultra: 13.6 TFLOPS theoretical FP32
- llama.cpp achieves: ~1.5 TFLOPS (11% utilization)
- Our engine: ~1.3 TFLOPS (9.5% utilization, 88% of llama.cpp)
- MLX on macOS: ~6-8 TFLOPS (45-59% utilization)

### Why GPU Utilization is Low
The AGX GPU has 64 cores but the Vulkan driver (Mesa Honeykrisp) does not
expose hardware-specific features:
1. **No cooperative matrix** (simdgroup_matrix in Metal, not in Vulkan yet)
2. **No texture-based matmul** (AGX optimized for texture sampling)
3. **Generic workgroup sizes** (not tuned for AGX's 32-wide SIMD)
4. **No async memory copies** (AGX has DMA engines not exposed via Vulkan)

### What MLX Does (the 6 TFLOPS target)
MLX uses Metal Performance Shaders with:
- `simdgroup_matrix_multiply_accumulate` — hardware 8x8 matrix multiply
- Optimal tile sizes for AGX (32x32 or 64x64 depending on register pressure)
- Threadgroup memory barriers optimized for AGX
- FP16 accumulation where safe

### What We Can Do in Vulkan
1. **Tune workgroup size** for AGX's 32-wide SIMD (currently generic)
2. **Optimize tile sizes** — experiment with 16x16, 32x32, 64x64
3. **Use subgroup operations** — `GL_KHR_shader_subgroup_arithmetic` for reductions
4. **FP16 compute** — `GL_EXT_shader_explicit_arithmetic_types_float16`
5. **Buffer-optimal access patterns** — coalesced reads matching AGX cache lines
6. **Reduce barrier count** — minimize `memoryBarrierShared()` calls

### Expected Gains
| Optimization | Expected Speedup | Effort |
|-------------|-----------------|--------|
| Workgroup size tuning | 1.2-1.5x | 2-4h |
| Tile size optimization | 1.3-1.8x | 4-8h |
| Subgroup reductions | 1.1-1.3x | 2-4h |
| FP16 compute path | 1.5-2.0x | 8-16h |
| Combined | **2-4x** | **20-40h** |

### Combined with current 21.7 TPS on 8B:
- 2x shaders → ~43 TPS on 8B single-user
- 3x shaders → ~65 TPS on 8B single-user
- 4x shaders → ~87 TPS on 8B single-user

### But...
The realistic ceiling on Vulkan without cooperative matrix is probably 2-3x,
not 4x. Getting from 1.3 TFLOPS to 4 TFLOPS would be exceptional.
The Metal/MLX ceiling (6-8 TFLOPS) requires hardware features not in Vulkan.

### Recommendation
1. Start with workgroup size tuning (2h) — easiest, measurable immediately
2. Then tile size optimization (4h) — biggest single lever
3. FP16 compute is the stretch goal (8h) — if Mesa supports it properly

### This is LONG-TERM work (Week 2+)
The current 21.7 TPS Q4 / 55.4 TPS Q4-0.5B is solid for fleet deployment.
Shader optimization is incremental from here.
