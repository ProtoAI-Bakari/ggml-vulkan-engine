# Feature Request: VK_KHR_cooperative_matrix for Honeykrisp (AGX)

## Summary
Request to expose `VK_KHR_cooperative_matrix` through Mesa's Honeykrisp
Vulkan driver for Apple AGX GPUs. AGX hardware has `simdgroup_matrix`
(8x8 matrix multiply-accumulate) which is exposed via Metal but not Vulkan.

## Motivation
LLM inference on Asahi Linux via Vulkan currently achieves 23 TPS on
Llama-3.1-8B (Q4_K_M) on M1 Ultra — 93% of llama.cpp. However, GPU
utilization is only 11% of theoretical FLOPS because GEMM must use
scalar FMA loops instead of hardware matrix multiply.

With cooperative matrix, matmul throughput could increase 3-5x, enabling
60-100+ TPS on 8B models — approaching Metal/MLX performance.

## Hardware Support
Apple AGX has `simdgroup_matrix_multiply_accumulate` which operates on
8x8 tiles with 32-thread simdgroups. This maps directly to
`VK_KHR_cooperative_matrix` with:
- `cooperativeMatrixSupportedStages = COMPUTE`
- `MSize = NSize = KSize = 8`
- `scope = SUBGROUP`

## Current State
- Honeykrisp exposes Vulkan 1.4.328 with full subgroup support
- `KHR_shader_float16_int8` is available
- `shader_integer_dot_product` is available
- No cooperative_matrix references exist in Mesa Asahi source

## References
- AGX simdgroup_matrix: Metal Shading Language Spec, chapter on SIMD-scoped operations
- Philip Turner's metal-benchmarks: https://github.com/philipturner/metal-benchmarks
- MLX Steel GEMM kernels using simdgroup_matrix: https://github.com/ml-explore/mlx
- Our Vulkan LLM engine: https://github.com/ProtoAI-Bakari/ggml-vulkan-engine

## Impact
This would benefit all Vulkan compute workloads on Asahi Linux:
- LLM inference (llama.cpp, vLLM, our engine)
- Scientific computing
- Machine learning training
- Any GEMM-heavy workload

## To File
GitLab issue at: https://gitlab.freedesktop.org/asahi/mesa/-/issues
