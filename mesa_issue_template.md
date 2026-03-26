---
name: Request for VK_KHR_cooperative_matrix Support on Apple Silicon (M1/M2/M3)
about: Feature request to enable cooperative matrix operations for improved ML inference performance
labels: enhancement, feature-request, apple-silicon, vulkan
assignees: 
---

**Device Information:**
- Hardware: Apple M1 Ultra / M1 Max / M2 / M3 series
- OS: macOS / Asahi Linux
- Driver: MoltenVK / Asahi Mesa driver
- Vulkan Version: 1.3.x
- Device Name:  (or similar)

**Current Status:**
Currently, Vulkan on Apple Silicon does not expose , despite the hardware having dedicated matrix acceleration units (AMX-like). This forces reliance on general-purpose compute shaders for matrix operations, resulting in suboptimal performance for large language models.

**Benchmark Data (Current Performance):**
Using a pure Vulkan inference engine (custom implementation based on ggml):
- Model: Llama-3.1-8B-Q4_K_M
- Hardware: M1 Ultra (128GB RAM)
- Current TPS: ~22 tokens/sec (using subgroup-based GEMV shaders)
- Bottleneck: General compute shaders lack specialized matrix multiply-accumulate instructions.

**Expected Improvement:**
With :
- Estimated TPS increase: 40-60% (based on NVIDIA/AMD analogues where cooperative matrices provide 2-3x speedup for FP16/INT8 GEMM).
- Reduced register pressure due to hardware-accelerated matrix accumulation.
- Better utilization of the Memory Controller and Neural Engine pathways.

**Relevant Extensions:**
- 
-  (already supported)
-  (if needed for atomic ops in kernels)

**Request:**
Please investigate enabling  support in the Asahi Mesa driver (and/or MoltenVK translation layer) for Apple Silicon GPUs. The hardware documentation suggests these units exist but are currently inaccessible via Vulkan.

**References:**
- Apple AMX Architecture Overview (public docs suggest matrix units exist)
- Vulkan Spec: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_cooperative_matrix.html
- Benchmark logs available upon request.

**Priority:** High for AI/ML workloads on Apple Silicon.
