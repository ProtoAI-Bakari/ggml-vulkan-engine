# GPU Optimization Recommendations from Coder Brain
## Date: 2026-03-25 05:30

### Context: 23.1 TPS → target 40+ TPS on 8B Q4_K_M

## HIGH-IMPACT (actionable within ggml)

### 1. Pipeline Caching + Dynamic Command Buffers
- Cache VkPipeline objects between tokens (ggml recreates per dispatch)
- Use dynamic rendering state (VK_EXT_extended_dynamic_state2)
- Expected: +2-3 TPS (eliminates pipeline creation per token)

### 2. Fuse matmul+dequant+silu in one shader
- 708 separate ops → merge chains like mul_mat→add→silu
- Single kernel launch per fused group
- Expected: +2-3 TPS

### 3. Pre-compiled SPIR-V with Specialization Constants
- ggml compiles shaders at runtime; pre-compile for common sizes
- VkPipelineCache for persistent shader cache
- Expected: +1-2 TPS on first token, marginal on sustained

### 4. Descriptor Buffer (VK_EXT_descriptor_buffer)
- Replace descriptor sets with GPU-visible buffer
- Eliminates descriptor pool/set management overhead
- Expected: +0.5-1 TPS

### 5. In-place KV cache update (skip ggml_cpy)
- Write K/V directly during QKV matmul shader
- Eliminates separate copy pass
- Expected: +0.5-1.5 TPS

### 6. Command Buffer Reuse
- Pre-record command buffers for decode (same structure each token)
- Only update push constants and input data
- Expected: +1-2 TPS

## TOTAL ESTIMATED: +7-12.5 TPS → 30-36 TPS

## KEY INSIGHT FROM BRAIN
"llama.cpp's advantage isn't raw GPU speed—it's CPU efficiency (tensor pools,
graph caching). By optimizing Vulkan infrastructure, you close that gap on
the GPU side and can EXCEED llama.cpp's TPS."

## REQUIRES: Modifying ggml-vulkan.cpp source code
These are all ggml Vulkan backend improvements, not changes to our wrapper.
Would benefit upstream llama.cpp + all ggml consumers.
