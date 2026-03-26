# Honeykrisp (Apple M1 Ultra) Vulkan Capabilities Matrix

**Date:** 2026-03-25  
**Device:** Apple M1 Ultra (G13D C0)  
**Vulkan Version:** 1.4.328  
**Driver Version:** 25.3.6

## Physical Device Properties

| Property | Value |
|----------|-------|
| Device Name | Apple M1 Ultra (G13D C0) |
| Device Type | INTEGRATED_GPU |
| Vendor ID | 0x10005 (Apple) |
| API Version | 1.4.328 |
| Driver Version | 25.3.6 |

## Memory Architecture

| Property | Value |
|----------|-------|
| Unified Memory Size | 63.19 GiB (67,853,352,960 bytes) |
| Memory Heaps | 1 (DEVICE_LOCAL only) |
| Memory Types | 2 |

### Memory Type 0 (Cached)
- **Flags:** DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED
- **Use Case:** Shader storage, cached host access
- **Tiling Support:** OPTIMAL (color, depth/stencil), LINEAR (color)

### Memory Type 1 (Uncached)
- **Flags:** DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT
- **Use Case:** Streaming uploads, low-latency CPU-GPU sync

## Shader Capabilities

| Capability | Supported |
|------------|-----------|
| shaderFloat16 (FP16) | ✅ YES |
| shaderInt8 (INT8) | ✅ YES |
| subgroupSize | 32 |

## Integer Dot Product Support

All integer dot product accelerations are **NOT SUPPORTED**:
- 8-bit unsigned/signed/mixed: ❌
- 4x8-bit packed: ❌
- 16-bit/32-bit/64-bit: ❌

**Implication:** LLM inference must use FP16 or FP32 GEMV kernels, not INT8 acceleration.

## Cooperative Matrix Support

| Feature | Supported |
|---------|-----------|
| VK_KHR_cooperative_matrix | ❌ NO |
| Tensor Cores (NVIDIA-style) | ❌ NO |

**Implication:** No hardware matrix multiply acceleration. Must use general-purpose compute shaders.

## Subgroup Operations

| Property | Value |
|----------|-------|
| subgroupSize | 32 |
| subgroupOperations | Basic (ballot, shuffle, etc.) |

**Implication:** Optimal SIMD width for LLM kernels is 32 threads.

## Descriptor Limits

| Limit | Value |
|-------|-------|
| maxStorageBufferRange | 2,147,483,647 (2GB) |
| maxPushConstantsSize | 256 bytes |
| maxBoundDescriptorSets | 8 |
| maxDescriptorSetStorageBuffers | 1,048,576 |
| maxDescriptorSetStorageBuffersDynamic | 32 |

## Image/Texture Limits

| Limit | Value |
|-------|-------|
| maxImageDimension2D | 16,384 |
| maxImageDimension3D | 16,384 |
| maxImageArrayLayers | 2,048 |

## Key Implications for LLM Inference

### ✅ Strengths
1. **Unified Memory:** 63GB device-local memory allows loading 120B models (60GB Q4)
2. **FP16 Support:** Enables half-precision GEMV kernels for 2x throughput vs FP32
3. **INT8 Support:** Can use INT8 dequantization in shaders (though no dot-product accel)
4. **High Bandwidth:** Theoretical 800 GB/s memory bandwidth
5. **Large Descriptor Space:** Can bind many tensors simultaneously

### ❌ Limitations
1. **No Tensor Cores:** No cooperative_matrix, must use general compute shaders
2. **No INT8 Dot Product:** INT8 dequant+GEMV must be implemented in shader code
3. **Small L1 Cache:** AGX has 8KB L1 per SIMD (tiny compared to CUDA's 128KB)
4. **Subgroup Size 32:** Optimal workgroup sizes are multiples of 32

### 🎯 Optimization Strategy
1. **FP16 GEMV Kernels:** Use FP16 for all matrix operations (2x throughput vs FP32)
2. **Register-Direct Loads:** AGX benefits from register loads over shared memory (UMA architecture)
3. **Workgroup Size 32:** Match subgroupSize for maximum occupancy
4. **Pre-allocate Buffers:** Use unified memory to avoid runtime allocations
5. **Minimize CB Re-record:** Cache command buffers to reduce CPU overhead

## Comparison: Honeykrisp vs NVIDIA RTX 4090

| Feature | Honeykrisp (M1 Ultra) | RTX 4090 |
|---------|----------------------|----------|
| Memory | 63GB Unified | 24GB GDDR6X |
| Bandwidth | 800 GB/s | 1,008 GB/s |
| Tensor Cores | ❌ No | ✅ Yes (4th gen) |
| FP16 Throughput | ~10 TFLOPS | ~1,300 TFLOPS (with Tensor) |
| INT8 Throughput | ~20 TFLOPS | ~1,300 TOPS (with Tensor) |
| Vulkan Support | 1.4.328 (Asahi) | 1.4.328 (NVIDIA) |
| Cooperative Matrix | ❌ No | ✅ Yes |

**Conclusion:** Honeykrisp trades raw throughput for unified memory capacity. Ideal for large models (120B+) that don't fit on consumer GPUs.

## Benchmark Targets

| Model | Quant | Expected TPS (Honeykrisp) | Expected TPS (llama.cpp Vulkan) |
|-------|-------|---------------------------|----------------------------------|
| Llama-3.1-8B | Q4_K_M | 22-28 TPS | 24.7 TPS |
| Qwen2.5-32B | Q4_K_M | 7-10 TPS | TBD |
| gpt-oss-120B | MXFP4 | 2-4 TPS | TBD |

## Files Modified

- `ggml_llama_gguf.c` - Vulkan dispatch pipeline
- `ggml_vulkan_engine_optimized.py` - Graph caching
- `benchmark_vulkan.py` - Performance measurement

## References

- Asahi Linux Vulkan Driver: https://asahilinux.org/
- Apple Silicon AGX Architecture: https://developer.apple.com/metal/
- Vulkan Spec: https://www.khronos.org/vulkan/
