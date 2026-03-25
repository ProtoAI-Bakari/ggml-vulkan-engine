# CRITICAL BLOCKER: Vulkan Matmul Numerical Precision

## Discovery (2026-03-24 18:20)

PyTorch Vulkan matmul on Mesa AGX (M1 Max) produces results that differ from CPU by:
- Max error: ~0.06 per matmul operation
- Mean error: ~0.008 per element
- Relative error: ~0.04% per matmul

## Impact
- Single matmul: barely noticeable
- 96 matmuls per token (24 layers * 4 matmuls): errors compound
- After full forward pass: output logits are corrupted -> gibberish text
- This is why EVERY Vulkan forward pass attempt produced "!!!!!!!" or random unicode

## Root Cause
- Mesa AGX Vulkan compute shaders use relaxed floating point precision
- The generic GEMM shader in PyTorch Vulkan is not IEEE 754 compliant
- Apple's AGX GPU prioritizes throughput over precision in compute shaders
- CUDA GPUs don't have this problem (they guarantee IEEE 754)

## Potential Fixes (in order of feasibility)
1. **Mixed precision accumulation**: Do matmul in float64, convert result to float32
   - Test: does float64 work on Vulkan? Likely not.
2. **Kahan summation in GEMM**: Compensated summation reduces error accumulation
   - Requires modifying the SPIR-V compute shader in PyTorch
3. **Tiled matmul with higher precision accumulator**: Split large matmuls into tiles
   - Accumulate partial results in higher precision
4. **Use Metal instead of Vulkan**: Metal Performance Shaders have proper precision
   - Requires macOS, not available on Asahi Linux
5. **Reduce number of Vulkan matmuls**: Only use Vulkan for the most compute-heavy ops
   - e.g., only gate_up (largest matmul), do everything else on CPU
6. **Wait for Mesa AGX driver improvements**: The driver is still maturing
   - Precision might improve in future Mesa releases

## Workaround for Now
- Accept CPU inference for small models (0.5B) where CPU is fast enough
- Focus Vulkan on larger models (8B+) where each matmul does MORE work
  and the precision error is proportionally smaller relative to signal
- The error is absolute (~0.06), not relative, so larger activations
  tolerate it better than small ones
