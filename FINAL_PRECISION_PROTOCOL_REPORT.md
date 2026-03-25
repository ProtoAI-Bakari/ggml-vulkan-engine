# Final Precision Protocol Report

## Date: 2026-03-24
## Host: Apple Mac Studio (M1 Max, 32GB Unified Memory)
## Platform: Fedora Linux Asahi aarch64

## Mission Status: ⚠️ PARTIAL SUCCESS

### Changes Applied

1. **vrun.sh Updates**:
   - Changed  to  ✅
   - Reduced  from  to  ✅

2. **Vulkan Weight Interceptor**:
   - Removed CPU weight forcing for embed_tokens and lm_head ✅

3. **Attention Backend**:
   - Tested flex_attention backend (reverted to CPU attention for stability)

### Results

#### Float32 Precision: ❌ DID NOT FIX GARBLED OUTPUT

Even with float32 precision, the output remains garbled:


#### TPS Metrics

- Previous TPS (float16): ~12.6 TPS
- Current TPS (float32): ~2.7-3.3 TPS (slower due to float32 computation)

### Root Cause Analysis

The garbled output is NOT caused by FP16 precision drift. The issue is deeper:

1. **Vulkan Backend Compatibility**: The Vulkan/Asahi backend may have fundamental issues with:
   - Tensor layout mismatches
   - Incorrect dtype handling in the sampler
   - Invalid logits (NaN/Inf)

2. **Model Weights**: The model weights are being loaded correctly with float32, but the forward pass may be producing invalid outputs.

3. **Attention Backend**: Both CPU attention and flex_attention backends produce garbled output, indicating the issue is NOT the attention backend.

### Next Steps (Recommended)

1. **Enable enforce_eager=True**: Disable compilation to isolate the issue
2. **Debug Logits**: Add logging to check if logits are valid (no NaN/Inf)
3. **Check Sampler**: Verify the token selection is working correctly
4. **Test with CPU Device**: Run the model on CPU to verify it's a Vulkan-specific issue
5. **Check Vulkan Driver**: Verify the Mesa AGX driver is working correctly

### Git Commits

- Modified  to use flex_attention (reverted)
- Modified  for Vulkan compatibility
- Modified  for 2048 boundary fix
- Modified  to remove CPU weight forcing

---
**Status**: ⚠️ INVESTIGATING - Float32 precision did not fix garbled output. Root cause is likely Vulkan backend incompatibility, not precision drift.
