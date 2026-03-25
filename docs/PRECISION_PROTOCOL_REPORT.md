# Precision Protocol Report

## Date: 2026-03-24
## Host: Apple Mac Studio (M1 Max, 32GB Unified Memory)
## Platform: Fedora Linux Asahi aarch64

## Mission: Fix Garbled Output (FP16 Precision Drift)

### Changes Applied

1. **vrun.sh Updates**:
   - Changed `--dtype float16` to `--dtype float32`
   - Reduced `TORCH_LOGS` from `"+dynamo,recompiles,graph_breaks"` to `"recompiles"`

2. **Attention Backend Investigation**:
   - Tested CPU attention backend (default)
   - Tested flex_attention backend
   - Both produced garbled output

### Findings

#### 1. Float32 Did NOT Fix Garbled Output

Even with float32 precision, the output remains garbled:
```
"thouse']]ervativeALTH tolbservice 方paralleled<TreeNodeŷ�箔envilleuth..."
```

#### 2. Root Cause: Vulkan Weight Interceptor

The `Vulkan Interceptor` in `default_loader.py` is keeping critical weights on CPU:
- `embed_tokens.weight`
- `lm_head.weight`

This causes dtype mismatches and incorrect tensor layouts during forward pass.

#### 3. Attention Backend

Both CPU attention and flex_attention backends produce garbled output, indicating the issue is NOT the attention backend but rather the weight loading or forward pass.

### TPS Metrics

- Previous TPS (with CPU attention): ~12.6 TPS
- Current TPS (with flex_attention): ~3.3 TPS (slower due to compilation overhead)

### Next Steps

1. **Fix Vulkan Weight Interceptor**: Remove the CPU weight forcing or ensure proper dtype handling
2. **Investigate Forward Pass**: Check if tensors are being moved correctly between CPU and Vulkan
3. **Test with enforce_eager=True**: Disable compilation to isolate the issue
4. **Check Model Loader**: Ensure all weights are loaded with correct dtype and device

### Git Commits

- Modified `vllm/platforms/vulkan.py` to use flex_attention (reverted)
- Modified `vllm/v1/attention/backends/cpu_attn.py` for Vulkan compatibility
- Modified `vllm/compilation/piecewise_backend.py` for 2048 boundary fix

---
**Status**: ⚠️ INVESTIGATING - Float32 precision did not fix garbled output. Root cause identified as Vulkan Weight Interceptor.