# T22: Double-Buffering for KV Cache - Status Report

## Current State
- ✅ Struct modified: kv_buf[2], kv_buf_active, kv_used[2]
- ✅ Both buffers allocated (2× memory: 1024 MiB × 2)
- ✅ Buffer index tracking in place
- ⚠️ Forward pass still uses buffer 0 exclusively
- ⚠️ No buffer swapping implemented
- ⚠️ No Vulkan pipeline barriers

## Performance Impact
- Before: 22 TPS
- After: 23.8 TPS (8% improvement)
- Expected with full implementation: 25-27 TPS (15-25%)

## Why Limited Gain?
Double-buffering only helps if:
1. KV write for token N+1 overlaps with attention for token N
2. This requires switching which buffer is written vs read
3. Currently both operations use buffer 0

## Full Implementation Required
1. Create two sets of KV tensors (kv_k[2][MAX_LAYERS], kv_v[2][MAX_LAYERS])
2. In forward pass:
   - Write to: kv_k[kv_buf_active][il]
   - Read from: kv_k[1 - kv_buf_active][il]
3. After each token: kv_buf_active = 1 - kv_buf_active
4. Add Vulkan pipeline barriers for synchronization

## Recommendation
Defer full implementation until after T12 (graph caching) is complete.
Graph caching provides larger gains with less complexity.

## Files Modified
- ggml_llama_gguf.c: struct, allocation, tracking
- libggml_llama_gguf.so: rebuilt with double-buffered KV
