# The Native FP16 Mission: Battle Plan

## Executive Summary

**Current State:** vLLM runs on Vulkan/Asahi with aggressive float32 casting, consuming ~645MB RSS + ~1GB Vulkan memory.

**Goal:** Enable native float16/bfloat16 support to halve memory usage (~320MB + ~500MB), enabling full GPU residency and 2-3x TPS improvement.

---

## 1. ROOT CAUSE ANALYSIS

### 1.1 PyTorch Vulkan Backend Dtype Support

**Source:** `/tmp/pytorch/aten/src/ATen/native/vulkan/api/Types.h`

```cpp
#define VK_FORALL_SCALAR_TYPES(_)               \
  _(uint8_t, VK_FORMAT_R8G8B8A8_UINT, Byte)     \
  _(int8_t, VK_FORMAT_R8G8B8A8_SINT, Char)      \
  _(int32_t, VK_FORMAT_R32G32B32A32_SINT, Int)  \
  _(bool, VK_FORMAT_R8G8B8A8_SINT, Bool)        \
  _(float, VK_FORMAT_R16G16B16A16_SFLOAT, Half) \  // ← float16 SUPPORTED
  _(float, VK_FORMAT_FLOAT4, Float)             \  // ← float32 (default)
  _(int8_t, VK_FORMAT_R8G8B8A8_SINT, QInt8)     \
  _(uint8_t, VK_FORMAT_R8G8B8A8_UINT, QUInt8)   \
  _(int32_t, VK_FORMAT_R32G32B32A32_SINT, QInt32)
```

**Key Findings:**
- ✅ **Half (float16) IS defined** in Vulkan backend
- ❌ **BFloat16 is NOT in the list** - completely missing
- ⚠️ **VK_FORMAT_FLOAT4 defaults to float32** unless `USE_VULKAN_FP16_INFERENCE` is defined

### 1.2 The Error Location

**Source:** `/tmp/pytorch/aten/src/ATen/native/vulkan/ops/Convert.h:57`

```cpp
static inline api::ScalarType convert_dtype(const c10::ScalarType dtype) {
  switch (dtype) {
    VK_FORALL_SCALAR_TYPES(DEFINE_CASE)
    default:
      TORCH_CHECK(false, "Not a supported Vulkan ScalarType!");  // ← ERROR HERE
  }
}
```

**Problem:** When PyTorch tries to convert `c10::ScalarType::Half` or `c10::ScalarType::BFloat16`, it fails because:
1. `BFloat16` is not in `VK_FORALL_SCALAR_TYPES`
2. `Half` may fail at shader compilation/runtime due to Asahi Mesa driver limitations

### 1.3 Asahi Mesa Driver Limitations

**Source:** System logs show "Mesa 25.1.0-asahi20250221 (Honeykrisp)"

**Known Issues:**
- Asahi's AGX Vulkan driver may not support `VK_FORMAT_R16G16B16A16_SFLOAT` for storage
- Shader compilation for FP16 may fail without proper extension support
- `VK_KHR_shader_float16_int8` extension may not be fully implemented

---

## 2. BATTLE PLAN: PHASED APPROACH

### Phase 1: Add BFloat16 Support to Vulkan Backend

**Objective:** Add `BFloat16` to PyTorch's Vulkan scalar type list.

**Files to Modify:**
1. `/tmp/pytorch/aten/src/ATen/native/vulkan/api/Types.h`
2. `/tmp/pytorch/aten/src/ATen/native/vulkan/ops/Convert.h`

**Changes:**

```cpp
// In Types.h - Add BFloat16 to VK_FORALL_SCALAR_TYPES
#define VK_FORALL_SCALAR_TYPES(_)               \
  _(uint8_t, VK_FORMAT_R8G8B8A8_UINT, Byte)     \
  _(int8_t, VK_FORMAT_R8G8B8A8_SINT, Char)      \
  _(int32_t, VK_FORMAT_R32G32B32A32_SINT, Int)  \
  _(bool, VK_FORMAT_R8G8B8A8_SINT, Bool)        \
  _(float, VK_FORMAT_R16G16B16A16_SFLOAT, Half) \
  _(bfloat16_t, VK_FORMAT_R32G32B32A32_SFLOAT, BFloat16) \  // ← ADD THIS
  _(float, VK_FORMAT_FLOAT4, Float)             \
  _(int8_t, VK_FORMAT_R8G8B8A8_SINT, QInt8)     \
  _(uint8_t, VK_FORMAT_R8G8B8A8_UINT, QUInt8)   \
  _(int32_t, VK_FORMAT_R32G32B32A32_SINT, QInt32)
```

**Note:** `bfloat16_t` is defined in `<c10/util/BFloat16.h>`. Map to `VK_FORMAT_R32G32B32A32_SFLOAT` as a 32-bit container (bfloat16 is 16-bit but stored in 32-bit for alignment).

**Build Command:**
```bash
cd /tmp/pytorch
python setup.py develop --cmake --cmake-defines="-DUSE_VULKAN=ON"
```

**Risk:** Medium - Requires PyTorch rebuild, may expose driver bugs.

---

### Phase 2: Enable FP16 Storage in Vulkan Backend

**Objective:** Force Vulkan backend to use `VK_FORMAT_R16G16B16A16_SFLOAT` instead of float32.

**Files to Modify:**
1. `/tmp/pytorch/aten/src/ATen/native/vulkan/api/Types.h`

**Changes:**

```cpp
// In Types.h - Force FP16 for Float type
// Remove the conditional, always use FP16
#define VK_FORMAT_FLOAT4 VK_FORMAT_R16G16B16A16_SFLOAT  // ← Always FP16
```

**Alternative (if rebuild not possible):** Set compile flag:
```bash
export CMAKE_ARGS="-DUSE_VULKAN_FP16_INFERENCE=ON"
python setup.py develop --cmake
```

**Risk:** High - May cause shader compilation failures on Asahi driver.

---

### Phase 3: vLLM Patch - Remove Float32 Shield

**Objective:** Once PyTorch supports FP16, remove Z-Alpha's aggressive float32 casting.

**Files to Modify:**
1. `~/GITDEV/vllm_0.17.1/vllm/v1/engine/core.py`

**Changes:**

```python
# Remove or comment out the _vulkan_shield_to function
# Let PyTorch handle dtype natively

# Instead, add permissive dtype handling
if os.environ.get('VLLM_PLATFORM') == 'vulkan':
    _orig_vulkan_to = torch.Tensor.to.__func__ if hasattr(torch.Tensor.to, '__func__') else torch.Tensor.to
    
    def _vulkan_shield_to(self, *args, **kwargs):
        device = kwargs.get('device') or (args[0] if args else None)
        
        if device and 'vulkan' in str(device):
            # Only convert truly unsupported types (int64, bool)
            if self.dtype == torch.int64:
                self = _orig_vulkan_to(self, torch.int32)
            elif self.dtype == torch.bool:
                self = _orig_vulkan_to(self, torch.int32)
            # Allow float16, bfloat16, float32 to pass through
            kwargs.pop('dtype', None)
            if len(args) > 1 and isinstance(args[1], torch.dtype):
                args = (args[0],)
            return _orig_vulkan_to(self, *args, **kwargs)
        
        return _orig_vulkan_to(self, *args, **kwargs)
    
    torch.Tensor.to = _vulkan_shield_to
    print("⚠️ VULKAN BRIDGE v9: Native FP16/BF16 Support Active")
```

**Risk:** Low - Only removes restrictions, doesn't add new code.

---

### Phase 4: Full GPU Residency - Move All Layers to Vulkan

**Objective:** Once memory is halved, move MLP and Norm layers to GPU.

**Files to Modify:**
1. `~/GITDEV/vllm_0.17.1/vllm/model_executor/model_loader/utils.py`

**Current Code:**
```python
if target_device.type == 'vulkan':
    for name, m in model.named_modules():
        if any(x in name for x in ["embed_tokens", "lm_head", "word_embeddings", 
                                    "norm", "layernorm", "layer_norm",
                                    "fc1", "fc2", "mlp"]):
            m.to('cpu')  # ← OFFLOADED
        elif any(x in name for x in ["self_attn", "q_proj", "k_proj", "v_proj", "o_proj"]):
            m.to('vulkan')  # ← ONLY THESE ON GPU
```

**New Code (after FP16 enabled):**
```python
if target_device.type == 'vulkan':
    for name, m in model.named_modules():
        # Keep ONLY embed_tokens and lm_head on CPU
        if any(x in name for x in ["embed_tokens", "lm_head"]):
            print(f"📍 Pinned to CPU: {name}")
            m.to('cpu')
        else:
            # Everything else (attention, MLP, norm) goes to Vulkan
            print(f"🚀 Vulkan: {name}")
            m.to('vulkan')
```

**Expected Memory Impact:**
- Before: ~645MB RSS (CPU) + ~1GB Vulkan (attention only)
- After: ~200MB RSS (CPU only embeddings) + ~500MB Vulkan (full model)
- **Total: ~700MB vs 1.6GB = 56% reduction**

**Risk:** Medium - May trigger VMA allocation errors if GPU memory is insufficient.

---

### Phase 5: KV Cache Tuning - Increase Memory Allocation

**Objective:** Increase KV cache from 1GB to 3-4GB for better throughput.

**Files to Modify:**
1. `~/GITDEV/vllm_0.17.1/vllm/v1/engine/core.py`
2. `~/AGENT/vrun.sh`

**Changes in core.py:**

```python
# Find the Vulkan memory limit check
if current_platform.device_type == "vulkan":
    # OLD: 256 MB limit
    # available_gpu_memory = [256 * 1024 * 1024]
    
    # NEW: 4GB limit (with FP16, we have room)
    available_gpu_memory = [4 * 1024 * 1024 * 1024]  # 4GB
    self.available_gpu_memory_for_kv_cache = 4 * 1024 * 1024 * 1024
```

**Changes in vrun.sh:**

```bash
# Increase gpu_memory_utilization
--gpu-memory-utilization 0.95 \  # ← Already high, but now we have FP16
--max-model-len 16384 \  # ← Double sequence length (was 8192)
```

**Expected Impact:**
- KV cache: 10,912 tokens → 43,648 tokens (4x)
- Max concurrency: 1.33x → 5.33x for 8192-token requests
- Throughput: ~1 token/s → ~3-5 tokens/s (less paging, more batching)

**Risk:** High - May cause OOM if model + KV cache exceeds available memory.

---

## 3. IMPLEMENTATION SEQUENCE

### Step 1: Test FP16 Support (No Code Changes)
```bash
# Test if PyTorch Vulkan can handle float16 tensors
python3 << 'EOF'
import torch
print("Vulkan available:", torch.is_vulkan_available())

# Test float16 tensor creation on Vulkan
try:
    t = torch.randn(10, 10, dtype=torch.float16).to('vulkan')
    print("✅ float16 on Vulkan: SUCCESS")
    print("Tensor device:", t.device)
    print("Tensor dtype:", t.dtype)
except Exception as e:
    print("❌ float16 on Vulkan: FAILED")
    print("Error:", str(e))

# Test bfloat16 tensor creation on Vulkan
try:
    t = torch.randn(10, 10, dtype=torch.bfloat16).to('vulkan')
    print("✅ bfloat16 on Vulkan: SUCCESS")
except Exception as e:
    print("❌ bfloat16 on Vulkan: FAILED")
    print("Error:", str(e))
EOF
```

### Step 2: Patch PyTorch Vulkan Backend (If Step 1 Fails)
1. Modify `Types.h` to add BFloat16
2. Rebuild PyTorch with `USE_VULKAN_FP16_INFERENCE=ON`
3. Test again

### Step 3: Update vLLM (After PyTorch Patch)
1. Remove float32 shield in `core.py`
2. Update layer offloading in `utils.py`
3. Increase KV cache limit in `core.py`

### Step 4: Stress Test
```bash
# Test with larger model and longer sequences
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen25",
    "messages": [{"role": "user", "content": "Test long context..."}],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

---

## 4. EXPECTED OUTCOMES

### Memory Footprint (Before vs After)

| Component | Current (FP32) | Target (FP16) | Reduction |
|-----------|---------------|---------------|-----------|
| CPU RSS | 645 MB | 200 MB | 69% |
| Vulkan GPU | 1,000 MB | 500 MB | 50% |
| **Total** | **1,645 MB** | **700 MB** | **57%** |

### Performance (Estimated)

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| TPS (tokens/sec) | 0.7-1.1 | 2.5-4.0 | 3-4x |
| KV Cache Capacity | 10,912 tokens | 43,648 tokens | 4x |
| Max Concurrent Requests | 1-2 | 5-8 | 4-5x |
| Max Sequence Length | 8,192 | 16,384 | 2x |

---

## 5. RISK MITIGATION

### Risk 1: PyTorch Rebuild Fails
**Mitigation:** Keep current float32 shield as fallback. Test in isolated environment.

### Risk 2: Asahi Driver FP16 Bugs
**Mitigation:** Use BFloat16 instead (stored as 32-bit, more compatible). Fallback to float32 if needed.

### Risk 3: OOM with Full GPU Residency
**Mitigation:** Gradual rollout - first half the model, then all. Monitor VMA allocations.

### Risk 4: KV Cache OOM
**Mitigation:** Start with 2GB, test, then increase to 4GB. Implement automatic fallback.

---

## 6. SUCCESS CRITERIA

✅ **Phase 1 Complete:** `torch.randn(10,10,dtype=torch.float16).to('vulkan')` works without error

✅ **Phase 2 Complete:** vLLM loads model with `--dtype float16` (no float32 shield)

✅ **Phase 3 Complete:** All layers (except embeddings) on Vulkan device

✅ **Phase 4 Complete:** KV cache > 2GB, TPS > 2.0 tokens/sec

✅ **Mission Complete:** Total memory < 800MB, TPS > 3.0 tokens/sec

---

## 7. AGENT ASSIGNMENTS

### Z-Alpha (Current Agent)
- Execute Phase 1: Test FP16 support
- Monitor PyTorch Vulkan backend behavior
- Document any Asahi-specific issues

### Qwen3-Next (M1 Ultra Partner)
- Review PyTorch source patches
- Provide adversarial testing scenarios
- Suggest alternative approaches if driver bugs block progress

### MiniMax 139B (Consultant)
- Analyze shader compilation errors
- Suggest Vulkan extension workarounds
- Review BFloat16 implementation strategy

---

## 8. TIMELINE

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| 1. Test FP16 | 30 min | None |
| 2. Patch PyTorch | 2-4 hours | Phase 1 results |
| 3. Update vLLM | 1 hour | Phase 2 complete |
| 4. Full GPU Residency | 1 hour | Phase 3 complete |
| 5. KV Cache Tuning | 30 min | Phase 4 complete |
| **Total** | **4-7 hours** | |

---

## 9. CONCLUSION

**The path to native FP16 on Vulkan/Asahi is clear:**

1. PyTorch's Vulkan backend **does support** float16 (Half) in theory
2. The Asahi Mesa driver may have **runtime limitations** with FP16 storage
3. **BFloat16** is the safer target - map to 32-bit storage for compatibility
4. Once dtype support is enabled, **memory halves** and **TPS triples**

**Critical Decision Point:** Start with testing FP16. If it fails, immediately pivot to BFloat16. Do not spend more than 2 hours debugging FP16 shader issues.

**End Game:** vLLM running full model on Vulkan with 700MB memory footprint and 3+ TPS.

---
**Document Version:** 1.0
**Last Updated:** 2026-03-23 22:15 UTC
**Author:** Z-Alpha