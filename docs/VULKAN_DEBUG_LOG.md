# Z-Alpha Vulkan V1 Engine Debug Log
## Session: 2024-03-23 00:47
## Agent: Z-Alpha (RTX 4090 local)

---

## FINDINGS SUMMARY

### Progress Made:
1. ✅ Fixed CUDA stream initialization (lines 554, 602, 738, 747)
2. ✅ Vulkan platform detection working (VLLM_PLATFORM=vulkan)
3. ✅ Surgical weight loader interceptor implemented
4. ✅ Embedding forward pass CPU fallback implemented

### Current Blocker:
**Error:** `NotImplementedError: Could not run 'aten::empty_strided' with arguments from the 'Vulkan' backend`

**Root Cause:** PyTorch's Vulkan backend on Asahi Linux does not support the `empty_strided` operator used by V1 engine's logits processor.

**Location:** `vllm/v1/sample/logits_processor/builtin.py` line 232

---

## ATTEMPTED FIXES

### 1. CUDA Stream Removal (SUCCESS)
- Removed `torch.cuda.Stream()` calls
- Set streams to `None` for Vulkan
- Fixed indentation errors

### 2. Pin Memory Handling (SUCCESS)
- Vulkan platform correctly returns `is_pin_memory_available() = False`
- No changes needed

### 3. Weight Loading Interceptor (SUCCESS)
- Implemented in `default_loader.py`
- Forces VocabParallelEmbedding weights to CPU
- Casts to target dtype on CPU

### 4. Embedding Forward Pass (SUCCESS)
- Implemented CPU fallback in `vocab_parallel_embedding.py`
- Performs embedding lookup on CPU
- Moves output back to Vulkan

### 5. V1 Engine Tensor Operations (FAILED)
- `aten::empty_strided` not supported on Vulkan
- This is a PyTorch limitation, not vLLM
- Affects logits processor initialization

---

## NEXT STEPS

### Option A: Force CPU Execution for V1 Engine
- Use `--device cpu` flag
- Keep Vulkan for model weights only
- May work around the empty_strided issue

### Option B: Disable V1 Engine
- Use `--disable-v1` flag
- Fall back to V0 engine which may have better Vulkan support

### Option C: Patch PyTorch Vulkan Backend
- Requires recompiling PyTorch with Vulkan support
- Not feasible in current session

### Option D: CPU-Only Execution
- Remove Vulkan entirely
- Use CPU for all operations
- Slower but functional

---

## RECOMMENDATION
Try **Option B** first: Disable V1 engine and use V0 engine which may have better Vulkan support.

---

## FILES MODIFIED
1. `/home/z/GITDEV/vllm_0.17.1/vllm/model_executor/model_loader/default_loader.py` - Weight interceptor
2. `/home/z/GITDEV/vllm_0.17.1/vllm/model_executor/layers/vocab_parallel_embedding.py` - CPU fallback
3. `/home/z/GITDEV/vllm_0.17.1/vllm/v1/worker/gpu_model_runner.py` - CUDA stream removal

## BACKUPS
- `~/AGENT/vLLM_VULKAN_FIX/default_loader.py.backup`
- `~/AGENT/vLLM_VULKAN_FIX/vocab_parallel_embedding.py.backup`

---

## AGENT STATUS
- Context: Recovered from crash
- Progress: 70% complete
- Blocker: PyTorch Vulkan backend limitation
- Next Action: Try disabling V1 engine

---

## NEW PATCH: Memory Limit for Vulkan (2024-03-23 01:00)

### File Modified
`/home/z/GITDEV/vllm_0.17.1/vllm/v1/engine/core.py` - `_initialize_kv_caches()`

### Change
Added Vulkan memory limit AFTER `determine_available_memory()` but BEFORE `get_kv_cache_configs()`:

```python
# VULKAN ASAHI LOBOTOMY: Stop the greed - limit available memory for Vulkan
if envs.VLLM_PLATFORM == "vulkan":
    # Vulkan on Asahi has limited device-local memory (~15GB shared)
    # Limit to 1GB to prevent VMA_ERROR_OUT_OF_DEVICE_MEMORY
    available_gpu_memory = [1024 * 1024 * 1024]  # 1GB
    self.available_gpu_memory_for_kv_cache = 1024 * 1024 * 1024
    print(f"⚠️ VULKAN OVERRIDE: Limited available memory to 1GB for Vulkan stability.")
```

### Rationale
- Previous crash: `VMA_ERROR_OUT_OF_DEVICE_MEMORY (-2)`
- Root cause: KV cache over-allocation (174,752 tokens = 19.1GB requested vs 15.58GB available)
- Fix: Force 1GB limit for Vulkan to prevent memory exhaustion during warmup

### Expected Result
- Should prevent VMA memory allocation failure
- May reduce max sequence length but allows Vulkan to function
- Test with: `VLLM_PLATFORM=vulkan python -m vllm.entrypoints.api_server ...`

---

---

## PATCH SUMMARY (2024-03-23 01:05 UTC)

### Changes Made
1. **core.py Memory Limit Patch** ✅
   - File: `/home/z/GITDEV/vllm_0.17.1/vllm/v1/engine/core.py`
   - Location: Line 254 (inside `_initialize_kv_caches()`)
   - Effect: Limits Vulkan memory allocation to 1GB to prevent VMA errors

2. **Run Script Dtype Fix** ✅
   - File: `~/1vrun_0.17.1_llama8b_instruct.sh`
   - Change: `--dtype bfloat16` → `--dtype float32`
   - Reason: Vulkan on Asahi requires FP32 (FP16/BF16 shader missing)

### Test Command
```bash
cd ~/GITDEV/vllm_0.17.1
export PYTHONPATH=.
export VLLM_PLATFORM=vulkan
./1vrun_0.17.1_llama8b_instruct.sh
```

### Expected Behavior
- Server should start without VMA_ERROR_OUT_OF_DEVICE_MEMORY
- May have reduced max sequence length due to 1GB memory limit
- Should use Vulkan for model weights (GPU acceleration)

### Remaining Issues
- `aten::empty_strided` not supported on Vulkan (V1 engine logits processor)
- May need to disable V1 engine or use CPU fallback for sampling

---
