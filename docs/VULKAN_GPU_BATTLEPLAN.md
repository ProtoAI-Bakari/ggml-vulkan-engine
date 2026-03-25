# VULKAN GPU INFERENCE BATTLEPLAN
## Adversarial Engineering Plan - No Excuses, Real GPU or Bust

### THE BRUTAL TRUTH
- Current "Vulkan" inference is 100% CPU. The GPU does NOTHING.
- 14-20 TPS on 0.5B model is pure CPU performance (M1 Max NEON/AMX)
- Every "Vulkan" attempt so far added overhead and made things SLOWER
- The PyTorch Vulkan backend is too limited for direct use in vLLM's forward pass
- We wasted time patching vLLM when the problem is in PyTorch

### THE GOAL
- 0.5B model: 40+ TPS (2x current CPU, achievable with GPU matmul)
- 8B model: 5+ TPS (vs ~1 TPS CPU-only, critical for real use)
- Full GPU compute for matrix operations
- CPU only for: embedding lookup, attention KV cache management, sampling

### WHY PREVIOUS ATTEMPTS FAILED
1. **Per-matmul CPU<->Vulkan roundtrip**: 2.6ms overhead per matmul at batch=1, 96 matmuls per token = 250ms overhead = 4 TPS max
2. **PyTorch Vulkan fp16 broken**: Packing.cpp:65 rejects fp16, forces float32 = 2x memory
3. **PyTorch Vulkan missing ops**: scalar division, negation, rsqrt, silu all broken = can't do RMSNorm on GPU
4. **vLLM model runner expects CUDA**: 30+ torch.cuda.* calls, CpuGpuBuffer hack, dtype assumptions everywhere

### THE THREE ATTACK VECTORS

---

## ATTACK 1: Fix PyTorch Vulkan Backend (HIGH IMPACT, MEDIUM EFFORT)
**Target: 100-300 hours compute time**
**Machines: sys12 M1 Max for testing, CUDA cluster for cross-reference**

The PyTorch Vulkan backend is the root cause. Fix it at the source.

#### Phase 1A: Fix fp16 support in Packing.cpp (50-100 hours)
- File: `~/GITDEV/pytorch/aten/src/ATen/native/vulkan/impl/Packing.cpp:65`
- The `get_nchw_to_image_shader` function rejects fp16
- Need to add SPIR-V shaders for fp16 packing/unpacking
- Reference: look at how CUDA backend handles fp16 in PyTorch
- Test: `torch.randn(4,4,dtype=torch.float16).to('vulkan')` should work
- **WIN**: Halves memory, doubles throughput, enables real fp16 compute

#### Phase 1B: Fix scalar division and rsqrt (20-50 hours)
- `1.0 / vulkan_tensor` fails with "Cannot access data pointer"
- `torch.reciprocal(vulkan_tensor)` also fails
- Need to add Vulkan compute shader for reciprocal/rsqrt
- File: `aten/src/ATen/native/vulkan/ops/` - add new op implementations
- **WIN**: Unlocks RMSNorm on GPU = entire transformer layer on GPU

#### Phase 1C: Fix negation and other missing ops (10-20 hours)
- `-tensor` fails but `tensor * (-1.0)` works
- `torch.nn.functional.silu` fails (workaround exists: `x * (exp(x)/(exp(x)+1))`)
- Audit all ops needed for transformer forward pass
- **WIN**: Clean Vulkan forward pass without workarounds

#### Phase 1D: Fix zero-copy for unified memory (50-100 hours)
- On M1/M2, CPU and GPU share physical memory
- `.to('vulkan')` currently does a BUFFER COPY (measured: 0.04-0.2us)
- Should be metadata-only change (zero-copy)
- File: `aten/src/ATen/native/vulkan/api/Resource.cpp`
- Need to use VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT for shared memory
- **WIN**: Eliminates ALL transfer overhead. CPU<->Vulkan becomes free.

**After Attack 1**: Full transformer layer on Vulkan, fp16, zero-copy. Expected 40-80 TPS on 0.5B, 5-10 TPS on 8B.

---

## ATTACK 2: MLP-Only Vulkan (QUICK WIN, LOW RISK)
**Target: 20-40 hours compute time**
**Can run IN PARALLEL with Attack 1**

Don't wait for PyTorch fixes. Bypass vLLM's gemm dispatch entirely and do Vulkan matmul directly in the Qwen2 MLP.

#### Phase 2A: Direct MLP Vulkan Override (10-20 hours)
- Override `Qwen2MLP.forward()` to do ALL 3 matmuls on Vulkan in one shot
- Manually manage weight cache (float32 Vulkan copies of gate_up and down weights)
- SiluAndMul already works on Vulkan (committed: b6704ec37)
- Flow: CPU input -> Vulkan (1 transfer) -> 3 matmuls + silu on Vulkan -> CPU output (1 transfer)
- Already PROVEN to work in isolation test (see session notes)
- The CRASH was because I modified gemm() which broke vLLM's model runner
- Fix: don't touch gemm, override MLP.forward directly

```python
class VulkanQwen2MLP(Qwen2MLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vk_gate_up_w = None
        self._vk_down_w = None

    def forward(self, x):
        if x.shape[0] <= 8:  # decode: CPU faster
            return super().forward(x)

        # Prefill: Vulkan MLP chain (2 transfers, 3 matmuls on GPU)
        orig_dtype = x.dtype
        if self._vk_gate_up_w is None:
            self._vk_gate_up_w = self.gate_up_proj.weight.cpu().float().to('vulkan')
            self._vk_down_w = self.down_proj.weight.cpu().float().to('vulkan')

        x_vk = x.cpu().float().to('vulkan')
        gate_up = torch.mm(x_vk, self._vk_gate_up_w.t())
        # Vulkan SiluAndMul
        d = gate_up.shape[-1] // 2
        a, b = gate_up[..., :d], gate_up[..., d:]
        exp_a = torch.exp(a)
        activated = (a * (exp_a / (exp_a + 1.0))) * b
        result = torch.mm(activated, self._vk_down_w.t())
        return result.cpu().to(orig_dtype)
```

#### Phase 2B: Monkey-patch at model load time (5-10 hours)
- After model loads, iterate all Qwen2MLP modules and replace forward
- Works for ANY Qwen2-based model without touching vLLM core code
- Can be done as a startup hook in vulkan.py

#### Phase 2C: Extend to QKV projection (5-10 hours)
- QKV projection is another matmul (hidden -> 3*hidden)
- Same pattern: override Qwen2Attention to do QKV matmul on Vulkan for batch>8
- Output projection (hidden -> hidden) can also go to Vulkan

**After Attack 2**: MLP on Vulkan for prefill. Expected 30+ TPS on 0.5B prefill, 15 TPS single decode (CPU MLP + CPU everything else).

---

## ATTACK 3: Custom Vulkan Compute Pipeline (NUCLEAR OPTION)
**Target: 200-500 hours compute time**
**Only if Attack 1 and 2 fail to deliver**

Bypass PyTorch Vulkan entirely. Write custom Vulkan compute shaders for the hot path.

#### Phase 3A: Custom GEMM shader (100-200 hours)
- Write SPIR-V/GLSL compute shader for matrix multiplication
- Optimized for Apple AGX GPU architecture (tiled, cache-friendly)
- Integrate via PyTorch custom op or direct Vulkan API calls
- Reference: llama.cpp's ggml-vulkan backend has working shaders

#### Phase 3B: Custom RMSNorm + SiLU shaders (50-100 hours)
- RMSNorm: reduce + normalize in one shader dispatch
- SiLU: fused SiluAndMul in one dispatch
- Fuse with GEMM for even less dispatch overhead

#### Phase 3C: Fused transformer layer shader (100-200 hours)
- One shader dispatch for entire MLP block
- Minimize Vulkan command buffer submissions
- Use push constants for dynamic shapes

**After Attack 3**: Full custom GPU pipeline. Expected 60+ TPS on 0.5B, 8-15 TPS on 8B.

---

### WORK DISTRIBUTION ACROSS FLEET

| Machine | Role | Task |
|---------|------|------|
| sys12 (M1 Max 32GB) | Primary test target | Run all experiments, test models |
| CUDA cluster (8x3090) | Smart brain | Run 122B model for agent, cross-reference CUDA behavior |
| M1 Ultra 128GB (x2) | Medium agents | Run 30B models as code-writing agents for Attack 1 |
| M2 Ultra 192GB (x2) | Large agents | Run 70B models for architecture decisions |
| Remaining Mac Studios | Parallel test | Test Vulkan changes on different hardware configs |

### EXECUTION ORDER
1. **IMMEDIATE (next session)**: Attack 2A - Direct MLP Vulkan override. 10 hours. Concrete code change, proven in isolation.
2. **PARALLEL**: Attack 1A - Fix PyTorch fp16. Start the agent on this while I work on 2A.
3. **WEEK 1**: Complete Attack 2, start Attack 1B (scalar division fix)
4. **WEEK 2-3**: Complete Attack 1, measure gains
5. **ONLY IF NEEDED**: Attack 3

### PERSISTENT STATE
- All code on git: `~/GITDEV/vllm_0.17.1` (stable-v0.17.1 branch)
- PyTorch source: `~/GITDEV/pytorch` (custom Vulkan build)
- Wheels: `~/WHEELS/`
- Knowledge base: `~/AGENT/VULKAN_KNOWLEDGE.md`
- Agent framework: `~/AGENT/v44_GPU.py` with bridge comms
- Memory: `~/.claude/projects/-home-z-AGENT/memory/`
- This plan: `~/AGENT/VULKAN_GPU_BATTLEPLAN.md`

### SUCCESS CRITERIA
- [ ] 0.5B model: 40+ TPS single user (2x CPU baseline)
- [ ] 8B model: 5+ TPS single user (5x CPU baseline)
- [ ] Coherent output on all models
- [ ] Server stable for 1000+ requests
- [ ] Works on ALL Mac Studio models in the fleet (M1 Max, M1 Ultra, M2 Ultra)
- [ ] Clean git commits with rollback points
