# SYS0 MULTI-AGENT VULKAN ACCELERATION BATTLEPLAN
## M1 Ultra 128GB — 2026-03-25

---

## SYSTEM STATE

| Resource | Value | Notes |
|----------|-------|-------|
| CPU | Apple M1 Ultra, 20 cores | 8 Firestorm + 4 Blizzard + 8 Icestorm |
| RAM | 126 GiB | 106 GiB free after OS + agents |
| GPU | Apple AGX (64 cores) via Vulkan 1.4 | Mesa Honeykrisp 25.3.6 |
| Vulkan Heap | 63.19 GiB (RAM/2 default) | Can push to ~100 GiB with HK_SYSMEM |
| PyTorch | 2.12.0a0+git5de8e44 with Vulkan | CONFIRMED WORKING |
| vLLM | 0.17.1 building from source NOW | vulkan-mlp-gpu branch (f95b8e4cb) |
| Models | Qwen2.5-1.5B, 3B, Llama-3.1-8B | All downloaded |

---

## WHAT WE KNOW (Consolidated from Sys4/Sys12 — 1000+ hours of work)

### Hard Facts — CPU vs GPU Performance

| Metric | CPU (NEON/AMX) | Vulkan GPU | Ratio |
|--------|---------------|------------|-------|
| Matmul batch=1 (1536x8960) | 0.5ms | 6.8ms | **CPU 14x faster** |
| Matmul batch=4 (QKV 1536-dim) | 6.8ms | 6.6ms | ~tie |
| Matmul batch=8 | 13ms | 7.1ms | **GPU 1.8x** |
| Matmul batch=16 | 24ms | 7.2ms | **GPU 3.4x** |
| Matmul batch=32 | 46ms | 9.2ms | **GPU 5x** |
| Matmul batch=64 | 91ms | 10.9ms | **GPU 8.3x** |
| Matmul batch=4096 | huge | ~10ms | **GPU 22.9x** |

**CRITICAL INSIGHT**: GPU wins at batch >= 4. Decode (batch=1) is CPU-faster.
Prefill and batch verification are where GPU shines.

### Achieved TPS on Sys4 (M1 Max 32GB)

| Model | Layers on GPU | Single TPS | Batch TPS | Coherent |
|-------|--------------|------------|-----------|----------|
| Qwen2.5-0.5B | 24/24 (all) | 25-26 | 29 | YES |
| Qwen2.5-1.5B | 28/28 (all) | 13.5 | - | YES |
| Qwen2.5-3B | 36/36 (all) | 7.6 | - | YES |
| Llama-8B | untested | - | - | - |

### Known Hard Constraints

1. **Vulkan image dimension limit ~16000**: Matmuls with >16000 rows produce garbage.
   FIX: Split gate_up into two halves. Already implemented.

2. **Vulkan matmul precision**: ~0.06 max error per matmul (AGX relaxed fp).
   BUT: Residual connections stabilize across ALL layers. 24-36 layers WORK.

3. **No fp16 matmul on Vulkan**: All matmul done in float32. 2x memory overhead.

4. **Missing Vulkan ops**: scalar division, rsqrt, negation, SiLU.
   FIX: CPU SiLU activation, CPU RMSNorm. Matmul only on GPU.

5. **Batch gating**: CPU for decode (batch<=4), Vulkan for prefill (batch>4).
   Commit `242418b3c`.

6. **Attention fast path**: QKV and output projections cached on CPU, bypass vllm dispatch.
   Commit `f95b8e4cb`. Gave +40% TPS boost.

### What M1 Ultra 128GB Unlocks

| Model | MLP Weight Size | Fits 63 GiB Vulkan? | All Layers? |
|-------|----------------|---------------------|-------------|
| Qwen2.5-0.5B | 1.2 GiB | YES | 24/24 |
| Qwen2.5-1.5B | 4.4 GiB | YES | 28/28 |
| Qwen2.5-3B | 10.8 GiB | YES | 36/36 |
| **Llama-3.1-8B** | **22.5 GiB** | **YES** | **32/32** |
| Llama-3.1-70B | ~140 GiB | NO (with HK_SYSMEM maybe) | ~28/80 |

**8B with ALL 32 layers on Vulkan GPU is now possible. This was impossible on M1 Max 32GB.**

---

## THE MISSION: 3 PARALLEL TRACKS

### TRACK 1: 8B Llama Full Vulkan (PRIMARY — this is why Sys0 exists)
**Goal**: First-ever 8B LLM running ALL layers on Vulkan GPU on Asahi Linux.
**Expected TPS**: 3-8 TPS single (based on 3B scaling), potentially 10-15 with optimizations.

### TRACK 2: Speculative Decoding (FORCE MULTIPLIER)
**Goal**: Exploit GPU's 22.9x batch advantage for verification.
**Why**: Decode is batch=1 (CPU-bound). Spec decode generates K candidates, verifies in batch=K+1.
If K=7 and acceptance=50%, effective throughput = 4 tokens per GPU batch = ~4x speedup.

vLLM 0.17.1 has 5 built-in spec decode methods:
- **ngram** (zero overhead, CPU-only proposer, works immediately)
- **draft_model** (small LLM on CPU proposes, big LLM on GPU verifies)
- **EAGLE** (draft heads on target model)
- **medusa** (multi-head parallel speculation)
- **MTP** (multi-token prediction)

**Start with ngram (zero risk), then try draft_model with 0.5B as drafter for 8B target.**

### TRACK 3: Performance Engineering (OPTIMIZE WHAT WE HAVE)
**Goal**: Squeeze maximum TPS from the current architecture before nuclear options.
- Profile per-token time breakdown (where does time go?)
- Tune batch gating threshold for M1 Ultra (maybe threshold=2 instead of 4)
- Test HK_SYSMEM to push Vulkan heap beyond 63 GiB
- Investigate VK_EXT_memory_budget for accurate runtime tracking
- Benchmark llama.cpp Vulkan as reference point

---

## MULTI-AGENT ARCHITECTURE

### Communication Protocol
All agents communicate through `~/AGENT/agent-comms-bridge.md`:
- **Append-only updates** (never edit others' sections)
- **Format**: `## [AGENT_NAME] UPDATE — [TIMESTAMP]`
- **Claim files before editing** (avoid collision)
- **Report findings immediately** (don't wait for task completion)

### Agent Roster (4-5 total)

```
AGENT 0 — LEAD COORDINATOR (this terminal)
  Role: Build management, integration, 8B Llama testing, commit workflow
  Owns: vllm build process, launch scripts, git workflow
  Files: llama.py (model-level patches), launch_*.sh, test scripts

AGENT 1 — VULKAN MEMORY & PERFORMANCE PROFILER
  Role: Memory profiling, batch size optimization, GPU utilization metrics
  Tasks:
    1. Profile Vulkan memory usage per model layer for 8B Llama
    2. Sweep batch thresholds (1,2,4,8,16) to find optimal crossover on Ultra
    3. Measure per-layer time breakdown: transfer vs compute vs overhead
    4. Test HK_SYSMEM values (64G, 96G, 112G) for max Vulkan heap
    5. Run vulkaninfo memory_budget extension tests
  Owns: profiling scripts, benchmark results

AGENT 2 — SPECULATIVE DECODE ENGINEER
  Role: Get spec decode working on Vulkan platform
  Tasks:
    1. Test ngram spec decode with current Vulkan setup (zero risk)
    2. Port rejection sampling from Triton GPU kernel to CPU (Vulkan has no Triton)
    3. Test draft_model with Qwen-0.5B as drafter for 8B target
    4. Measure acceptance rates and effective TPS improvement
    5. Investigate if verification batch can run on Vulkan (batch=K+1)
  Owns: spec_decode config, rejection sampling port
  Files: vllm/v1/spec_decode/*, test_spec_decode_vulkan.py

AGENT 3 — LLAMA.CPP VULKAN BENCHMARK (short-lived)
  Role: Get ground truth performance numbers from llama.cpp Vulkan
  Tasks:
    1. Build llama.cpp with Vulkan backend
    2. Run 8B Llama on Vulkan with all layers
    3. Benchmark TPS at various batch sizes
    4. Compare with our vLLM Vulkan numbers
    5. Report shader quality / precision findings
  Owns: llama.cpp build dir, benchmark scripts

AGENT 4 — (OPTIONAL) PYTORCH VULKAN IMPROVEMENTS
  Role: Fix PyTorch Vulkan backend issues at the source
  Tasks:
    1. Enable VK_EXT_memory_budget in Resource.cpp
    2. Test fp16 matmul on newer Mesa (25.3.6 might have fixes)
    3. Increase VMA block size for UMA systems
    4. Profile Vulkan command buffer overhead per matmul
  Owns: ~/GITDEV/pytorch source modifications
```

### Work Dependency Graph

```
[vLLM Build Completes]
        |
        v
    +---+---+
    |       |
    v       v
  [Agent 0: 8B Llama Test]    [Agent 1: Memory Profile]
    |                               |
    v                               v
  [8B baseline TPS]          [Optimal batch threshold]
    |                               |
    +-------+-----------------------+
            |
            v
    [Agent 2: Spec Decode on 8B]  ← needs working 8B baseline first
            |
            v
    [Spec decode TPS numbers]
            |
            v
    [DECISION: Ship Track 1+2 results or go to Track 3 nuclear options]
```

### Independent (can start NOW, before build):
- Agent 3: llama.cpp benchmark (separate build, no vLLM dependency)
- Agent 4: PyTorch Vulkan improvements (separate source tree)

---

## WHAT TO TELL THE SYS12 AGENT

Send this to the Sys12 agent for the knowledge transfer file:

```
Create ~/AGENT/Sys12-Sys0-agent-comms-bridge.md with ALL of the following:

1. EXACT CPU vs GPU benchmark numbers at every batch size you tested
   - Matrix dimensions tested (e.g., 1536x8960, 4096x14336)
   - Transfer overhead measurements (CPU->Vulkan, Vulkan->CPU)
   - Include both M1 Max and any M1 Ultra numbers if available

2. PRECISION DATA
   - Max error, mean error, cosine similarity per matmul at each matrix size
   - The 16000 row breakpoint details (exact dimensions where it breaks)
   - Whether Kahan summation helped (you said marginal: 0.054->0.057)
   - Residual connection stability proof (cosine=1.000 through 32 layers)

3. VULKAN DRIVER QUIRKS
   - Driver degradation after repeated crashes (need reboot)
   - VMA memory behavior (reclaim broken? or was that the 2.6GB myth?)
   - HK_SYSMEM env var behavior (what values tested, results)
   - Mesa 25.3.6 specific issues vs earlier versions

4. VLLM INTEGRATION LESSONS LEARNED
   - EngineCore subprocess + Vulkan: what works, what doesn't
   - The torch.cuda.Stream fix needed in gpu_model_runner.py
   - Platform detection: VLLM_PLATFORM=vulkan flow
   - The _C.abi3.so symbol mismatch between torch versions
   - Which files MUST come from the same build as _C.abi3.so

5. MODEL-SPECIFIC FINDINGS
   - Qwen2 gate_up splitting: where to split, cosine after split
   - Llama gate_proj vs up_proj: separate weights, no split needed?
   - CPU SiLU vs Vulkan SiLU: why CPU SiLU is needed
   - Attention fast path: what it caches, where the speedup comes from

6. WHAT DIDN'T WORK (critical for avoiding wasted time)
   - fp16 on Vulkan (Packing.cpp rejects it)
   - Vulkan RMSNorm (scalar division crashes)
   - General gemm dispatch (broke vLLM model runner)
   - Multiple Vulkan contexts (worse than single context)
   - CUDAGraph on Vulkan (obviously doesn't work)

7. PERFORMANCE PROJECTIONS FOR 8B ON M1 ULTRA
   - Your estimated TPS based on scaling from 0.5B/1.5B/3B
   - Memory requirements (MLP weights + attention weights + KV cache)
   - Expected bottlenecks (CPU attention? transfer overhead? Vulkan compute?)
```

---

## IMMEDIATE EXECUTION PLAN

### Phase 0: BUILD (happening NOW, ~15 min remaining)
- vLLM building from source at ~/GITDEV/vllm_0.17.1 (SSH-initiated)
- vulkan-mlp-gpu branch checked out at ~/GITDEV/vllm-vulkan
- Once build completes, apply patches and test

### Phase 1: VALIDATE (30 min after build)
- Agent 0: Test Qwen-0.5B smoke test (known working config)
- Agent 0: Test Qwen-1.5B with all 28 layers
- Agent 0: Test Llama-8B with 32 layers — THE MAIN EVENT

### Phase 2: PARALLEL AGENTS (once 8B baseline established)
- Spawn Agent 1 (memory profiler) and Agent 2 (spec decode)
- Spawn Agent 3 (llama.cpp) immediately (independent)
- Each agent reads this plan + comms bridge before starting

### Phase 3: OPTIMIZE (hours 2-6)
- Apply Agent 1 findings (optimal batch threshold)
- Test Agent 2 spec decode (ngram first, then draft model)
- Compare with Agent 3 llama.cpp numbers
- Commit winning configuration

### Phase 4: REPORT (hour 6+)
- Comprehensive TPS table for all models on M1 Ultra
- Spec decode improvement factor
- Comparison with llama.cpp Vulkan and MLX (ground truth)
- Decision: ship or continue to Attack 3 (custom shaders)

---

## SUCCESS CRITERIA

- [ ] Llama-3.1-8B running ALL 32 MLP layers on Vulkan GPU
- [ ] Coherent text output (temp=0: factual, temp=0.7: creative)
- [ ] Measured TPS for 8B (target: 3-8 TPS single, 10+ batch)
- [ ] Spec decode working (ngram at minimum)
- [ ] Spec decode TPS improvement measured (target: 2-4x over baseline)
- [ ] llama.cpp Vulkan benchmark for comparison
- [ ] All results committed and documented

---

## COORDINATION RULES

1. **One agent per file** — claim in bridge before editing
2. **No agent touches another's build** — separate working dirs
3. **Bridge updates every 15 min** — or immediately on breakthrough/blocker
4. **Git workflow: only Agent 0 commits** — others propose patches via bridge
5. **Kill condition**: If 8B crashes Vulkan driver, STOP ALL AGENTS, reboot, restart with smaller test first
