# Sys0 Agent Communications Bridge
# ALL agents on Sys0: READ THIS BEFORE DOING ANYTHING. Append updates. Never edit others' sections.
# Format: [AGENT_NAME] [TIMESTAMP] [STATUS] message

---

## SYS0 AGENT 0 (LEAD) — 2026-03-25 00:40

### SYSTEM STATE
- **Machine**: M1 Ultra 128GB, Fedora Asahi Remix 42, 20 cores, 126 GiB RAM
- **Vulkan**: 1.4.328, Mesa Honeykrisp 25.3.6, 63.19 GiB heap (GPU0: Apple M1 Ultra G13D C0)
- **PyTorch**: 2.12.0a0+git5de8e44 with Vulkan — CONFIRMED WORKING in venv
- **venv**: ~/.venv-vLLM_0.17.1_Stable (Python 3.12)
- **IP**: 10.255.255.128

### BUILD IN PROGRESS (DO NOT START ANOTHER BUILD)
- **WHO**: Sys4 LEAD agent initiated via SSH
- **WHAT**: `pip install -e .` at ~/GITDEV/vllm_0.17.1 (non-git copy with patches applied)
- **BRANCH**: vulkan-mlp-gpu (f95b8e4cb) also available at ~/GITDEV/vllm-vulkan (git clone)
- **STATUS**: Compiling C extensions (oneDNN + _C.abi3.so), ~11 cc1plus procs, ETA ~5-10 min
- **PID**: ninja PID 41729, parent bash PID 38086

### WHAT I ALREADY DID
1. Installed all system deps (python3.12, vulkan-tools, cmake, ninja, etc)
2. Created venv with Vulkan PyTorch wheel from ~/WHEELS/
3. Verified torch.is_vulkan_available() = True
4. Verified Vulkan mm works (64x128 @ 128x64, roundtrip OK)
5. Cloned repo to ~/GITDEV/vllm-vulkan, checked out vulkan-mlp-gpu branch
6. Downloaded Llama-3.1-8B-Instruct to ~/models/
7. Downloaded Qwen2.5-1.5B and 3B already at ~/models/
8. Wrote battleplan at ~/AGENT/SYS0_MULTI_AGENT_BATTLEPLAN.md

### MISTAKES I MADE (so future agents don't repeat)
1. Tried to build vllm myself without checking if SSH build was already running
2. Mixed wheel-installed vllm with GITDEV source files (API mismatch: _C.abi3.so vs Python)
3. Didn't establish this bridge file immediately when told another agent was active

### WHAT'S NEXT (once build completes)
1. Verify vllm import works with Vulkan
2. Smoke test Qwen-0.5B or 1.5B (known working config)
3. TEST LLAMA-3.1-8B WITH ALL 32 LAYERS ON VULKAN — the main event
4. Spawn parallel agents for spec decode, profiling, llama.cpp comparison

### FILES I OWN (other agents: don't edit)
- ~/AGENT/SYS0_MULTI_AGENT_BATTLEPLAN.md
- ~/AGENT/Sys0-agent-comms-bridge.md (this file — but others APPEND)
- ~/AGENT/test_vulkan_sys0.py
- Git workflow (only I commit)

### MODELS AVAILABLE
- ~/models/Qwen2.5-1.5B-Instruct
- ~/models/Qwen2.5-3B-Instruct
- ~/models/Llama-3.1-8B-Instruct

### WAITING FOR
- Sys12 agent to create ~/AGENT/Sys12-Sys0-agent-comms-bridge.md with all CPU vs GPU findings
- Build to complete at ~/GITDEV/vllm_0.17.1

---

## LEAD — 2026-03-25 01:44 — THRESHOLD=0 RESULT

VLLM_VK_BATCH_THRESHOLD=0 (GPU always, even batch=1): **4.1 TPS** on 8B 200tok
Slightly SLOWER than threshold=4 (4.3 TPS). Per-layer roundtrip overhead dominates.
Raw matmul wins at batch=1 but full pipeline has too many dispatches per token.

Moving to Task 2: llama.cpp F16 benchmark.

## LEAD — 2026-03-25 01:50 — F16 BENCHMARK COMPLETE (DECISION POINT)

### llama.cpp Vulkan 8B Llama on M1 Ultra — ALL quantizations

| Quant | Size | Prefill TPS | Decode TPS | vs our vLLM FP32 |
|-------|------|------------|------------|-------------------|
| Q4_K_M | 4.6G | 137 | **24.8** | 5.7x faster |
| Q8_0 | 8.0G | 169 | **22.8** | 5.3x faster |
| **F16** | 15.0G | 171 | **13.9** | **3.2x faster** |

### ANALYSIS
Even at F16 (comparable to our FP32 precision-wise), llama.cpp shaders are 3.2x faster.
The gap breaks down as:
- **Shader quality**: ~3.2x (F16 vs FP32 = buffer-based tiled matmul vs image-based generic)
- **Quantization**: additional ~1.8x on top (Q4 decode 24.8 / F16 decode 13.9)
- Combined: 5.7x total gap

### DECISION: BOTH PATHS MATTER
1. **Better shaders** (Path B: ggml backend) → 3.2x improvement → ~14 TPS
2. **+ Quantization** (Q4/Q8 in our pipeline) → additional 1.8x → ~25 TPS
3. Both together = match llama.cpp performance within vLLM

### IMMEDIATE NEXT: Start ggml prototype (Path B)

## LEAD — 2026-03-25 01:51 — SIGNAL SYSTEM ESTABLISHED

### REAL-TIME AGENT COORDINATION
Installed inotify-tools. Signal protocol at ~/AGENT/.signals/protocol.md

**WORKER: Set up your watcher on session start:**
```bash
inotifywait -m -e modify ~/AGENT/.signals/LEAD_TO_WORKER &
```

**Before ANY benchmark/inference:**
1. echo "TIMESTAMP|CLAIM|description" > ~/AGENT/.signals/WORKER_TO_LEAD
2. Wait 2 seconds
3. Run benchmark
4. echo "TIMESTAMP|DONE|description" > ~/AGENT/.signals/WORKER_TO_LEAD

I'll do the same with LEAD_TO_WORKER. This prevents GPU contention.

### GPU STATUS: FREE
I'm done with all benchmarks for now. Worker can run.
