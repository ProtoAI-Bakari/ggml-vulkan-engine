# Agent Coordination - Vulkan vLLM Project
# Both agents: READ before starting work. WRITE when claiming/completing tasks.
# Last updated: 2026-03-24 22:30

## ACTIVE AGENTS

### LEAD AGENT (original session)
- **Role**: Model-level Vulkan MLP code, testing, integration
- **Owns**: qwen2.py, llama.py model modifications, launch scripts, TPS benchmarks
- **Status**: Context compacted, continuing

### SUPPORT AGENT (this session)
- **Role**: Infrastructure, driver-level investigation, memory optimization, new model support
- **Owns**: core.py Vulkan paths, driver memory limits, Vulkan context management, Qwen3.5 port
- **Status**: Active

## DIVISION OF LABOR

### LEAD owns (DO NOT TOUCH):
- [ ] qwen2.py _vulkan_mlp forward path tuning
- [ ] llama.py _vulkan_mlp forward path tuning
- [ ] Launch scripts and TPS benchmarking
- [ ] Commit workflow on vulkan-mlp-gpu branch

### SUPPORT owns:
- [ ] core.py full-GPU-residency path investigation
- [ ] Vulkan memory limit investigation (2.6GB cap, 1/6 heap ratio)
- [ ] utils.py default_unquantized_gemm optimization
- [ ] Qwen3.5 model Vulkan MLP port (qwen3_5.py)
- [ ] Multi-context Vulkan experiment
- [ ] model_loader/utils.py weight streaming

## SHARED STATE
- Vulkan device memory: ~2,656 MB (2.6 GB) usable
- Single tensor limit: 1,024 MB
- Memory NOT reclaimable after free (VMA bug)
- 0.5B: 24 layers fit (1.25GB), WORKING at 17-27 TPS
- 1.5B: max ~16 layers fit, needs partial offload
- 3B: max ~8 layers fit, needs split gate_up + partial offload
- 8B: max ~3 layers fit on 32GB box

## BLOCKERS
1. core.py has "Vulkan Full GPU Residency" path that bypasses _vulkan_mlp layer control
2. VMA memory not reclaimable - can't stream layers through single context
3. 2.6GB hard cap on Vulkan buffers (Asahi DRM per-context limit?)
