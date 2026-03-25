# Session Summary - 2026-03-24

## What Was Accomplished

### Hybrid CPU/Vulkan Dispatch Working (commit 6a7fd1e5d)
- **20+ TPS sustained** with 10 concurrent requests
- **22 TPS burst** on single requests
- **Coherent English output** verified
- Server stable under load, no crashes

### Key Changes (3 files on top of eb73a8290)
1. **`vllm/model_executor/layers/utils.py`** - Hybrid gemm: CPU for decode (batch<=8), Vulkan for prefill (batch>8) with weight caching
2. **`vllm/v1/engine/core.py`** - dtype shield: ALL non-float32 -> float32 before Vulkan, int tensors stay on CPU
3. **`vllm/v1/attention/backends/cpu_attn.py`** - AMX check try/except (Intel-only)

### PyTorch Changes (separate, in ~/GITDEV/pytorch)
- Descriptor pool: 1024 -> 65536 (enables large matmuls on Vulkan)
- FREE_DESCRIPTOR_SET_BIT flag added
- Wheel saved: ~/WHEELS/torch-2.12.0a0+git5de8e44-vulkan_pool_65536-cp312-cp312-linux_aarch64.whl

### Performance Data
- Vulkan matmul is 6-81x faster than CPU for batch>1
- CPU matmul is 2.6x faster than Vulkan for batch=1
- Transfer overhead CPU<->Vulkan is ~0.04-0.2us (negligible on unified memory)
- The bottleneck for decode is Vulkan dispatch overhead per matmul, not transfer

### Agent Framework (v44_GPU.py)
- Bridge comms working (JSONL file-based)
- Control file for inject/kill/prompt-update
- Auto-status every 10 turns, mandatory Claude check-in every 50
- Safety guards: no pkill python, auto-kill before server launch
- Intelligence plan written: ~/AGENT/AGENT_INTELLIGENCE_PLAN.md

## Server Running
```bash
# Server is running on port 8000
curl http://localhost:8000/v1/models
# Log: ~/AGENT/LOGS/hybrid_20260324_150041.log
# Commit: 6a7fd1e5d
```

## Next Steps
1. Test with 8B model (~/models/Llama-3.1-8B-FP8) - Vulkan benefit should be larger
2. Fix PyTorch Vulkan fp16 in Packing.cpp - halves memory, enables real GPU fp16
3. MLP-block-level Vulkan dispatch (2 transfers/layer vs 6)
4. Multi-agent framework with dedicated roles
5. PyTorch zero-copy on unified memory (the ultimate fix)
