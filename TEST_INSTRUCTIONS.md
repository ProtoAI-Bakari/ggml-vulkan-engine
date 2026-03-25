# Vulkan vLLM Test Instructions
# Updated: 2026-03-24 22:52

## Quick Test Commands (Direct Python, NO server)

### Test 0.5B (all 24 layers on GPU, fastest)
```bash
cd ~/GITDEV/vllm_0.17.1
VLLM_PLATFORM=vulkan PYTORCH_ENABLE_MPS_FALLBACK=1 VLLM_USE_V1=1 VLLM_VK_MLP_LAYERS=24 \
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen2.5-0.5B-Instruct', dtype='float16', enforce_eager=True, max_model_len=256, enable_chunked_prefill=False, gpu_memory_utilization=0.01)
out = llm.generate(['The capital of France is'], SamplingParams(temperature=0, max_tokens=30))
print(out[0].outputs[0].text)
"
```

### Test 1.5B (14 layers on GPU)
```bash
cd ~/GITDEV/vllm_0.17.1
VLLM_PLATFORM=vulkan PYTORCH_ENABLE_MPS_FALLBACK=1 VLLM_USE_V1=1 VLLM_VK_MLP_LAYERS=14 \
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='/home/z/models/Qwen2.5-1.5B-Instruct', dtype='float16', enforce_eager=True, max_model_len=256, enable_chunked_prefill=False, gpu_memory_utilization=0.01)
out = llm.generate(['The capital of France is'], SamplingParams(temperature=0, max_tokens=30))
print(out[0].outputs[0].text)
"
```

### Test 3B (8 layers on GPU)
```bash
cd ~/GITDEV/vllm_0.17.1
VLLM_PLATFORM=vulkan PYTORCH_ENABLE_MPS_FALLBACK=1 VLLM_USE_V1=1 VLLM_VK_MLP_LAYERS=8 \
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='/home/z/models/Qwen2.5-3B-Instruct', dtype='float16', enforce_eager=True, max_model_len=256, enable_chunked_prefill=False, gpu_memory_utilization=0.01)
out = llm.generate(['The capital of France is'], SamplingParams(temperature=0, max_tokens=30))
print(out[0].outputs[0].text)
"
```

## Key Environment Variables
- `VLLM_PLATFORM=vulkan` — force Vulkan platform
- `VLLM_VK_MLP_LAYERS=N` — how many MLP layers go to GPU (rest stay CPU)
- `VLLM_USE_V1=1` — use V1 engine

## Layer Budget (2.6 GB Vulkan memory)
| Model | Max GPU layers | Total layers | Expected TPS |
|-------|---------------|--------------|-------------|
| 0.5B  | 24 (all)      | 24           | 17-27       |
| 1.5B  | 14            | 28           | 6-8         |
| 3B    | 8             | 36           | 2-4         |
| 8B    | 3             | 32           | <2 (needs M1 Ultra) |

## Files Modified
- `vllm/model_executor/models/qwen2.py` — Vulkan MLP with split gate_up + CPU SiLU
- `vllm/model_executor/models/llama.py` — Vulkan MLP for Llama
- `vllm/model_executor/layers/utils.py` — Disabled general Vulkan cache
- `vllm/model_executor/model_loader/utils.py` — Selective offload messaging
- `vllm/v1/engine/core.py` — Vulkan bridge, dtype shield, memory limits
