#!/bin/bash
source /home/z/.venv-vLLM_0.17.1_Stable/bin/activate
cd /home/z/GITDEV/vllm_0.17.1
export PYTHONPATH=/home/z/GITDEV/vllm_0.17.1:$PYTHONPATH
export VLLM_PLATFORM=vulkan
export OMP_NUM_THREADS=10
exec vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --dtype float16 --max-model-len 1024 \
  --enforce-eager --gpu-memory-utilization 0.8
