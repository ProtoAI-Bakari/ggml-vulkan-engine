#!/bin/bash
# Launch vLLM and keep it running. For benchmarking.
set -e
find /home/z/GITDEV/vllm_0.17.1 -name "__pycache__" -exec rm -rf {} + 2>/dev/null
pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f "bin/vllm" 2>/dev/null || true
sleep 2

source /home/z/.venv-vLLM_0.17.1_Stable/bin/activate
cd /home/z/GITDEV/vllm_0.17.1
export PYTHONPATH=/home/z/GITDEV/vllm_0.17.1:$PYTHONPATH
export VLLM_PLATFORM=vulkan
export OMP_NUM_THREADS=10

LOG=/home/z/AGENT/LOGS/vulkan_gpu_$(date +%s).log
echo "Logging to $LOG"

exec vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --dtype float16 --max-model-len 1024 \
  --enforce-eager --gpu-memory-utilization 0.8 \
  2>&1 | tee "$LOG"
