#!/bin/bash
# Vulkan hybrid dispatch server launcher
set -e
# Clear stale bytecode
find /home/z/GITDEV/vllm_0.17.1 -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Kill old servers (exclude our own PID)
for pid in $(pgrep -f EngineCore 2>/dev/null); do [ "$pid" != "$$" ] && kill -9 "$pid" 2>/dev/null; done
for pid in $(pgrep -f "bin/vllm" 2>/dev/null); do [ "$pid" != "$$" ] && kill -9 "$pid" 2>/dev/null; done
sleep 2

source /home/z/.venv-vLLM_0.17.1_Stable/bin/activate
cd /home/z/GITDEV/vllm_0.17.1

export PYTHONPATH=/home/z/GITDEV/vllm_0.17.1:$PYTHONPATH
export VLLM_PLATFORM=vulkan
export OMP_NUM_THREADS=10

LOGFILE=/home/z/AGENT/LOGS/hybrid_$(date +%Y%m%d_%H%M%S).log
echo "Logging to: $LOGFILE"
echo "Starting vLLM Vulkan hybrid server..."

exec vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --dtype float16 --max-model-len 1024 \
  --enforce-eager --gpu-memory-utilization 0.8 \
  2>&1 | tee "$LOGFILE"
