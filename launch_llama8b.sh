#!/bin/bash
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

LOG=/home/z/AGENT/LOGS/llama8b_$(date +%s).log
echo "Logging to $LOG"

vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --dtype float16 --max-model-len 256 \
  --enforce-eager --gpu-memory-utilization 0.9 \
  > "$LOG" 2>&1 &
PID=$!
echo "PID=$PID"

for i in $(seq 1 120); do
  if ! kill -0 $PID 2>/dev/null; then
    echo "DIED at ${i}s"
    echo "=== ERROR ==="
    tail -30 "$LOG"
    exit 1
  fi
  if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "READY at ${i}s"
    echo "=== TEST ==="
    curl -s http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":20,"temperature":0}' \
      | python3 -c "import sys,json;r=json.load(sys.stdin);print('OUTPUT:', r['choices'][0]['message']['content'])"
    exit 0
  fi
  sleep 1
done
echo "TIMEOUT"
tail -30 "$LOG"
