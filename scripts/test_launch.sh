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

LOG=/home/z/AGENT/LOGS/test_$(date +%s).log
echo "Logging to $LOG"

vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --dtype float16 --max-model-len 1024 \
  --enforce-eager --gpu-memory-utilization 0.8 \
  > "$LOG" 2>&1 &
PID=$!
echo "vllm PID=$PID"

for i in $(seq 1 90); do
  if ! kill -0 $PID 2>/dev/null; then
    echo "DIED at ${i}s"
    echo "=== LAST 30 LINES ==="
    tail -30 "$LOG"
    exit 1
  fi
  if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "READY at ${i}s"
    echo "=== TESTING ==="
    curl -s http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"2+2?"}],"max_tokens":5,"temperature":0}' \
      | python3 -c "import sys,json;r=json.load(sys.stdin);print('OUTPUT:', r['choices'][0]['message']['content'])"
    echo "=== TPS ==="
    sleep 3
    grep throughput "$LOG" | tail -3
    kill $PID 2>/dev/null
    exit 0
  fi
  sleep 1
done
echo "TIMEOUT"
tail -30 "$LOG"
kill $PID 2>/dev/null
