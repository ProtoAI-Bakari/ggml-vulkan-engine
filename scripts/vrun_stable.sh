#!/bin/bash
# vLLM Vulkan Server - STABLE MODE (enforce-eager for Vulkan)
cd ~/AGENT
export VLLM_USE_VULKAN=1
export VLLM_PLATFORM=vulkan
export VLLM_LOGGING_LEVEL=INFO

MODEL_PATH="/home/z/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
LOGFILE="/home/z/AGENT/LOGS/vulkan_server_stable_20260323_225922.log"
mkdir -p /home/z/AGENT/LOGS

echo "🚀 STARTING vLLM VULKAN - ENFORCE EAGER MODE (STABLE)"
/home/z/.venv-vLLM_0.17.1_Stable/bin/vllm serve     ""     --host 0.0.0.0     --port 8000     --dtype float16     --max-model-len 8192     --gpu-memory-utilization 0.95     --served-model-name qwen25     --enforce-eager     2>&1 | tee "" &
echo  > /home/z/AGENT/LOGS/server.pid
echo "Server started, PID: "
echo "Log: "
