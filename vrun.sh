#!/bin/bash
export VLLM_USE_V1=1
export OMP_NUM_THREADS=10
export MAX_JOBS=10
MODEL="/home/z/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"

vllm serve "$MODEL" \
    --host 0.0.0.0 --port 8000 --dtype float16 \
    --max-model-len 1024 \
    --max-num-batched-tokens 256 \
    --enforce-eager \
    --served-model-name qwen25 2>&1 | tee ~/AGENT/LOGS/vllm_server.log
