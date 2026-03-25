#!/bin/bash
# Test inference on running vLLM Vulkan server
echo "Testing vLLM Vulkan inference..."
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5,
        "temperature": 0
    }' 2>&1

echo ""
echo "Checking process status..."
ps aux | grep -E "EngineCore|APIServer" | grep -v grep