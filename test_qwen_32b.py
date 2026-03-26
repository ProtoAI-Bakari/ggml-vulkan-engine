#!/usr/bin/env python3
from ggml_vllm_backend import GgmlLLM, SamplingParams

try:
    llm = GgmlLLM("/home/z/models/gguf/Qwen2.5-32B-Instruct-Q4_K_M.gguf")
    r = llm.generate("What is quantum computing?", params=SamplingParams(temperature=0, max_tokens=10))
    print(f"{r.tps:.0f} TPS: {r.text}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
