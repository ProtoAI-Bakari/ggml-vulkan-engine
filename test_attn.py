import os, time
os.environ['VLLM_PLATFORM'] = 'vulkan'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['VLLM_USE_V1'] = '1'
os.environ['VLLM_VK_MLP_LAYERS'] = '24'
os.environ['VLLM_VK_BATCH_THRESHOLD'] = '4'
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="float16", enforce_eager=True, max_model_len=256, enable_chunked_prefill=False, gpu_memory_utilization=0.01)
print("LOADED")
llm.generate(["Hi"], SamplingParams(temperature=0, max_tokens=5))
p = SamplingParams(temperature=0, max_tokens=100)
for i in range(3):
    t0 = time.time()
    out = llm.generate(["Write a detailed essay about artificial intelligence:"], p)
    t1 = time.time()
    toks = len(out[0].outputs[0].token_ids)
    print(f"RUN {i+1}: {toks} tok in {t1-t0:.2f}s = {toks/(t1-t0):.1f} TPS")
print(f"TEXT: {out[0].outputs[0].text[:100]}")
