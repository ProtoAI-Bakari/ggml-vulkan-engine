
import time
import torch
from ggml_model_wrapper import GgmlLLM

print('=== T05 CPU TIMING BREAKDOWN ===')

# Load model
start = time.perf_counter()
model = GgmlLLM('/home/z/models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf', device='vulkan')
load_time = time.perf_counter() - start
print(f'Model load: {load_time*1000:.1f}ms')

# Warmup
for _ in range(3):
    model.generate('Warmup', max_tokens=5)

# Timed inference with granular breakdown
prompt = 'What is 2 + 2?'
start = time.perf_counter()

# Tokenization
tok_start = time.perf_counter()
tokens = model.tokenizer.encode(prompt)
tok_time = time.perf_counter() - tok_start
print(f'Tokenization: {tok_time*1000:.2f}ms ({len(tokens)} tokens)')

# Prefill
prefill_start = time.perf_counter()
out = model.generate(prompt, max_tokens=1)
prefill_time = time.perf_counter() - prefill_start
print(f'Prefill (1 token): {prefill_time*1000:.2f}ms')

# Decode timing (10 tokens)
decode_times = []
for i in range(10):
    start_t = time.perf_counter()
    out = model.generate(prompt, max_tokens=i+2)
    end_t = time.perf_counter()
    decode_times.append((end_t - start_t) * 1000)
    print(f'Decode token {i+2}: {decode_times[-1]:.2f}ms')

avg_decode = sum(decode_times[1:]) / len(decode_times[1:])  # skip first (includes prefill)
print(f'
Average decode per token: {avg_decode:.2f}ms ({1000/avg_decode:.1f} TPS)')
print(f'Load time: {load_time*1000:.1f}ms')
