import re

with open('v32sys4.py', 'r') as f:
    code = f.read()

# Re-inject global counters
if "TOTAL_TOKENS_GEN = 0" not in code:
    code = code.replace("client = OpenAI", "TOTAL_TOKENS_GEN = 0\nTOTAL_WORDS_GEN = 0\nclient = OpenAI")

# Find the loop and replace with telemetry version
old_loop_start = """            print(f"\\n[Z-Alpha]: ", end="", flush=True)
            full_content = ""
            try:
                resp = client.chat.completions.create(model=MODEL_NAME, messages=history, stream=True, max_tokens=4096)
                for chunk in resp:
                    if chunk.choices and chunk.choices[0].delta.content:
                        c = chunk.choices[0].delta.content
                        print(c, end="", flush=True); full_content += c"""

new_loop_start = """            print(f"\\n[Z-Alpha]: ", end="", flush=True)
            full_content = ""
            start_time = time.perf_counter()
            first_token_time = None
            tokens_in_run = 0
            
            try:
                resp = client.chat.completions.create(model=MODEL_NAME, messages=history, stream=True, max_tokens=8192)
                for chunk in resp:
                    if chunk.choices and chunk.choices[0].delta.content:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        c = chunk.choices[0].delta.content
                        print(c, end="", flush=True)
                        full_content += c
                        tokens_in_run += 1
                
                end_time = time.perf_counter()
                if tokens_in_run > 0:
                    ttft = (first_token_time - start_time) * 1000
                    total_time = end_time - first_token_time
                    tps = tokens_in_run / total_time if total_time > 0 else 0
                    itl = (total_time / tokens_in_run) * 1000
                    words = len(full_content.split())
                    print(f"\\n\\n[📊 STATS | TTFT: {ttft:.2f}ms | TPS: {tps:.2f} | ITL: {itl:.2f}ms | Tokens: {tokens_in_run} | Words: {words}]")"""

if old_loop_start in code:
    code = code.replace(old_loop_start, new_loop_start)
    with open('v32sys4.py', 'w') as f:
        f.write(code)
    print("✅ Telemetry Restored.")
else:
    print("❌ Could not find loop block. Check script name or manual edit required.")
