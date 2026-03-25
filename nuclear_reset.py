import os

# 1. Hard-Lock vrun.sh to 0.5B ONLY
vrun_path = os.path.expanduser("~/AGENT/vrun.sh")
vrun_content = """#!/bin/bash
cd ~/AGENT
rm -rf ~/.cache/torch/inductor
rm -rf ~/.cache/vllm/torch_compile_cache

export MAX_JOBS=4
export OMP_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=4
export VLLM_USE_VULKAN=1
export VLLM_PLATFORM=vulkan
export VLLM_USE_V1=1
export TORCH_LOGS="recompiles"

# HARD-LOCKED TINY MODEL
MODEL="/home/z/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"

/home/z/.venv-vLLM_0.17.1_Stable/bin/vllm serve "$MODEL" --host 0.0.0.0 --port 8000 --dtype float16 --max-model-len 8192 --gpu-memory-utilization 0.8 --served-model-name qwen25 2>&1 | tee ~/AGENT/LOGS/vulkan_warrior.log
"""
with open(vrun_path, "w") as f: f.write(vrun_content)
os.chmod(vrun_path, 0o755)

# 2. Kill the Retry Loop in the Agent Code
agent_path = os.path.expanduser("~/AGENT/v32agent.py")
with open(agent_path, "r") as f:
    code = f.read()

# Force the command_loop_count to kill the process instead of retrying
code = code.replace('if command_loop_count >= 3:', 'if command_loop_count >= 2:\n                    print("🚨 LOOP DETECTED. EXITING.")\n                    os._exit(1)')
# Force the 5-second sleep in the bash tool
code = code.replace('time.sleep(5)', 'time.sleep(2)')

with open(agent_path, "w") as f: f.write(code)

# 3. Aggressive System Prompt
prompt_path = os.path.expanduser("~/AGENT/SYSTEM_PROMPT.txt")
prompt_content = """You are Z-Alpha, an AGGRESSIVE Code Warrior.
RULE 1: Use 'timeout 45s bash vrun.sh' to start the server.
RULE 2: Check the log EVERY 5 SECONDS with 'tail -n 20'.
RULE 3: If you see 'ImportError' or gibberish, call 'call_qwen3' (.4) IMMEDIATELY.
RULE 4: Do not repeat any command more than twice. Load the 0.5B model ONLY."""
with open(prompt_path, "w") as f: f.write(prompt_content)

print("✅ BRAIN SURGERY COMPLETE. vrun.sh locked to 0.5B. Retry loop killed.")
