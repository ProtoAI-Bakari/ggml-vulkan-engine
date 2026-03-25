import os

agent_path = os.path.expanduser("~/AGENT/v32agent.py")
with open(agent_path, "r") as f:
    code = f.read()

# 1. BRAIN SURGERY: Make loop_count=2 FATAL. No more 5x repetitions.
code = code.replace('if command_loop_count >= 3:', 
                    'if command_loop_count >= 2:\n                    print("🚨 LOOP DETECTED. OS-LEVEL EXIT.")\n                    os._exit(1)')

# 2. SENSORY UPGRADE: Reduce internal wait time to prevent "Thought Ghosting"
code = code.replace('time.sleep(5)', 'time.sleep(1)')

with open(agent_path, "w") as f:
    f.write(code)

# 3. MISSION RE-WRITE: Hard-lock the SYSTEM_PROMPT.txt
prompt_path = os.path.expanduser("~/AGENT/SYSTEM_PROMPT.txt")
prompt_content = """You are Z-Alpha, an AGGRESSIVE Code Warrior.
RULE 1: You are FORBIDDEN from loading any model larger than 0.5B. 
RULE 2: If a command takes more than 5 seconds, YOU MUST WAIT. Do not repeat commands.
RULE 3: If you see gibberish, call .4 (call_qwen3) immediately.
"""
with open(prompt_path, "w") as f:
    f.write(prompt_content)

print("✅ LOOP KILLER APPLIED. Agent will now self-destruct if it repeats itself.")
