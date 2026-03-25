import os

path = "v33agent.py"
if not os.path.exists(path):
    print(f"Error: {path} not found. Ensure you did 'cp cont_agent_v32.py v33agent.py' first.")
    exit(1)

with open(path, "r") as f:
    src = f.read()

# 1. FIX THE BASH TOOL (3 Min Timeout + 15s Heartbeat)
old_bash = """def execute_bash(command: str) -> str:
    if re.search(r'\\btimeout\\s+\\d+', command) or re.search(r'\\bsleep\\s+\\d+', command) or command.strip().endswith('&'):
        return "[SYSTEM OVERRIDE ERROR]: Command rejected! You violated the ANTI-HANG DIRECTIVE. Remove 'timeout', 'sleep', and background '&'. Just run the script directly. The Python wrapper will handle the 45s timeout automatically."

    try:
        wrapped_cmd = f"timeout 45 bash -c {subprocess.list2cmdline([command])}"
        res = subprocess.run(wrapped_cmd, shell=True, capture_output=True, text=True)
        out = res.stdout if res.stdout else res.stderr

        output_str = out.strip() if out else ""
        truncated = False
        if len(output_str) > 250000:
            output_str = output_str[:250000]
            truncated = True

        if res.returncode == 124:
            return output_str + "\\n\\n[AGENT SYSTEM]: Command timed out after 45s. GHOST LOAD DETECTED. You failed. Analyze the logs above."

        if truncated:
            return output_str + "\\n\\n[AGENT SYSTEM WARNING]: The output was too massive and was truncated. If you are looking for an error or traceback, IT WAS CUT OFF. Use `tail -n 200` on the log file to see the actual error."

        return output_str if output_str else "Done."
    except Exception as e: return f"Error: {e}\""""

new_bash = """def execute_bash(command: str) -> str:
    if re.search(r'\\btimeout\\s+\\d+', command) or re.search(r'\\bsleep\\s+\\d+', command) or command.strip().endswith('&'):
        return "[SYSTEM OVERRIDE ERROR]: Violation! Remove 'timeout/sleep/&'. Python handles the 180s cap."

    print(f"[*] Executing (Max 180s)...")
    try:
        proc = subprocess.Popen(f"timeout 180 bash -c {subprocess.list2cmdline([command])}", 
                                shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        full_output = []
        start_time = time.time()
        last_poll_time = start_time
        while True:
            if time.time() - last_poll_time > 15:
                print(f"    [Still Crunching... {int(time.time() - start_time)}s elapsed]")
                last_poll_time = time.time()
            line = proc.stdout.readline()
            if not line and proc.poll() is not None: break
            if line: full_output.append(line)
        out = "".join(full_output).strip()
        if proc.returncode == 124:
            return out + "\\n\\n[💀 FATAL: Command killed after 180s. Possible Ghost Load.]"
        return out if out else "Done."
    except Exception as e: return f"Error: {e}\""""

src = src.replace(old_bash, new_bash)

# 2. ADD TURN COUNTER & 10-RUN PAUSE
src = src.replace("current_ctx_tokens = 0", "current_ctx_tokens = 0\\n    turn_counter = 0")
src = src.replace("while True:\\n        try:", "while True:\\n        try:\\n            turn_counter += 1")

old_action_check = 'if not tool_calls:\\n                print("\\n[🤖 Z-Alpha has no more actions to take.]")'
new_action_check = """if turn_counter % 10 == 0:
                print(f"\\n[⏸️ 10-RUN CHECKPOINT: Turn {turn_counter}]")
                user_msg = get_multiline_input("Z-Alpha is pausing for your input. Press Enter to continue or type new instructions:")
                if user_msg:
                    history.append({"role": "user", "content": f"[ARCHITECT INTERVENTION]: {user_msg}"})
                    continue

            if not tool_calls:
                print("\\n[🤖 Z-Alpha has no more actions to take.]")"""

src = src.replace(old_action_check, new_action_check)

# 3. UPDATE PROMPT TIMEOUT TEXT
src = src.replace("automatically timeout after 45 seconds", "automatically timeout after 180 seconds (3 minutes)")

with open(path, "w") as f:
    f.write(src)

print("✅ v33agent.py successfully patched.")
