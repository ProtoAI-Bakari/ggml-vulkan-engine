import os

path = "v33agent.py"
with open(path, "r") as f:
    src = f.read()

# 1. SURGERY: Update the execute_bash function for 180s timeout + 15s log polling
old_bash = '''def execute_bash(command: str) -> str:
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
    except Exception as e: return f"Error: {e}"'''

new_bash = '''def execute_bash(command: str) -> str:
    # Lifted timeout to 180s (3 mins)
    if re.search(r'\\btimeout\\s+\\d+', command) or re.search(r'\\bsleep\\s+\\d+', command) or command.strip().endswith('&'):
        return "[SYSTEM OVERRIDE ERROR]: Violation! Remove 'timeout/sleep/&'. Python handles the 180s cap."

    print(f"[*] Executing (Max 180s)...")
    try:
        # Using Popen for real-time log watching and polling
        proc = subprocess.Popen(f"timeout 180 bash -c {subprocess.list2cmdline([command])}", 
                                shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        full_output = []
        start_time = time.time()
        last_poll_time = start_time
        
        while True:
            # Poll for updates every 15 seconds as requested
            if time.time() - last_poll_time > 15:
                print(f"    [Still Running... {int(time.time() - start_time)}s elapsed]")
                last_poll_time = time.time()

            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                full_output.append(line)

        out = "".join(full_output).strip()
        
        if proc.returncode == 124:
            return out + "\\n\\n[💀 FATAL: Command killed after 180s. Possible Ghost Load.]"
        
        return out if out else "Done."
    except Exception as e: return f"Error: {e}"'''

src = src.replace(old_bash, new_bash)

# 2. SURGERY: Update the system prompt text within the run_cli function
src = src.replace("automatically timeout after 45 seconds", "automatically timeout after 180 seconds (3 minutes)")

with open(path, "w") as f:
    f.write(src)

print("✅ v33agent.py Surgically Updated:")
print(" - Timeout lifted to 180 seconds.")
print(" - Log/Process watching set to 15-second intervals.")
