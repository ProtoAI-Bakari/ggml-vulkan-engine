import os

path = "v33agent.py"
with open(path, "r") as f:
    lines = f.readlines()

fixed_lines = []
for line in lines:
    # Fix the mashed 'current_ctx_tokens' line
    if "current_ctx_tokens = 0\\n" in line:
        fixed_lines.append("    current_ctx_tokens = 0\n")
        fixed_lines.append("    turn_counter = 0\n")
    # Fix the mashed 'while True' line
    elif "while True:\\n" in line:
        fixed_lines.append("    while True:\n")
        fixed_lines.append("        try:\n")
        fixed_lines.append("            turn_counter += 1\n")
    # Fix the mashed '10-run checkpoint' block
    elif "if turn_counter % 10 == 0:\\n" in line:
        fixed_lines.append("            if turn_counter % 10 == 0:\n")
        fixed_lines.append("                print(f\"\\n[⏸️ 10-RUN CHECKPOINT: Turn {turn_counter}]\")\n")
        fixed_lines.append("                user_msg = get_multiline_input(\"Z-Alpha is pausing for your input. Press Enter to continue:\")\n")
        fixed_lines.append("                if user_msg:\n")
        fixed_lines.append("                    history.append({\"role\": \"user\", \"content\": f\"[ARCHITECT INTERVENTION]: {user_msg}\"})\n")
        fixed_lines.append("                    continue\n")
        fixed_lines.append("\n")
        fixed_lines.append("            if not tool_calls:\n")
        fixed_lines.append("                print(\"\\n[🤖 Z-Alpha has no more actions to take.]\")\n")
    else:
        fixed_lines.append(line)

with open(path, "w") as f:
    f.writelines(fixed_lines)

print("✅ v33agent.py syntax repaired. Mashup lines expanded into proper blocks.")
