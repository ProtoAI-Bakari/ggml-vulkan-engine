import re

with open("cont_agent_v22.py", "r") as f:
    src = f.read()

old_parser_block = '''            # Frankenstein Fallback: Model used <tool_call> but stuffed XML parameters inside it
            elif "<parameter=command>" in raw_json:
                cmd_match = re.search(r'<parameter=command>\\s*(.*?)\\s*</parameter>', raw_json, re.DOTALL)
                if cmd_match:
                    tcs.append({"id": f"xml_{time.time()}", "name": "execute_bash", "arguments": json.dumps({"command": cmd_match.group(1).strip()})})'''

new_parser_block = '''            # Frankenstein Fallback: Model used <tool_call> but stuffed XML parameters inside it
            elif "<parameter=command>" in raw_json:
                cmd_match = re.search(r'<parameter=command>\\s*(.*?)\\s*</parameter>', raw_json, re.DOTALL)
                if cmd_match:
                    tcs.append({"id": f"xml_{time.time()}", "name": "execute_bash", "arguments": json.dumps({"command": cmd_match.group(1).strip()})})
            
            # Nuclear Fallback: The JSON is completely destroyed, but we know it's a bash command
            elif "execute_bash" in raw_json:
                # Rip everything out that isn't the tool name, trying to find the core payload
                # This catches {"name": "execute_bash", "<parameter>\ncommand>\n [PAYLOAD] \n</parameter>
                # It just looks for the longest contiguous block of code.
                lines = raw_json.split('\\n')
                # Filter out lines that look like broken JSON/XML headers or footers
                payload_lines = [l for l in lines if not re.match(r'^\\s*(\\{|"name"|<parameter|</parameter|\\}|</function|<function)', l)]
                clean_cmd = "\\n".join(payload_lines).strip()
                if clean_cmd:
                    tcs.append({"id": f"xml_{time.time()}", "name": "execute_bash", "arguments": json.dumps({"command": clean_cmd})})'''

src = src.replace(old_parser_block, new_parser_block)

with open("cont_agent_v22.py", "w") as f:
    f.write(src)

print("✅ Nuclear Fallback injected. The parser is now practically invincible.")
