import os
path = os.path.expanduser("~/AGENT/v32agent.py")
with open(path, "r") as f:
    code = f.read()

qwen3_tool = """
def call_qwen3(query: str) -> str:
    \"\"\"DUO CALL: Mandatory for all C++, Linker, or 500 Internal Server Errors.\"\"\"
    url = "http://10.255.255.4:8000/v1/chat/completions"
    payload = {"model": "qwen3-next-coder", "messages": [{"role": "user", "content": query}], "temperature": 0.1}
    try:
        return requests.post(url, json=payload, timeout=300).json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"DUO CALL TO .4 FAILED: {e}"
"""

if "def call_qwen3" not in code:
    code = code.replace("def ask_human", qwen3_tool + "\ndef ask_human")
    code = code.replace('"ask_human": ask_human', '"call_qwen3": call_qwen3, "ask_human": ask_human')
    # Update TOOLS JSON
    new_json = '{"type": "function", "function": {"name": "call_qwen3", "description": "Call the high-level Qwen3-Next-Coder at .4 for complex logic/C++ fixes.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},\n    '
    code = code.replace('{"type": "function", "function": {"name": "ask_human"', new_json + '{"type": "function", "function": {"name": "ask_human"')

with open(path, "w") as f:
    f.write(code)
print("✅ DUO TOOL INJECTED: Z-Alpha can now talk to .4")
