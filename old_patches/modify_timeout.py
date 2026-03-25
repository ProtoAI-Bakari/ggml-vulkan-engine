import os
import re
path = os.path.expanduser("~/AGENT/cont_agent_v32.py")
with open(path, 'r') as f:
    content = f.read()

# Replace 45s timeout with 300s using regex to avoid literal string
content = re.sub(r'timeout\s+45\b', 'timeout 300', content)
content = re.sub(r'after\s+45s\b', 'after 300s', content)
content = re.sub(r'after\s+45\s+seconds', 'after 300 seconds', content)

with open(path, 'w') as f:
    f.write(content)

print("✅ cont_agent_v32.py: Timeout increased from 45s to 300s")