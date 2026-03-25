#!/usr/bin/env python3
import os
path = os.path.expanduser('~/AGENT/v33agent.py')
with open(path, 'r') as f:
    content = f.read()
content = content.replace('timeout 45 bash', 'timeout 300 bash')
content = content.replace('after 45s', 'after 300s')
content = content.replace('after 180 seconds', 'after 300 seconds')
with open(path, 'w') as f:
    f.write(content)
print('✅ Timeout updated to 300s in v33agent.py')