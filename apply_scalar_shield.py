import os
import re

# 1. Self-Modification: Increase Agent Timeout to 600s
agent_path = os.path.expanduser("~/AGENT/cont_agent_v33.py")
if os.path.exists(agent_path):
    with open(agent_path, 'r') as f:
        agent_code = f.read()
    # Change timeout 45 or 180 to 600
    agent_code = re.sub(r"timeout\s*=\s*\d+", "timeout=600", agent_code)
    with open(agent_path, 'w') as f:
        f.write(agent_code)
    print("✅ Agent: Heartbeat increased to 600s.")
else:
    print(f"⚠️ Agent file not found at {agent_path}")

# 2. Shield Upgrade: Handle int64 AND bool for Vulkan
core_path = os.path.expanduser("~/GITDEV/vllm_0.17.1/vllm/v1/engine/core.py")
if os.path.exists(core_path):
    with open(core_path, 'r') as f:
        core_lines = f.readlines()

    new_bridge = '''
import torch
if os.environ.get('VLLM_PLATFORM') == 'vulkan':
    _orig_to = torch.Tensor.to
    def _vulkan_shield_to(self, *args, **kwargs):
        device = kwargs.get('device') or (args[0] if args else None)
        # If moving to Vulkan, downcast forbidden types
        if device and 'vulkan' in str(device):
            if self.dtype == torch.int64 or self.dtype == torch.bool:
                # Vulkan only supports int32 for metadata
                return _orig_to(self.to(torch.int32), *args, **kwargs)
        return _orig_to(self, *args, **kwargs)
    torch.Tensor.to = _vulkan_shield_to
    print("⚠️ VULKAN BRIDGE v5: Scalar Shield (int64/bool) Active")
'''

    content = "".join(core_lines)
    if "VULKAN BRIDGE" in content:
        content = re.sub(r"import torch\s+if os\.environ\.get\('VLLM_PLATFORM'\) == 'vulkan':.*?Active.*?\)\"", new_bridge, content, flags=re.DOTALL)
    else:
        # Insert after imports
        content = content.replace("from vllm.version import __version__ as VLLM_VERSION", 
                                 "from vllm.version import __version__ as VLLM_VERSION\n" + new_bridge)

    with open(core_path, 'w') as f:
        f.write(content)
    print("✅ core.py: Scalar Shield v5 (int64 + bool) Installed.")
else:
    print(f"⚠️ core.py file not found at {core_path}")
