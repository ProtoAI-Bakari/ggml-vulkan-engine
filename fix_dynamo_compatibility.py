#!/usr/bin/env python3
"""
Fix: Make Vulkan dtype shield compatible with TorchDynamo/CUDAGraph
Add @torch.compiler.disable to prevent Dynamo from tracing the shield
"""

core_path = "/home/z/GITDEV/vllm_0.17.1/vllm/v1/engine/core.py"

with open(core_path, 'r') as f:
    content = f.read()

# Add torch.compiler.disable decorator to the shield function
old_function = '''    def _vulkan_shield_to(self, *args, **kwargs):'''
new_function = '''    @torch.compiler.disable  # Prevent Dynamo from tracing the shield
    def _vulkan_shield_to(self, *args, **kwargs):'''

if old_function in content and new_function not in content:
    content = content.replace(old_function, new_function)
    with open(core_path, 'w') as f:
        f.write(content)
    print("✅ core.py updated - Added @torch.compiler.disable to dtype shield")
else:
    print("⚠️ Pattern not found or already patched")
    if "@torch.compiler.disable" in content:
        print("  (Already has torch.compiler.disable)")