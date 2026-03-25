#!/usr/bin/env python3
"""Patch vLLM Vulkan platform to disable pin_memory."""

import re

vulkan_py_path = "/home/z/GITDEV/vllm_0.17.1/vllm/platforms/vulkan.py"

with open(vulkan_py_path, 'r') as f:
    content = f.read()

# Find the location to insert the new method (after get_model_runner_cls)
pattern = r'(@classmethod\s+def get_model_runner_cls\(cls, vllm_config\):.*?return "vllm\.v1\.worker\.gpu_model_runner\.GPUModelRunner")'

replacement = r'''\1
    
    @classmethod
    def is_pin_memory_available(cls) -> bool:
        """Vulkan on Asahi Linux does not support pinned memory.
        
        Pinning memory requires CUDA or other accelerator backends.
        Vulkan on Asahi (Apple Silicon on Linux) lacks this capability.
        """
        return False'''

new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

if new_content == content:
    print("ERROR: Pattern not found in vulkan.py")
    exit(1)

with open(vulkan_py_path, 'w') as f:
    f.write(new_content)

print("SUCCESS: Patched vulkan.py to disable pin_memory for Vulkan")
