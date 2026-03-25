#!/usr/bin/env python3
"""Fix for vLLM Vulkan: torch.Event() not supported on CPU backend"""

import os

file_path = '/home/z/GITDEV/vllm_0.17.1/vllm/v1/worker/gpu_model_runner.py'

with open(file_path, 'r') as f:
    content = f.read()

# Find and replace the problematic code - just the event creation line
old_code = """        # Event on the copy stream so we can synchronize the non-blocking copy.
        self.async_copy_ready_event = torch.Event()"""

new_code = """        # Event on the copy stream so we can synchronize the non-blocking copy.
        # Only create event for CUDA (Vulkan/CPU don't support torch.Event)
        is_vulkan = os.environ.get('VLLM_PLATFORM') == 'vulkan'
        self.async_copy_ready_event = None if is_vulkan else torch.Event()"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"✅ Fixed torch.Event() CPU backend issue in gpu_model_runner.py")
    print(f"   File: {file_path}")
else:
    print(f"❌ Could not find the target code in {file_path}")
    print("   Manual intervention required.")