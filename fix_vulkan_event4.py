#!/usr/bin/env python3
"""Fix for vLLM Vulkan: torch.Event() synchronize() on None"""

import os

file_path = '/home/z/GITDEV/vllm_0.17.1/vllm/v1/worker/gpu_model_runner.py'

with open(file_path, 'r') as f:
    content = f.read()

# Find and replace the problematic code - the synchronize call
old_code = """        max_gen_len = self.sampled_token_ids_cpu.shape[-1]
        self.async_copy_ready_event.synchronize()"""

new_code = """        max_gen_len = self.sampled_token_ids_cpu.shape[-1]
        # Only synchronize if event exists (CUDA only)
        if self.async_copy_ready_event is not None:
            self.async_copy_ready_event.synchronize()"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"✅ Fixed async_copy_ready_event.synchronize() issue in gpu_model_runner.py")
    print(f"   File: {file_path}")
else:
    print(f"❌ Could not find the target code in {file_path}")
    print("   Manual intervention required.")