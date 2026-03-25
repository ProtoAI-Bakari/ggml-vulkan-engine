#!/usr/bin/env python3
"""Fix for vLLM Vulkan: async_copy_ready_event assertion in gpu_input_batch.py"""

import os

file_path = '/home/z/GITDEV/vllm_0.17.1/vllm/v1/worker/gpu_input_batch.py'

with open(file_path, 'r') as f:
    content = f.read()

# Find and replace the problematic code
old_code = """            if sampled_token_ids is None:
                assert self.async_copy_ready_event is not None
                self.async_copy_ready_event.synchronize()
                sampled_token_ids = self.sampled_token_ids_cpu.tolist()"""

new_code = """            if sampled_token_ids is None:
                # Only synchronize if event exists (CUDA only)
                if self.async_copy_ready_event is not None:
                    self.async_copy_ready_event.synchronize()
                sampled_token_ids = self.sampled_token_ids_cpu.tolist()"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"✅ Fixed async_copy_ready_event assertion in gpu_input_batch.py")
    print(f"   File: {file_path}")
else:
    print(f"❌ Could not find the target code in {file_path}")
    print("   Manual intervention required.")