#!/usr/bin/env python3
"""Fix for vLLM Vulkan: torch.Event() not supported on CPU backend - Part 2"""

import os

file_path = '/home/z/GITDEV/vllm_0.17.1/vllm/v1/worker/gpu_model_runner.py'

with open(file_path, 'r') as f:
    content = f.read()

# Find and replace the problematic code - the event recording part
old_code = """        # Use async stream only for CUDA (not Vulkan)
        if async_output_copy_stream is not None and not is_vulkan:
            with torch.cuda.stream(async_output_copy_stream):
                async_output_copy_stream.wait_stream(default_stream)
            self.async_copy_ready_event.record()
        elif is_vulkan:
            # For Vulkan, record event immediately since no async stream
            self.async_copy_ready_event.record()"""

new_code = """        # Use async stream only for CUDA (not Vulkan)
        # Only record event if it exists (CUDA only)
        if async_output_copy_stream is not None and not is_vulkan and self.async_copy_ready_event is not None:
            with torch.cuda.stream(async_output_copy_stream):
                async_output_copy_stream.wait_stream(default_stream)
            self.async_copy_ready_event.record()"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"✅ Fixed async_copy_ready_event.record() issue in gpu_model_runner.py")
    print(f"   File: {file_path}")
else:
    print(f"❌ Could not find the target code in {file_path}")
    print("   Manual intervention required.")