#!/usr/bin/env python3
"""Fix for vLLM Vulkan: sampled_token_ids_cpu attribute error"""

import os

file_path = '/home/z/GITDEV/vllm_0.17.1/vllm/v1/worker/gpu_model_runner.py'

with open(file_path, 'r') as f:
    content = f.read()

# Find and replace the problematic code
old_code = """        # Initiate the copy on a separate stream, but do not synchronize it.
        # Only get CUDA stream if not using Vulkan (Vulkan doesn't have CUDA streams)
        is_vulkan = os.environ.get('VLLM_PLATFORM') == 'vulkan'
        default_stream = None if is_vulkan else torch.cuda.current_stream()
        if async_output_copy_stream is not None:
            with torch.cuda.stream(async_output_copy_stream):
                if not is_vulkan:
                    async_output_copy_stream.wait_stream(default_stream)
            self.sampled_token_ids_cpu = self._sampled_token_ids.to(
                "cpu", non_blocking=True
            )
            self._logprobs_tensors_cpu = (
                self._logprobs_tensors.to_cpu_nonblocking()
                if self._logprobs_tensors
                else None
            )
            if os.environ.get('VLLM_PLATFORM') != 'vulkan':
                self.async_copy_ready_event.record()"""

new_code = """        # Initiate the copy on a separate stream, but do not synchronize it.
        # Only get CUDA stream if not using Vulkan (Vulkan doesn't have CUDA streams)
        is_vulkan = os.environ.get('VLLM_PLATFORM') == 'vulkan'
        default_stream = None if is_vulkan else torch.cuda.current_stream()
        
        # Always create sampled_token_ids_cpu, even on Vulkan
        self.sampled_token_ids_cpu = self._sampled_token_ids.to(
            "cpu", non_blocking=True
        )
        self._logprobs_tensors_cpu = (
            self._logprobs_tensors.to_cpu_nonblocking()
            if self._logprobs_tensors
            else None
        )
        
        # Use async stream only for CUDA (not Vulkan)
        if async_output_copy_stream is not None and not is_vulkan:
            with torch.cuda.stream(async_output_copy_stream):
                async_output_copy_stream.wait_stream(default_stream)
            self.async_copy_ready_event.record()
        elif is_vulkan:
            # For Vulkan, record event immediately since no async stream
            self.async_copy_ready_event.record()"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"✅ Fixed sampled_token_ids_cpu attribute issue in gpu_model_runner.py")
    print(f"   File: {file_path}")
else:
    print(f"❌ Could not find the target code in {file_path}")
    print("   Manual intervention required.")