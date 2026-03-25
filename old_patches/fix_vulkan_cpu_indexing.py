#!/usr/bin/env python3
"""Fix Vulkan CPU tensor indexing issue"""
import re

runner_path = "/home/z/GITDEV/vllm_0.17.1/vllm/v1/worker/gpu_model_runner.py"

with open(runner_path, 'r') as f:
    content = f.read()

# Fix the indexing issue - ensure indices are on CPU when indexing CPU tensors
old_code = '''        self.input_ids.gpu.scatter_(
            dim=0,
            index=sampled_tokens_index_tensor,
            src=self.input_batch.prev_sampled_token_ids[
                prev_common_req_indices_tensor, 0
            ],
        )'''

new_code = '''        # VULKAN FIX: Ensure indices are on CPU when indexing CPU tensors
        cpu_indices = prev_common_req_indices_tensor.cpu() if prev_common_req_indices_tensor.device.type != 'cpu' else prev_common_req_indices_tensor
        src_tensor = self.input_batch.prev_sampled_token_ids[cpu_indices, 0]
        self.input_ids.gpu.scatter_(
            dim=0,
            index=sampled_tokens_index_tensor,
            src=src_tensor,
        )'''

new_content = content.replace(old_code, new_code)

with open(runner_path, 'w') as f:
    f.write(new_content)

print("✅ Vulkan CPU indexing fix applied")
