#!/usr/bin/env python3
"""Patch core.py to ALLOW native FP16 on Vulkan"""
import re

core_path = "/home/z/GITDEV/vllm_0.17.1/vllm/v1/engine/core.py"

with open(core_path, 'r') as f:
    content = f.read()

# Replace the aggressive float32-only logic with FP16-allowing logic
old_logic = '''        # If moving TO Vulkan, FORCE float32 for ALL tensors
        if device and 'vulkan' in str(device):
            # AGGRESSIVE: Convert ANY non-float32 dtype to float32
            if self.dtype != torch.float32:
                # Log conversion for debugging
                if self.dtype not in (torch.int32, torch.int64, torch.bool, torch.float16, torch.half, torch.bfloat16):
                    pass  # Silent for common types
                self = _orig_vulkan_to(self, torch.float32)'''

new_logic = '''        # If moving TO Vulkan, allow native dtypes (FP16 NOW SUPPORTED!)
        if device and 'vulkan' in str(device):
            # ✅ NATIVE FP16 SUPPORT ENABLED (PyTorch C++ rebuild complete)
            # Only convert truly unsupported types, keep float16/native dtypes
            if self.dtype in (torch.int64, torch.bool):
                # Vulkan metadata needs int32, bool not supported
                self = _orig_vulkan_to(self, torch.int32)
            elif self.dtype not in (torch.float32, torch.float16, torch.half, torch.bfloat16, torch.int32):
                # Convert unknown types to float32 as fallback
                self = _orig_vulkan_to(self, torch.float32)
            # float16/half, bfloat16, float32, int32 pass through unchanged'''

content = content.replace(old_logic, new_logic)

# Update the version message
content = content.replace(
    'print("⚠️ VULKAN BRIDGE v8: AGGRESSIVE float32-ONLY Shield Active")',
    'print("✅ VULKAN BRIDGE v9: NATIVE FP16 SUPPORT ENABLED")'
)

with open(core_path, 'w') as f:
    f.write(content)

print("✅ core.py patched - Native FP16 now allowed on Vulkan")
