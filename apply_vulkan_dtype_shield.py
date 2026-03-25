#!/usr/bin/env python3
"""Apply Vulkan dtype shield fix to core.py"""
import re

core_path = "/home/z/GITDEV/vllm_0.17.1/vllm/v1/engine/core.py"

with open(core_path, 'r') as f:
    content = f.read()

# Find the import torch line and add our shield after it
vulkan_shield = '''

# VULKAN ASAHI FIX: dtype shield for Vulkan device transfers
import os
if os.environ.get('VLLM_PLATFORM') == 'vulkan':
    _orig_to = torch.Tensor.to
    
    def _vulkan_shield_to(self, *args, **kwargs):
        device = kwargs.get('device') or (args[0] if args else None)
        dtype = kwargs.get('dtype') or (args[1] if len(args) > 1 else None)
        
        # If moving to Vulkan, enforce float32 dtype
        if device and 'vulkan' in str(device):
            # Handle forbidden types
            if self.dtype == torch.int64:
                # Vulkan only supports int32 for metadata
                self = self.to(torch.int32)
            elif self.dtype == torch.bool:
                self = self.to(torch.int32)
            elif self.dtype == torch.float16 or self.dtype == torch.half:
                # Vulkan FP16 shader missing - force float32
                self = self.to(torch.float32)
            # bfloat16 is kept as-is but may not work on all Vulkan implementations
            
            # Remove dtype from kwargs if we've already converted
            if dtype is not None:
                kwargs.pop('dtype', None)
        
        return _orig_to(self, *args, **kwargs)
    
    torch.Tensor.to = _vulkan_shield_to
    print("⚠️ VULKAN BRIDGE v6: Full dtype Shield (int64/bool/fp16->fp32) Active")

'''

# Find "import torch" and insert after it
pattern = r'(import torch)'
replacement = r'\1' + vulkan_shield

new_content = re.sub(pattern, replacement, content, count=1)

with open(core_path, 'w') as f:
    f.write(new_content)

print("✅ Vulkan dtype shield applied to core.py")
