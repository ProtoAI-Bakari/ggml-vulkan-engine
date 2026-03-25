#!/usr/bin/env python3
"""Fix Vulkan dtype shield - ensure conversion happens BEFORE calling original to()"""
import re

core_path = "/home/z/GITDEV/vllm_0.17.1/vllm/v1/engine/core.py"

with open(core_path, 'r') as f:
    content = f.read()

# Find and replace the _vulkan_shield_to function
old_shield = '''# VULKAN ASAHI FIX: dtype shield for Vulkan device transfers
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
    print("⚠️ VULKAN BRIDGE v6: Full dtype Shield (int64/bool/fp16->fp32) Active")'''

new_shield = '''# VULKAN ASAHI FIX: dtype shield for Vulkan device transfers
import os
if os.environ.get('VLLM_PLATFORM') == 'vulkan':
    # Save original to method BEFORE any patching
    _orig_vulkan_to = torch.Tensor.to.__func__ if hasattr(torch.Tensor.to, '__func__') else torch.Tensor.to
    
    def _vulkan_shield_to(self, *args, **kwargs):
        device = kwargs.get('device') or (args[0] if args else None)
        
        # If moving TO Vulkan, enforce float32 dtype FIRST
        if device and 'vulkan' in str(device):
            # Convert self to float32 if needed BEFORE calling original to()
            if self.dtype == torch.int64:
                self = _orig_vulkan_to(self, torch.int32)
            elif self.dtype == torch.bool:
                self = _orig_vulkan_to(self, torch.int32)
            elif self.dtype == torch.float16 or self.dtype == torch.half:
                self = _orig_vulkan_to(self, torch.float32)
            
            # Now call original to() with device only (no dtype to avoid conflict)
            kwargs.pop('dtype', None)
            if len(args) > 1:
                args = (args[0],)  # Remove dtype from args
            return _orig_vulkan_to(self, *args, **kwargs)
        
        # For non-vulkan devices, use normal to()
        return _orig_vulkan_to(self, *args, **kwargs)
    
    torch.Tensor.to = _vulkan_shield_to
    print("⚠️ VULKAN BRIDGE v7: Pre-convert Shield (dtype->fp32 BEFORE to()) Active")'''

new_content = content.replace(old_shield, new_shield)

with open(core_path, 'w') as f:
    f.write(new_content)

print("✅ Vulkan shield v7 applied")
