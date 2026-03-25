#!/usr/bin/env python3
"""Vulkan shield v8 - AGGRESSIVE float32 conversion for ALL tensors"""
import re

core_path = "/home/z/GITDEV/vllm_0.17.1/vllm/v1/engine/core.py"

with open(core_path, 'r') as f:
    content = f.read()

old_shield = '''# VULKAN ASAHI FIX: dtype shield for Vulkan device transfers
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

new_shield = '''# VULKAN ASAHI FIX: AGGRESSIVE dtype shield - Vulkan ONLY supports float32
import os
if os.environ.get('VLLM_PLATFORM') == 'vulkan':
    # Save original to method BEFORE any patching
    _orig_vulkan_to = torch.Tensor.to.__func__ if hasattr(torch.Tensor.to, '__func__') else torch.Tensor.to
    
    def _vulkan_shield_to(self, *args, **kwargs):
        device = kwargs.get('device') or (args[0] if args else None)
        dtype_arg = kwargs.get('dtype')
        
        # If moving TO Vulkan, FORCE float32 for ALL tensors
        if device and 'vulkan' in str(device):
            # AGGRESSIVE: Convert ANY non-float32 dtype to float32
            if self.dtype != torch.float32:
                # Log conversion for debugging
                if self.dtype not in (torch.int32, torch.int64, torch.bool, torch.float16, torch.half, torch.bfloat16):
                    pass  # Silent for common types
                self = _orig_vulkan_to(self, torch.float32)
            
            # Remove dtype from kwargs - we've already converted
            kwargs.pop('dtype', None)
            if len(args) > 1 and isinstance(args[1], torch.dtype):
                args = (args[0],)  # Remove dtype from args
            
            return _orig_vulkan_to(self, *args, **kwargs)
        
        # For non-vulkan devices, use normal to()
        return _orig_vulkan_to(self, *args, **kwargs)
    
    torch.Tensor.to = _vulkan_shield_to
    print("⚠️ VULKAN BRIDGE v8: AGGRESSIVE float32-ONLY Shield Active")'''

new_content = content.replace(old_shield, new_shield)

with open(core_path, 'w') as f:
    f.write(new_content)

print("✅ Vulkan shield v8 applied - AGGRESSIVE float32 conversion")
