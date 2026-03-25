import os
path = os.path.expanduser("~/GITDEV/vllm_0.17.1/vllm/v1/engine/core.py")
with open(path, 'r') as f:
    content = f.read()

# Remove any existing bridge code
content = content.replace("""
import torch
if os.environ.get('VLLM_PLATFORM') == 'vulkan':
    # 1. Scalar Shield: Force all integer transfers to int32 (FIXED RECURSION)
    _orig_to = torch.Tensor.to
    def _vulkan_to(self, *args, **kwargs):
        # Check if already int32 to avoid recursion
        if self.dtype == torch.int64:
            # Use the original function directly to avoid recursion
            return torch.Tensor.to(torch.Tensor.to(self, torch.int32), *args, **kwargs)
        return torch.Tensor.to(self, *args, **kwargs)
    torch.Tensor.to = _vulkan_to
    
    # 2. Kernel Bridge: Fallback for missing Vulkan kernels (as_strided)
    _orig_as_strided = torch.Tensor.as_strided
    def _vulkan_as_strided(self, size, stride, storage_offset=0):
        try:
            return _orig_as_strided(self, size, stride, storage_offset)
        except NotImplementedError:
            # Fallback: move to CPU, perform operation, return
            cpu_tensor = self.cpu()
            return _orig_as_strided(cpu_tensor, size, stride, storage_offset)
    torch.Tensor.as_strided = _vulkan_as_strided
    
    print("⚠️ VULKAN BRIDGE: Global Scalar and Kernel Shield Active (v3 - Recursion Fixed)")
""", "")

# Inject the updated Global Vulkan Bridge at the top of the file
bridge_code = """
import torch
if os.environ.get('VLLM_PLATFORM') == 'vulkan':
    # 1. Scalar Shield: Force all integer transfers to int32 (FIXED RECURSION)
    # Save the original unpatched method
    _orig_to = torch.Tensor.to.__func__
    def _vulkan_to(self, *args, **kwargs):
        # Check if already int32 to avoid recursion
        if self.dtype == torch.int64:
            # Convert to int32 first, then apply original to
            int32_tensor = _orig_to(self, torch.int32)
            return _orig_to(int32_tensor, *args, **kwargs)
        return _orig_to(self, *args, **kwargs)
    torch.Tensor.to = _vulkan_to
    
    # 2. Kernel Bridge: Fallback for missing Vulkan kernels (as_strided)
    _orig_as_strided = torch.Tensor.as_strided.__func__
    def _vulkan_as_strided(self, size, stride, storage_offset=0):
        try:
            return _orig_as_strided(self, size, stride, storage_offset)
        except NotImplementedError:
            # Fallback: move to CPU, perform operation, return
            cpu_tensor = self.cpu()
            return _orig_as_strided(cpu_tensor, size, stride, storage_offset)
    torch.Tensor.as_strided = _vulkan_as_strided
    
    print("⚠️ VULKAN BRIDGE: Global Scalar and Kernel Shield Active (v4 - Recursion Fixed)")
"""

# Find a good insertion point (after imports, around line 20)
lines = content.split('\n')
insert_idx = 20
for i, line in enumerate(lines[:30]):
    if line.strip().startswith('import ') or line.strip().startswith('from '):
        insert_idx = i + 1

lines.insert(insert_idx, bridge_code)
content = '\n'.join(lines)

with open(path, 'w') as f:
    f.write(content)
print("✅ core.py: Updated Vulkan Bridge (v4) Installed - Recursion Fixed.")