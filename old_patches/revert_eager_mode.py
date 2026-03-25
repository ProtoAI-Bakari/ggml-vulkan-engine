#!/usr/bin/env python3
"""
Revert to --enforce-eager mode due to TorchDynamo compatibility issues with Vulkan
"""

vrun_path = "/home/z/AGENT/vrun.sh"

with open(vrun_path, 'r') as f:
    content = f.read()

# Add back --enforce-eager flag
old_content = '    --dtype float16 \\'
new_content = '    --enforce-eager --dtype float16 \\'

if old_content in content:
    content = content.replace(old_content, new_content)
    
    # Update the echo message
    content = content.replace(
        'echo "dtype: float16 (NATIVE VULKAN) - CUDAGraph ENABLED!"',
        'echo "dtype: float16 (NATIVE VULKAN) - Eager Mode (CUDAGraph incompatible with Vulkan/Asahi)"'
    )
    
    with open(vrun_path, 'w') as f:
        f.write(content)
    print("✅ vrun.sh reverted - --enforce-eager restored (CUDAGraph not compatible)")
else:
    print("⚠️ Could not find pattern in vrun.sh")