#!/usr/bin/env python3
"""
Test removing --enforce-eager to enable CUDAGraph for Vulkan
This could unlock 10x+ performance improvement!
"""

vrun_path = "/home/z/AGENT/vrun.sh"

with open(vrun_path, 'r') as f:
    content = f.read()

# Remove --enforce-eager flag
old_content = '    --enforce-eager --dtype float16 \\'
new_content = '    --dtype float16 \\'

if old_content in content:
    content = content.replace(old_content, new_content)
    
    # Update the echo message
    content = content.replace(
        'echo "dtype: float16 (NATIVE VULKAN SUPPORT)"',
        'echo "dtype: float16 (NATIVE VULKAN) - CUDAGraph ENABLED!"'
    )
    
    with open(vrun_path, 'w') as f:
        f.write(content)
    print("✅ vrun.sh updated - --enforce-eager REMOVED, CUDAGraph ENABLED")
else:
    print("⚠️ Could not find --enforce-eager in vrun.sh")
    print("Current content:")
    print(content[content.find('vllm serve'):content.find('vllm serve')+500])