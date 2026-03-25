#!/usr/bin/env python3
"""
REMOVE THE ENTIRE VULKAN DTYPE SHIELD - Ripping out the global monkey-patch
This will allow TorchDynamo to compile CUDAGraph successfully!
"""

core_path = "/home/z/GITDEV/vllm_0.17.1/vllm/v1/engine/core.py"

with open(core_path, 'r') as f:
    content = f.read()

# Find and remove the entire dtype shield block
shield_start = "# VULKAN ASAHI FIX: AGGRESSIVE dtype shield - Vulkan ONLY supports float32"
shield_end = 'print("✅ VULKAN BRIDGE v9: NATIVE FP16 SUPPORT ENABLED")'

start_idx = content.find(shield_start)
end_idx = content.find(shield_end)

if start_idx != -1 and end_idx != -1:
    # Include the print statement and newline
    end_idx = end_idx + len(shield_end) + 1
    
    # Remove the entire block
    before = content[:start_idx]
    after = content[end_idx:]
    
    # Clean up any extra newlines
    new_content = before.rstrip() + '\n\n' + after.lstrip()
    
    with open(core_path, 'w') as f:
        f.write(new_content)
    
    print("✅ core.py CLEANED - Vulkan dtype shield REMOVED completely")
    print("   TorchDynamo can now trace native torch.Tensor.to()")
else:
    print("⚠️ Could not find shield block to remove")
    print(f"   start_idx: {start_idx}, end_idx: {end_idx}")