#!/usr/bin/env python3
"""Update Vulkan dtype shield to ALLOW native FP16 (PyTorch C++ rebuild complete)"""
import re

core_path = "/home/z/GITDEV/vllm_0.17.1/vllm/v1/engine/core.py"

with open(core_path, 'r') as f:
    content = f.read()

# Find and replace the old shield with FP16-ALLOWING version
old_pattern = r'''elif self\.dtype == torch\.float16 or self\.dtype == torch\.half:
                # Vulkan FP16 shader missing - force float32
                self = self\.to\(torch\.float32\)'''

new_code = '''elif self.dtype == torch.float16 or self.dtype == torch.half:
                # ✅ NATIVE FP16 VULKAN NOW SUPPORTED (PyTorch C++ rebuild complete)
                # Keep float16 as-is - Vulkan backend handles it natively
                pass'''

new_content = re.sub(old_pattern, new_code, content)

if new_content != content:
    with open(core_path, 'w') as f:
        f.write(new_content)
    print("✅ Vulkan dtype shield updated - FP16 NOW ALLOWED on Vulkan")
else:
    print("⚠️ Pattern not found, checking file...")
    # Try a broader search
    if "Vulkan FP16 shader missing" in content:
        content = content.replace(
            "# Vulkan FP16 shader missing - force float32\n                self = self.to(torch.float32)",
            "# ✅ NATIVE FP16 VULKAN NOW SUPPORTED\n                # Keep float16 as-is"
        )
        with open(core_path, 'w') as f:
            f.write(content)
        print("✅ Vulkan dtype shield updated (alternative method) - FP16 NOW ALLOWED")
    else:
        print("⚠️ Could not find pattern to replace")