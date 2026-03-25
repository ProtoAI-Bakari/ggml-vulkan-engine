#!/usr/bin/env python3
"""
FIX: Increase Vulkan KV Cache from 1GB to 8GB
More memory = longer context = better performance
"""

core_path = "/home/z/GITDEV/vllm_0.17.1/vllm/v1/engine/core.py"

with open(core_path, 'r') as f:
    content = f.read()

# Find and replace the 1GB memory limit with 8GB
old_memory_limit = '1 * 1024 * 1024 * 1024'
new_memory_limit = '8 * 1024 * 1024 * 1024'

if old_memory_limit in content:
    content = content.replace(old_memory_limit, new_memory_limit)
    print(f"✅ core.py updated - KV cache increased from 1GB to 8GB")
else:
    # Try alternative pattern
    old_pattern = 'Limited available memory to 1GB'
    new_pattern = 'Limited available memory to 8GB'
    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        print(f"✅ core.py updated - Memory limit increased to 8GB (alternative pattern)")
    else:
        print("⚠️ Could not find memory limit pattern, checking for Vulkan overrides...")
        # Search for Vulkan memory override
        if "VULKAN OVERRIDE" in content or "vulkan" in content.lower():
            # Find the line with the memory estimate
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if '1GB' in line or '1 * 1024' in line or '1073741824' in line:
                    print(f"Found at line {i+1}: {line.strip()}")
                    # Try to replace
                    if '1GB' in line:
                        lines[i] = line.replace('1GB', '8GB')
                        content = '\n'.join(lines)
                        print(f"✅ Updated line {i+1} to 8GB")
                        break
        else:
            print("⚠️ No Vulkan memory limit found - may be using default")

with open(core_path, 'w') as f:
    f.write(content)