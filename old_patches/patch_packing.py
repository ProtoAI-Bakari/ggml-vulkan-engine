#!/usr/bin/env python3
"""
Vulkan FP16 Injection Script
Patches Packing.cpp to add Half dtype support alongside Float
"""

import re

packing_path = "/tmp/pytorch/aten/src/ATen/native/vulkan/impl/Packing.cpp"

print(f"🔧 Reading {packing_path}...")

with open(packing_path, 'r') as f:
    content = f.read()

# Track changes
changes_made = []

# PATCH 1: get_nchw_to_image_shader - Add Half support alongside Float
# Find the pattern: if (v_dst.dtype() == api::kFloat)
old_pattern_1 = r'(  if \(v_dst\.dtype\(\) == api::kFloat\) \{)'
new_pattern_1 = r'  if (v_dst.dtype() == api::kFloat || v_dst.dtype() == api::kHalf) {'

if re.search(old_pattern_1, content):
    content = re.sub(old_pattern_1, new_pattern_1, content)
    changes_made.append("✅ Patch 1: Added Half to get_nchw_to_image_shader Float check")
else:
    print("⚠️  Patch 1 NOT applied - pattern not found")

# PATCH 2: get_image_to_nchw_shader - Add Half support alongside Float
# Find the pattern: if (v_src.dtype() == api::kFloat)
old_pattern_2 = r'(  if \(v_src\.dtype\(\) == api::kFloat\) \{)'
new_pattern_2 = r'  if (v_src.dtype() == api::kFloat || v_src.dtype() == api::kHalf) {'

if re.search(old_pattern_2, content):
    content = re.sub(old_pattern_2, new_pattern_2, content)
    changes_made.append("✅ Patch 2: Added Half to get_image_to_nchw_shader Float check")
else:
    print("⚠️  Patch 2 NOT applied - pattern not found")

# Write the patched file
print(f"💾 Writing patched file...")
with open(packing_path, 'w') as f:
    f.write(content)

print(f"\n{'='*60}")
print(f"🎯 PATCH SUMMARY")
print(f"{'='*60}")
for change in changes_made:
    print(change)

print(f"\n{'='*60}")
print(f"🔍 VERIFICATION - Searching for 'Half' in Packing.cpp")
print(f"{'='*60}")

# Now verify the changes
import subprocess
result = subprocess.run(
    ['grep', '-n', '-C', '3', 'Half', packing_path],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print(result.stdout)
else:
    print("⚠️  No 'Half' references found - patch may have failed")

print(f"\n{'='*60}")
print(f"📊 ORIGINAL ERROR LOCATION")
print(f"{'='*60}")
print("Error was at line 65: VK_THROW(\"Unsupported dtype!\")")
print("This should now only trigger for truly unsupported dtypes")
print(f"{'='*60}")

print(f"\n✅ Python patching complete. No compilation needed yet.")
print(f"📝 Next step: grep to verify C++ syntax, then STOP")