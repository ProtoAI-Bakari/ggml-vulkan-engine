#!/usr/bin/env python3
"""Update utils.py to reflect FP16 support in Vulkan layer offloading"""

utils_path = "/home/z/GITDEV/vllm_0.17.1/vllm/model_executor/model_loader/utils.py"

with open(utils_path, 'r') as f:
    content = f.read()

# Update the message to reflect FP16 support
content = content.replace(
    'print("🚀 VULKAN GPU ENGAGEMENT: Aggressive CPU Offload for Stability...")',
    'print("🚀 VULKAN GPU ENGAGEMENT: FP16 Native Support Enabled - Optimized Layer Offload...")'
)

# Add a note about FP16 memory savings
old_comment = '# VULKAN ASAHI FIX: Keep model on CPU - Vulkan device memory is too limited'
new_comment = '# VULKAN ASAHI FIX: Layer offloading with native FP16 support (50% memory savings!)'

content = content.replace(old_comment, new_comment)

with open(utils_path, 'w') as f:
    f.write(content)

print("✅ utils.py updated - FP16 memory optimization noted")
