#!/usr/bin/env python3
"""
Apply Vulkan Weight Interceptor to vLLM default_loader.py
Forces ALL weights to Vulkan device for FULL GPU RESIDENCY
"""

loader_path = '/home/z/GITDEV/vllm_0.17.1/vllm/model_executor/model_loader/default_loader.py'

# Read the file
with open(loader_path, 'r') as f:
    lines = f.readlines()

# Find the DefaultModelLoader class and add the interceptor method after __init__
insert_idx = None
in_init = False
for i, line in enumerate(lines):
    if 'class DefaultModelLoader' in line:
        print(f"Found DefaultModelLoader class at line {i+1}")
    if 'def __init__(self, load_config' in line:
        in_init = True
        print(f"Found __init__ at line {i+1}")
    elif in_init and line.strip() and not line.startswith('        ') and not line.startswith('    ') and not line.startswith('\t'):
        # End of __init__ method - insert before this line
        insert_idx = i
        print(f"Insertion point found at line {i+1}")
        break
    elif in_init and 'def _' in line and line.startswith('        '):
        # Found a private method inside __init__ - continue
        pass

if insert_idx is None:
    # Fallback: find the first public method after __init__
    for i, line in enumerate(lines):
        if 'def _prepare_weights' in line:
            insert_idx = i
            print(f"Fallback insertion point at line {i+1}")
            break

if insert_idx is None:
    print("ERROR: Could not find insertion point")
    exit(1)

# Create the interceptor method
interceptor_code = '''
    def _vulkan_weight_interceptor(self, weights_iterator, model_config):
        """
        Vulkan Weight Interceptor for FULL GPU RESIDENCY on Asahi Linux
        Forces ALL weights to Vulkan device to prevent 12.6 TPS gibberish
        """
        logger.info("🔥 VULKAN INTERCEPTOR: Intercepting weight loading...")
        
        intercepted_weights = []
        for name, param in weights_iterator:
            # Force weights to Vulkan device
            if hasattr(param, 'to'):
                try:
                    param = param.to("vulkan")
                    logger.info(f"  ✓ Moved {name} to Vulkan")
                except Exception as e:
                    logger.warning(f"  ⚠ Failed to move {name} to Vulkan: {e}")
            intercepted_weights.append((name, param))
        
        logger.info(f"🔥 VULKAN INTERCEPTOR: Intercepted {len(intercepted_weights)} weight tensors")
        return iter(intercepted_weights)

'''

# Insert the interceptor method
lines.insert(insert_idx, interceptor_code)

# Now find where weights are loaded and add the interceptor call
# Look for patterns like "for name, param in weights_iterator:"
for i, line in enumerate(lines):
    if 'for name, param in weights_iterator:' in line:
        # Add the interceptor call before the for loop
        indent = len(line) - len(line.lstrip())
        interceptor_call = ' ' * indent + 'weights_iterator = self._vulkan_weight_interceptor(weights_iterator, model_config)\n'
        lines.insert(i, interceptor_call)
        print(f"Added interceptor call at line {i+1}")
        break

# Write the modified file
with open(loader_path, 'w') as f:
    f.writelines(lines)

print(f"✅ Successfully applied Vulkan weight interceptor to {loader_path}")
print(f"   Inserted at line {insert_idx+1}")