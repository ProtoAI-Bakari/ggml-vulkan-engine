# Read the file
with open('/home/z/GITDEV/vllm_0.17.1/vllm/model_executor/custom_op.py', 'r') as f:
    content = f.read()

# Find and replace the dispatch_forward method
old_code = '''        elif current_platform.is_out_of_tree():
            return self.forward_oot
        else:
            return self.forward_cuda'''

new_code = '''        elif current_platform.is_out_of_tree():
            return self.forward_oot
        elif current_platform.__class__.__name__ == 'VulkanPlatform':
            # Vulkan doesn't support custom ops, use native PyTorch implementation
            return self.forward_native
        else:
            return self.forward_cuda'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('/home/z/GITDEV/vllm_0.17.1/vllm/model_executor/custom_op.py', 'w') as f:
        f.write(content)
    print("Patched custom_op.py to use forward_native for Vulkan")
else:
    print("Could not find the code to patch")
