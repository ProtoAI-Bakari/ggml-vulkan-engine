# Read the file
with open('/home/z/GITDEV/vllm_0.17.1/vllm/model_executor/model_loader/utils.py', 'r') as f:
    content = f.read()

# Find the end of process_weights_after_loading function and add embed_tokens fix
old_code = '''    for _, module in model.named_modules():
        if isinstance(module, (Attention, MLAAttention)) and hasattr(
            module, "process_weights_after_loading"
        ):'''

new_code = '''    # VULKAN: Force embed_tokens back to CPU after all processing
    if target_device.type == 'vulkan':
        for name, m in model.named_modules():
            if any(x in name for x in ["embed_tokens", "word_embeddings", "lm_head"]):
                if hasattr(m, 'parameters') and list(m.parameters()):
                    first_param = next(m.parameters())
                    if first_param.device.type != 'cpu':
                        print(f"🔧 FORCING {name} back to CPU", flush=True)
                        m.to('cpu')
    
    for _, module in model.named_modules():
        if isinstance(module, (Attention, MLAAttention)) and hasattr(
            module, "process_weights_after_loading"
        ):'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('/home/z/GITDEV/vllm_0.17.1/vllm/model_executor/model_loader/utils.py', 'w') as f:
        f.write(content)
    print("Patched utils.py to force embed_tokens to CPU")
else:
    print("Could not find the code to patch")
