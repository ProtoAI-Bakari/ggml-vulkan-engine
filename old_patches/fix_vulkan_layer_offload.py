#!/usr/bin/env python3
"""
FIX: Full GPU Residency - Remove ALL CPU offloading
Every layer goes to Vulkan - no more ping-pong latency!
"""

utils_path = "/home/z/GITDEV/vllm_0.17.1/vllm/model_executor/model_loader/utils.py"

with open(utils_path, 'r') as f:
    content = f.read()

# Remove the aggressive CPU offloading logic completely
old_offload_logic = '''    # VULKAN ASAHI FIX: Layer offloading with native FP16 support (50% memory savings!)
    
    if target_device.type == 'vulkan':
        print("🚀 VULKAN GPU ENGAGEMENT: FP16 Native Support Enabled - Optimized Layer Offload...")
        for name, m in model.named_modules():
            # Keep these on CPU - they don't need GPU acceleration
            if any(x in name for x in ["embed_tokens", "lm_head", "word_embeddings", 
                                        "norm", "layernorm", "layer_norm",
                                        "fc1", "fc2", "mlp"]):
                print(f"📍 Pinned to CPU: {name}")
                m.to('cpu')
            elif hasattr(m, 'weight'):
                # Only attention layers go to Vulkan
                if any(x in name for x in ["self_attn", "q_proj", "k_proj", "v_proj", "o_proj"]):
                    print(f"🚀 Vulkan: {name}")
                    m.to('vulkan')
                else:
                    print(f"📍 Pinned to CPU (non-attention): {name}")
                    m.to('cpu')'''

new_offload_logic = '''    # ✅ FULL GPU RESIDENCY - Everything on Vulkan!
    # No CPU offloading - eliminates ping-pong latency
    if target_device.type == 'vulkan':
        print("🚀 VULKAN FULL GPU RESIDENCY: All layers on Vulkan - Zero CPU offloading!")
        # Let PyTorch handle device placement normally
        # The target_device is already 'vulkan', so model.to(target_device) will work
        # We just need to ensure no modules are explicitly moved to CPU'''

content = content.replace(old_offload_logic, new_offload_logic)

# Also remove the embed_tokens CPU fix at the end
old_embed_fix = '''    # VULKAN: Force embed_tokens back to CPU after all processing
    if hasattr(model, 'get_input_embeddings'):
        embed_tokens = model.get_input_embeddings()
        if target_device.type == 'vulkan':
            # VULKAN ASAHI FIX: Keep on CPU, don't try to move to Vulkan
            # Vulkan device memory is too limited on Asahi
            embed_tokens.to('cpu')'''

new_embed_fix = '''    # ✅ FULL GPU RESIDENCY: embed_tokens stays on Vulkan
    # No CPU forcing - everything on GPU'''

content = content.replace(old_embed_fix, new_embed_fix)

with open(utils_path, 'w') as f:
    f.write(content)

print("✅ utils.py updated - FULL GPU RESIDENCY enabled (no CPU offloading)")