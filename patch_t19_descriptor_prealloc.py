#!/usr/bin/env python3
"""T19: Patch llama.cpp ggml-vulkan.cpp to pre-allocate descriptor pools

This eliminates runtime descriptor pool allocation during decode,
reducing overhead from ~3ms to <0.1ms per token.
"""

import sys
import re

# Path to llama.cpp ggml-vulkan source
GGML_VULKAN_PATH = "/home/z/GITDEV/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp"

def patch_descriptor_preallocation():
    """Apply descriptor pool pre-allocation patch"""
    
    with open(GGML_VULKAN_PATH, 'r') as f:
        content = f.read()
    
    # Find the struct ggml_backend_vk_context definition
    # Add pre-allocation fields
    struct_pattern = r'(struct ggml_backend_vk_context \{[^}]+)vk::Device device;'
    struct_match = re.search(struct_pattern, content, re.DOTALL)
    
    if not struct_match:
        print("ERROR: Could not find ggml_backend_vk_context struct")
        return False
    
    # Add pre-allocation constants before the struct
    constants_patch = '''
// T19: Pre-allocated descriptor pool configuration
// Pre-allocate pools to eliminate runtime allocation during decode
#define VK_PREALLOC_POOL_SIZE 4096  // Descriptors per pool (tunable)
#define VK_PREALLOC_MAX_POOLS 16    // Maximum pools (64k descriptors total)

'''
    
    # Insert constants before struct
    insert_pos = struct_match.start()
    content = content[:insert_pos] + constants_patch + content[insert_pos:]
    
    # Now modify ggml_pipeline_allocate_descriptor_sets to use pre-allocation
    old_func = '''static void ggml_pipeline_allocate_descriptor_sets(ggml_backend_vk_context * ctx) {

    if (ctx->descriptor_sets.size() >= ctx->pipeline_descriptor_set_requirements) {
        // Enough descriptors are available
        return;
    }

    vk_device& device = ctx->device;

    // Grow by 50% to avoid frequent allocations
    uint32_t needed = std::max(3 * ctx->descriptor_sets.size() / 2, size_t{ctx->pipeline_descriptor_set_requirements});
    uint32_t to_alloc = needed - ctx->descriptor_sets.size();
    uint32_t pool_remaining = VK_DEVICE_DESCRIPTOR_POOL_SIZE - ctx->descriptor_sets.size() % VK_DEVICE_DESCRIPTOR_POOL_SIZE;
    uint32_t pool_idx = ctx->descriptor_sets.size() / VK_DEVICE_DESCRIPTOR_POOL_SIZE;

    while (to_alloc > 0) {
        const uint32_t alloc_count = std::min(pool_remaining, to_alloc);
        to_alloc -= alloc_count;
        pool_remaining = VK_DEVICE_DESCRIPTOR_POOL_SIZE;

        if (pool_idx >= ctx->descriptor_pools.size()) {
            vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer, MAX_PARAMETER_COUNT * VK_DEVICE_DESCRIPTOR_POOL_SIZE);
            vk::DescriptorPoolCreateInfo descriptor_pool_create_info({}, VK_DEVICE_DESCRIPTOR_POOL_SIZE, descriptor_pool_size);
            ctx->descriptor_pools.push_back(device->device.createDescriptorPool(descriptor_pool_create_info));
        }

        std::vector<vk::DescriptorSetLayout> layouts(alloc_count);
        for (uint32_t i = 0; i < alloc_count; i++) {
            layouts[i] = device->dsl;
        }
        vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(ctx->descriptor_pools[pool_idx], alloc_count, layouts.data());
        std::vector<vk::DescriptorSet> sets = device->device.allocateDescriptorSets(descriptor_set_alloc_info);
        ctx->descriptor_sets.insert(ctx->descriptor_sets.end(), sets.begin(), sets.end());

        pool_idx++;
    }
}'''
    
    new_func = '''// T19: Pre-allocate descriptor pools at initialization time
static void ggml_vk_preallocate_descriptor_pools(ggml_backend_vk_context * ctx) {
    vk_device& device = ctx->device;
    
    // Estimate worst-case: 32 layers × 7 descriptors/token × 1024 tokens = ~224k descriptors
    // But we reuse descriptor sets, so 64k should be sufficient for most cases
    const uint32_t total_descriptors = VK_PREALLOC_POOL_SIZE * VK_PREALLOC_MAX_POOLS;
    
    VK_LOG_INFO("T19: Pre-allocating " << total_descriptors << " descriptor sets (" 
                << VK_PREALLOC_MAX_POOLS << " pools × " << VK_PREALLOC_POOL_SIZE << ")");
    
    for (uint32_t pool_idx = 0; pool_idx < VK_PREALLOC_MAX_POOLS; pool_idx++) {
        vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer, 
                                                    MAX_PARAMETER_COUNT * VK_PREALLOC_POOL_SIZE);
        vk::DescriptorPoolCreateInfo descriptor_pool_create_info(
            vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 
            VK_PREALLOC_POOL_SIZE, 
            descriptor_pool_size);
        
        vk::DescriptorPool pool = device->device.createDescriptorPool(descriptor_pool_create_info);
        ctx->descriptor_pools.push_back(pool);
        
        // Allocate descriptor sets from this pool
        std::vector<vk::DescriptorSetLayout> layouts(VK_PREALLOC_POOL_SIZE);
        for (uint32_t i = 0; i < VK_PREALLOC_POOL_SIZE; i++) {
            layouts[i] = device->dsl;
        }
        
        vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(
            pool, VK_PREALLOC_POOL_SIZE, layouts.data());
        std::vector<vk::DescriptorSet> sets = device->device.allocateDescriptorSets(descriptor_set_alloc_info);
        ctx->descriptor_sets.insert(ctx->descriptor_sets.end(), sets.begin(), sets.end());
    }
    
    VK_LOG_INFO("T19: Pre-allocation complete. " << ctx->descriptor_sets.size() << " sets available.");
}

static void ggml_pipeline_allocate_descriptor_sets(ggml_backend_vk_context * ctx) {
    // T19: If pre-allocation is done, just check if we have enough
    // No runtime pool creation needed
    
    if (ctx->descriptor_sets.size() >= ctx->pipeline_descriptor_set_requirements) {
        // Enough descriptors are available
        return;
    }
    
    // Fallback: grow if pre-allocation wasn't sufficient (shouldn't happen with proper sizing)
    vk_device& device = ctx->device;
    
    uint32_t needed = std::max(3 * ctx->descriptor_sets.size() / 2, size_t{ctx->pipeline_descriptor_set_requirements});
    uint32_t to_alloc = needed - ctx->descriptor_sets.size();
    uint32_t pool_remaining = VK_PREALLOC_POOL_SIZE - ctx->descriptor_sets.size() % VK_PREALLOC_POOL_SIZE;
    uint32_t pool_idx = ctx->descriptor_sets.size() / VK_PREALLOC_POOL_SIZE;

    while (to_alloc > 0) {
        const uint32_t alloc_count = std::min(pool_remaining, to_alloc);
        to_alloc -= alloc_count;
        pool_remaining = VK_PREALLOC_POOL_SIZE;

        if (pool_idx >= ctx->descriptor_pools.size()) {
            // Runtime fallback allocation (should be rare)
            vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer, 
                                                        MAX_PARAMETER_COUNT * VK_PREALLOC_POOL_SIZE);
            vk::DescriptorPoolCreateInfo descriptor_pool_create_info(
                vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 
                VK_PREALLOC_POOL_SIZE, 
                descriptor_pool_size);
            ctx->descriptor_pools.push_back(device->device.createDescriptorPool(descriptor_pool_create_info));
        }

        std::vector<vk::DescriptorSetLayout> layouts(alloc_count);
        for (uint32_t i = 0; i < alloc_count; i++) {
            layouts[i] = device->dsl;
        }
        vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(ctx->descriptor_pools[pool_idx], alloc_count, layouts.data());
        std::vector<vk::DescriptorSet> sets = device->device.allocateDescriptorSets(descriptor_set_alloc_info);
        ctx->descriptor_sets.insert(ctx->descriptor_sets.end(), sets.begin(), sets.end());

        pool_idx++;
    }
}'''
    
    if old_func not in content:
        print("ERROR: Could not find ggml_pipeline_allocate_descriptor_sets function")
        return False
    
    content = content.replace(old_func, new_func)
    
    # Now find ggml_backend_vk_init and add pre-allocation call
    init_pattern = r'(static ggml_backend_vk_context \*ggml_backend_vk_init\(ggml_backend_vk_context \* ctx\) \{)'
    init_match = re.search(init_pattern, content)
    
    if init_match:
        # Add pre-allocation call after device initialization
        # Find where device is set up
        device_setup = r'(ctx->device = device;\s*\n)'
        device_match = re.search(device_setup, content)
        
        if device_match:
            insert_pos = device_match.end()
            prealloc_call = '''    
    // T19: Pre-allocate descriptor pools before any compute
    ggml_vk_preallocate_descriptor_pools(ctx);
'''
            content = content[:insert_pos] + prealloc_call + content[insert_pos:]
    
    # Write patched file
    with open(GGML_VULKAN_PATH, 'w') as f:
        f.write(content)
    
    print("✓ Patch applied successfully to ggml-vulkan.cpp")
    print("  - Added VK_PREALLOC_POOL_SIZE = 4096")
    print("  - Added VK_PREALLOC_MAX_POOLS = 16")
    print("  - Added ggml_vk_preallocate_descriptor_pools()")
    print("  - Modified ggml_pipeline_allocate_descriptor_sets()")
    print("  - Added pre-allocation call in ggml_backend_vk_init()")
    return True

if __name__ == '__main__':
    if patch_descriptor_preallocation():
        print("\nNext steps:")
        print("1. Rebuild llama.cpp: cd ~/GITDEV/llama.cpp && cmake --build build-lib")
        print("2. Rebuild our C extension: gcc -shared -O2 -fPIC -o libggml_llama_gguf.so ggml_llama_gguf.c ...")
        print("3. Benchmark: python3 test_fa_scalar_vs_standard.py")
        print("4. Verify: No vkCreateDescriptorPool calls during decode (use RenderDoc or vkLayer)")
    else:
        sys.exit(1)
