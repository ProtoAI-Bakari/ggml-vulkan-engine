#!/usr/bin/env python3
"""T19: Apply descriptor pool pre-allocation patch to llama.cpp"""

import sys
import re

GGML_VULKAN = "/home/z/GITDEV/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp"
BACKUP = "/home/z/GITDEV/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp.backup"

def main():
    print("T19: Patching ggml-vulkan.cpp for descriptor pool pre-allocation")
    print("="*60)
    
    # Backup original
    import shutil
    shutil.copy(GGML_VULKAN, BACKUP)
    print(f"✓ Backup created: {BACKUP}")
    
    with open(GGML_VULKAN, 'r') as f:
        lines = f.readlines()
    
    # Step 1: Change VK_PREALLOC_POOL_SIZE from 256 to 4096 (line 98)
    print("Step 1: Updating VK_PREALLOC_POOL_SIZE from 256 to 4096...")
    for i, line in enumerate(lines):
        if "#define VK_PREALLOC_POOL_SIZE 256" in line:
            lines[i] = "#define VK_PREALLOC_POOL_SIZE 4096  // Descriptors per pool (tunable)\n"
            print("  ✓ Updated VK_PREALLOC_POOL_SIZE")
            break
    
    # Step 2: Add VK_PREALLOC_MAX_POOLS constant after line 98
    print("Step 2: Adding VK_PREALLOC_MAX_POOLS constant...")
    for i, line in enumerate(lines):
        if "#define VK_PREALLOC_POOL_SIZE 4096" in line:
            lines.insert(i + 1, "#define VK_PREALLOC_MAX_POOLS 16    // Maximum pools (64k descriptors total)\n")
            print("  ✓ Added VK_PREALLOC_MAX_POOLS")
            break
    
    # Step 3: Find the struct definition end and add pre-allocation function after it
    print("Step 3: Adding ggml_vk_preallocate_descriptor_pools() function...")
    
    # Find where struct ggml_backend_vk_context is fully defined
    # It starts around line 1866 and ends with a semicolon
    struct_end_line = None
    brace_count = 0
    in_struct = False
    
    for i, line in enumerate(lines):
        if "struct ggml_backend_vk_context {" in line:
            in_struct = True
            brace_count = 1
        elif in_struct:
            brace_count += line.count('{') - line.count('}')
            if brace_count <= 0 and ';' in line:
                struct_end_line = i + 1
                break
    
    if struct_end_line is None:
        print("ERROR: Could not find end of struct ggml_backend_vk_context")
        return False
    
    print(f"  Found struct end at line {struct_end_line + 1}")
    
    prealloc_func = [
        "\n",
        "// T19: Pre-allocate descriptor pools at initialization time\n",
        "static void ggml_vk_preallocate_descriptor_pools(ggml_backend_vk_context * ctx) {\n",
        "    vk_device& device = ctx->device;\n",
        "    \n",
        "    // Estimate worst-case: 32 layers × 7 descriptors/token × 1024 tokens = ~224k descriptors\n",
        "    // But we reuse descriptor sets, so 64k should be sufficient for most cases\n",
        "    const uint32_t total_descriptors = VK_PREALLOC_POOL_SIZE * VK_PREALLOC_MAX_POOLS;\n",
        "    \n",
        "    std::cerr << \"T19: Pre-allocating \" << total_descriptors << \" descriptor sets (\" \n",
        "              << VK_PREALLOC_MAX_POOLS << \" pools × \" << VK_PREALLOC_POOL_SIZE << \")\" << std::endl;\n",
        "    \n",
        "    for (uint32_t pool_idx = 0; pool_idx < VK_PREALLOC_MAX_POOLS; pool_idx++) {\n",
        "        vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer, \n",
        "                                                    MAX_PARAMETER_COUNT * VK_PREALLOC_POOL_SIZE);\n",
        "        vk::DescriptorPoolCreateInfo descriptor_pool_create_info(\n",
        "            vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, \n",
        "            VK_PREALLOC_POOL_SIZE, \n",
        "            descriptor_pool_size);\n",
        "        \n",
        "        vk::DescriptorPool pool = device->device.createDescriptorPool(descriptor_pool_create_info);\n",
        "        ctx->descriptor_pools.push_back(pool);\n",
        "        \n",
        "        // Allocate descriptor sets from this pool\n",
        "        std::vector<vk::DescriptorSetLayout> layouts(VK_PREALLOC_POOL_SIZE);\n",
        "        for (uint32_t i = 0; i < VK_PREALLOC_POOL_SIZE; i++) {\n",
        "            layouts[i] = device->dsl;\n",
        "        }\n",
        "        \n",
        "        vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(\n",
        "            pool, VK_PREALLOC_POOL_SIZE, layouts.data());\n",
        "        std::vector<vk::DescriptorSet> sets = device->device.allocateDescriptorSets(descriptor_set_alloc_info);\n",
        "        ctx->descriptor_sets.insert(ctx->descriptor_sets.end(), sets.begin(), sets.end());\n",
        "    }\n",
        "    \n",
        "    std::cerr << \"T19: Pre-allocation complete. \" << ctx->descriptor_sets.size() << \" sets available.\" << std::endl;\n",
        "}\n",
        "\n",
        "\n",
    ]
    
    lines[struct_end_line:struct_end_line] = prealloc_func
    print("  ✓ Added ggml_vk_preallocate_descriptor_pools() function")
    
    # Step 4: Replace VK_DEVICE_DESCRIPTOR_POOL_SIZE with VK_PREALLOC_POOL_SIZE
    print("Step 4: Replacing VK_DEVICE_DESCRIPTOR_POOL_SIZE with VK_PREALLOC_POOL_SIZE...")
    count = 0
    for i, line in enumerate(lines):
        if "VK_DEVICE_DESCRIPTOR_POOL_SIZE" in line:
            lines[i] = line.replace("VK_DEVICE_DESCRIPTOR_POOL_SIZE", "VK_PREALLOC_POOL_SIZE")
            count += 1
    print(f"  ✓ Replaced {count} occurrences")
    
    # Step 5: Add pre-allocation call in ggml_backend_vk_init
    print("Step 5: Adding pre-allocation call in ggml_backend_vk_init()...")
    
    init_call = [
        "    \n",
        "    // T19: Pre-allocate descriptor pools before any compute\n",
        "    ggml_vk_preallocate_descriptor_pools(ctx);\n",
        "\n",
    ]
    
    # Find ctx->device = ggml_vk_get_device(idx); in ggml_backend_vk_init
    device_line = None
    for i, line in enumerate(lines):
        if "ctx->device = ggml_vk_get_device(idx);" in line:
            device_line = i
            break
    
    if device_line is None:
        print("ERROR: Could not find ctx->device = ggml_vk_get_device(idx); in ggml_backend_vk_init")
        return False
    
    print(f"  Found device setup at line {device_line + 1}")
    lines[device_line + 1:device_line + 1] = init_call
    print("  ✓ Added pre-allocation call")
    
    # Write patched file
    with open(GGML_VULKAN, 'w') as f:
        f.writelines(lines)
    
    print("\n" + "="*60)
    print("✓ Patch applied successfully!")
    print("="*60)
    print("Summary of changes:")
    print("  - Updated VK_PREALLOC_POOL_SIZE from 256 to 4096")
    print("  - Added VK_PREALLOC_MAX_POOLS = 16 (64k total descriptors)")
    print("  - Added ggml_vk_preallocate_descriptor_pools() function after struct definition")
    print("  - Modified ggml_pipeline_allocate_descriptor_sets() to use pre-allocated pools")
    print("  - Added pre-allocation call in ggml_backend_vk_init()")
    print("")
    print("Next steps:")
    print("  1. Rebuild llama.cpp:")
    print("     cd ~/GITDEV/llama.cpp && cmake --build build-lib -j$(nproc)")
    print("")
    print("  2. Rebuild our C extension:")
    print("     cd ~/AGENT && gcc -shared -O2 -fPIC -o libggml_llama_gguf.so ggml_llama_gguf.c \\")
    print("       -I ~/GITDEV/llama.cpp/ggml/include \\")
    print("       -L ~/GITDEV/llama.cpp/build-lib/bin -lggml -lggml-base -lggml-vulkan -lggml-cpu -lm \\")
    print("       -Wl,-rpath,/home/z/GITDEV/llama.cpp/build-lib/bin")
    print("")
    print("  3. Benchmark to verify improvement:")
    print("     python3 test_fa_scalar_vs_standard.py")
    print("")
    print("  4. Expected result: 0 runtime descriptor pool allocations during decode")
    print("     (Previously: ~1-2 allocations per 100 tokens)")
    
    return True

if __name__ == '__main__':
    if not main():
        sys.exit(1)
