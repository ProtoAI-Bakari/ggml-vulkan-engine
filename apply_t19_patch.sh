#!/bin/bash
# T19: Apply descriptor pool pre-allocation patch to llama.cpp ggml-vulkan.cpp

GGML_VULKAN="/home/z/GITDEV/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp"
BACKUP="/home/z/GITDEV/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp.backup"

echo "T19: Patching ggml-vulkan.cpp for descriptor pool pre-allocation"
echo "================================================================"

# Backup original
cp "$GGML_VULKAN" "$BACKUP"
echo "✓ Backup created: $BACKUP"

# Step 1: Add constants after line 117 (before struct forward declaration)
echo "Step 1: Adding pre-allocation constants..."
sed -i '117a\
// T19: Pre-allocated descriptor pool configuration\n\
#define VK_PREALLOC_POOL_SIZE 4096  // Descriptors per pool (tunable)\n\
#define VK_PREALLOC_MAX_POOLS 16    // Maximum pools (64k descriptors total)\n\
' "$GGML_VULKAN"

# Step 2: Add pre-allocation function before ggml_pipeline_allocate_descriptor_sets
echo "Step 2: Adding ggml_vk_preallocate_descriptor_pools() function..."

# Find line number of ggml_pipeline_allocate_descriptor_sets
LINE_NUM=$(grep -n "^static void ggml_pipeline_allocate_descriptor_sets" "$GGML_VULKAN" | cut -d: -f1)

if [ -z "$LINE_NUM" ]; then
    echo "ERROR: Could not find ggml_pipeline_allocate_descriptor_sets function"
    exit 1
fi

echo "  Found function at line $LINE_NUM"

# Insert pre-allocation function before it
sed -i "${LINE_NUM}i\
// T19: Pre-allocate descriptor pools at initialization time\n\
static void ggml_vk_preallocate_descriptor_pools(ggml_backend_vk_context * ctx) {\n\
    vk_device& device = ctx->device;\n\
    \n\
    // Estimate worst-case: 32 layers × 7 descriptors/token × 1024 tokens = ~224k descriptors\n\
    // But we reuse descriptor sets, so 64k should be sufficient for most cases\n\
    const uint32_t total_descriptors = VK_PREALLOC_POOL_SIZE * VK_PREALLOC_MAX_POOLS;\n\
    \n\
    VK_LOG_INFO(\"T19: Pre-allocating \" << total_descriptors << \" descriptor sets (\" \n\
                << VK_PREALLOC_MAX_POOLS << \" pools × \" << VK_PREALLOC_POOL_SIZE << \")\");\n\
    \n\
    for (uint32_t pool_idx = 0; pool_idx < VK_PREALLOC_MAX_POOLS; pool_idx++) {\n\
        vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer, \n\
                                                    MAX_PARAMETER_COUNT * VK_PREALLOC_POOL_SIZE);\n\
        vk::DescriptorPoolCreateInfo descriptor_pool_create_info(\n\
            vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, \n\
            VK_PREALLOC_POOL_SIZE, \n\
            descriptor_pool_size);\n\
        \n\
        vk::DescriptorPool pool = device->device.createDescriptorPool(descriptor_pool_create_info);\n\
        ctx->descriptor_pools.push_back(pool);\n\
        \n\
        // Allocate descriptor sets from this pool\n\
        std::vector<vk::DescriptorSetLayout> layouts(VK_PREALLOC_POOL_SIZE);\n\
        for (uint32_t i = 0; i < VK_PREALLOC_POOL_SIZE; i++) {\n\
            layouts[i] = device->dsl;\n\
        }\n\
        \n\
        vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(\n\
            pool, VK_PREALLOC_POOL_SIZE, layouts.data());\n\
        std::vector<vk::DescriptorSet> sets = device->device.allocateDescriptorSets(descriptor_set_alloc_info);\n\
        ctx->descriptor_sets.insert(ctx->descriptor_sets.end(), sets.begin(), sets.end());\n\
    }\n\
    \n\
    VK_LOG_INFO(\"T19: Pre-allocation complete. \" << ctx->descriptor_sets.size() << \" sets available.\");\n\
}\n\
\n\
" "$GGML_VULKAN"

# Step 3: Modify ggml_pipeline_allocate_descriptor_sets to use VK_PREALLOC_POOL_SIZE
echo "Step 3: Modifying ggml_pipeline_allocate_descriptor_sets()..."

# Replace VK_DEVICE_DESCRIPTOR_POOL_SIZE with VK_PREALLOC_POOL_SIZE
sed -i 's/VK_DEVICE_DESCRIPTOR_POOL_SIZE/VK_PREALLOC_POOL_SIZE/g' "$GGML_VULKAN"

# Step 4: Add pre-allocation call in ggml_backend_vk_init after device setup
echo "Step 4: Adding pre-allocation call in ggml_backend_vk_init()..."

# Find line where ctx->device = device; appears in ggml_backend_vk_init
INIT_LINE=$(grep -n "ctx->device = device;" "$GGML_VULKAN" | head -1 | cut -d: -f1)

if [ -z "$INIT_LINE" ]; then
    echo "ERROR: Could not find ctx->device = device; in ggml_backend_vk_init"
    exit 1
fi

echo "  Found device setup at line $INIT_LINE"

# Insert pre-allocation call after device setup
sed -i "${INIT_LINE}a\
    \n\
    // T19: Pre-allocate descriptor pools before any compute\n\
    ggml_vk_preallocate_descriptor_pools(ctx);\n\
" "$GGML_VULKAN"

echo ""
echo "✓ Patch applied successfully!"
echo ""
echo "Summary of changes:"
echo "  - Added VK_PREALLOC_POOL_SIZE = 4096"
echo "  - Added VK_PREALLOC_MAX_POOLS = 16 (64k total descriptors)"
echo "  - Added ggml_vk_preallocate_descriptor_pools() function"
echo "  - Modified ggml_pipeline_allocate_descriptor_sets() to use pre-allocated pools"
echo "  - Added pre-allocation call in ggml_backend_vk_init()"
echo ""
echo "Next steps:"
echo "  1. Rebuild llama.cpp:"
echo "     cd ~/GITDEV/llama.cpp && cmake --build build-lib -j$(nproc)"
echo ""
echo "  2. Rebuild our C extension:"
echo "     cd ~/AGENT && gcc -shared -O2 -fPIC -o libggml_llama_gguf.so ggml_llama_gguf.c \\"
echo "       -I ~/GITDEV/llama.cpp/ggml/include \\"
echo "       -L ~/GITDEV/llama.cpp/build-lib/bin -lggml -lggml-base -lggml-vulkan -lggml-cpu -lm \\"
echo "       -Wl,-rpath,/home/z/GITDEV/llama.cpp/build-lib/bin"
echo ""
echo "  3. Benchmark to verify improvement:"
echo "     python3 test_fa_scalar_vs_standard.py"
echo ""
echo "  4. Expected result: 0 runtime descriptor pool allocations during decode"
echo "     (Previously: ~1-2 allocations per 100 tokens)"
