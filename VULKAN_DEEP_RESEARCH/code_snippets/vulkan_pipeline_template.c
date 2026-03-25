// Minimal Vulkan Compute Pipeline Boilerplate
// This is the bare minimum to dispatch a compute shader in Vulkan
// Useful as a reference for building custom kernels outside of ggml

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================
// STEP 1: Create compute pipeline
// ============================================================
VkPipeline create_compute_pipeline(
    VkDevice device,
    VkPipelineLayout layout,
    const uint32_t* spirv_code,
    size_t spirv_size,
    const VkSpecializationInfo* spec_info)  // For tile sizes, etc.
{
    VkShaderModuleCreateInfo shader_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv_size,
        .pCode = spirv_code,
    };
    VkShaderModule shader;
    vkCreateShaderModule(device, &shader_info, NULL, &shader);

    VkComputePipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
            .pSpecializationInfo = spec_info,
        },
        .layout = layout,
    };

    VkPipeline pipeline;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline);
    vkDestroyShaderModule(device, shader, NULL);
    return pipeline;
}

// ============================================================
// STEP 2: Create pipeline layout with push constants
// ============================================================
VkPipelineLayout create_pipeline_layout(
    VkDevice device,
    VkDescriptorSetLayout dsl,
    uint32_t push_constant_size)
{
    VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = push_constant_size,
    };

    VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &dsl,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };

    VkPipelineLayout layout;
    vkCreatePipelineLayout(device, &layout_info, NULL, &layout);
    return layout;
}

// ============================================================
// STEP 3: Create descriptor set layout (buffer bindings)
// ============================================================
VkDescriptorSetLayout create_descriptor_set_layout(
    VkDevice device,
    uint32_t binding_count)  // Number of storage buffer bindings
{
    VkDescriptorSetLayoutBinding* bindings =
        calloc(binding_count, sizeof(VkDescriptorSetLayoutBinding));

    for (uint32_t i = 0; i < binding_count; i++) {
        bindings[i] = (VkDescriptorSetLayoutBinding){
            .binding = i,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        };
    }

    VkDescriptorSetLayoutCreateInfo dsl_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = binding_count,
        .pBindings = bindings,
    };

    VkDescriptorSetLayout dsl;
    vkCreateDescriptorSetLayout(device, &dsl_info, NULL, &dsl);
    free(bindings);
    return dsl;
}

// ============================================================
// STEP 4: Allocate and update descriptor set
// ============================================================
VkDescriptorSet allocate_and_update_descriptor_set(
    VkDevice device,
    VkDescriptorPool pool,
    VkDescriptorSetLayout dsl,
    VkBuffer* buffers,        // Array of buffer handles
    VkDeviceSize* offsets,    // Array of offsets
    VkDeviceSize* sizes,      // Array of sizes
    uint32_t count)
{
    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &dsl,
    };

    VkDescriptorSet ds;
    vkAllocateDescriptorSets(device, &alloc_info, &ds);

    VkDescriptorBufferInfo* buffer_infos =
        calloc(count, sizeof(VkDescriptorBufferInfo));
    VkWriteDescriptorSet* writes =
        calloc(count, sizeof(VkWriteDescriptorSet));

    for (uint32_t i = 0; i < count; i++) {
        buffer_infos[i] = (VkDescriptorBufferInfo){
            .buffer = buffers[i],
            .offset = offsets[i],
            .range = sizes[i],
        };
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = ds,
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buffer_infos[i],
        };
    }

    vkUpdateDescriptorSets(device, count, writes, 0, NULL);
    free(buffer_infos);
    free(writes);
    return ds;
}

// ============================================================
// STEP 5: Record and submit compute dispatch
// ============================================================
void dispatch_compute(
    VkCommandBuffer cmd,
    VkPipeline pipeline,
    VkPipelineLayout layout,
    VkDescriptorSet ds,
    const void* push_constants,
    uint32_t push_constant_size,
    uint32_t wg_x, uint32_t wg_y, uint32_t wg_z)
{
    // Bind pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    // Bind descriptor set
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        layout, 0, 1, &ds, 0, NULL);

    // Push constants (dimensions, strides, etc.)
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, push_constant_size, push_constants);

    // Dispatch workgroups
    vkCmdDispatch(cmd, wg_x, wg_y, wg_z);
}

// ============================================================
// STEP 6: Pipeline barrier between dispatches
// ============================================================
void insert_compute_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1, &barrier,
        0, NULL,
        0, NULL);
}

// ============================================================
// SPECIALIZATION CONSTANTS SETUP (for tile sizes)
// ============================================================
// Use this pattern to set BLOCK_SIZE, BM, BN, BK, etc.
typedef struct {
    uint32_t block_size;  // id=0
    uint32_t bm;          // id=1
    uint32_t bn;          // id=2
    uint32_t bk;          // id=3
    uint32_t wm;          // id=4
    uint32_t wn;          // id=5
    uint32_t tm;          // id=6
    uint32_t tn;          // id=7
    uint32_t warp;        // id=8
} SpecConstants;

VkSpecializationInfo create_spec_info(const SpecConstants* constants) {
    static VkSpecializationMapEntry entries[] = {
        {0, offsetof(SpecConstants, block_size), sizeof(uint32_t)},
        {1, offsetof(SpecConstants, bm),         sizeof(uint32_t)},
        {2, offsetof(SpecConstants, bn),         sizeof(uint32_t)},
        {3, offsetof(SpecConstants, bk),         sizeof(uint32_t)},
        {4, offsetof(SpecConstants, wm),         sizeof(uint32_t)},
        {5, offsetof(SpecConstants, wn),         sizeof(uint32_t)},
        {6, offsetof(SpecConstants, tm),         sizeof(uint32_t)},
        {7, offsetof(SpecConstants, tn),         sizeof(uint32_t)},
        {8, offsetof(SpecConstants, warp),       sizeof(uint32_t)},
    };

    static VkSpecializationInfo info = {
        .mapEntryCount = 9,
        .pMapEntries = entries,
        .dataSize = sizeof(SpecConstants),
        .pData = NULL,
    };
    info.pData = constants;
    return info;
}

// ============================================================
// BUFFER ALLOCATION (Apple Silicon unified memory)
// ============================================================
// On Apple Silicon, device-local memory IS host-visible
// So we can allocate once and get both GPU access and CPU mapping
VkBuffer create_unified_buffer(
    VkDevice device,
    VkPhysicalDevice phys_device,
    VkDeviceSize size,
    VkDeviceMemory* memory,
    void** mapped_ptr)
{
    VkBufferCreateInfo buf_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    VkBuffer buffer;
    vkCreateBuffer(device, &buf_info, NULL, &buffer);

    VkMemoryRequirements mem_req;
    vkGetBufferMemoryRequirements(device, buffer, &mem_req);

    // Find memory type that is both device-local and host-visible
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys_device, &mem_props);

    uint32_t mem_type = UINT32_MAX;
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((mem_req.memoryTypeBits & (1 << i)) &&
            (mem_props.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
            mem_type = i;
            break;
        }
    }

    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_req.size,
        .memoryTypeIndex = mem_type,
    };

    vkAllocateMemory(device, &alloc_info, NULL, memory);
    vkBindBufferMemory(device, buffer, *memory, 0);

    // Map for CPU access (stays mapped forever on unified memory)
    vkMapMemory(device, *memory, 0, VK_WHOLE_SIZE, 0, mapped_ptr);

    return buffer;
}
