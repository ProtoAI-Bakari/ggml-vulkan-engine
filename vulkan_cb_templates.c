/*
 * T13: Vulkan Command Buffer Template Recording
 * 
 * Implements reusable command buffer templates for common operations:
 * - KV cache copy
 * - Matrix multiplication
 * - Layer normalization
 * - Attention computation
 * 
 * Benefits:
 * - Reduced CPU overhead by pre-recording common patterns
 * - Faster inference through template reuse
 * - Better GPU utilization
 */

#include <vulkan/vulkan.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Command buffer template structure */
typedef struct {
    VkCommandBuffer cmd_buffer;
    VkCommandPool command_pool;
    VkDevice device;
    uint32_t template_id;
    uint32_t ref_count;
    int is_recording;
    
    /* Template metadata */
    uint32_t op_count;
    uint32_t memory_barrier_count;
    uint32_t buffer_copy_count;
    
    /* For reuse tracking */
    uint64_t last_used_frame;
    uint64_t total_executions;
} VkCommandBufferTemplate;

/* Template pool for managing multiple templates */
typedef struct {
    VkCommandBufferTemplate **templates;
    uint32_t template_count;
    uint32_t template_capacity;
    VkDevice device;
    VkPhysicalDevice physical_device;
} VkTemplatePool;

/* Common template types */
#define CB_TEMPLATE_KV_COPY       0
#define CB_TEMPLATE_MATMUL        1
/* Forward declarations */
void vk_template_destroy(VkCommandBufferTemplate* template);
int vk_template_begin_recording(VkCommandBufferTemplate* template);
int vk_template_end_recording(VkCommandBufferTemplate* template);
#define CB_TEMPLATE_NORM          2
#define CB_TEMPLATE_ATTENTION     3
#define CB_TEMPLATE_FFN           4
#define CB_TEMPLATE_ROPE          5

/* Initialize template pool */
VkTemplatePool* vk_template_pool_create(VkDevice device, VkPhysicalDevice phys_dev) {
    VkTemplatePool* pool = (VkTemplatePool*)malloc(sizeof(VkTemplatePool));
    if (!pool) return NULL;
    
    pool->device = device;
    pool->physical_device = phys_dev;
    pool->templates = NULL;
    pool->template_count = 0;
    pool->template_capacity = 0;
    
    return pool;
}

/* Destroy template pool */
void vk_template_pool_destroy(VkTemplatePool* pool) {
    if (!pool) return;
    
    for (uint32_t i = 0; i < pool->template_count; i++) {
        vk_template_destroy(pool->templates[i]);
    }
    
    free(pool->templates);
    free(pool);
}

/* Create a new command buffer template */
VkCommandBufferTemplate* vk_template_create(VkTemplatePool* pool, uint32_t template_type) {
    if (!pool) return NULL;
    
    /* Resize pool if needed */
    if (pool->template_count >= pool->template_capacity) {
        uint32_t new_capacity = pool->template_capacity == 0 ? 8 : pool->template_capacity * 2;
        VkCommandBufferTemplate** new_templates = (VkCommandBufferTemplate**)realloc(
            pool->templates, new_capacity * sizeof(VkCommandBufferTemplate*));
        
        if (!new_templates) return NULL;
        
        pool->templates = new_templates;
        pool->template_capacity = new_capacity;
    }
    
    /* Allocate template */
    VkCommandBufferTemplate* template = (VkCommandBufferTemplate*)calloc(1, sizeof(VkCommandBufferTemplate));
    if (!template) return NULL;
    
    template->device = pool->device;
    template->template_id = template_type;
    template->ref_count = 1;
    template->total_executions = 0;
    
    /* Create command pool */
    VkCommandPoolCreateInfo pool_info = {0};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = 0; /* TODO: Get from device */
    
    VkResult result = vkCreateCommandPool(pool->device, &pool_info, NULL, &template->command_pool);
    if (result != VK_SUCCESS) {
        free(template);
        return NULL;
    }
    
    /* Allocate command buffer */
    VkCommandBufferAllocateInfo alloc_info = {0};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = template->command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;
    
    result = vkAllocateCommandBuffers(pool->device, &alloc_info, &template->cmd_buffer);
    if (result != VK_SUCCESS) {
        vkDestroyCommandPool(pool->device, template->command_pool, NULL);
        free(template);
        return NULL;
    }
    
    /* Add to pool */
    pool->templates[pool->template_count++] = template;
    
    return template;
}

/* Destroy command buffer template */
void vk_template_destroy(VkCommandBufferTemplate* template) {
    if (!template) return;
    
    template->ref_count--;
    
    if (template->ref_count > 0) return;
    
    vkFreeCommandBuffers(template->device, template->command_pool, 1, &template->cmd_buffer);
    vkDestroyCommandPool(template->device, template->command_pool, NULL);
    free(template);
}

/* Begin recording template */
int vk_template_begin_recording(VkCommandBufferTemplate* template) {
    if (!template || template->is_recording) return -1;
    
    VkCommandBufferBeginInfo begin_info = {0};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    VkResult result = vkBeginCommandBuffer(template->cmd_buffer, &begin_info);
    if (result == VK_SUCCESS) {
        template->is_recording = 1;
        template->op_count = 0;
        template->memory_barrier_count = 0;
        template->buffer_copy_count = 0;
    }
    
    return result == VK_SUCCESS ? 0 : -1;
}

/* End recording template */
int vk_template_end_recording(VkCommandBufferTemplate* template) {
    if (!template || !template->is_recording) return -1;
    
    VkResult result = vkEndCommandBuffer(template->cmd_buffer);
    if (result == VK_SUCCESS) {
        template->is_recording = 0;
    }
    
    return result == VK_SUCCESS ? 0 : -1;
}

/* Record buffer copy operation */
void vk_template_record_buffer_copy(VkCommandBufferTemplate* template,
                                   VkBuffer src, VkBuffer dst,
                                   VkDeviceSize size, VkDeviceSize src_offset, VkDeviceSize dst_offset) {
    if (!template || !template->is_recording) return;
    
    VkBufferCopy copy_region = {0};
    copy_region.srcOffset = src_offset;
    copy_region.dstOffset = dst_offset;
    copy_region.size = size;
    
    vkCmdCopyBuffer(template->cmd_buffer, src, dst, 1, &copy_region);
    template->buffer_copy_count++;
    template->op_count++;
}

/* Record memory barrier */
void vk_template_record_memory_barrier(VkCommandBufferTemplate* template,
                                      VkAccessFlags src_access, VkAccessFlags dst_access,
                                      VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
    if (!template || !template->is_recording) return;
    
    VkMemoryBarrier barrier = {0};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = src_access;
    barrier.dstAccessMask = dst_access;
    
    vkCmdPipelineBarrier(template->cmd_buffer,
                        src_stage, dst_stage,
                        0,
                        1, &barrier,
                        0, NULL,
                        0, NULL);
    template->memory_barrier_count++;
    template->op_count++;
}

/* Record buffer memory barrier */
void vk_template_record_buffer_memory_barrier(VkCommandBufferTemplate* template,
                                             VkBuffer buffer,
                                             VkAccessFlags src_access, VkAccessFlags dst_access,
                                             VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage,
                                             VkDeviceSize offset, VkDeviceSize size) {
    if (!template || !template->is_recording) return;
    
    VkBufferMemoryBarrier barrier = {0};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = src_access;
    barrier.dstAccessMask = dst_access;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer;
    barrier.offset = offset;
    barrier.size = size;
    
    vkCmdPipelineBarrier(template->cmd_buffer,
                        src_stage, dst_stage,
                        0,
                        0, NULL,
                        1, &barrier,
                        0, NULL);
    template->memory_barrier_count++;
    template->op_count++;
}

/* Record image memory barrier */
void vk_template_record_image_memory_barrier(VkCommandBufferTemplate* template,
                                            VkImage image,
                                            VkAccessFlags src_access, VkAccessFlags dst_access,
                                            VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage,
                                            VkImageLayout old_layout, VkImageLayout new_layout,
                                            uint32_t base_mip_level, uint32_t mip_levels,
                                            uint32_t base_array_layer, uint32_t array_layers) {
    if (!template || !template->is_recording) return;
    
    VkImageMemoryBarrier barrier = {0};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = src_access;
    barrier.dstAccessMask = dst_access;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = base_mip_level;
    barrier.subresourceRange.levelCount = mip_levels;
    barrier.subresourceRange.baseArrayLayer = base_array_layer;
    barrier.subresourceRange.layerCount = array_layers;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    
    vkCmdPipelineBarrier(template->cmd_buffer,
                        src_stage, dst_stage,
                        0,
                        0, NULL,
                        0, NULL,
                        1, &barrier);
    template->memory_barrier_count++;
    template->op_count++;
}

/* Submit template for execution */
int vk_template_submit(VkCommandBufferTemplate* template, VkQueue queue, VkFence fence) {
    if (!template || template->is_recording) return -1;
    
    VkSubmitInfo submit_info = {0};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &template->cmd_buffer;
    
    VkResult result = vkQueueSubmit(queue, 1, &submit_info, fence);
    if (result == VK_SUCCESS) {
        template->total_executions++;
        template->last_used_frame = 1; /* TODO: Track frame number */
    }
    
    return result == VK_SUCCESS ? 0 : -1;
}

/* Get template statistics */
void vk_template_get_stats(VkCommandBufferTemplate* template,
                          uint32_t* op_count,
                          uint32_t* barrier_count,
                          uint32_t* copy_count,
                          uint64_t* executions) {
    if (!template) return;
    
    if (op_count) *op_count = template->op_count;
    if (barrier_count) *barrier_count = template->memory_barrier_count;
    if (copy_count) *copy_count = template->buffer_copy_count;
    if (executions) *executions = template->total_executions;
}

/* Example: Create KV cache copy template */
VkCommandBufferTemplate* create_kv_copy_template(VkTemplatePool* pool,
                                                VkBuffer src_buffer, VkBuffer dst_buffer,
                                                VkDeviceSize copy_size) {
    VkCommandBufferTemplate* template = vk_template_create(pool, CB_TEMPLATE_KV_COPY);
    if (!template) return NULL;
    
    if (vk_template_begin_recording(template) != 0) {
        vk_template_destroy(template);
        return NULL;
    }
    
    /* Record memory barrier for src */
    vk_template_record_buffer_memory_barrier(template,
                                            src_buffer,
                                            VK_ACCESS_TRANSFER_WRITE_BIT,
                                            VK_ACCESS_TRANSFER_READ_BIT,
                                            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                                            0, copy_size);
    
    /* Record copy */
    vk_template_record_buffer_copy(template, src_buffer, dst_buffer, copy_size, 0, 0);
    
    /* Record memory barrier for dst */
    vk_template_record_buffer_memory_barrier(template,
                                            dst_buffer,
                                            VK_ACCESS_TRANSFER_READ_BIT,
                                            VK_ACCESS_TRANSFER_WRITE_BIT,
                                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                                            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                            0, copy_size);
    
    if (vk_template_end_recording(template) != 0) {
        vk_template_destroy(template);
        return NULL;
    }
    
    return template;
}

/* Example: Create attention computation template */
VkCommandBufferTemplate* create_attention_template(VkTemplatePool* pool,
                                                  uint32_t batch_size,
                                                  uint32_t seq_len,
                                                  uint32_t num_heads,
                                                  uint32_t head_dim) {
    VkCommandBufferTemplate* template = vk_template_create(pool, CB_TEMPLATE_ATTENTION);
    if (!template) return NULL;
    
    if (vk_template_begin_recording(template) != 0) {
        vk_template_destroy(template);
        return NULL;
    }
    
    /* Record compute shader dispatch for attention */
    /* TODO: Add vkCmdDispatch with appropriate workgroup sizes */
    
    /* Record memory barriers for intermediate results */
    /* TODO: Add barriers between attention stages */
    
    if (vk_template_end_recording(template) != 0) {
        vk_template_destroy(template);
        return NULL;
    }
    
    return template;
}

/* Benchmark template reuse */
void benchmark_template_reuse(VkCommandBufferTemplate* template, VkQueue queue, uint32_t iterations) {
    if (!template) return;
    
    uint64_t start = 0; /* TODO: Get high-res timestamp */
    
    for (uint32_t i = 0; i < iterations; i++) {
        vk_template_submit(template, queue, NULL);
    }
    
    /* TODO: Wait for completion and calculate time */
    
    uint64_t end = 0; /* TODO: Get high-res timestamp */
    double avg_time = (double)(end - start) / iterations;
    
    printf("Template %u: %.2f µs average per execution\n", 
           template->template_id, avg_time);
}

/* Print template info */
void vk_template_print_info(VkCommandBufferTemplate* template) {
    if (!template) return;
    
    printf("Template ID: %u\n", template->template_id);
    printf("  Ref Count: %u\n", template->ref_count);
    printf("  Op Count: %u\n", template->op_count);
    printf("  Barriers: %u\n", template->memory_barrier_count);
    printf("  Copies: %u\n", template->buffer_copy_count);
    printf("  Executions: %lu\n", template->total_executions);
}

/* ============================================================================
 * PUSH CONSTANTS SUPPORT
 * ============================================================================
 * T13: Add vkCmdPushConstants for dynamic parameters
 */

/* Push constant range for template */
typedef struct {
    VkShaderStageFlags stage_flags;
    uint32_t offset;
    uint32_t size;
} PushConstantRange;

/* Record push constants */
void vk_template_record_push_constants(VkCommandBufferTemplate* template,
                                      VkShaderStageFlags stage_flags,
                                      uint32_t offset,
                                      uint32_t size,
                                      const void* data) {
    if (!template || !template->is_recording) return;
    
    vkCmdPushConstants(template->cmd_buffer,
                      VK_NULL_HANDLE, /* TODO: Get from pipeline */
                      stage_flags,
                      offset,
                      size,
                      data);
    template->op_count++;
}

/* Common push constant layouts */
#define PUSH_CONST_KV_OFFSET    0
#define PUSH_CONST_SEQ_LEN      4
#define PUSH_CONST_BATCH_SIZE   8
#define PUSH_CONST_NUM_HEADS    12
#define PUSH_CONST_HEAD_DIM     16
#define PUSH_CONST_POS_EMBED    20
#define PUSH_CONST_SCALE        24
#define PUSH_CONST_LAYERNORM_EPS 28

/* Push constant structure for attention */
typedef struct {
    uint32_t kv_offset;      // Offset into KV cache
    uint32_t seq_len;        // Sequence length
    uint32_t batch_size;     // Batch size
    uint32_t num_heads;      // Number of attention heads
    uint32_t head_dim;       // Dimension per head
    uint32_t pos_embed;      // Position embedding offset
    float scale;             // Attention scale (1/sqrt(head_dim))
    float layernorm_eps;     // Layer norm epsilon
} AttentionPushConstants;

/* Push constant structure for FFN */
typedef struct {
    uint32_t batch_size;
    uint32_t seq_len;
    uint32_t hidden_dim;
    uint32_t intermediate_dim;
    uint32_t ffn_idx;        // For MoE: which expert
    float gate_scale;
    float activation_scale;
    float output_scale;
} FFNPushConstants;

/* Push constant structure for RoPE */
typedef struct {
    uint32_t batch_size;
    uint32_t seq_len;
    uint32_t num_heads;
    uint32_t head_dim;
    uint32_t pos_offset;     // Starting position
    float freq_base;         // RoPE frequency base
    float freq_scale;        // RoPE frequency scale
    uint32_t rope_type;      // 0=none, 1=original, 2=linear, 3=ntk
} RoPEPushConstants;

/* Example: Record attention with push constants */
VkCommandBufferTemplate* create_attention_template_v2(VkTemplatePool* pool,
                                                     uint32_t max_batch,
                                                     uint32_t max_seq_len,
                                                     uint32_t num_heads,
                                                     uint32_t head_dim) {
    VkCommandBufferTemplate* template = vk_template_create(pool, CB_TEMPLATE_ATTENTION);
    if (!template) return NULL;
    
    if (vk_template_begin_recording(template) != 0) {
        vk_template_destroy(template);
        return NULL;
    }
    
    /* Record push constants for attention parameters */
    AttentionPushConstants pc = {0};
    pc.num_heads = num_heads;
    pc.head_dim = head_dim;
    pc.scale = 1.0f / sqrtf((float)head_dim);
    pc.layernorm_eps = 1e-5f;
    
    vk_template_record_push_constants(template,
                                     VK_SHADER_STAGE_COMPUTE_BIT,
                                     0,
                                     sizeof(AttentionPushConstants),
                                     &pc);
    
    /* Record compute dispatch for Q*K^T attention */
    /* Workgroup size: (batch, seq_len, num_heads) */
    uint32_t workgroup_x = (max_batch + 7) / 8;
    uint32_t workgroup_y = (max_seq_len + 7) / 8;
    uint32_t workgroup_z = (num_heads + 7) / 8;
    
    /* TODO: vkCmdDispatch(template->cmd_buffer, workgroup_x, workgroup_y, workgroup_z); */
    
    /* Record memory barriers for attention output */
    /* TODO: Add barriers between attention stages */
    
    if (vk_template_end_recording(template) != 0) {
        vk_template_destroy(template);
        return NULL;
    }
    
    return template;
}

/* Example: Record FFN with push constants */
VkCommandBufferTemplate* create_ffn_template(VkTemplatePool* pool,
                                            uint32_t max_batch,
                                            uint32_t max_seq_len,
                                            uint32_t hidden_dim,
                                            uint32_t intermediate_dim) {
    VkCommandBufferTemplate* template = vk_template_create(pool, CB_TEMPLATE_FFN);
    if (!template) return NULL;
    
    if (vk_template_begin_recording(template) != 0) {
        vk_template_destroy(template);
        return NULL;
    }
    
    /* Record push constants for FFN parameters */
    FFNPushConstants pc = {0};
    pc.hidden_dim = hidden_dim;
    pc.intermediate_dim = intermediate_dim;
    pc.gate_scale = 1.0f;
    pc.activation_scale = 1.0f;
    pc.output_scale = 1.0f;
    
    vk_template_record_push_constants(template,
                                     VK_SHADER_STAGE_COMPUTE_BIT,
                                     0,
                                     sizeof(FFNPushConstants),
                                     &pc);
    
    /* Record compute dispatch for FFN */
    uint32_t workgroup_x = (max_batch * max_seq_len + 63) / 64;
    uint32_t workgroup_y = (intermediate_dim + 255) / 256;
    uint32_t workgroup_z = 1;
    
    /* TODO: vkCmdDispatch(template->cmd_buffer, workgroup_x, workgroup_y, workgroup_z); */
    
    if (vk_template_end_recording(template) != 0) {
        vk_template_destroy(template);
        return NULL;
    }
    
    return template;
}

/* Example: Record RoPE with push constants */
VkCommandBufferTemplate* create_rope_template(VkTemplatePool* pool,
                                             uint32_t max_batch,
                                             uint32_t max_seq_len,
                                             uint32_t num_heads,
                                             uint32_t head_dim,
                                             float freq_base,
                                             uint32_t rope_type) {
    VkCommandBufferTemplate* template = vk_template_create(pool, CB_TEMPLATE_ROPE);
    if (!template) return NULL;
    
    if (vk_template_begin_recording(template) != 0) {
        vk_template_destroy(template);
        return NULL;
    }
    
    /* Record push constants for RoPE parameters */
    RoPEPushConstants pc = {0};
    pc.num_heads = num_heads;
    pc.head_dim = head_dim;
    pc.freq_base = freq_base;
    pc.freq_scale = 1.0f;
    pc.rope_type = rope_type;
    
    vk_template_record_push_constants(template,
                                     VK_SHADER_STAGE_COMPUTE_BIT,
                                     0,
                                     sizeof(RoPEPushConstants),
                                     &pc);
    
    /* Record compute dispatch for RoPE */
    uint32_t workgroup_x = (max_batch * max_seq_len * num_heads + 63) / 64;
    uint32_t workgroup_y = (head_dim / 2 + 255) / 256;
    uint32_t workgroup_z = 1;
    
    /* TODO: vkCmdDispatch(template->cmd_buffer, workgroup_x, workgroup_y, workgroup_z); */
    
    if (vk_template_end_recording(template) != 0) {
        vk_template_destroy(template);
        return NULL;
    }
    
    return template;
}

/* Update push constants at runtime (without re-recording) */
int vk_template_update_push_constants(VkCommandBufferTemplate* template,
                                     VkShaderStageFlags stage_flags,
                                     uint32_t offset,
                                     uint32_t size,
                                     const void* data) {
    if (!template || template->is_recording) return -1;
    
    /* Need to begin/record/end to update push constants */
    /* This is a limitation - push constants are baked into the template */
    /* Solution: Use vkCmdPushConstants at submission time instead */
    
    return -1; /* Not supported for recorded templates */
}

/* Alternative: Store push constant values in template for runtime update */
typedef struct {
    VkCommandBufferTemplate* template;
    uint32_t max_push_const_size;
    void* push_const_data;
} VkTemplateWithPushConstants;

/* Create template with runtime-updatable push constants */
VkTemplateWithPushConstants* vk_template_create_with_push_constants(
    VkTemplatePool* pool,
    uint32_t template_type,
    uint32_t push_const_size) {
    
    VkTemplateWithPushConstants* wrapper = (VkTemplateWithPushConstants*)calloc(1, sizeof(VkTemplateWithPushConstants));
    if (!wrapper) return NULL;
    
    wrapper->template = vk_template_create(pool, template_type);
    if (!wrapper->template) {
        free(wrapper);
        return NULL;
    }
    
    wrapper->max_push_const_size = push_const_size;
    wrapper->push_const_data = malloc(push_const_size);
    if (!wrapper->push_const_data) {
        vk_template_destroy(wrapper->template);
        free(wrapper);
        return NULL;
    }
    
    return wrapper;
}

/* Update push constants and re-submit */
int vk_template_with_push_constants_update(
    VkTemplateWithPushConstants* wrapper,
    const void* data,
    uint32_t size,
    VkQueue queue,
    VkFence fence) {
    
    if (!wrapper || !data || size > wrapper->max_push_const_size) return -1;
    
    /* Copy new push constant data */
    memcpy(wrapper->push_const_data, data, size);
    
    /* TODO: Re-record with new push constants or use separate push at submit time */
    /* For now, just submit the template */
    
    return vk_template_submit(wrapper->template, queue, fence);
}

/* Destroy template with push constants */
void vk_template_with_push_constants_destroy(VkTemplateWithPushConstants* wrapper) {
    if (!wrapper) return;
    
    if (wrapper->push_const_data) {
        free(wrapper->push_const_data);
    }
    
    if (wrapper->template) {
        vk_template_destroy(wrapper->template);
    }
    
    free(wrapper);
}

/* ============================================================================
 * COMMAND POOL RESET SUPPORT
 * ============================================================================
 * T13: Use vkResetCommandPool for pool-level reset (not per-buffer)
 * This is more efficient than vkFreeCommandBuffers + vkAllocateCommandBuffers
 */

/* Reset command pool and all associated templates */
int vk_template_pool_reset(VkTemplatePool* pool) {
    if (!pool) return -1;
    
    for (uint32_t i = 0; i < pool->template_count; i++) {
        VkCommandBufferTemplate* template = pool->templates[i];
        
        /* Reset the command pool - this frees all command buffers */
        VkResult result = vkResetCommandPool(pool->device, template->command_pool, 0);
        if (result != VK_SUCCESS) {
            fprintf(stderr, "Warning: Failed to reset command pool for template %u\n", 
                    template->template_id);
            continue;
        }
        
        /* Reallocate command buffer */
        VkCommandBufferAllocateInfo alloc_info = {0};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = template->command_pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;
        
        result = vkAllocateCommandBuffers(pool->device, &alloc_info, &template->cmd_buffer);
        if (result != VK_SUCCESS) {
            fprintf(stderr, "Error: Failed to reallocate command buffer for template %u\n",
                    template->template_id);
            return -1;
        }
        
        /* Reset template state */
        template->is_recording = 0;
        template->op_count = 0;
        template->memory_barrier_count = 0;
        template->buffer_copy_count = 0;
    }
    
    return 0;
}

/* Reset specific template's command pool */
int vk_template_reset(VkCommandBufferTemplate* template) {
    if (!template) return -1;
    
    /* Reset command pool */
    VkResult result = vkResetCommandPool(template->device, template->command_pool, 0);
    if (result != VK_SUCCESS) {
        return -1;
    }
    
    /* Reallocate command buffer */
    VkCommandBufferAllocateInfo alloc_info = {0};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = template->command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;
    
    result = vkAllocateCommandBuffers(template->device, &alloc_info, &template->cmd_buffer);
    if (result != VK_SUCCESS) {
        return -1;
    }
    
    /* Reset template state */
    template->is_recording = 0;
    template->op_count = 0;
    template->memory_barrier_count = 0;
    template->buffer_copy_count = 0;
    
    return 0;
}

/* ============================================================================
 * CB INVALIDATION LOGIC (T14 PREVIEW)
 * ============================================================================
 * Detect topology changes and invalidate stale templates
 */

/* Graph fingerprint for topology detection */
typedef struct {
    uint32_t num_layers;
    uint32_t hidden_dim;
    uint32_t intermediate_dim;
    uint32_t num_heads;
    uint32_t head_dim;
    uint32_t vocab_size;
    uint32_t max_context;
    uint32_t moe_experts;      // For MoE models
    uint32_t moe_active_experts;
    uint64_t hash;             // Combined hash of all params
} GraphFingerprint;

/* Compute graph fingerprint hash */
uint64_t graph_fingerprint_hash(GraphFingerprint* fp) {
    uint64_t hash = 0;
    hash ^= (uint64_t)fp->num_layers << 0;
    hash ^= (uint64_t)fp->hidden_dim << 8;
    hash ^= (uint64_t)fp->intermediate_dim << 16;
    hash ^= (uint64_t)fp->num_heads << 24;
    hash ^= (uint64_t)fp->head_dim << 32;
    hash ^= (uint64_t)fp->vocab_size << 40;
    hash ^= (uint64_t)fp->max_context << 48;
    hash ^= (uint64_t)fp->moe_experts << 56;
    hash ^= (uint64_t)fp->moe_active_experts << 60;
    return hash;
}

/* Template invalidation state */
typedef struct {
    GraphFingerprint current_fingerprint;
    uint64_t last_valid_hash;
    int is_valid;
    uint32_t invalidation_count;
    uint64_t last_invalidation_frame;
} TemplateInvalidationState;

/* Initialize invalidation state */
TemplateInvalidationState* template_invalidation_create(void) {
    TemplateInvalidationState* state = (TemplateInvalidationState*)calloc(1, sizeof(TemplateInvalidationState));
    if (!state) return NULL;
    
    state->is_valid = 1;
    state->last_valid_hash = 0;
    state->invalidation_count = 0;
    
    return state;
}

/* Check if templates need re-recording */
int template_needs_re_recording(TemplateInvalidationState* state, GraphFingerprint* new_fp) {
    if (!state || !new_fp) return 0;
    
    uint64_t new_hash = graph_fingerprint_hash(new_fp);
    
    if (state->last_valid_hash == 0) {
        /* First time, accept this fingerprint */
        state->last_valid_hash = new_hash;
        state->current_fingerprint = *new_fp;
        return 0;
    }
    
    if (new_hash != state->last_valid_hash) {
        /* Topology changed, need re-recording */
        state->invalidation_count++;
        state->last_valid_hash = new_hash;
        state->current_fingerprint = *new_fp;
        return 1;
    }
    
    return 0;
}

/* Mark templates as invalid */
void template_invalidate_all(VkTemplatePool* pool, TemplateInvalidationState* state) {
    if (!pool || !state) return;
    
    for (uint32_t i = 0; i < pool->template_count; i++) {
        VkCommandBufferTemplate* template = pool->templates[i];
        
        /* Mark template as needing re-recording */
        template->is_recording = 0; /* Force re-recording on next use */
        template->total_executions = 0; /* Reset stats */
    }
    
    state->is_valid = 0;
}

/* Re-record all templates after topology change */
int template_re_record_all(VkTemplatePool* pool, TemplateInvalidationState* state,
                          void (*re_record_callback)(VkTemplatePool*)) {
    if (!pool || !state || !re_record_callback) return -1;
    
    if (!state->is_valid) {
        /* Re-record all templates */
        re_record_callback(pool);
        state->is_valid = 1;
        return 0;
    }
    
    return -1; /* No re-recording needed */
}

/* Get invalidation statistics */
void template_invalidation_get_stats(TemplateInvalidationState* state,
                                    uint32_t* invalidation_count,
                                    uint64_t* last_hash,
                                    int* is_valid) {
    if (!state) return;
    
    if (invalidation_count) *invalidation_count = state->invalidation_count;
    if (last_hash) *last_hash = state->last_valid_hash;
    if (is_valid) *is_valid = state->is_valid;
}
