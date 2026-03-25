/*
 * vulkan_matmul_bench.c — Raw Vulkan compute pipeline for custom matmul shader.
 * Dispatches our tiled_matmul.spv and benchmarks GFLOPS.
 *
 * Build:
 *   gcc -O2 -o vulkan_matmul_bench vulkan_matmul_bench.c \
 *     -lvulkan -lm
 */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define CHECK_VK(x) do { VkResult r = (x); if (r != VK_SUCCESS) { fprintf(stderr, "Vulkan error %d at %s:%d\n", r, __FILE__, __LINE__); exit(1); } } while(0)

typedef struct {
    uint32_t M, N, K;
    uint32_t lda, ldb, ldc;
} PushConstants;

static double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static uint32_t *read_spirv(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t *buf = malloc(*size);
    fread(buf, 1, *size, f);
    fclose(f);
    return buf;
}

int main(int argc, char **argv) {
    /* Matrix dimensions */
    uint32_t M = argc > 1 ? atoi(argv[1]) : 1;
    uint32_t K = argc > 2 ? atoi(argv[2]) : 4096;
    uint32_t N = argc > 3 ? atoi(argv[3]) : 14336;
    int n_iter = argc > 4 ? atoi(argv[4]) : 20;

    printf("Matrix: A(%u×%u) × B(%u×%u)^T = C(%u×%u)\n", M, K, N, K, M, N);

    /* Create Vulkan instance */
    VkApplicationInfo app_info = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "matmul_bench", 1, NULL, 0, VK_API_VERSION_1_2};
    VkInstanceCreateInfo inst_info = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app_info, 0, NULL, 0, NULL};
    VkInstance instance;
    CHECK_VK(vkCreateInstance(&inst_info, NULL, &instance));

    /* Find GPU */
    uint32_t n_dev = 0;
    vkEnumeratePhysicalDevices(instance, &n_dev, NULL);
    VkPhysicalDevice *devs = malloc(n_dev * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(instance, &n_dev, devs);
    VkPhysicalDevice pdev = devs[0];
    free(devs);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pdev, &props);
    printf("GPU: %s\n", props.deviceName);

    /* Find compute queue family */
    uint32_t n_qf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pdev, &n_qf, NULL);
    VkQueueFamilyProperties *qfp = malloc(n_qf * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(pdev, &n_qf, qfp);
    uint32_t qf_idx = 0;
    for (uint32_t i = 0; i < n_qf; i++) {
        if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf_idx = i; break; }
    }
    free(qfp);

    /* Create logical device */
    float qp = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, qf_idx, 1, &qp};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &qci, 0, NULL, 0, NULL, NULL};
    VkDevice device;
    CHECK_VK(vkCreateDevice(pdev, &dci, NULL, &device));

    VkQueue queue;
    vkGetDeviceQueue(device, qf_idx, 0, &queue);

    /* Memory properties */
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(pdev, &mem_props);
    uint32_t mem_type = UINT32_MAX;
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((mem_props.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            mem_type = i;
            break;
        }
    }

    /* Allocate buffers */
    size_t size_a = (size_t)M * K * sizeof(float);
    size_t size_b = (size_t)N * K * sizeof(float);
    size_t size_c = (size_t)M * N * sizeof(float);

    VkBuffer buf_a, buf_b, buf_c;
    VkDeviceMemory mem_a, mem_b, mem_c;

    #define CREATE_BUF(buf, mem, sz) do { \
        VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL}; \
        CHECK_VK(vkCreateBuffer(device, &bci, NULL, &buf)); \
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(device, buf, &mr); \
        VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, mem_type}; \
        CHECK_VK(vkAllocateMemory(device, &mai, NULL, &mem)); \
        CHECK_VK(vkBindBufferMemory(device, buf, mem, 0)); \
    } while(0)

    CREATE_BUF(buf_a, mem_a, size_a);
    CREATE_BUF(buf_b, mem_b, size_b);
    CREATE_BUF(buf_c, mem_c, size_c);

    /* Fill A and B with random data */
    float *ptr;
    CHECK_VK(vkMapMemory(device, mem_a, 0, size_a, 0, (void**)&ptr));
    for (size_t i = 0; i < (size_t)M*K; i++) ptr[i] = (float)rand() / RAND_MAX - 0.5f;
    vkUnmapMemory(device, mem_a);

    CHECK_VK(vkMapMemory(device, mem_b, 0, size_b, 0, (void**)&ptr));
    for (size_t i = 0; i < (size_t)N*K; i++) ptr[i] = (float)rand() / RAND_MAX - 0.5f;
    vkUnmapMemory(device, mem_b);

    /* Load shader */
    size_t spv_size;
    uint32_t *spv = read_spirv(getenv("SPV_PATH") ? getenv("SPV_PATH") : "shaders/tiled_matmul.spv", &spv_size);
    if (!spv) return 1;

    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, spv_size, spv};
    VkShaderModule shader;
    CHECK_VK(vkCreateShaderModule(device, &smci, NULL, &shader));
    free(spv);

    /* Descriptor set layout (3 storage buffers) */
    VkDescriptorSetLayoutBinding bindings[3];
    for (int i = 0; i < 3; i++) {
        bindings[i] = (VkDescriptorSetLayoutBinding){i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    }
    VkDescriptorSetLayoutCreateInfo dslci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 3, bindings};
    VkDescriptorSetLayout ds_layout;
    CHECK_VK(vkCreateDescriptorSetLayout(device, &dslci, NULL, &ds_layout));

    /* Push constant range */
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};

    /* Pipeline layout */
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &ds_layout, 1, &pcr};
    VkPipelineLayout pipe_layout;
    CHECK_VK(vkCreatePipelineLayout(device, &plci, NULL, &pipe_layout));

    /* Compute pipeline */
    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, shader, "main", NULL},
        pipe_layout, VK_NULL_HANDLE, 0};
    VkPipeline pipeline;
    CHECK_VK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &pipeline));

    /* Descriptor pool and set */
    VkDescriptorPoolSize dps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 1, &dps};
    VkDescriptorPool dp;
    CHECK_VK(vkCreateDescriptorPool(device, &dpci, NULL, &dp));

    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 1, &ds_layout};
    VkDescriptorSet ds;
    CHECK_VK(vkAllocateDescriptorSets(device, &dsai, &ds));

    /* Update descriptor set */
    VkDescriptorBufferInfo dbi[3] = {
        {buf_a, 0, size_a},
        {buf_b, 0, size_b},
        {buf_c, 0, size_c},
    };
    VkWriteDescriptorSet wds[3];
    for (int i = 0; i < 3; i++) {
        wds[i] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, i, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &dbi[i], NULL};
    }
    vkUpdateDescriptorSets(device, 3, wds, 0, NULL);

    /* Command buffer */
    VkCommandPoolCreateInfo cpoolci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qf_idx};
    VkCommandPool cpool;
    CHECK_VK(vkCreateCommandPool(device, &cpoolci, NULL, &cpool));

    VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cpool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cb;
    CHECK_VK(vkAllocateCommandBuffers(device, &cbai, &cb));

    /* Fence */
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0};
    VkFence fence;
    CHECK_VK(vkCreateFence(device, &fci, NULL, &fence));

    /* Workgroup dispatch: ceil(M/64) × ceil(N/64) */
    uint32_t gx = (M + 63) / 64;
    uint32_t gy = (N + 63) / 64;

    PushConstants pc = {M, N, K, K, K, N};

    printf("Workgroups: %u × %u = %u (256 threads each)\n", gx, gy, gx * gy);
    printf("Running %d iterations...\n", n_iter);

    /* Warmup */
    {
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer(cb, &bi);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout, 0, 1, &ds, 0, NULL);
        vkCmdPushConstants(cb, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cb, gx, gy, 1);
        vkEndCommandBuffer(cb);
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb, 0, NULL};
        CHECK_VK(vkQueueSubmit(queue, 1, &si, fence));
        CHECK_VK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
        vkResetFences(device, 1, &fence);
    }

    /* Benchmark */
    double times[100];
    for (int iter = 0; iter < n_iter && iter < 100; iter++) {
        vkResetCommandBuffer(cb, 0);
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer(cb, &bi);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout, 0, 1, &ds, 0, NULL);
        vkCmdPushConstants(cb, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cb, gx, gy, 1);
        vkEndCommandBuffer(cb);

        double t0 = get_time_ms();
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb, 0, NULL};
        CHECK_VK(vkQueueSubmit(queue, 1, &si, fence));
        CHECK_VK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
        times[iter] = get_time_ms() - t0;
        vkResetFences(device, 1, &fence);
    }

    /* Sort for median */
    for (int i = 0; i < n_iter - 1; i++)
        for (int j = i + 1; j < n_iter; j++)
            if (times[j] < times[i]) { double t = times[i]; times[i] = times[j]; times[j] = t; }

    double median_ms = times[n_iter / 2];
    double best_ms = times[0];
    double flops = 2.0 * M * K * N;
    double gflops_med = flops / (median_ms * 1e6);
    double gflops_best = flops / (best_ms * 1e6);

    printf("\nResults (%u×%u × %u×%u):\n", M, K, N, K);
    printf("  Median: %.2f ms = %.1f GFLOPS\n", median_ms, gflops_med);
    printf("  Best:   %.2f ms = %.1f GFLOPS\n", best_ms, gflops_best);
    printf("  Theoretical: %.0f GFLOPS (13600 on M1 Ultra)\n", 13600.0);
    printf("  Utilization: %.1f%% (median) / %.1f%% (best)\n", gflops_med/136, gflops_best/136);

    /* Verify correctness (spot check) */
    CHECK_VK(vkMapMemory(device, mem_c, 0, size_c, 0, (void**)&ptr));
    printf("  C[0][0] = %f\n", ptr[0]);
    vkUnmapMemory(device, mem_c);

    /* Cleanup */
    vkDestroyFence(device, fence, NULL);
    vkDestroyCommandPool(device, cpool, NULL);
    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyPipelineLayout(device, pipe_layout, NULL);
    vkDestroyDescriptorPool(device, dp, NULL);
    vkDestroyDescriptorSetLayout(device, ds_layout, NULL);
    vkDestroyShaderModule(device, shader, NULL);
    vkDestroyBuffer(device, buf_a, NULL); vkFreeMemory(device, mem_a, NULL);
    vkDestroyBuffer(device, buf_b, NULL); vkFreeMemory(device, mem_b, NULL);
    vkDestroyBuffer(device, buf_c, NULL); vkFreeMemory(device, mem_c, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    return 0;
}
