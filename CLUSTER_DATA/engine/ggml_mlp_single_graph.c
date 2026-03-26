/*
 * ggml_mlp_single_graph.c — ALL 32 MLP layers in ONE ggml compute graph.
 * ONE Vulkan dispatch for the entire MLP chain. No CPU round-trips.
 *
 * Build:
 *   gcc -shared -O2 -fPIC -o libggml_mlp_single.so ggml_mlp_single_graph.c \
 *     -I ~/GITDEV/llama.cpp/ggml/include \
 *     -L ~/GITDEV/llama.cpp/build-lib/bin \
 *     -lggml -lggml-base -lggml-vulkan -lggml-cpu -lm \
 *     -Wl,-rpath,/home/z/GITDEV/llama.cpp/build-lib/bin
 */
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-vulkan.h>
#include <ggml-cpu.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define MAX_LAYERS 64
/* 32 layers × (gate_mm + up_mm + silu + mul + down_mm) = 160 nodes + input + intermediates */
#define GRAPH_SIZE 8192

typedef struct {
    ggml_backend_t backend_vk;
    ggml_backend_t backend_cpu;
    ggml_backend_sched_t sched;

    int n_layers;
    int hidden_dim;
    int intermediate;

    /* Weight tensors on Vulkan — persistent across calls */
    struct ggml_tensor *gate_w[MAX_LAYERS];
    struct ggml_tensor *up_w[MAX_LAYERS];
    struct ggml_tensor *down_w[MAX_LAYERS];

    /* Weight contexts and buffers (one per layer for memory management) */
    struct ggml_context *w_ctx[MAX_LAYERS];
    ggml_backend_buffer_t w_buf[MAX_LAYERS];
} mlp_engine_t;

mlp_engine_t *mlp_engine_init(int n_layers, int hidden_dim, int intermediate) {
    mlp_engine_t *e = calloc(1, sizeof(mlp_engine_t));
    e->n_layers = n_layers;
    e->hidden_dim = hidden_dim;
    e->intermediate = intermediate;

    e->backend_vk = ggml_backend_vk_init(0);
    if (!e->backend_vk) { fprintf(stderr, "[mlp] Vulkan init FAILED\n"); free(e); return NULL; }

    e->backend_cpu = ggml_backend_cpu_init();

    ggml_backend_t backends[2] = { e->backend_vk, e->backend_cpu };
    e->sched = ggml_backend_sched_new(backends, NULL, 2, GRAPH_SIZE, false, false);

    return e;
}

int mlp_engine_load_layer(mlp_engine_t *e, int layer,
                          const float *gate_data, const float *up_data, const float *down_data) {
    int K = e->hidden_dim;
    int N = e->intermediate;

    struct ggml_init_params p = { .mem_size = 64*1024*1024, .mem_buffer = NULL, .no_alloc = true };
    e->w_ctx[layer] = ggml_init(p);

    /* gate_w: (N, K)  — ggml_new_tensor_2d(type, ne0=K, ne1=N) */
    e->gate_w[layer] = ggml_new_tensor_2d(e->w_ctx[layer], GGML_TYPE_F32, K, N);
    e->up_w[layer]   = ggml_new_tensor_2d(e->w_ctx[layer], GGML_TYPE_F32, K, N);
    /* down_w: (K, N)  — transposed: ggml_mul_mat(down_w, act) = act @ down_w.T */
    e->down_w[layer] = ggml_new_tensor_2d(e->w_ctx[layer], GGML_TYPE_F32, N, K);

    e->w_buf[layer] = ggml_backend_alloc_ctx_tensors(e->w_ctx[layer], e->backend_vk);
    if (!e->w_buf[layer]) {
        fprintf(stderr, "[mlp] Vulkan alloc FAILED layer %d\n", layer);
        return -1;
    }

    ggml_backend_tensor_set(e->gate_w[layer], gate_data, 0, (size_t)K * N * sizeof(float));
    ggml_backend_tensor_set(e->up_w[layer],   up_data,   0, (size_t)K * N * sizeof(float));
    ggml_backend_tensor_set(e->down_w[layer], down_data, 0, (size_t)N * K * sizeof(float));

    return 0;
}

/*
 * Forward pass: ONE ggml graph for ALL layers. ONE Vulkan dispatch.
 *
 * For each layer:
 *   gate = input @ gate_w.T         (ggml_mul_mat)
 *   up   = input @ up_w.T           (ggml_mul_mat)
 *   act  = silu(gate) * up           (ggml_silu + ggml_mul)
 *   output = act @ down_w.T          (ggml_mul_mat)
 *   input = output                   (chain to next layer)
 *
 * input/output: (M, hidden_dim) float32
 */
int mlp_engine_forward(mlp_engine_t *e, int M, const float *input, float *output) {
    int K = e->hidden_dim;
    int N = e->intermediate;

    /* Context for the compute graph — needs enough space for all tensor metadata.
     * Each layer creates ~5 tensors. 32 layers = 160 tensors + overhead. */
    size_t ctx_size = (size_t)(e->n_layers * 6 + 4) * ggml_tensor_overhead() + 1024*1024;
    struct ggml_init_params gp = { .mem_size = ctx_size, .mem_buffer = NULL, .no_alloc = true };
    struct ggml_context *ctx = ggml_init(gp);
    if (!ctx) { fprintf(stderr, "[mlp] context init failed (need %zu bytes)\n", ctx_size); return -1; }

    /* Build the ENTIRE computation graph */
    struct ggml_tensor *cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_set_name(cur, "input");

    for (int i = 0; i < e->n_layers; i++) {
        /* gate = cur @ gate_w.T */
        struct ggml_tensor *gate = ggml_mul_mat(ctx, e->gate_w[i], cur);
        /* up = cur @ up_w.T */
        struct ggml_tensor *up = ggml_mul_mat(ctx, e->up_w[i], cur);
        /* act = silu(gate) * up  (element-wise) */
        struct ggml_tensor *silu_gate = ggml_silu(ctx, gate);
        struct ggml_tensor *act = ggml_mul(ctx, silu_gate, up);
        /* down = act @ down_w.T */
        struct ggml_tensor *down = ggml_mul_mat(ctx, e->down_w[i], act);

        cur = down;  /* chain to next layer */
    }
    ggml_set_name(cur, "output");

    /* Build forward graph — single graph for ALL layers */
    struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
    ggml_build_forward_expand(graph, cur);

    /* Allocate and compute — ONE dispatch */
    ggml_backend_sched_reset(e->sched);
    if (!ggml_backend_sched_alloc_graph(e->sched, graph)) {
        fprintf(stderr, "[mlp] graph alloc failed\n");
        ggml_free(ctx);
        return -1;
    }

    /* Set input data */
    struct ggml_tensor *input_tensor = ggml_graph_node(graph, 0);
    /* Find the actual input tensor (first node's src) */
    /* Actually, we need to find "input" tensor by name or by the graph's leaf */
    /* The input tensor is a leaf (no src), find it */
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        struct ggml_tensor *node = ggml_graph_node(graph, i);
        /* Check sources */
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            struct ggml_tensor *src = node->src[s];
            if (src && strcmp(src->name, "input") == 0) {
                ggml_backend_tensor_set(src, input, 0, (size_t)M * K * sizeof(float));
                goto input_set;
            }
        }
    }
input_set:

    /* COMPUTE — ONE Vulkan dispatch for ALL 32 layers */
    enum ggml_status status = ggml_backend_sched_graph_compute(e->sched, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[mlp] compute failed: %d\n", status);
        ggml_free(ctx);
        return -1;
    }

    /* Read output */
    ggml_backend_tensor_get(cur, output, 0, (size_t)M * K * sizeof(float));

    ggml_free(ctx);
    return 0;
}

void mlp_engine_free(mlp_engine_t *e) {
    for (int i = 0; i < e->n_layers; i++) {
        if (e->w_buf[i]) ggml_backend_buffer_free(e->w_buf[i]);
        if (e->w_ctx[i]) ggml_free(e->w_ctx[i]);
    }
    if (e->sched) ggml_backend_sched_free(e->sched);
    if (e->backend_cpu) ggml_backend_free(e->backend_cpu);
    if (e->backend_vk) ggml_backend_free(e->backend_vk);
    free(e);
}
