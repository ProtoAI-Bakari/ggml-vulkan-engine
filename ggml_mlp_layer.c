/*
 * ggml_mlp_layer.c — Single MLP layer as ONE ggml graph (5 ops, 1 dispatch).
 * gate_mm + up_mm + silu + mul + down_mm in one Vulkan submission.
 *
 * Build:
 *   gcc -shared -O2 -fPIC -o libggml_mlp_layer.so ggml_mlp_layer.c \
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
#define GRAPH_SIZE 512

typedef struct {
    ggml_backend_t backend_vk;
    ggml_backend_t backend_cpu;
    ggml_backend_sched_t sched;

    int n_layers;
    int hidden_dim;
    int intermediate;

    struct ggml_tensor *gate_w[MAX_LAYERS];
    struct ggml_tensor *up_w[MAX_LAYERS];
    struct ggml_tensor *down_w[MAX_LAYERS];
    struct ggml_context *w_ctx[MAX_LAYERS];
    ggml_backend_buffer_t w_buf[MAX_LAYERS];
    int loaded;
} mlp_layer_engine_t;

mlp_layer_engine_t *mlp_layer_init(int n_layers, int hidden_dim, int intermediate) {
    mlp_layer_engine_t *e = calloc(1, sizeof(mlp_layer_engine_t));
    e->n_layers = n_layers;
    e->hidden_dim = hidden_dim;
    e->intermediate = intermediate;
    e->loaded = 0;

    e->backend_vk = ggml_backend_vk_init(0);
    if (!e->backend_vk) { free(e); return NULL; }
    e->backend_cpu = ggml_backend_cpu_init();

    ggml_backend_t backends[2] = { e->backend_vk, e->backend_cpu };
    e->sched = ggml_backend_sched_new(backends, NULL, 2, GRAPH_SIZE, false, false);
    return e;
}

int mlp_layer_load(mlp_layer_engine_t *e, int layer,
                   const float *gate_data, const float *up_data, const float *down_data) {
    int K = e->hidden_dim;
    int N = e->intermediate;
    struct ggml_init_params p = { .mem_size = 64*1024*1024, .mem_buffer = NULL, .no_alloc = true };
    e->w_ctx[layer] = ggml_init(p);
    e->gate_w[layer] = ggml_new_tensor_2d(e->w_ctx[layer], GGML_TYPE_F32, K, N);
    e->up_w[layer]   = ggml_new_tensor_2d(e->w_ctx[layer], GGML_TYPE_F32, K, N);
    e->down_w[layer] = ggml_new_tensor_2d(e->w_ctx[layer], GGML_TYPE_F32, N, K);
    e->w_buf[layer] = ggml_backend_alloc_ctx_tensors(e->w_ctx[layer], e->backend_vk);
    if (!e->w_buf[layer]) return -1;
    ggml_backend_tensor_set(e->gate_w[layer], gate_data, 0, (size_t)K*N*sizeof(float));
    ggml_backend_tensor_set(e->up_w[layer], up_data, 0, (size_t)K*N*sizeof(float));
    ggml_backend_tensor_set(e->down_w[layer], down_data, 0, (size_t)N*K*sizeof(float));
    e->loaded++;
    return 0;
}

/*
 * Run ONE layer's MLP as a single ggml graph:
 *   gate = x @ gate_w.T
 *   up   = x @ up_w.T
 *   act  = silu(gate) * up
 *   out  = act @ down_w.T
 *
 * x: (M, hidden_dim) float32
 * out: (M, hidden_dim) float32
 */
int mlp_layer_forward(mlp_layer_engine_t *e, int layer, int M,
                      const float *x_data, float *out_data) {
    int K = e->hidden_dim;
    int N = e->intermediate;

    size_t ctx_size = 8 * ggml_tensor_overhead() + 256*1024;
    struct ggml_init_params gp = { .mem_size = ctx_size, .mem_buffer = NULL, .no_alloc = true };
    struct ggml_context *ctx = ggml_init(gp);

    struct ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_set_name(x, "x");

    struct ggml_tensor *gate = ggml_mul_mat(ctx, e->gate_w[layer], x);
    struct ggml_tensor *up   = ggml_mul_mat(ctx, e->up_w[layer], x);
    struct ggml_tensor *act  = ggml_mul(ctx, ggml_silu(ctx, gate), up);
    struct ggml_tensor *out  = ggml_mul_mat(ctx, e->down_w[layer], act);
    ggml_set_name(out, "out");

    struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_sched_reset(e->sched);
    if (!ggml_backend_sched_alloc_graph(e->sched, graph)) {
        ggml_free(ctx);
        return -1;
    }

    /* Find and set input tensor */
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        struct ggml_tensor *node = ggml_graph_node(graph, i);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            struct ggml_tensor *src = node->src[s];
            if (src && strcmp(src->name, "x") == 0) {
                ggml_backend_tensor_set(src, x_data, 0, (size_t)M*K*sizeof(float));
                goto done_set;
            }
        }
    }
done_set:

    enum ggml_status status = ggml_backend_sched_graph_compute(e->sched, graph);
    if (status != GGML_STATUS_SUCCESS) { ggml_free(ctx); return -2; }

    ggml_backend_tensor_get(out, out_data, 0, (size_t)M*K*sizeof(float));
    ggml_free(ctx);
    return 0;
}

void mlp_layer_free(mlp_layer_engine_t *e) {
    for (int i = 0; i < e->n_layers; i++) {
        if (e->w_buf[i]) ggml_backend_buffer_free(e->w_buf[i]);
        if (e->w_ctx[i]) ggml_free(e->w_ctx[i]);
    }
    if (e->sched) ggml_backend_sched_free(e->sched);
    if (e->backend_cpu) ggml_backend_free(e->backend_cpu);
    if (e->backend_vk) ggml_backend_free(e->backend_vk);
    free(e);
}
