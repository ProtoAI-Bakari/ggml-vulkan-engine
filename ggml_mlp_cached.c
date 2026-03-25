/*
 * ggml_mlp_cached.c — Pre-built graph, reused across tokens. Zero graph creation overhead.
 *
 * Build:
 *   gcc -shared -O2 -fPIC -o libggml_mlp_cached.so ggml_mlp_cached.c \
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
    int batch_size;  /* Fixed batch size for the pre-built graph */

    /* Weight tensors on Vulkan */
    struct ggml_tensor *gate_w[MAX_LAYERS];
    struct ggml_tensor *up_w[MAX_LAYERS];
    struct ggml_tensor *down_w[MAX_LAYERS];
    struct ggml_context *w_ctx[MAX_LAYERS];
    ggml_backend_buffer_t w_buf[MAX_LAYERS];

    /* Pre-built graph (reused across tokens) */
    struct ggml_context *graph_ctx;
    struct ggml_cgraph *graph;
    struct ggml_tensor *input_tensor;   /* Pointer to input node */
    struct ggml_tensor *output_tensor;  /* Pointer to output node */
    int graph_built;
    int graph_allocated;
} mlp_cached_t;

mlp_cached_t *mlp_cached_init(int n_layers, int hidden_dim, int intermediate, int batch_size) {
    mlp_cached_t *e = calloc(1, sizeof(mlp_cached_t));
    e->n_layers = n_layers;
    e->hidden_dim = hidden_dim;
    e->intermediate = intermediate;
    e->batch_size = batch_size;

    e->backend_vk = ggml_backend_vk_init(0);
    if (!e->backend_vk) { free(e); return NULL; }
    e->backend_cpu = ggml_backend_cpu_init();

    ggml_backend_t backends[2] = { e->backend_vk, e->backend_cpu };
    e->sched = ggml_backend_sched_new(backends, NULL, 2, GRAPH_SIZE, false, false);
    return e;
}

int mlp_cached_load(mlp_cached_t *e, int layer,
                    const float *gate_data, const float *up_data, const float *down_data) {
    int K = e->hidden_dim, N = e->intermediate;
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
    return 0;
}

/* Build the graph ONCE for a specific layer. Reuse on every forward call. */
int mlp_cached_build_graph(mlp_cached_t *e, int layer) {
    int K = e->hidden_dim, N = e->intermediate, M = e->batch_size;

    /* Free old graph if exists */
    if (e->graph_ctx) { ggml_free(e->graph_ctx); e->graph_ctx = NULL; }
    e->graph_built = 0;
    e->graph_allocated = 0;

    size_t ctx_size = 16 * ggml_tensor_overhead() + 512*1024;
    struct ggml_init_params gp = { .mem_size = ctx_size, .mem_buffer = NULL, .no_alloc = true };
    e->graph_ctx = ggml_init(gp);
    if (!e->graph_ctx) return -1;

    e->input_tensor = ggml_new_tensor_2d(e->graph_ctx, GGML_TYPE_F32, K, M);
    ggml_set_name(e->input_tensor, "x");

    struct ggml_tensor *gate = ggml_mul_mat(e->graph_ctx, e->gate_w[layer], e->input_tensor);
    struct ggml_tensor *up   = ggml_mul_mat(e->graph_ctx, e->up_w[layer], e->input_tensor);
    struct ggml_tensor *act  = ggml_mul(e->graph_ctx, ggml_silu(e->graph_ctx, gate), up);
    e->output_tensor = ggml_mul_mat(e->graph_ctx, e->down_w[layer], act);
    ggml_set_name(e->output_tensor, "out");

    e->graph = ggml_new_graph_custom(e->graph_ctx, GRAPH_SIZE, false);
    ggml_build_forward_expand(e->graph, e->output_tensor);

    e->graph_built = 1;
    return 0;
}

/* Forward: reuse pre-built graph, only set input data and compute.
 * MUCH faster than building a new graph each time. */
int mlp_cached_forward(mlp_cached_t *e, int layer, const float *x_data, float *out_data) {
    int K = e->hidden_dim, M = e->batch_size;

    /* Build graph for this layer if not cached */
    if (!e->graph_built) {
        if (mlp_cached_build_graph(e, layer) != 0) return -1;
    }

    /* Allocate graph on first use (scheduler handles Vulkan buffer allocation) */
    ggml_backend_sched_reset(e->sched);
    if (!ggml_backend_sched_alloc_graph(e->sched, e->graph)) return -2;

    /* Set input data */
    ggml_backend_tensor_set(e->input_tensor, x_data, 0, (size_t)M*K*sizeof(float));

    /* Compute — single Vulkan dispatch for 5 ops */
    enum ggml_status status = ggml_backend_sched_graph_compute(e->sched, e->graph);
    if (status != GGML_STATUS_SUCCESS) return -3;

    /* Read output */
    ggml_backend_tensor_get(e->output_tensor, out_data, 0, (size_t)M*K*sizeof(float));
    return 0;
}

/* Forward with layer switching — rebuilds graph when layer changes */
static int _last_layer = -1;
int mlp_cached_forward_layer(mlp_cached_t *e, int layer, int M,
                             const float *x_data, float *out_data) {
    /* Rebuild graph if layer changed or batch size changed */
    if (layer != _last_layer || M != e->batch_size) {
        e->batch_size = M;
        e->graph_built = 0;
        e->graph_allocated = 0;
        if (mlp_cached_build_graph(e, layer) != 0) return -1;
        _last_layer = layer;
    }

    return mlp_cached_forward(e, layer, x_data, out_data);
}

void mlp_cached_free(mlp_cached_t *e) {
    if (e->graph_ctx) ggml_free(e->graph_ctx);
    for (int i = 0; i < e->n_layers; i++) {
        if (e->w_buf[i]) ggml_backend_buffer_free(e->w_buf[i]);
        if (e->w_ctx[i]) ggml_free(e->w_ctx[i]);
    }
    if (e->sched) ggml_backend_sched_free(e->sched);
    if (e->backend_cpu) ggml_backend_free(e->backend_cpu);
    if (e->backend_vk) ggml_backend_free(e->backend_vk);
    free(e);
}
