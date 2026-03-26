/*
 * ggml_mlp_chain.c — Run full 32-layer MLP chain in one C call
 * Eliminates Python overhead between layers.
 * Build: gcc -shared -O2 -o libggml_mlp_chain.so ggml_mlp_chain.c \
 *        -I ~/GITDEV/llama.cpp/ggml/include \
 *        -L ~/GITDEV/llama.cpp/build-lib/bin \
 *        -lggml -lggml-base -lm
 */
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-vulkan.h>
#include <ggml-cpu.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define MAX_LAYERS 64
#define MAX_GRAPH_SIZE 4096

typedef struct {
    ggml_backend_t backend_vk;
    ggml_backend_t backend_cpu;
    ggml_backend_sched_t sched;

    int n_layers;
    int hidden_dim;     // K (e.g., 4096 for 8B)
    int intermediate;   // N (e.g., 14336 for 8B)

    // Weight tensors (on Vulkan)
    struct ggml_tensor *gate_w[MAX_LAYERS];
    struct ggml_tensor *up_w[MAX_LAYERS];
    struct ggml_tensor *down_w[MAX_LAYERS];

    // Weight contexts and buffers
    struct ggml_context *w_ctx[MAX_LAYERS];
    ggml_backend_buffer_t w_buf[MAX_LAYERS];
} mlp_chain_t;

// Initialize the chain
mlp_chain_t *mlp_chain_init(int n_layers, int hidden_dim, int intermediate) {
    mlp_chain_t *chain = calloc(1, sizeof(mlp_chain_t));
    chain->n_layers = n_layers;
    chain->hidden_dim = hidden_dim;
    chain->intermediate = intermediate;

    chain->backend_vk = ggml_backend_vk_init(0);
    chain->backend_cpu = ggml_backend_cpu_init();

    ggml_backend_t backends[2] = {chain->backend_vk, chain->backend_cpu};
    chain->sched = ggml_backend_sched_new(backends, NULL, 2, MAX_GRAPH_SIZE, false, false);

    return chain;
}

// Load weights for one layer
int mlp_chain_load_layer(mlp_chain_t *chain, int layer,
                         const float *gate_data, const float *up_data, const float *down_data) {
    int K = chain->hidden_dim;
    int N = chain->intermediate;

    struct ggml_init_params params = {
        .mem_size = 64 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    chain->w_ctx[layer] = ggml_init(params);

    chain->gate_w[layer] = ggml_new_tensor_2d(chain->w_ctx[layer], GGML_TYPE_F32, K, N);
    chain->up_w[layer] = ggml_new_tensor_2d(chain->w_ctx[layer], GGML_TYPE_F32, K, N);
    chain->down_w[layer] = ggml_new_tensor_2d(chain->w_ctx[layer], GGML_TYPE_F32, N, K);

    chain->w_buf[layer] = ggml_backend_alloc_ctx_tensors(chain->w_ctx[layer], chain->backend_vk);
    if (!chain->w_buf[layer]) {
        fprintf(stderr, "[mlp_chain] Failed to alloc layer %d on Vulkan\n", layer);
        return -1;
    }

    ggml_backend_tensor_set(chain->gate_w[layer], gate_data, 0, (size_t)K * N * sizeof(float));
    ggml_backend_tensor_set(chain->up_w[layer], up_data, 0, (size_t)K * N * sizeof(float));
    ggml_backend_tensor_set(chain->down_w[layer], down_data, 0, (size_t)N * K * sizeof(float));

    return 0;
}

// Run full MLP chain: processes input through all layers
// input: (M, hidden_dim), output: (M, hidden_dim)
// Each layer: gate=input@gate_w.T, up=input@up_w.T, act=silu(gate)*up, output=act@down_w.T
int mlp_chain_forward(mlp_chain_t *chain, int M,
                      const float *input, float *output) {
    int K = chain->hidden_dim;
    int N = chain->intermediate;

    // We process one layer at a time to minimize graph size
    // Copy input to working buffer
    float *current = (float *)malloc((size_t)M * K * sizeof(float));
    memcpy(current, input, (size_t)M * K * sizeof(float));

    float *gate_out = (float *)malloc((size_t)M * N * sizeof(float));
    float *up_out = (float *)malloc((size_t)M * N * sizeof(float));
    float *act_out = (float *)malloc((size_t)M * N * sizeof(float));

    for (int layer = 0; layer < chain->n_layers; layer++) {
        // Build graph for this layer: 3 matmuls
        struct ggml_init_params gp = {
            .mem_size = 64 * 1024 * 1024,
            .mem_buffer = NULL,
            .no_alloc = true,
        };
        struct ggml_context *ctx = ggml_init(gp);

        struct ggml_tensor *t_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        ggml_set_name(t_in, "in");

        struct ggml_tensor *t_gate = ggml_mul_mat(ctx, chain->gate_w[layer], t_in);
        ggml_set_name(t_gate, "gate");
        struct ggml_tensor *t_up = ggml_mul_mat(ctx, chain->up_w[layer], t_in);
        ggml_set_name(t_up, "up");

        // Build graph with both matmuls
        struct ggml_cgraph *graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, t_gate);
        ggml_build_forward_expand(graph, t_up);

        ggml_backend_sched_reset(chain->sched);
        if (!ggml_backend_sched_alloc_graph(chain->sched, graph)) {
            fprintf(stderr, "[mlp_chain] Failed to alloc graph for layer %d\n", layer);
            ggml_free(ctx);
            free(current); free(gate_out); free(up_out); free(act_out);
            return -1;
        }

        ggml_backend_tensor_set(t_in, current, 0, (size_t)M * K * sizeof(float));
        ggml_backend_sched_graph_compute(chain->sched, graph);

        ggml_backend_tensor_get(t_gate, gate_out, 0, (size_t)M * N * sizeof(float));
        ggml_backend_tensor_get(t_up, up_out, 0, (size_t)M * N * sizeof(float));

        ggml_free(ctx);

        // SiLU activation on CPU: silu(gate) * up
        for (size_t i = 0; i < (size_t)M * N; i++) {
            float g = gate_out[i];
            act_out[i] = (g / (1.0f + expf(-g))) * up_out[i];
        }

        // Down projection
        gp.mem_size = 64 * 1024 * 1024;
        ctx = ggml_init(gp);

        struct ggml_tensor *t_act = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);
        ggml_set_name(t_act, "act");
        struct ggml_tensor *t_down = ggml_mul_mat(ctx, chain->down_w[layer], t_act);
        ggml_set_name(t_down, "down");

        graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, t_down);

        ggml_backend_sched_reset(chain->sched);
        ggml_backend_sched_alloc_graph(chain->sched, graph);

        ggml_backend_tensor_set(t_act, act_out, 0, (size_t)M * N * sizeof(float));
        ggml_backend_sched_graph_compute(chain->sched, graph);

        ggml_backend_tensor_get(t_down, current, 0, (size_t)M * K * sizeof(float));

        ggml_free(ctx);

        // Note: in a real transformer, we'd add residual + norm here
        // For now, just pass through to next layer
    }

    memcpy(output, current, (size_t)M * K * sizeof(float));

    free(current);
    free(gate_out);
    free(up_out);
    free(act_out);
    return 0;
}

void mlp_chain_free(mlp_chain_t *chain) {
    for (int i = 0; i < chain->n_layers; i++) {
        if (chain->w_buf[i]) ggml_backend_buffer_free(chain->w_buf[i]);
        if (chain->w_ctx[i]) ggml_free(chain->w_ctx[i]);
    }
    ggml_backend_sched_free(chain->sched);
    ggml_backend_free(chain->backend_cpu);
    ggml_backend_free(chain->backend_vk);
    free(chain);
}
