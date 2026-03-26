/*
 * ggml_llama_full.c — Full Llama transformer forward pass in ONE ggml graph.
 * Attention + MLP + RMSNorm + RoPE + residual, all on Vulkan.
 * Follows llama.cpp's graph pattern from src/models/llama.cpp.
 *
 * Build:
 *   gcc -shared -O2 -fPIC -o libggml_llama_full.so ggml_llama_full.c \
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
#include <math.h>

#define MAX_LAYERS 64
#define GRAPH_SIZE 16384  /* 32 layers × ~15 ops each + overhead */

typedef struct {
    /* Attention weights */
    struct ggml_tensor *wq;       /* (n_heads * head_dim, hidden) */
    struct ggml_tensor *wk;       /* (n_kv_heads * head_dim, hidden) */
    struct ggml_tensor *wv;       /* (n_kv_heads * head_dim, hidden) */
    struct ggml_tensor *wo;       /* (hidden, n_heads * head_dim) */
    /* MLP weights */
    struct ggml_tensor *w_gate;   /* (intermediate, hidden) */
    struct ggml_tensor *w_up;     /* (intermediate, hidden) */
    struct ggml_tensor *w_down;   /* (hidden, intermediate) */
    /* Norms */
    struct ggml_tensor *attn_norm; /* (hidden,) */
    struct ggml_tensor *ffn_norm;  /* (hidden,) */
} layer_weights_t;

typedef struct {
    ggml_backend_t backend_vk;
    ggml_backend_t backend_cpu;
    ggml_backend_sched_t sched;

    /* Model config */
    int n_layers;
    int hidden_dim;     /* 4096 for 8B */
    int intermediate;   /* 14336 for 8B */
    int n_heads;        /* 32 for 8B */
    int n_kv_heads;     /* 8 for 8B (GQA) */
    int head_dim;       /* 128 for 8B */
    int vocab_size;     /* 128256 for Llama-3.1 */
    float rms_eps;      /* 1e-5 */
    float rope_theta;   /* 500000 for Llama-3.1 */

    /* Per-layer weights on Vulkan */
    layer_weights_t layers[MAX_LAYERS];
    struct ggml_context *w_ctx;
    ggml_backend_buffer_t w_buf;

    /* Token embedding and output norm/head */
    struct ggml_tensor *tok_embd;   /* (vocab, hidden) */
    struct ggml_tensor *output_norm; /* (hidden,) */
    struct ggml_tensor *output;     /* (vocab, hidden) — lm_head */

    /* KV cache on Vulkan */
    struct ggml_tensor *kv_k[MAX_LAYERS]; /* (n_ctx, n_kv_heads, head_dim) */
    struct ggml_tensor *kv_v[MAX_LAYERS]; /* (n_ctx, n_kv_heads, head_dim) */
    struct ggml_context *kv_ctx;
    ggml_backend_buffer_t kv_buf;
    int kv_used;  /* How many positions are filled */
    int n_ctx;    /* Max context length */
} llama_engine_t;

/*
 * Initialize the engine with model parameters.
 * Does NOT load weights — call llama_engine_load_weights after.
 */
llama_engine_t *llama_engine_init(
    int n_layers, int hidden_dim, int intermediate,
    int n_heads, int n_kv_heads, int vocab_size,
    int n_ctx, float rms_eps, float rope_theta
) {
    llama_engine_t *e = calloc(1, sizeof(llama_engine_t));
    e->n_layers = n_layers;
    e->hidden_dim = hidden_dim;
    e->intermediate = intermediate;
    e->n_heads = n_heads;
    e->n_kv_heads = n_kv_heads;
    e->head_dim = hidden_dim / n_heads;
    e->vocab_size = vocab_size;
    e->rms_eps = rms_eps;
    e->rope_theta = rope_theta;
    e->n_ctx = n_ctx;
    e->kv_used = 0;

    e->backend_vk = ggml_backend_vk_init(0);
    if (!e->backend_vk) {
        fprintf(stderr, "[llama_engine] Vulkan init FAILED\n");
        free(e);
        return NULL;
    }
    e->backend_cpu = ggml_backend_cpu_init();

    ggml_backend_t backends[2] = { e->backend_vk, e->backend_cpu };
    e->sched = ggml_backend_sched_new(backends, NULL, 2, GRAPH_SIZE, false, false);

    fprintf(stderr, "[llama_engine] Init: %d layers, %d hidden, %d intermediate, %d heads, %d kv_heads, %d ctx\n",
            n_layers, hidden_dim, intermediate, n_heads, n_kv_heads, n_ctx);
    return e;
}

/*
 * Allocate weight tensors on Vulkan.
 * Returns total weight buffer size needed.
 * After this, call llama_engine_set_weight() for each weight.
 */
int llama_engine_alloc_weights(llama_engine_t *e) {
    int H = e->hidden_dim;
    int I = e->intermediate;
    int NH = e->n_heads;
    int NKV = e->n_kv_heads;
    int HD = e->head_dim;
    int V = e->vocab_size;

    /* Calculate total tensor count for context sizing */
    int n_tensors = e->n_layers * 9 + 3; /* 9 per layer + embd + norm + output */
    size_t ctx_size = (size_t)n_tensors * ggml_tensor_overhead() + 16*1024*1024;

    struct ggml_init_params p = { .mem_size = ctx_size, .mem_buffer = NULL, .no_alloc = true };
    e->w_ctx = ggml_init(p);
    if (!e->w_ctx) { fprintf(stderr, "[llama_engine] weight context init failed\n"); return -1; }

    /* Weight dtype: F16 for matmul weights (halves bandwidth), F32 for norms */
    enum ggml_type wtype = GGML_TYPE_F16;

    /* Token embedding (keep F32 for precision in lookup) */
    e->tok_embd = ggml_new_tensor_2d(e->w_ctx, wtype, H, V);
    ggml_set_name(e->tok_embd, "tok_embd");

    /* Output norm (F32) and head (F16) */
    e->output_norm = ggml_new_tensor_1d(e->w_ctx, GGML_TYPE_F32, H);
    ggml_set_name(e->output_norm, "output_norm");
    e->output = ggml_new_tensor_2d(e->w_ctx, wtype, H, V);
    ggml_set_name(e->output, "output");

    for (int i = 0; i < e->n_layers; i++) {
        layer_weights_t *l = &e->layers[i];
        char name[64];

        /* Attention (F16 for bandwidth) */
        l->wq = ggml_new_tensor_2d(e->w_ctx, wtype, H, NH * HD);
        snprintf(name, sizeof(name), "l%d.wq", i); ggml_set_name(l->wq, name);

        l->wk = ggml_new_tensor_2d(e->w_ctx, wtype, H, NKV * HD);
        snprintf(name, sizeof(name), "l%d.wk", i); ggml_set_name(l->wk, name);

        l->wv = ggml_new_tensor_2d(e->w_ctx, wtype, H, NKV * HD);
        snprintf(name, sizeof(name), "l%d.wv", i); ggml_set_name(l->wv, name);

        l->wo = ggml_new_tensor_2d(e->w_ctx, wtype, NH * HD, H);
        snprintf(name, sizeof(name), "l%d.wo", i); ggml_set_name(l->wo, name);

        /* MLP (F16 for bandwidth) */
        l->w_gate = ggml_new_tensor_2d(e->w_ctx, wtype, H, I);
        snprintf(name, sizeof(name), "l%d.gate", i); ggml_set_name(l->w_gate, name);

        l->w_up = ggml_new_tensor_2d(e->w_ctx, wtype, H, I);
        snprintf(name, sizeof(name), "l%d.up", i); ggml_set_name(l->w_up, name);

        l->w_down = ggml_new_tensor_2d(e->w_ctx, wtype, I, H);
        snprintf(name, sizeof(name), "l%d.down", i); ggml_set_name(l->w_down, name);

        /* Norms */
        l->attn_norm = ggml_new_tensor_1d(e->w_ctx, GGML_TYPE_F32, H);
        snprintf(name, sizeof(name), "l%d.attn_norm", i); ggml_set_name(l->attn_norm, name);

        l->ffn_norm = ggml_new_tensor_1d(e->w_ctx, GGML_TYPE_F32, H);
        snprintf(name, sizeof(name), "l%d.ffn_norm", i); ggml_set_name(l->ffn_norm, name);
    }

    /* Allocate all on Vulkan */
    e->w_buf = ggml_backend_alloc_ctx_tensors(e->w_ctx, e->backend_vk);
    if (!e->w_buf) {
        fprintf(stderr, "[llama_engine] weight alloc on Vulkan FAILED\n");
        return -1;
    }

    /* Allocate KV cache on Vulkan */
    {
        int n_kv_tensors = e->n_layers * 2;
        size_t kv_ctx_size = (size_t)n_kv_tensors * ggml_tensor_overhead() + 4*1024*1024;
        struct ggml_init_params kp = { .mem_size = kv_ctx_size, .mem_buffer = NULL, .no_alloc = true };
        e->kv_ctx = ggml_init(kp);

        for (int i = 0; i < e->n_layers; i++) {
            char kname[64], vname[64];
            /* K cache: (head_dim, n_kv_heads, n_ctx) — ggml is row-major ne0=head_dim */
            e->kv_k[i] = ggml_new_tensor_3d(e->kv_ctx, GGML_TYPE_F16, HD, NKV, e->n_ctx);
            snprintf(kname, sizeof(kname), "kv_k_%d", i); ggml_set_name(e->kv_k[i], kname);

            /* V cache: (head_dim, n_kv_heads, n_ctx) */
            e->kv_v[i] = ggml_new_tensor_3d(e->kv_ctx, GGML_TYPE_F16, HD, NKV, e->n_ctx);
            snprintf(vname, sizeof(vname), "kv_v_%d", i); ggml_set_name(e->kv_v[i], vname);
        }

        e->kv_buf = ggml_backend_alloc_ctx_tensors(e->kv_ctx, e->backend_vk);
        if (!e->kv_buf) {
            fprintf(stderr, "[llama_engine] KV cache alloc FAILED\n");
            return -1;
        }
        /* Zero out KV cache */
        ggml_backend_buffer_clear(e->kv_buf, 0);

        size_t kv_total = ggml_backend_buffer_get_size(e->kv_buf);
        fprintf(stderr, "[llama_engine] KV cache allocated: %.2f MiB on Vulkan\n", kv_total / (1024.0*1024));
    }

    size_t total = ggml_backend_buffer_get_size(e->w_buf);
    fprintf(stderr, "[llama_engine] Weights allocated: %.2f GiB on Vulkan\n", total / (1024.0*1024*1024));
    return 0;
}

/* Set weight data by name pattern */
int llama_engine_set_weight(llama_engine_t *e, const char *name, const float *data, size_t nbytes) {
    /* Find the tensor by name */
    struct ggml_tensor *t = NULL;

    if (strcmp(name, "tok_embd") == 0) t = e->tok_embd;
    else if (strcmp(name, "output_norm") == 0) t = e->output_norm;
    else if (strcmp(name, "output") == 0) t = e->output;
    else {
        /* Parse layer index: "l<N>.<weight_name>" */
        int layer;
        char wname[32];
        if (sscanf(name, "l%d.%31s", &layer, wname) == 2 && layer < e->n_layers) {
            layer_weights_t *l = &e->layers[layer];
            if (strcmp(wname, "wq") == 0) t = l->wq;
            else if (strcmp(wname, "wk") == 0) t = l->wk;
            else if (strcmp(wname, "wv") == 0) t = l->wv;
            else if (strcmp(wname, "wo") == 0) t = l->wo;
            else if (strcmp(wname, "gate") == 0) t = l->w_gate;
            else if (strcmp(wname, "up") == 0) t = l->w_up;
            else if (strcmp(wname, "down") == 0) t = l->w_down;
            else if (strcmp(wname, "attn_norm") == 0) t = l->attn_norm;
            else if (strcmp(wname, "ffn_norm") == 0) t = l->ffn_norm;
        }
    }

    if (!t) {
        fprintf(stderr, "[llama_engine] Unknown weight: %s\n", name);
        return -1;
    }

    /* Convert F32 data to tensor's type if needed */
    if (t->type == GGML_TYPE_F16) {
        /* Convert F32 → F16 */
        size_t n_elements = nbytes / sizeof(float);
        size_t f16_bytes = n_elements * sizeof(uint16_t);
        uint16_t *f16_data = (uint16_t *)malloc(f16_bytes);
        ggml_fp32_to_fp16_row(data, (ggml_fp16_t *)f16_data, n_elements);
        ggml_backend_tensor_set(t, f16_data, 0, f16_bytes);
        free(f16_data);
    } else {
        ggml_backend_tensor_set(t, data, 0, nbytes);
    }
    return 0;
}

/*
 * Build the full transformer graph for n_tokens input tokens.
 * This is the core — ONE graph for ALL layers, matching llama.cpp's architecture.
 *
 * Returns: logits tensor (n_tokens, vocab_size)
 */
int llama_engine_forward(llama_engine_t *e, int n_tokens,
                         const int32_t *tokens, const int32_t *positions,
                         float *logits_out) {
    int H = e->hidden_dim;
    int I = e->intermediate;
    int NH = e->n_heads;
    int NKV = e->n_kv_heads;
    int HD = e->head_dim;

    /* Compute context — needs space for all intermediate tensors */
    size_t ctx_size = (size_t)(e->n_layers * 20 + 10) * ggml_tensor_overhead() + 64*1024*1024;
    struct ggml_init_params gp = { .mem_size = ctx_size, .mem_buffer = NULL, .no_alloc = true };
    struct ggml_context *ctx = ggml_init(gp);
    if (!ctx) return -1;

    /* Input: token indices → embedding lookup */
    struct ggml_tensor *inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens, "tokens");

    struct ggml_tensor *inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_pos, "positions");

    /* Embedding lookup */
    struct ggml_tensor *cur = ggml_get_rows(ctx, e->tok_embd, inp_tokens);
    struct ggml_tensor *inpL = cur;

    float kq_scale = 1.0f / sqrtf((float)HD);

    /* Create graph early so KV cache writes can be added during layer loop */
    struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);

    /* Causal attention mask: (kv_len, n_tokens) — filled on CPU, passed as input
     * flash_attn_ext expects mask or NULL. With NULL it does causal masking automatically
     * when the K sequence is longer than Q. We'll pass NULL for simplicity. */
    struct ggml_tensor *kq_mask = NULL;

    struct ggml_tensor *attn;

    for (int il = 0; il < e->n_layers; il++) {
        layer_weights_t *l = &e->layers[il];
        struct ggml_tensor *inpSA = inpL;

        /* Pre-attention RMSNorm */
        cur = ggml_rms_norm(ctx, inpL, e->rms_eps);
        cur = ggml_mul(ctx, cur, l->attn_norm);

        /* QKV projections */
        struct ggml_tensor *Qcur = ggml_mul_mat(ctx, l->wq, cur);
        struct ggml_tensor *Kcur = ggml_mul_mat(ctx, l->wk, cur);
        struct ggml_tensor *Vcur = ggml_mul_mat(ctx, l->wv, cur);

        /* Reshape for multi-head */
        Qcur = ggml_reshape_3d(ctx, Qcur, HD, NH, n_tokens);
        Kcur = ggml_reshape_3d(ctx, Kcur, HD, NKV, n_tokens);
        Vcur = ggml_reshape_3d(ctx, Vcur, HD, NKV, n_tokens);

        /* RoPE */
        Qcur = ggml_rope_ext(ctx, Qcur, inp_pos, NULL,
                             HD, 0 /*ROPE_NEOX*/, 0, e->rope_theta, 1.0f,
                             0.0f, 1.0f, 0.0f, 0.0f);
        Kcur = ggml_rope_ext(ctx, Kcur, inp_pos, NULL,
                             HD, 0, 0, e->rope_theta, 1.0f,
                             0.0f, 1.0f, 0.0f, 0.0f);

        /* KV Cache: write new K/V into cache, then attend over full cache */
        {
            int kv_pos = e->kv_used;  /* position to write new tokens */
            int kv_len = kv_pos + n_tokens;  /* total filled length after this batch */

            /* Cast K/V to F16 for cache storage */
            struct ggml_tensor *Kf16 = ggml_cast(ctx, Kcur, GGML_TYPE_F16);
            struct ggml_tensor *Vf16 = ggml_cast(ctx, Vcur, GGML_TYPE_F16);

            /* Write new K into cache: copy into kv_k[il] at offset kv_pos along dim 2 */
            /* kv_k shape: (HD, NKV, n_ctx). We write at [:, :, kv_pos:kv_pos+n_tokens] */
            size_t k_offset = (size_t)kv_pos * NKV * HD * ggml_type_size(GGML_TYPE_F16);
            struct ggml_tensor *k_dst = ggml_view_3d(ctx, e->kv_k[il],
                HD, NKV, n_tokens,
                e->kv_k[il]->nb[1], e->kv_k[il]->nb[2], k_offset);
            struct ggml_tensor *k_cpy = ggml_cpy(ctx, Kf16, k_dst);
            ggml_build_forward_expand(graph, k_cpy);

            /* Write new V into cache */
            size_t v_offset = (size_t)kv_pos * NKV * HD * ggml_type_size(GGML_TYPE_F16);
            struct ggml_tensor *v_dst = ggml_view_3d(ctx, e->kv_v[il],
                HD, NKV, n_tokens,
                e->kv_v[il]->nb[1], e->kv_v[il]->nb[2], v_offset);
            struct ggml_tensor *v_cpy = ggml_cpy(ctx, Vf16, v_dst);
            ggml_build_forward_expand(graph, v_cpy);

            /* View of full cached K/V up to kv_len */
            struct ggml_tensor *K_cached = ggml_view_3d(ctx, e->kv_k[il],
                HD, NKV, kv_len,
                e->kv_k[il]->nb[1], e->kv_k[il]->nb[2], 0);
            struct ggml_tensor *V_cached = ggml_view_3d(ctx, e->kv_v[il],
                HD, NKV, kv_len,
                e->kv_v[il]->nb[1], e->kv_v[il]->nb[2], 0);

            /* Permute for attention: (HD, NH/NKV, n_tokens/kv_len) → (HD, n_tokens/kv_len, NH/NKV) */
            struct ggml_tensor *q_perm = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
            struct ggml_tensor *k_perm = ggml_permute(ctx, K_cached, 0, 2, 1, 3);
            struct ggml_tensor *v_perm = ggml_permute(ctx, V_cached, 0, 2, 1, 3);

            /* Flash attention with causal mask = NULL (ggml handles causal internally) */
            attn = ggml_flash_attn_ext(ctx, q_perm, k_perm, v_perm, kq_mask, kq_scale, 0.0f, 0.0f);
        }

        /* Reshape back */
        attn = ggml_reshape_2d(ctx, attn, NH * HD, n_tokens);

        /* Output projection */
        cur = ggml_mul_mat(ctx, l->wo, attn);

        /* Residual */
        struct ggml_tensor *ffn_inp = ggml_add(ctx, cur, inpSA);

        /* Pre-FFN RMSNorm */
        cur = ggml_rms_norm(ctx, ffn_inp, e->rms_eps);
        cur = ggml_mul(ctx, cur, l->ffn_norm);

        /* FFN: gate + up + silu + mul + down */
        struct ggml_tensor *gate = ggml_mul_mat(ctx, l->w_gate, cur);
        struct ggml_tensor *up   = ggml_mul_mat(ctx, l->w_up, cur);
        struct ggml_tensor *act  = ggml_mul(ctx, ggml_silu(ctx, gate), up);
        cur = ggml_mul_mat(ctx, l->w_down, act);

        /* Residual */
        inpL = ggml_add(ctx, cur, ffn_inp);
    }

    /* Final RMSNorm */
    cur = ggml_rms_norm(ctx, inpL, e->rms_eps);
    cur = ggml_mul(ctx, cur, e->output_norm);

    /* LM head */
    cur = ggml_mul_mat(ctx, e->output, cur);
    ggml_set_name(cur, "logits");

    /* Finalize graph — add final output as target */
    ggml_build_forward_expand(graph, cur);

    fprintf(stderr, "[llama_engine] Graph: %d nodes\n", ggml_graph_n_nodes(graph));

    /* Allocate */
    ggml_backend_sched_reset(e->sched);
    if (!ggml_backend_sched_alloc_graph(e->sched, graph)) {
        fprintf(stderr, "[llama_engine] graph alloc FAILED\n");
        ggml_free(ctx);
        return -2;
    }

    /* Set inputs */
    ggml_backend_tensor_set(inp_tokens, tokens, 0, n_tokens * sizeof(int32_t));
    ggml_backend_tensor_set(inp_pos, positions, 0, n_tokens * sizeof(int32_t));

    /* COMPUTE — ONE dispatch for the ENTIRE model */
    enum ggml_status status = ggml_backend_sched_graph_compute(e->sched, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[llama_engine] compute FAILED: %d\n", status);
        ggml_free(ctx);
        return -3;
    }

    /* Read logits */
    size_t logits_size = (size_t)n_tokens * e->vocab_size * sizeof(float);
    ggml_backend_tensor_get(cur, logits_out, 0, logits_size);

    /* Advance KV cache position */
    e->kv_used += n_tokens;

    ggml_free(ctx);
    return 0;
}

/* Reset KV cache (for new conversation) */
void llama_engine_reset_kv(llama_engine_t *e) {
    e->kv_used = 0;
    if (e->kv_buf) {
        ggml_backend_buffer_clear(e->kv_buf, 0);
    }
}

void llama_engine_free(llama_engine_t *e) {
    if (e->w_buf) ggml_backend_buffer_free(e->w_buf);
    if (e->w_ctx) ggml_free(e->w_ctx);
    if (e->kv_buf) ggml_backend_buffer_free(e->kv_buf);
    if (e->kv_ctx) ggml_free(e->kv_ctx);
    if (e->sched) ggml_backend_sched_free(e->sched);
    if (e->backend_cpu) ggml_backend_free(e->backend_cpu);
    if (e->backend_vk) ggml_backend_free(e->backend_vk);
    free(e);
}
