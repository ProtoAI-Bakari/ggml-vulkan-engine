#!/usr/bin/env python3
import re

# Read file
with open('ggml_llama_gguf.c', 'r') as f:
    content = f.read()

# 1. Add MoE fields to layer_t struct
layer_t_pattern = r'(typedef struct \{\n    struct ggml_tensor \*wq, \*wk, \*wv, \*wo;\n    struct ggml_tensor \*w_gate, \*w_up, \*w_down;\n    struct ggml_tensor \*attn_norm, \*ffn_norm;\n\})'
layer_t_replacement = '''typedef struct {
    struct ggml_tensor *wq, *wk, *wv, *wo;
    struct ggml_tensor *w_gate, *w_up, *w_down;
    struct ggml_tensor *attn_norm, *ffn_norm;
    /* MoE expert weights (optional) */
    struct ggml_tensor *ffn_gate_exps, *ffn_up_exps, *ffn_down_exps;
    struct ggml_tensor *ffn_gate_inp;
    int expert_count;
    int expert_used;
}'''
content = re.sub(layer_t_pattern, layer_t_replacement, content)

# 2. Add MoE loading after tensor loading loop
tensor_load_end = r'(    fclose\(fp\);\n\n    fprintf\(stderr, "\[gguf\] Loaded %d tensors from GGUF\\n", loaded\);)'
tensor_load_replacement = '''    fclose(fp);

    /* Load MoE expert weights if present (T07) */
    for (int il = 0; il < e->n_layers; il++) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps", il);
        struct ggml_tensor *t = ggml_get_tensor(e->w_ctx, name);
        if (t) {
            e->layers[il].expert_count = t->ne[2];
            e->layers[il].expert_used = 4; /* Default for gpt-oss */
            e->layers[il].ffn_gate_exps = t;
            snprintf(name, sizeof(name), "blk.%d.ffn_up_exps", il);
            e->layers[il].ffn_up_exps = ggml_get_tensor(e->w_ctx, name);
            snprintf(name, sizeof(name), "blk.%d.ffn_down_exps", il);
            e->layers[il].ffn_down_exps = ggml_get_tensor(e->w_ctx, name);
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp", il);
            e->layers[il].ffn_gate_inp = ggml_get_tensor(e->w_ctx, name);
            fprintf(stderr, "[gguf] Layer %d: MoE detected (%d experts)\\n", il, e->layers[il].expert_count);
        }
    }

    fprintf(stderr, "[gguf] Loaded %d tensors from GGUF", loaded);'''
content = re.sub(tensor_load_end, tensor_load_replacement, content)

# Write file
with open('ggml_llama_gguf.c', 'w') as f:
    f.write(content)

print("MoE fields added to layer_t")
print("MoE loading code added after tensor loading")
