#!/usr/bin/env python3
"""Apply MoE support to ggml_llama_gguf.c"""

with open('ggml_llama_gguf.c', 'r') as f:
    lines = f.readlines()

# 1. Add MoE fields to layer_t (after line with "attn_norm, *ffn_norm;")
for i, line in enumerate(lines):
    if 'attn_norm, *ffn_norm;' in line and i < 50:
        # Insert after this line
        indent = '    '
        moe_fields = [
            indent + '/* MoE expert weights (optional) */\n',
            indent + 'struct ggml_tensor *ffn_gate_exps, *ffn_up_exps, *ffn_down_exps;\n',
            indent + 'struct ggml_tensor *ffn_gate_inp;\n',
            indent + 'int expert_count;\n',
            indent + 'int expert_used;\n',
        ]
        for j, field in enumerate(moe_fields):
            lines.insert(i + 1 + j, field)
        break

# 2. Add MoE loading after tensor loading (find "fclose(fp);")
for i, line in enumerate(lines):
    if 'fclose(fp);' in line and i > 200:
        # Insert MoE loading code before fclose
        moe_load_code = [
            '\n',
            '    /* Load MoE expert weights if present (T07) */\n',
            '    for (int il = 0; il < e->n_layers; il++) {\n',
            '        char name[64];\n',
            '        snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps", il);\n',
            '        struct ggml_tensor *t = ggml_get_tensor(e->w_ctx, name);\n',
            '        if (t) {\n',
            '            e->layers[il].expert_count = t->ne[2];\n',
            '            e->layers[il].expert_used = 4; /* Default for gpt-oss */\n',
            '            e->layers[il].ffn_gate_exps = t;\n',
            '            snprintf(name, sizeof(name), "blk.%d.ffn_up_exps", il);\n',
            '            e->layers[il].ffn_up_exps = ggml_get_tensor(e->w_ctx, name);\n',
            '            snprintf(name, sizeof(name), "blk.%d.ffn_down_exps", il);\n',
            '            e->layers[il].ffn_down_exps = ggml_get_tensor(e->w_ctx, name);\n',
            '            snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp", il);\n',
            '            e->layers[il].ffn_gate_inp = ggml_get_tensor(e->w_ctx, name);\n',
            '            fprintf(stderr, "[gguf] Layer %d: MoE detected (%d experts)\\n", il, e->layers[il].expert_count);\n',
            '        }\n',
            '    }\n',
        ]
        for j, code_line in enumerate(moe_load_code):
            lines.insert(i + j, code_line)
        break

# 3. Replace dense FFN with MoE-aware FFN (find "struct ggml_tensor *gate = ggml_mul_mat(ctx, l->w_gate, cur);")
for i, line in enumerate(lines):
    if 'struct ggml_tensor *gate = ggml_mul_mat(ctx, l->w_gate, cur);' in line and i > 300:
        # Replace this block (4 lines)
        moe_ffn_code = [
            '        /* MoE FFN if experts present, else dense FFN (T07) */\n',
            '        if (l->ffn_gate_exps) {\n',
            '            /* MoE path: router -> topK -> expert FFN -> weighted sum */\n',
            '            int N_EXPERT = l->expert_count;\n',
            '            int N_USED = l->expert_used;\n',
            '            int H = e->hidden_dim;\n',
            '            \n',
            '            /* Step 1: Router (gate_inp) -> expert logits [N_EXPERT, n_tokens] */\n',
            '            struct ggml_tensor *logits = ggml_mul_mat(ctx, l->ffn_gate_inp, cur);\n',
            '            \n',
            '            /* Step 2: Softmax for expert selection probabilities */\n',
            '            struct ggml_tensor *probs = ggml_soft_max(ctx, logits);\n',
            '            \n',
            '            /* Step 3: TopK selection (get indices of top 4 experts) */\n',
            '            struct ggml_tensor *selected_experts = ggml_argsort_top_k(ctx, probs, N_USED);\n',
            '            \n',
            '            /* Step 4: Get expert weights for selected experts */\n',
            '            struct ggml_tensor *weights = ggml_get_rows(ctx, probs, selected_experts);\n',
            '            weights = ggml_reshape_2d(ctx, weights, N_USED, n_tokens);\n',
            '            weights = ggml_soft_max(ctx, weights); /* Normalize weights */\n',
            '            weights = ggml_reshape_3d(ctx, weights, 1, N_USED, n_tokens);\n',
            '            \n',
            '            /* Step 5: Repeat cur for each selected expert [H, N_USED, n_tokens] */\n',
            '            struct ggml_tensor *cur_expert = ggml_repeat_4d(ctx, cur, H, N_USED, n_tokens, 1);\n',
            '            \n',
            '            /* Step 6: Multiply by weights */\n',
            '            cur_expert = ggml_mul(ctx, cur_expert, weights);\n',
            '            \n',
            '            /* Step 7: Expert up proj (fused gate+up for efficiency) */\n',
            '            struct ggml_tensor *gate_up_exps = ggml_mul_mat_id(ctx, l->ffn_up_exps, cur_expert, selected_experts);\n',
            '            \n',
            '            /* Step 8: Split gate and up */\n',
            '            int FF = gate_up_exps->ne[0] / 2;\n',
            '            struct ggml_tensor *gate_expert = ggml_view_3d(ctx, gate_up_exps, FF, N_USED, n_tokens, \n',
            '                gate_up_exps->nb[1], gate_up_exps->nb[2], 0);\n',
            '            struct ggml_tensor *up_expert = ggml_view_3d(ctx, gate_up_exps, FF, N_USED, n_tokens,\n',
            '                gate_up_exps->nb[1], gate_up_exps->nb[2], FF * gate_up_exps->nb[0]);\n',
            '            \n',
            '            /* Step 9: SwiGLU activation */\n',
            '            struct ggml_tensor *gate_act = ggml_silu(ctx, gate_expert);\n',
            '            struct ggml_tensor *act = ggml_mul(ctx, gate_act, up_expert);\n',
            '            \n',
            '            /* Step 10: Expert down proj */\n',
            '            struct ggml_tensor *down_expert = ggml_mul_mat_id(ctx, l->ffn_down_exps, act, selected_experts);\n',
            '            \n',
            '            /* Step 11: Weighted sum across experts */\n',
            '            down_expert = ggml_mul(ctx, down_expert, weights);\n',
            '            cur = ggml_sum_rows(ctx, down_expert);\n',
            '            cur = ggml_reshape_2d(ctx, cur, H, n_tokens);\n',
            '        } else {\n',
            '            /* Dense FFN (original path) */\n',
            '            struct ggml_tensor *gate = ggml_mul_mat(ctx, l->w_gate, cur);\n',
            '            struct ggml_tensor *up   = ggml_mul_mat(ctx, l->w_up, cur);\n',
            '            cur = ggml_mul(ctx, ggml_silu(ctx, gate), up);\n',
            '            cur = ggml_mul_mat(ctx, l->w_down, cur);\n',
            '        }\n',
        ]
        # Remove the old 4 lines
        del lines[i:i+4]
        # Insert new code
        for j, code_line in enumerate(moe_ffn_code):
            lines.insert(i + j, code_line)
        break

with open('ggml_llama_gguf.c', 'w') as f:
    f.writelines(lines)

print("MoE support applied successfully")
