#!/usr/bin/env python3
import re

# Read file
with open('ggml_llama_gguf.c', 'r') as f:
    content = f.read()

# Find and replace the dense FFN section with MoE-aware FFN
dense_ffn_pattern = r'''        struct ggml_tensor \*gate = ggml_mul_mat\(ctx, l->w_gate, cur\);
        struct ggml_tensor \*up   = ggml_mul_mat\(ctx, l->w_up, cur\);
        cur = ggml_mul\(ctx, ggml_silu\(ctx, gate\), up\);
        cur = ggml_mul_mat\(ctx, l->w_down, cur\);'''

moe_ffn_replacement = '''        /* MoE FFN if experts present, else dense FFN (T07) */
        if (l->ffn_gate_exps) {
            /* MoE path: router -> topK -> expert FFN -> weighted sum */
            int N_EXPERT = l->expert_count;
            int N_USED = l->expert_used;
            int H = e->hidden_dim;
            
            /* Step 1: Router (gate_inp) -> expert logits [N_EXPERT, n_tokens] */
            struct ggml_tensor *logits = ggml_mul_mat(ctx, l->ffn_gate_inp, cur);
            
            /* Step 2: Softmax for expert selection probabilities */
            struct ggml_tensor *probs = ggml_soft_max(ctx, logits);
            
            /* Step 3: TopK selection (get indices of top 4 experts) */
            struct ggml_tensor *selected_experts = ggml_argsort_top_k(ctx, probs, N_USED);
            
            /* Step 4: Get expert weights for selected experts */
            struct ggml_tensor *weights = ggml_get_rows(ctx, probs, selected_experts);
            weights = ggml_reshape_2d(ctx, weights, N_USED, n_tokens);
            weights = ggml_soft_max(ctx, weights); /* Normalize weights */
            weights = ggml_reshape_3d(ctx, weights, 1, N_USED, n_tokens);
            
            /* Step 5: Repeat cur for each selected expert [H, N_USED, n_tokens] */
            struct ggml_tensor *cur_expert = ggml_repeat_4d(ctx, cur, H, N_USED, n_tokens, 1);
            
            /* Step 6: Multiply by weights */
            cur_expert = ggml_mul(ctx, cur_expert, weights);
            
            /* Step 7: Expert up proj (fused gate+up for efficiency) */
            struct ggml_tensor *gate_up_exps = ggml_mul_mat_id(ctx, l->ffn_up_exps, cur_expert, selected_experts);
            struct ggml_tensor *gate_up_exps_b = NULL; /* No bias for now */
            if (gate_up_exps_b) {
                gate_up_exps = ggml_add_id(ctx, gate_up_exps, gate_up_exps_b, selected_experts);
            }
            
            /* Step 8: Split gate and up */
            int FF = gate_up_exps->ne[0] / 2;
            struct ggml_tensor *gate_expert = ggml_view_3d(ctx, gate_up_exps, FF, N_USED, n_tokens, 
                gate_up_exps->nb[1], gate_up_exps->nb[2], 0);
            struct ggml_tensor *up_expert = ggml_view_3d(ctx, gate_up_exps, FF, N_USED, n_tokens,
                gate_up_exps->nb[1], gate_up_exps->nb[2], FF * gate_up_exps->nb[0]);
            
            /* Step 9: SwiGLU activation */
            struct ggml_tensor *gate_act = ggml_silu(ctx, gate_expert);
            struct ggml_tensor *act = ggml_mul(ctx, gate_act, up_expert);
            
            /* Step 10: Expert down proj */
            struct ggml_tensor *down_expert = ggml_mul_mat_id(ctx, l->ffn_down_exps, act, selected_experts);
            
            /* Step 11: Weighted sum across experts */
            down_expert = ggml_mul(ctx, down_expert, weights);
            cur = ggml_sum_rows(ctx, down_expert);
            cur = ggml_reshape_2d(ctx, cur, H, n_tokens);
        } else {
            /* Dense FFN (original path) */
            struct ggml_tensor *gate = ggml_mul_mat(ctx, l->w_gate, cur);
            struct ggml_tensor *up   = ggml_mul_mat(ctx, l->w_up, cur);
            cur = ggml_mul(ctx, ggml_silu(ctx, gate), up);
            cur = ggml_mul_mat(ctx, l->w_down, cur);
        }'''

content = re.sub(dense_ffn_pattern, moe_ffn_replacement, content)

# Write file
with open('ggml_llama_gguf.c', 'w') as f:
    f.write(content)

print("MoE FFN forward pass added")
