#!/usr/bin/env python3
"""Apply MoE fix to ggml_llama_gguf.c"""

# Read the file
with open('ggml_llama_gguf.c', 'r') as f:
    content = f.read()

# Find and replace the MoE section (lines 393-448)
old_moe_code = '''        /* MoE FFN if experts present, else dense FFN (T07) */
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
        } else {'''

new_moe_code = '''        /* MoE FFN if experts present, else dense FFN (T07 - FIXED) */
        if (l->ffn_gate_exps) {
            /* MoE path: router -> topK -> per-token expert routing -> weighted sum */
            int N_EXPERT = l->expert_count;
            int N_USED = l->expert_used;
            int H = e->hidden_dim;
            
            /* Step 1: Router (gate_inp) -> expert logits [N_EXPERT, n_tokens] */
            struct ggml_tensor *router_logits = ggml_mul_mat(ctx, l->ffn_gate_inp, cur);
            
            /* Step 2: TopK selection - get top-N_USED expert indices per token */
            /* argsort_top_k returns [N_USED, n_tokens] of int32 indices */
            struct ggml_tensor *topk_indices = ggml_argsort_top_k(ctx, router_logits, N_USED);
            
            /* Step 3: Get router weights for selected experts */
            /* Get the logits for selected experts: [N_USED, n_tokens] */
            struct ggml_tensor *topk_logits = ggml_get_rows(ctx, router_logits, topk_indices);
            topk_logits = ggml_reshape_2d(ctx, topk_logits, N_USED, n_tokens);
            
            /* Softmax to get expert weights */
            struct ggml_tensor *expert_weights = ggml_soft_max(ctx, topk_logits);
            
            /* Step 4: Initialize output accumulator */
            struct ggml_tensor *moe_out = ggml_dup(ctx, cur);
            
            /* Step 5: For each expert rank, compute contribution */
            for (int k = 0; k < N_USED; k++) {
                /* Get expert indices for this rank: [n_tokens] */
                struct ggml_tensor *expert_idx_k = ggml_view_1d(ctx, topk_indices, n_tokens, k * topk_indices->nb[1]);
                
                /* Get weights for this rank: [n_tokens] */
                struct ggml_tensor *weights_k = ggml_view_1d(ctx, expert_weights, n_tokens, k * expert_weights->nb[1]);
                
                /* Expand weights to [1, n_tokens] for broadcasting */
                weights_k = ggml_reshape_2d(ctx, weights_k, 1, n_tokens);
                
                /* Step 6: Expert up proj using ggml_mul_mat_id */
                /* ffn_up_exps: [H, FF, N_EXPERT], cur: [H, n_tokens], expert_idx_k: [n_tokens] */
                struct ggml_tensor *up_out = ggml_mul_mat_id(ctx, l->ffn_up_exps, cur, expert_idx_k);
                
                /* Step 7: Split gate and up (fused gate+up) */
                int FF = l->ffn_up_exps->ne[0] / 2;
                struct ggml_tensor *gate_out = ggml_view_2d(ctx, up_out, FF, n_tokens, 
                    up_out->nb[1], 0);
                struct ggml_tensor *up_out_split = ggml_view_2d(ctx, up_out, FF, n_tokens,
                    up_out->nb[1], FF * up_out->nb[0]);
                
                /* Step 8: SwiGLU activation */
                struct ggml_tensor *gate_act = ggml_silu(ctx, gate_out);
                struct ggml_tensor *act = ggml_mul(ctx, gate_act, up_out_split);
                
                /* Step 9: Expert down proj */
                /* ffn_down_exps: [FF, H, N_EXPERT], act: [FF, n_tokens] */
                struct ggml_tensor *down_out = ggml_mul_mat_id(ctx, l->ffn_down_exps, act, expert_idx_k);
                
                /* Step 10: Apply expert weight and accumulate */
                down_out = ggml_mul(ctx, down_out, weights_k);
                moe_out = ggml_add(ctx, moe_out, down_out);
            }
            
            cur = moe_out;
        } else {'''

if old_moe_code not in content:
    print("ERROR: Could not find old MoE code block!")
    exit(1)

content = content.replace(old_moe_code, new_moe_code)

# Write back
with open('ggml_llama_gguf.c', 'w') as f:
    f.write(content)

print("MoE fix applied successfully!")
