/* MoE FFN if experts present, else dense FFN (T07 - FIXED) */
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
        } else {
            /* Dense FFN (original path) */
            struct ggml_tensor *gate = ggml_mul_mat(ctx, l->w_gate, cur);
            struct ggml_tensor *up   = ggml_mul_mat(ctx, l->w_up, cur);
            cur = ggml_mul(ctx, ggml_silu(ctx, gate), up);
            cur = ggml_mul_mat(ctx, l->w_down, cur);
        }
