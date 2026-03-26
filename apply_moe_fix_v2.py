#!/usr/bin/env python3
"""Apply correct MoE fix to ggml_llama_gguf.c based on llama.cpp implementation"""

# Read the file
with open('ggml_llama_gguf.c', 'r') as f:
    lines = f.readlines()

# Find the MoE section (around line 393-448)
start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if '/* MoE FFN if experts present, else dense FFN (T07 - FIXED) */' in line:
        start_idx = i
    if start_idx and '        } else {' in line and i > start_idx + 10:
        end_idx = i
        break

if start_idx is None or end_idx is None:
    print("ERROR: Could not find MoE section!")
    exit(1)

print(f"Found MoE section at lines {start_idx+1} to {end_idx+1}")

# New correct MoE implementation based on llama.cpp
new_moe_code = '''        /* MoE FFN if experts present, else dense FFN (T07 - CORRECTED) */
        if (l->ffn_gate_exps) {
            /* MoE path: router -> topK -> expert FFN -> weighted sum */
            int N_EXPERT = l->expert_count;
            int N_USED = l->expert_used;
            int H = e->hidden_dim;
            
            /* Step 1: Router (gate_inp) -> expert logits [N_EXPERT, n_tokens] */
            struct ggml_tensor *logits = ggml_mul_mat(ctx, l->ffn_gate_inp, cur);
            
            /* Step 2: TopK selection - get top-N_USED expert indices per token */
            /* argsort_top_k returns [N_USED, n_tokens] of int32 indices */
            struct ggml_tensor *selected_experts = ggml_argsort_top_k(ctx, logits, N_USED);
            
            /* Step 3: Get expert weights for selected experts */
            struct ggml_tensor *weights = ggml_get_rows(ctx, logits, selected_experts);
            weights = ggml_reshape_2d(ctx, weights, N_USED, n_tokens);
            weights = ggml_soft_max(ctx, weights); /* Normalize weights */
            weights = ggml_reshape_3d(ctx, weights, 1, N_USED, n_tokens);
            
            /* Step 4: Reshape cur to [H, 1, n_tokens] for mul_mat_id */
            struct ggml_tensor *cur_3d = ggml_reshape_3d(ctx, cur, H, 1, n_tokens);
            
            /* Step 5: Expert up proj (fused gate+up) using mul_mat_id */
            /* ffn_up_exps: [H, FF*2, N_EXPERT], cur_3d: [H, 1, n_tokens], selected_experts: [N_USED, n_tokens] */
            struct ggml_tensor *gate_up_exps = ggml_mul_mat_id(ctx, l->ffn_up_exps, cur_3d, selected_experts);
            /* Output: [FF*2, N_USED, n_tokens] */
            
            /* Step 6: Split gate and up */
            int FF = gate_up_exps->ne[0] / 2;
            struct ggml_tensor *gate_expert = ggml_view_3d(ctx, gate_up_exps, FF, N_USED, n_tokens, 
                gate_up_exps->nb[1], gate_up_exps->nb[2], 0);
            struct ggml_tensor *up_expert = ggml_view_3d(ctx, gate_up_exps, FF, N_USED, n_tokens,
                gate_up_exps->nb[1], gate_up_exps->nb[2], FF * gate_up_exps->nb[0]);
            
            /* Step 7: SwiGLU activation */
            struct ggml_tensor *gate_act = ggml_silu(ctx, gate_expert);
            struct ggml_tensor *act = ggml_mul(ctx, gate_act, up_expert);
            /* act: [FF, N_USED, n_tokens] */
            
            /* Step 8: Expert down proj using mul_mat_id */
            /* ffn_down_exps: [FF, H, N_EXPERT], act: [FF, N_USED, n_tokens] */
            struct ggml_tensor *down_experts = ggml_mul_mat_id(ctx, l->ffn_down_exps, act, selected_experts);
            /* Output: [H, N_USED, n_tokens] */
            
            /* Step 9: Apply expert weights */
            down_experts = ggml_mul(ctx, down_experts, weights);
            
            /* Step 10: Sum across experts (view each expert and add) */
            struct ggml_tensor *moe_out = ggml_view_3d(ctx, down_experts, H, 1, n_tokens,
                down_experts->nb[1], down_experts->nb[2], 0);
            for (int k = 1; k < N_USED; k++) {
                struct ggml_tensor *expert_k = ggml_view_3d(ctx, down_experts, H, 1, n_tokens,
                    down_experts->nb[1], down_experts->nb[2], k * down_experts->nb[1]);
                moe_out = ggml_add(ctx, moe_out, expert_k);
            }
            cur = moe_out;
        } else {'''

# Replace the old code
new_lines = lines[:start_idx] + [new_moe_code + '\n'] + lines[end_idx:]

# Write back
with open('ggml_llama_gguf.c', 'w') as f:
    f.writelines(new_lines)

print("MoE fix applied successfully!")
