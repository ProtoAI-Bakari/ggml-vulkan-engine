#!/usr/bin/env python3
"""Patch vLLM utils.py to fix dtype mismatch in scatter_add."""

import re

utils_py_path = "/home/z/GITDEV/vllm_0.17.1/vllm/model_executor/layers/utils.py"

with open(utils_py_path, 'r') as f:
    content = f.read()

# Find and fix the get_token_bin_counts_and_mask function
old_code = '''def get_token_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = torch.zeros(
        (num_seqs, vocab_size + 1), dtype=torch.long, device=tokens.device
    )
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask'''

new_code = '''def get_token_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = torch.zeros(
        (num_seqs, vocab_size + 1), dtype=torch.long, device=tokens.device
    )
    # FIX: Ensure dtype matches bin_counts (torch.long) for scatter_add
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens, dtype=torch.long))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask'''

if old_code not in content:
    print("ERROR: Could not find the target code block")
    exit(1)

new_content = content.replace(old_code, new_code)

with open(utils_py_path, 'w') as f:
    f.write(new_content)

print("SUCCESS: Patched utils.py to fix dtype mismatch in scatter_add")
