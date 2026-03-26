# Model Benchmark Results — 2026-03-25

## Test: 2 prompts (Python linked list, C SIMD matrix multiply)

| Model | Node | Avg TPS | Avg TTFT | Avg Tokens | Code% | Type |
|-------|------|---------|----------|------------|-------|------|
| Qwen3-Coder-Next-8bit | mlx-4 | 27.1 | 0.50s | 754 | 100% | MoE 512e/10a |
| Qwen3.5-122B (was mlx-6) | - | 27.6 | 7.05s | 1251 | 100% | MoE 256e/8a |
| Coder-Next-4bit | mlx-7 | 25.3 | 12.23s | 323 | 100% | MoE 512e/10a |
| GLM-4.7-Flash-8bit | mlx-5 | 24.0 | 0.84s | 3216 | 100% | Dense |
| 235B-Thinking | mlx-2 | 16.2 | 3.31s | 624 | 100% | MoE 256e/8a |
| CUDA 122B-FP8 | .11 | 120+ | <1s | - | - | MoE 256e/8a |

## Key Findings
1. Coder-Next-8bit has best TTFT (0.50s) — instant response
2. Coder-Next-4bit has 12s TTFT — painful for agents
3. GLM-4.7-Flash produces 4549 tokens for a linked list — verbose
4. 122B models too slow for MLX agent work (0.3 TPS prefill)
5. CUDA cluster: 1518 TPS prefill, 120 TPS gen — 4-5x faster than any MLX
