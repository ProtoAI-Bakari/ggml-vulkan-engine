# Fleet Optimization Plan v1

## Current Fleet (after today's swaps)
| Node | Model | TPS | Role | Issues |
|------|-------|-----|------|--------|
| mlx-2 (M2U 192G) | GLM-4.7-Flash-8bit | 42 | ARCHITECT | Verbose — too many tokens |
| mlx-3 (M2U 192G) | Coder-Next-4bit | 50 | CODER | 12s TTFT |
| mlx-4 (M1U 128G) | Coder-Next-8bit | 30 | CODER | Best TTFT (0.5s) but slow gen |
| mlx-5 (M1U 128G) | GLM-4.7-Flash-8bit | 42 | DESIGNER | Good balance |
| mlx-6 (M1U 128G) | Coder-Next-4bit | 50 | CODER | 12s TTFT |
| mlx-7 (M1U 128G) | Coder-Next-4bit | 51 | FAST-CODER | 12s TTFT |
| CUDA .11 (8x3090) | Qwen3.5-122B-FP8 | 120 | MASTER BRAIN | Best by far |

## Proposed Optimization
| Node | New Model | Expected TPS | Role |
|------|-----------|-------------|------|
| mlx-2 (M2U 192G) | GLM-4.7-Flash-8bit | 42-60 | ARCHITECT (query only, not task runner) |
| mlx-3 (M2U 192G) | Qwen3.5-35B-A3B-4bit | 80-100 | FAST ENGINEER |
| mlx-4 (M1U 128G) | Qwen3-Coder-30B-A3B-4bit | 80-100 | FAST CODER |
| mlx-5 (M1U 128G) | gpt-oss-120b-MXFP4-Q8 | 45-60 | SMART CODER |
| mlx-6 (M1U 128G) | Qwen3.5-35B-A3B-4bit | 80-100 | FAST REVIEWER |
| mlx-7 (M1U 128G) | Qwen3-Coder-30B-A3B-4bit | 80-100 | FAST CODER |

## Target: 100+ TPS per node on small MoE models
## Combined fleet throughput: ~500+ TPS across 6 MLX nodes + 120 TPS CUDA
