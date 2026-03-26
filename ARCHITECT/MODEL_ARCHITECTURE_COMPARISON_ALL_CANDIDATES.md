# Model Architecture Comparison — All Candidates

## Currently Deployed
| Model | Size | Type | Layers | Hidden | Experts | Active | Node | TPS |
|-------|------|------|--------|--------|---------|--------|------|-----|
| GLM-4.7-Flash-8bit | 32GB | Dense | 47 | 2048 | 0 | 0 | mlx-2,5 | 42 |
| Qwen3-Coder-Next-8bit | 76GB | MoE | 48 | 2048 | 512 | 10 | mlx-4 | 30 |
| Qwen3-Coder-Next-4bit | 40GB | MoE | 48 | 2048 | 512 | 10 | mlx-3,6,7 | 51 |
| Qwen3.5-122B-FP8 | 119GB | MoE | 48 | 3072 | 256 | 8 | CUDA | 120 |

## Sweet Spot Candidates (40-80GB, MoE, fast)
| Model | Size | Type | Layers | Hidden | Experts | Active | Predicted TPS |
|-------|------|------|--------|--------|---------|--------|---------------|
| gpt-oss-120b-MXFP4-Q8 | 58GB | MoE | 36 | 2880 | 128 | 4 | 45-60 (proven) |
| Qwen3.5-122B-A10B-4bit | 63GB | MoE | 48 | 3072 | 256 | 8 | 40 |
| Qwen3-Coder-Next-4bit | 40GB | MoE | 48 | 2048 | 512 | 10 | 51 (proven) |

## Smaller Candidates (targeting 100+ TPS)
| Model | Size | Type | Layers | Hidden | Experts | Active | Predicted TPS |
|-------|------|------|--------|--------|---------|--------|---------------|
| Qwen3.5-35B-A3B-4bit | 18GB | MoE | 40 | 2048 | 256 | 8 | 80-100+ |
| Qwen3.5-35B-A3B-8bit | 36GB | MoE | 40 | 2048 | 256 | 8 | 50-60 |
| Qwen3-Coder-30B-A3B-4bit | 18GB | MoE | 48 | 2048 | 128 | 8 | 80-100+ |
| Qwen3-Coder-30B-A3B-8bit | 32GB | MoE | 48 | 2048 | 128 | 8 | 50-60 |
| GLM-4.7-Flash-4bit | 18GB | Dense | 47 | 2048 | 0 | 0 | 80+ |

## 100 TPS Target
To hit 100 TPS on MLX M1 Ultra (340 GB/s real bandwidth):
- Active params per token must be < 3.4GB (340/100)
- 4-bit model with 3B active params: 3B × 0.5 bytes = 1.5GB → 340/1.5 = 226 TPS theoretical
- With overhead (KV cache, attention): ~100-120 TPS realistic
- Best candidates: Qwen3.5-35B-A3B-4bit, Qwen3-Coder-30B-A3B-4bit
