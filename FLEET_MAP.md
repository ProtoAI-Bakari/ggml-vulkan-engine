# FLEET MAP — Updated 2026-03-25

## COMPUTE NODES
| IP | Machine | RAM | GPU | Current Role | Available |
|----|---------|-----|-----|-------------|-----------|
| .11 | CUDA Cluster | 8x24GB | 8x3090 (192GB VRAM) | 122B reasoning brain | YES — multi-GPU |
| .4 | M1 Ultra 128G | 128GB | 64-core AGX | Qwen3-Coder-Next | YES — can dual-model |
| .5 | M1 Ultra 128G | 128GB | 64-core AGX | IDLE | YES — deploy model |
| .6 | M1 Ultra 128G | 128GB | 64-core AGX | IDLE | YES — deploy model |
| .7 | M1 Ultra 128G | 128GB | 64-core AGX | IDLE | YES — deploy model |
| .128 | M1 Ultra 128G | 128GB | 64-core AGX (Vulkan) | ggml engine dev | YES |
| MBP | M3 Max 128G | 128GB | 40-core GPU | MiniMaxM2 (waste) | YES — redeploy |

## TOTAL FLEET CAPACITY
- 5x M1 Ultra 128G = 640GB unified memory
- 1x M3 Max 128G = 128GB
- 1x CUDA cluster = 192GB VRAM + system RAM
- **Total: ~960GB+ of model-capable memory**

## PROPOSED DEPLOYMENT
| Node | Model | Role | TPS (est) |
|------|-------|------|-----------|
| .11 | Qwen3.5-122B-A10B | Architecture/Reasoning brain | 30-40 |
| .4 | Qwen3-Coder-Next-32B | Code generation | 30-40 |
| .5 | DeepSeek-R1-0528-32B-Q4 | Reasoning/debugging | 30-40 |
| .6 | Qwen2.5-Coder-32B-Q4 | Code review/testing | 30-40 |
| .7 | gpt-oss-120b-mxfp4 | Large model testing | 15-20 |
| .128 | Vulkan engine (any GGUF) | Vulkan dev/benchmark | 22 |
| MBP | Codestral-25.01-22B-Q4 | Fast code assistant | 40-50 |

## AGENT ARCHITECTURE
Each node runs:
1. MLX/llama.cpp inference server (port 8000)
2. OMNIAGENT_v4 (local task execution)
3. HTTP API for cross-node task dispatch

Agents communicate via:
- HTTP /v1/chat/completions (model queries)
- SSH + shared NFS (/repo, /slowrepo) for file sharing
- ~/AGENT/agent-comms-bridge.md for coordination
