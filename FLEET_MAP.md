# Fleet Map — Updated 2026-03-25

## Brain Servers (MLX/vLLM inference)

| System | IP | Model | Port | TPS | Role |
|--------|-----|-------|------|-----|------|
| **Sys0** | 127.0.0.1 | Qwen2.5-3B Q4_K_M | 8081 | 22 | Fast ggml rewrite server |
| **Sys4** | 10.255.255.4 | Qwen3-Coder-Next-8bit | 8000 | ~50 | Primary coder brain |
| **Sys5** | 10.255.255.5 | gpt-oss-120b-MXFP4-Q8 | 8000 | 61 | Heavy reasoning |
| **Sys6** | 10.255.255.6 | gpt-oss-120b-MXFP4-Q8 | 8000 | 61 | Heavy reasoning |
| **Sys7** | 10.255.255.7 | gpt-oss-120b-MXFP4-Q8 | 8000 | 61 | Heavy reasoning |
| **Cluster** | 10.255.255.11 | Qwen3.5-122B-A10B-FP8 | 8000 | ~30 | Architecture brain (8x3090) |
| **MBP164** | 192.168.1.164 | MiniMaxM2 | 8765 | ~20 | Mobile brain |

## Agent Fleet

| Agent | Script | Brain | Status |
|-------|--------|-------|--------|
| OmniAgent Main | OMNIAGENT_v4_focused.py | All brains | Primary on Sys0 |
| OmniAgent Sys4 | agents/OMNIAGENT_v4_sys4.py | Coder (.4) + 122B (.11) | Secondary |
| OmniAgent Cluster2 | agents/OMNIAGENT_v4_cluster2.py | 122B (.11) + Coder (.4) | Secondary |

## Hardware

| System | Chip | RAM | Disk | OS |
|--------|------|-----|------|-----|
| Sys0 | M1 Ultra | 128GB | 2TB | Asahi Linux (Fedora 42) |
| Sys4 | M1 Ultra | 128GB | ? | macOS |
| Sys5 | M1 Ultra | 128GB | 926GB | macOS |
| Sys6 | M1 Ultra | 128GB | 1.8TB | macOS |
| Sys7 | M1 Ultra | 128GB | 1.8TB | macOS |
| Cluster | 8x3090 | 256GB | ? | Linux |

## Quick Commands

```bash
# Check all brains
./deploy_fleet.sh status

# Start/stop new fleet
./deploy_fleet.sh start
./deploy_fleet.sh stop

# Query any brain
python3 brain_bridge.py --brain sys5 "your question"
python3 brain_bridge.py --brain sys5,sys6,sys7 --discuss "topic"
```
