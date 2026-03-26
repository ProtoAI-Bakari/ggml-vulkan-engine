# ProtoAI-Bakari Agentic Cluster — Tool Guide

## Quick Reference

| Tool | Command | Where to Run |
|------|---------|-------------|
| **Commander** | `commander` | Sys10 or Sys0 |
| **Deployer** | `deployer` | Sys10 or Sys0 |
| **Dashboard** | `python3 cluster_dashboard.py` | Any node |
| **CUDA Audit** | `cuda_audit` | Sys0, Sys10, or Sys3 |
| **Flight Deck** | `vllmv12` | .11 (CUDA head) |

---

## swarm_commander.py — Command Center
Controls the swarm from a single terminal.

```bash
commander          # From Sys10 (SSH alias)
python3 ~/AGENT/swarm_commander.py  # From Sys0

# Key commands:
fleet              # Show all nodes + status
act                # Agent activity (local + distributed)
go                 # Launch all agents with auto-go
agentstop all      # Kill all agents
kick 3             # Restart specific agent
council "question" # Ask architect+engineer+designer
reconcile          # Sync task queue with git
logs mlx-5         # Tail node's logs
tasks              # Task queue summary
```

## cluster_deployer.py — Fleet Deployment
Syncs code, installs deps, launches agents across all nodes.

```bash
deployer           # From Sys10

# Key commands:
status             # Check all nodes: server, agent, files, deps
deploy             # FULL: deps + sync + launch on ALL nodes
sync               # Push agent files to all nodes
sync mlx-5 mlx-7   # Push to specific nodes only
launch             # Start agents on all nodes
stop               # Stop agents on all nodes
restart mlx-3      # Stop + sync + relaunch specific node
deps               # Install Python dependencies everywhere
logs mlx-5         # Tail agent log
```

## cluster_dashboard.py — Real-Time TUI
Live-updating view of the entire cluster.

```bash
python3 ~/AGENT/cluster_dashboard.py           # Full dashboard (5s refresh)
python3 ~/AGENT/cluster_dashboard.py --interval 2  # Faster refresh
python3 ~/AGENT/cluster_dashboard.py --compact  # Minimal view

# Shows: Node status, MLX/CUDA servers, agent state, memory, disk, request count
# Colors: Catppuccin Macchiato theme
# Exit: Ctrl+C
```

## cuda_cluster_audit_v16.sh — CUDA Cluster Health
Full audit of RDMA, kernel modules, GPU state, NFS mounts.

```bash
cuda_audit                        # Full audit from any node
bash ~/AGENT/scripts/cuda_cluster_audit_v16.sh --load-all   # Load all perf modules
bash ~/AGENT/scripts/cuda_cluster_audit_v16.sh --aliases    # Print module aliases

# Module management aliases (after sourcing):
cuda_perf_on       # Load gdrdrv+xpmem+knem+peermem on all CUDA nodes
cuda_perf_off      # Unload all
cuda_enable_gdr    # Load gdrdrv only
cuda_disable_peer  # Unload peermem only
cuda_mods          # Quick 4/4 module check
```

## Architecture

```
Sys10 (Workstation)
  └── commander / deployer / dashboard
        │
        ├── mlx-2 (ARCHITECT)  ← local agent + 235B brain
        ├── mlx-3 (ENGINEER)   ← local agent + 122B brain
        ├── mlx-4 (CODER)      ← local agent + Coder-Next brain
        ├── mlx-5 (DESIGNER)   ← local agent + GLM-4.7 brain
        ├── mlx-6 (REVIEWER)   ← local agent + 122B brain
        ├── mlx-7 (FAST-CODER) ← local agent + Coder-Next-4b brain
        │
        └── CUDA Cluster (.11-.14)
              └── Qwen3.5-122B-FP8 at TP8 (heavy lifting)
```

Each Mac Studio runs its OWN agent using its OWN local MLX brain.
CUDA cluster available to all agents for heavy code generation.
"""
