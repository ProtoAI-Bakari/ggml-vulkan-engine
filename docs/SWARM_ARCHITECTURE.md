# SWARM ARCHITECTURE — Multi-Agent Multi-Machine Framework

## The Problem
One agent on one machine is too slow. We have 7 machines with ~960GB total memory.
We need multiple agents working in PARALLEL on DIFFERENT subtasks, all coordinated.

## Architecture: Hub-and-Spoke with Autonomous Workers

```
                    ┌─────────────────┐
                    │   LEAD AGENT    │
                    │  (Claude/Opus)  │
                    │  Orchestrator   │
                    └────────┬────────┘
                             │ HTTP dispatch
            ┌────────────────┼────────────────┐
            │                │                │
     ┌──────┴──────┐  ┌─────┴──────┐  ┌──────┴──────┐
     │  WORKER .4  │  │ WORKER .5  │  │ WORKER .6   │
     │ Coder Brain │  │ Reasoner   │  │ Code Review │
     │ OMNIAGENT   │  │ OMNIAGENT  │  │ OMNIAGENT   │
     └─────────────┘  └────────────┘  └─────────────┘
            │                │                │
     ┌──────┴──────┐  ┌─────┴──────┐  ┌──────┴──────┐
     │ MLX Server  │  │ MLX Server │  │ MLX Server  │
     │ Qwen-Coder  │  │ DeepSeek-R1│  │ Qwen-Coder  │
     └─────────────┘  └────────────┘  └─────────────┘
```

## Each Node Runs:
1. **MLX inference server** (port 8000) — the brain
2. **OMNIAGENT_v4** — autonomous task executor
3. **Task queue** — reads from shared NFS or HTTP API
4. **Results publisher** — writes to shared bridge file

## How It Works:

### Step 1: Lead Decomposes Task
Lead (Claude or OMNIAGENT on .11) breaks a big task into subtasks:
```
TASK: "Add MoE support to ggml engine"
  → SUBTASK-1: "Write MoE routing C code" → .4 (coder)
  → SUBTASK-2: "Design expert weight layout" → .5 (reasoner)
  → SUBTASK-3: "Write test harness" → .6 (coder)
  → SUBTASK-4: "Benchmark current MoE in llama.cpp" → .128 (compute)
```

### Step 2: Workers Execute Independently
Each OMNIAGENT:
- Reads its assigned subtask
- Queries its LOCAL model for solutions
- Executes code locally (compile, test)
- Writes results to shared storage

### Step 3: Lead Collects and Integrates
Lead reads all results, merges code, resolves conflicts, tests integration.

## Communication Protocol:
- **Task assignment**: Write JSON to `/repo/swarm/tasks/{node}.json`
- **Results**: Write to `/repo/swarm/results/{node}_{task_id}.json`
- **Status**: Each node writes heartbeat to `/repo/swarm/status/{node}.json`
- **Code sharing**: All code on shared NFS (`/repo/AGENT/` or `/slowrepo/`)

## OMNIAGENT_v4 Config Per Node:

### .4 (Coder)
```python
PRIMARY_IP = "10.255.255.4"  # Query self for code
CODER_IP = "10.255.255.4"   # Self
PORT = "8000"
ROLE = "coder"
TASK_DIR = "/repo/swarm/tasks/sys4/"
```

### .5 (Reasoner)
```python
PRIMARY_IP = "10.255.255.5"  # Query self for reasoning
CODER_IP = "10.255.255.4"   # Ask .4 for code help
PORT = "8000"
ROLE = "reasoner"
TASK_DIR = "/repo/swarm/tasks/sys5/"
```

## Parallel Task Categories:
| Category | Best Node | Why |
|----------|-----------|-----|
| Write C code | .4 (Coder) | Dedicated code model |
| Write shaders (GLSL) | .4 (Coder) | Code generation |
| Architecture design | .11 (122B) | Deep reasoning |
| Debug crashes | .11 (122B) | Root cause analysis |
| Compile & test | .128 (Sys0) | Vulkan hardware |
| Benchmark | .128 (Sys0) | GPU access |
| Code review | .6 (if online) | Second coder |
| Research (web) | .11 or MBP | Internet access |

## Scaling: 5 Agents = 5x Throughput
With 5 nodes running OMNIAGENT simultaneously:
- 5 subtasks execute in parallel
- Each takes ~5-10 min instead of 25-50 min sequential
- Total sprint velocity: 5x faster

## Quick Start:
1. Boot .5, .6, .7
2. Deploy MLX models on each
3. Copy OMNIAGENT_v4 to each with node-specific config
4. Create shared task directory on NFS
5. Lead dispatches tasks → Agents execute → Lead integrates
