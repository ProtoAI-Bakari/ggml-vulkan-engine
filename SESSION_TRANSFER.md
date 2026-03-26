# Session Transfer — 2026-03-26 03:15

## WHAT'S RUNNING RIGHT NOW

### Fleet (12 nodes)
- **sys1** (.128): Asahi M1 Ultra 128G — ggml Vulkan server on :8080 (22 TPS Llama-8B), Main agent on CUDA brain
- **sys2** (.2): M2 Ultra 192G — GLM-4.7-Flash-8bit, agent running, T16 claimed
- **sys3** (.3): M2 Ultra 192G — Qwen3-Coder-Next-8bit, agent running, T28 claimed
- **sys4** (.4): M1 Ultra 128G — Qwen3-Coder-Next-8bit, agent running, T44 claimed
- **sys5** (.5): M1 Ultra 128G — Qwen3-Coder-Next-8bit, agent running, T38 claimed
- **sys6** (.6): M1 Ultra 128G — Qwen3-Coder-Next-4bit, agent running, T42 claimed
- **sys7** (.7): M1 Ultra 128G — Qwen3-Coder-Next-4bit, agent running, T34 claimed
- **cuda-sys1** (.11): 8x3090 TP8 — Qwen3.5-122B-A10B-FP8, 42K ctx, FP8 KV, FlashInfer, max_num_seqs=4
- **z4090** (.10): RTX 4090 — qwen3-coder-30b at 68 TPS, orchestrator deployed

### Services
- Task API: sys1:9091 (central task locking, flock-based)
- Task server: tmux session `taskserver` on sys1
- Vulkan server: port 8080 on sys1 (ggml engine, Llama-8B Q4_K_M)
- Orchestrator: tmux session `orchestrator` on z4090

### Key Files
- `~/AGENT/OMNIAGENT_v4_focused.py` — agent code (all nodes)
- `~/AGENT/task_server.py` — central task API
- `~/AGENT/cluster_dashboard.py` — TUI dashboard v3
- `~/AGENT/cluster_orchestrator.py` — z4090 orchestrator
- `~/AGENT/fleet_health_check.py` — agent health diagnostic
- `~/AGENT/swarm_commander.py` — CLI commander
- `~/AGENT/cluster_deployer.py` — fleet deployer
- `~/AGENT/ggml_llama_gguf.c` — Vulkan C engine (22 TPS)
- `~/AGENT/ggml_vllm_backend.py` — Python wrapper
- `~/AGENT/ggml_server.py` — HTTP server for Vulkan engine
- `~/AGENT/TASK_QUEUE_v5.md` — master task queue (37 DONE, ~40 READY)

## WHAT WAS FIXED THIS SESSION

### Critical Fixes
1. **Vulkan engine gibberish** — TWO bugs: (a) llama.cpp NULL deref in matmul pipeline for Apple Vulkan, (b) our engine mixed ggml_gallocr with ggml_backend_sched. Now 22.7 TPS coherent.
2. **Stop token leak** — `<|eot_id|>` showed in Streamlit. Fixed: check stop IDs before streaming.
3. **Task claiming race condition** — central HTTP API with flock locking on sys1:9091
4. **Agent GO_PROMPT loop** — was reloading full GO_PROMPT every turn in no-TTY. Now: first turn only, subsequent turns get task-aware nudge.
5. **v4/v5 mismatch** — all references to TASK_QUEUE_v4 purged
6. **Path mismatch** — /home/z vs /Users/z fixed in read_file/write_file
7. **System prompt bloat** — cut from 4251 to 232 tokens (TTFT 57s → 20s)
8. **Context overflow** — trim history on 400 error instead of infinite retry
9. **Multi-claim bug** — agents hoarding 2-8 tasks. Released duplicates, 1 per agent.

### Infrastructure Built
- Dashboard v3: TPS, Reqs, GPU%, Mem, Disk, Arch (layers/hidden/MoE), Task column from API
- Fleet health checker: detects looping, errors, stalls across all nodes
- Central task API with progress tracking (%, timestamps)
- push_changes tool for remote agents to SCP files back to sys1
- update_progress tool for agents to report completion %
- Cluster orchestrator (580 lines, dry-run default, guardrails)
- 2.1TB disk freed across sys2/3/5/z4090

## KNOWN ISSUES / TODO
1. **Agent TPS is 0.5-2 effective** — 30-50s TTFT due to tool schema overhead (17 tools × ~280 chars each). Raw model is 48-58 TPS but agent overhead kills it. Fix: reduce tool count or make descriptions shorter.
2. **Agents claim wrong tasks** — the BLOCKED check uses partial agent name match that sometimes fails. Fix the regex in task_server.py.
3. **Stream errors when CUDA is down** — agents retry forever. The context overflow fix handles 400 but not connection errors well enough.
4. **sys1 agent shares GPU with Vulkan server** — both compete for M1 Ultra GPU. sys1 agent should use CUDA only.
5. **Commander still uses old node names** (mlx-X). Added NODE_ALIASES but not fully tested.
6. **No agent on z4090** — orchestrator runs but no OMNIAGENT. Could be another worker.
7. **Remote agents can't git commit** — push_changes SCPs files to sys1 but nobody commits them.
8. **ask_claude escalation** — added but untested. Triggers after 3 parse failures.

## PASSWORDS/AUTH
- MLX nodes (.2-.7): `sshpass -f ~/DEV/authpass`
- CUDA nodes (.11-.14) + z4090 (.10): `sshpass -p z`
- NEVER echo passwords in chat. Use `cat ~/DEV/authpass | sudo -S` for sudo.

## COMMANDS
```bash
# Dashboard
python3 cluster_dashboard.py

# Commander
python3 swarm_commander.py

# Health check
python3 fleet_health_check.py

# Deploy to all nodes
python3 cluster_deployer.py restart

# Wype a node (kills all python/ray/vllm)
sshpass -f ~/DEV/authpass ssh z@10.255.255.X "source ~/.zshrc; wype; clean"

# View agent log
tail -f ~/AGENT/LOGS/main_trace.log                    # sys1
sshpass -f ~/DEV/authpass ssh z@10.255.255.X "tail -f ~/AGENT/LOGS/agent_trace.log"  # remote

# Task queue
curl http://localhost:9091/tasks/summary
```

## GIT STATUS
- Branch: master, 100+ commits this session
- GitHub: ProtoAI-Bakari/agentic-cluster (PUBLIC — passwords in history, needs scrub)
- Last commit: fleet_health_check.py
