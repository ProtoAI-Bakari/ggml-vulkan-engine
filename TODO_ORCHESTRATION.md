# ORCHESTRATION TODO — What's NOT done yet
# Updated: 2026-03-26T10:10

## CRITICAL — Agents can't operate reliably without these

- [ ] Agents still don't call push_changes consistently — code written but never synced to sys1
- [ ] Agents still don't call complete_task — tasks stay IN_PROGRESS forever
- [ ] sys3 model shows "qwen-local" not real model — MLX server crashed, agent uses fallback name
- [ ] Task queue has only 6 READY tasks — fleet will idle soon, need new task generation
- [ ] No automatic task release for IN_PROGRESS >30min (orphaned tasks)
- [ ] Heartbeat files (.heartbeat) not created on remote nodes — old code still deployed on some
- [ ] Agent log rotation not working on remote nodes
- [ ] No way to verify agent actually DID the work before marking DONE

## HIGH — Operational quality

- [ ] Commander still shows old node names (mlx-0, cuda-1) in NODES dict
- [ ] Dashboard sys1 Arch column shows "-" (should show Llama-8B architecture)
- [ ] Dashboard sys1 Reqs still "-" (ggml_server doesn't log HTTP reqs)
- [ ] No automated compile-test after push_changes lands on sys1
- [ ] No git auto-push to GitHub (90+ local commits not pushed)
- [ ] Task server not systemd (tmux session, dies on reboot)
- [ ] ggml_server not systemd (tmux/sg hack)
- [ ] agent0 test runner not running (was killed during restarts)
- [ ] Watchdog v2 on z4090 status unknown (other Claude session manages it)
- [ ] Orchestrator on z4090 not integrated with fleet ops

## MEDIUM — Would improve productivity

- [ ] No shared filesystem — agents work on stale copies
- [ ] SSH still password-based — should be key auth
- [ ] No centralized logging (each node has own logs)
- [ ] No A/B testing of different prompts
- [ ] Agents don't read KNOWLEDGE_BASE before starting tasks
- [ ] No cross-agent code review
- [ ] No branch-per-task git workflow
- [ ] No rollback mechanism for bad agent code
- [ ] No alerting (Slack/webhook) when agent completes task or errors
- [ ] count_fleet_tokens.py doesn't aggregate all token types correctly

## LOW — Nice to have

- [ ] No Prometheus/Grafana fleet dashboards
- [ ] No cost tracking for CUDA brain usage
- [ ] No agent efficiency metrics (useful output per token)
- [ ] No historical trend data (TPS over time, tasks/hour over time)
- [ ] No agent memory between sessions
- [ ] scripts/mount_shared.sh created but not deployed
- [ ] systemd service files created but not installed
- [ ] Node standardization (aliases, .zprofile) partially done

## DONE THIS SESSION (for reference)
- [x] Vulkan engine fixed (22 TPS coherent)
- [x] Dashboard v3 with TPS, Reqs, GPU%, Mem, Arch, Task columns
- [x] Central task API with flock locking (task_server.py)
- [x] Fleet health check v2 with heartbeat, stagnancy, completions
- [x] Token counter (count_fleet_tokens.py)
- [x] Agent0 test runner on sys1
- [x] Autonomous watchdog on z4090
- [x] Cluster orchestrator (580 lines)
- [x] 114/150 issues fixed from ISSUES_v1
- [x] 100 new issues documented in ISSUES_v2
- [x] Parse failure nuclear fallback (12/12 tests pass)
- [x] JSON key fixer regex (was crashing all agents)
- [x] sys6/7 switched from 4-bit to 8-bit
- [x] 2.1TB disk freed across fleet
- [x] llama.cpp Vulkan NULL deref fix (upstream-ready)
- [x] GO_PROMPT loop fixed (was reloading every turn)
- [x] System prompt 4251→757 tokens (82% reduction)
- [x] Tool schema 18→10 tools (55% reduction)
- [x] 58/85 engineering tasks completed by agents
- [x] ~8M tokens crunched across fleet
- [x] 95+ git commits this session
