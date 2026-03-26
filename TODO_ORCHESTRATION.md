# ORCHESTRATION & AGENTIC FRAMEWORK TODO
# Created: 2026-03-26 10:10
# 58/85 Vulkan tasks DONE, 118/118 infra issues DONE

## P0: Fleet Can't Self-Heal
1. [ ] Watchdog nohup restart broken on macOS (tmux quoting)
2. [ ] sys1 hangs on CUDA brain stream stall (timeout added, needs testing)
3. [ ] Mock server port conflicts (agents hit stubs not real MLX)
4. [ ] Zero-token loop auto-fix (detection added, no remediation)
5. [ ] 397B OOM on 192GB — need PP2 across sys2+sys3

## P1: Agents Waste Compute  
6. [ ] Agents call ./claim_task.sh in bash bypassing BLOCKED logic
7. [ ] CUDA 122B still outputs think blocks sometimes
8. [ ] Agent task hoarding (BLOCKED check name matching)
9. [ ] No task priority (random READY selection)
10. [ ] complete_task never called (tasks stuck at 90%)
11. [ ] Context trim loses task assignment
12. [ ] GLM-4.7 on sys2 returns empty responses
13. [ ] No file cache between turns

## P2: Orchestration Gaps
14. [ ] No PP/TP distributed serving for large models
15. [ ] No compile+test pipeline (C code written, .so never rebuilt)
16. [ ] C engine T79 broke backend_sched (CPU backend removed)
17. [ ] Watchdog doesn't sync code before restart
18. [ ] No load balancing across agents
19. [ ] Dashboard task column shows turn count not task ID
20. [ ] No shared git across fleet
21. [ ] Fleet restart script (sync code + start all agents correctly)

## P3: Intelligence
22. [ ] Agents don't read KNOWLEDGE_BASE before tasks
23. [ ] No peer review of agent code
24. [ ] No cross-agent coordination or file sharing
25. [ ] No planning phase before coding
26. [ ] No request routing (hard→122B, easy→30B)

## P4: Production
27. [ ] Passwords in git history
28. [ ] No TLS/firewall between nodes
29. [ ] Task/Vulkan servers in tmux not systemd
30. [ ] No backup schedule
31. [ ] SSH should be key-based

## P5: Scaling
32. [ ] CUDA sys2/vm3/vm4 idle (no agents)
33. [ ] z4090 4090 running model but no agent
34. [ ] No auto-discovery of fleet members
35. [ ] No model registry
36. [ ] No batching/preemption for GPU throughput

## QUICK WINS (<30min each)
- [ ] Fleet restart script with code sync
- [ ] Auto complete_task after push_changes
- [ ] Watchdog auto-restart Vulkan server
- [ ] Token count aggregation endpoint
- [ ] PP2 launch script for 397B on sys2+sys3
- [ ] Add --host 0.0.0.0 to all MLX launches
