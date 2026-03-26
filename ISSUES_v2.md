# CLUSTER ISSUES TRACKER v2 — OPERATIONAL EXCELLENCE
# 100 items: making agents PRODUCE CODE, not LOOP
# Updated: 2026-03-26T10:05

## P0: AGENTS MUST STOP LOOPING (1-20)

1. [OPEN] Agent reads entire TASK_QUEUE instead of grep READY — wastes 2-3 turns per restart
2. [OPEN] Agent claims task then re-reads queue instead of working on claimed task
3. [OPEN] Agent output has <think> blocks consuming 50%+ of max_tokens — need to suppress or truncate
4. [OPEN] Agent generates 500+ token reasoning before a 20-token tool call
5. [OPEN] Agent calls claim_task on DONE tasks sequentially instead of skipping
6. [OPEN] Agent context trims but loses what files it already read/wrote
7. [OPEN] Agent restarts lose all progress — no checkpoint mechanism
8. [OPEN] Agent writes code to wrong path (/home/z vs /Users/z vs ~) despite fixes
9. [OPEN] Agent generates multiline heredoc in write_file JSON — still breaks parser
10. [OPEN] Agent calls ask_cuda_brain for trivial questions instead of just executing
11. [OPEN] Agent stuck in "Thinking..." for >60s — no timeout on model generation
12. [OPEN] Agent generates duplicate code (rewrites file it already wrote 5 turns ago)
13. [OPEN] Agent doesn't verify its code compiles before calling complete_task
14. [OPEN] Agent doesn't run tests after writing code
15. [OPEN] Agent claims multiple tasks when API is slow (race condition still exists)
16. [OPEN] sys2 accumulates absurd turn counts (2446) without producing files
17. [OPEN] Agents on macOS can't run Vulkan tests (only sys1 has Asahi)
18. [OPEN] Agent forgets its task after context trim even with nudge
19. [OPEN] TTFT >25s on some nodes despite slim 757-token prompt
20. [OPEN] Agent produces <tool_call> without closing </tool_call> tag

## P1: DASHBOARD & MONITORING (21-40)

21. [OPEN] sys2 Task column shows "t2446" — turn count confused with task ID (FIXED this session)
22. [OPEN] sys3 Model column sometimes shows "-" despite server being UP
23. [OPEN] sys1 Reqs column still "-" — ggml_server doesn't count HTTP requests properly
24. [OPEN] sys1 GPU% always "-" — no Vulkan GPU utilization metric on Asahi
25. [OPEN] Dashboard Arch column truncated with "..." for some models
26. [OPEN] Dashboard refresh takes 8-12s due to SSH to all nodes (should be <3s)
27. [OPEN] Commander node names still show mlx-0, cuda-1 in some places
28. [OPEN] Commander "Agents: 1 running" even when 7 are running
29. [OPEN] fleet_health_check v2 "NO HEARTBEAT" on all remote nodes
30. [OPEN] Watchdog v2 and fleet_health_check overlap — need single source of truth
31. [OPEN] No real-time token/s meter — only average TPS from logs
32. [OPEN] No way to see agent conversation context (what it "knows")
33. [OPEN] Dashboard doesn't show WHICH task the agent completed (only claimed)
34. [OPEN] Task progress % never updates (agents don't call update_progress)
35. [OPEN] count_fleet_tokens.py doesn't parse all log formats correctly
36. [OPEN] No GPU memory utilization tracking (separate from GPU compute %)
37. [OPEN] sys7 shows 56G free disk — running low
38. [OPEN] sys3 disk dropped from 507G to 329G — 170G used in one session
39. [OPEN] No alert when disk drops below 100G
40. [OPEN] Dashboard shows stale TPS from old log entries (not real-time)

## P2: TASK MANAGEMENT (41-60)

41. [OPEN] Task queue has only 6 READY tasks left — need to generate new tasks
42. [OPEN] Completed tasks have messy DONE format with duplicate timestamps
43. [OPEN] No way to add new tasks via API (only manually edit file)
44. [OPEN] Task dependencies not enforced reliably (T58 should wait for T57)
45. [OPEN] No task priority within same phase — agents pick randomly
46. [OPEN] Agents complete tasks without verification (mark DONE = actually done?)
47. [OPEN] Task history log (task_history.jsonl) not being read by any tool
48. [OPEN] No way to re-open a DONE task if it was completed incorrectly
49. [OPEN] Task server /redirect endpoint not tested
50. [OPEN] No periodic task queue health check (detect orphaned IN_PROGRESS)
51. [OPEN] Agent claims task but doesn't start working for 5+ turns
52. [OPEN] No task timeout — IN_PROGRESS tasks stay forever if agent dies
53. [OPEN] Multiple agents claiming same task within 1 second (race window)
54. [OPEN] Complete_task doesn't require push_changes first
55. [OPEN] No way to see how long a task has been IN_PROGRESS
56. [OPEN] POST /block endpoint not tested or used by agents
57. [OPEN] Task queue file growing with accumulated metadata per entry
58. [OPEN] No way to bulk-release stale IN_PROGRESS tasks via API
59. [OPEN] Agents should generate sub-tasks for large tasks
60. [OPEN] No task estimation accuracy tracking (estimated vs actual time)

## P3: CODE QUALITY (61-75)

61. [OPEN] No code review before merge — agents push directly to master
62. [OPEN] No branch per feature/task — everything on master
63. [OPEN] No automated compile check after push_changes
64. [OPEN] agent0 test runner only checks coherency — not agent code quality
65. [OPEN] No linting/formatting enforcement on agent-produced code
66. [OPEN] Agents produce duplicate functions in same file
67. [OPEN] No diff review in push_changes — blind overwrite
68. [OPEN] Agent-produced C code may not compile on Asahi (different from macOS)
69. [OPEN] No test coverage tracking for engine code
70. [OPEN] Agent writes Python 3.12 features that might not work on all nodes
71. [OPEN] No documentation generated for agent-produced code
72. [OPEN] Agent sometimes leaves debug print statements in code
73. [OPEN] No memory safety check for C code changes
74. [OPEN] Agent overwrites other agent's work when pushing same file
75. [OPEN] No git blame tracking — can't tell which agent wrote which line

## P4: INFRASTRUCTURE (76-90)

76. [OPEN] No shared filesystem — agents work on stale local copies
77. [OPEN] SSH key auth not set up — still using passwords
78. [OPEN] task_server runs in tmux — should be systemd (service file exists)
79. [OPEN] ggml_server runs via sg video hack — should be systemd
80. [OPEN] No centralized log aggregation (each node has own logs)
81. [OPEN] No log rotation on remote nodes — logs grow unbounded
82. [OPEN] No backup automation (manual tarball only)
83. [OPEN] No git auto-push to GitHub
84. [OPEN] z4090 orchestrator not connected to fleet health system
85. [OPEN] NFS mount script created but not tested/deployed
86. [OPEN] No monitoring for MLX server crashes on remote nodes
87. [OPEN] Agent deployer still uses old SYNC_FILES list
88. [OPEN] No way to rolling-restart agents without downtime
89. [OPEN] sys7 only 56G free — needs cleanup or model swap
90. [OPEN] No resource limits — agent can OOM the node

## P5: INTELLIGENCE & RESILIENCE (91-100)

91. [OPEN] Agents don't learn from failed approaches — repeat same mistakes
92. [OPEN] No cross-agent knowledge sharing — agent on sys4 can't see sys5's findings
93. [OPEN] No adaptive prompt — same instructions for code tasks and doc tasks
94. [OPEN] Agent doesn't adapt behavior based on model (8bit vs 4bit capabilities)
95. [OPEN] No circuit breaker per model — if model produces garbage, agent retries forever
96. [OPEN] No A/B testing — can't compare agent productivity across different prompts
97. [OPEN] Watchdog restarts agents but doesn't fix the ROOT CAUSE of failures
98. [OPEN] No escalation path — stuck agent just dies and restarts
99. [OPEN] No post-mortem analysis of failed task attempts
100. [OPEN] No feedback loop — completed task quality never verified by another agent

## SUMMARY
- 100 items across 5 priority levels
- P0 (20): Stop looping, make agents produce
- P1 (20): Dashboard and monitoring accuracy
- P2 (20): Task management reliability
- P3 (15): Code quality and safety
- P4 (15): Infrastructure hardening
- P5 (10): Intelligence and resilience
