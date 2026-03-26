# CLUSTER ISSUES TRACKER v1
# 100 issues to fix — prioritized
# Updated: 2026-03-26 04:00

## P0: CRITICAL (agents broken/looping)

1. [OPEN] JSON parse failures from embedded newlines in write_file content — heredocs break JSON
2. [OPEN] Agents try to claim DONE tasks in sequence instead of skipping to READY
3. [OPEN] BLOCKED response doesn't always stop agent from re-claiming
4. [OPEN] Context overflow on CUDA (42K limit) causes infinite retry loop
5. [OPEN] Agents lose task assignment after context trim — nudge can't find task
6. [OPEN] Remote agents can't compile/test code — only sys1 has the Vulkan GPU
7. [OPEN] GO_PROMPT on first turn eats too many tokens reading full queue
8. [OPEN] Tool schema sent every turn adds ~525 tokens overhead even when not needed
9. [OPEN] Multiple agents claim same task when API is slow/unreachable
10. [OPEN] Stream errors during CUDA restart cause all agents using .11 to loop

## P1: HIGH (agents inefficient)

11. [OPEN] Agents read TASK_QUEUE_v5.md fully instead of grep READY | head -3
12. [OPEN] Agents re-read files they already read 5 turns ago
13. [OPEN] No file cache — agent reads same file every turn after context trim
14. [OPEN] Agent writes code but never calls push_changes to sync to sys1
15. [OPEN] Agent writes code but never calls complete_task when done
16. [OPEN] Agent calls ask_cuda_brain for simple questions (wastes shared resource)
17. [OPEN] Agent generates 2000 tokens of reasoning before a 20-token tool call
18. [OPEN] write_file with large content (>500 lines) always fails JSON parse
19. [OPEN] Agent doesn't update_progress — task stays at 0% forever
20. [OPEN] No way to tell agent "stop working on this task, do X instead"

## P2: MEDIUM (dashboard/monitoring gaps)

21. [OPEN] Dashboard TPS column shows 0 for sys1 (ggml server logs differently)
22. [OPEN] Dashboard Reqs column shows 0 for sys1
23. [OPEN] Dashboard Arch column fails on some models (binary log issue)
24. [OPEN] Dashboard GPU% uses powermetrics (needs sudo, adds 500ms per node)
25. [OPEN] Commander still uses old node names (mlx-0, cuda-1) not sys1, cuda-sys1
26. [OPEN] Commander agent detection shows wrong count in "Agents: 1 running"
27. [OPEN] Fleet health checker doesn't detect "agent writing but no files produced"
28. [OPEN] No dashboard column for "last file written" or "last commit"
29. [OPEN] Task server timestamps accumulate — "| 0% | started:... | 0% | started:..."
30. [OPEN] Task server BLOCKED check regex fragile with bracket-heavy agent names

## P3: AGENT LIFECYCLE

31. [OPEN] No graceful agent shutdown — pkill -9 loses in-flight work
32. [OPEN] Agent doesn't save state between restarts — loses context
33. [OPEN] Agent can't resume a partially-completed task after restart
34. [OPEN] No heartbeat mechanism — can't tell if agent is alive vs thinking
35. [OPEN] Agent log grows unbounded — no rotation
36. [OPEN] agent_trace.log has ANSI escape codes making grep/analysis painful
37. [OPEN] No way to inject a message into a running agent's context
38. [OPEN] Agents on macOS use /home/z paths that don't exist (needs /Users/z)
39. [OPEN] Agent name patching (sed on OMNIAGENT) is fragile — can corrupt file
40. [OPEN] No version tracking — can't tell which code version an agent is running

## P4: TASK MANAGEMENT

41. [OPEN] Task queue file gets corrupted by concurrent writes from API + agents
42. [OPEN] Duplicate timestamps accumulate on task entries from multiple claims
43. [OPEN] No task dependencies — T57 should block T58 but doesn't
44. [OPEN] No task priority — agents pick random READY tasks, not highest priority
45. [OPEN] No way to assign a task to a specific agent/node
46. [OPEN] No estimated completion time tracking
47. [OPEN] No task history — can't see who worked on what and when
48. [OPEN] claim_task.sh falls back to local sed when API is unreachable — creates split-brain
49. [OPEN] complete_task doesn't verify the agent actually did the work
50. [OPEN] No way to mark a task as BLOCKED with a reason via the API

## P5: CODE QUALITY/TESTING

51. [OPEN] No automated test suite for OMNIAGENT changes
52. [OPEN] No CI/CD — changes go live without testing
53. [OPEN] agent0 test runner doesn't test new code from remote agents
54. [OPEN] No regression test when OMNIAGENT is modified
55. [OPEN] No smoke test after fleet restart
56. [OPEN] Vulkan coherency test only runs 5 prompts — need 50+
57. [OPEN] No load test for task server under concurrent agent pressure
58. [OPEN] No test for push_changes actually landing on sys1
59. [OPEN] No test for claim_task race condition prevention
60. [OPEN] No golden output comparison for different model versions

## P6: INFRASTRUCTURE

61. [OPEN] No shared filesystem — agents work on stale local copies
62. [OPEN] SSH auth uses passwords in scripts — should be key-based
63. [OPEN] No centralized logging — each node has its own logs
64. [OPEN] No log aggregation or search across fleet
65. [OPEN] No alerting — watchdog detects issues but no push notifications
66. [OPEN] No dashboard persistence — state lost on restart
67. [OPEN] No backup schedule for ~/AGENT directory
68. [OPEN] No git auto-push — commits stay local until manual push
69. [OPEN] task_server.py runs in tmux — should be systemd service
70. [OPEN] ggml_server.py runs via sg video hack — should be proper service

## P7: SECURITY

71. [OPEN] Passwords in git history (sshpass -p z, authpass contents)
72. [OPEN] GitHub repo is PUBLIC — all code + IPs + hostnames exposed
73. [OPEN] SSH accepts password auth on all nodes — should be key-only
74. [OPEN] No firewall rules — all ports open between nodes
75. [OPEN] Agent can execute arbitrary bash commands on any node
76. [OPEN] No audit trail for destructive agent actions (rm, pkill)
77. [OPEN] No sandboxing — agents have full filesystem access
78. [OPEN] CUDA cluster settings modifiable by any SSH session
79. [OPEN] Orchestrator has write access to everything (no read-only mode)
80. [OPEN] No TLS between nodes — all traffic is plaintext HTTP

## P8: OBSERVABILITY

81. [OPEN] No Prometheus/Grafana for fleet metrics
82. [OPEN] No token usage tracking per agent per task
83. [OPEN] No cost tracking for CUDA brain usage
84. [OPEN] No agent efficiency metric (useful output per token)
85. [OPEN] No task completion velocity tracking
86. [OPEN] No heatmap of which tools are used most/least
87. [OPEN] No error rate per model/node correlation
88. [OPEN] No TTFT trend over time
89. [OPEN] No disk usage trend alerting
90. [OPEN] No network latency monitoring between nodes

## P9: AGENT INTELLIGENCE

91. [OPEN] Agents don't learn from previous task completions
92. [OPEN] No memory between agent sessions (context lost on restart)
93. [OPEN] Agents can't read other agents' work to avoid duplication
94. [OPEN] No peer review — agent writes code, nobody checks it
95. [OPEN] Agents don't consult KNOWLEDGE_BASE.md before starting tasks
96. [OPEN] No planning phase — agents jump into code without architecture
97. [OPEN] Agents don't update KNOWLEDGE_BASE.md with findings
98. [OPEN] No cross-agent communication (agent on sys2 can't ask agent on sys4)
99. [OPEN] Agents don't adapt prompts based on what worked before
100. [OPEN] No abort mechanism — agent keeps working on impossible task forever

## SUMMARY
- P0 Critical: 10 issues (agents broken)
- P1 High: 10 issues (agents inefficient)
- P2 Medium: 10 issues (monitoring gaps)
- P3 Agent Lifecycle: 10 issues
- P4 Task Management: 10 issues
- P5 Code Quality: 10 issues
- P6 Infrastructure: 10 issues
- P7 Security: 10 issues
- P8 Observability: 10 issues
- P9 Agent Intelligence: 10 issues
