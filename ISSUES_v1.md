# CLUSTER ISSUES TRACKER v1
# 100 issues to fix — prioritized
# Updated: 2026-03-26T09:15

## P0: CRITICAL (agents broken/looping)

1. [IN_PROGRESS by LEAD_CLAUDE | 60% | started:2026-03-26T04:08] JSON parse failures from embedded newlines — PARTIAL FIX: heredoc instruction added
2. [IN_PROGRESS by LEAD_CLAUDE | 50% | started:2026-03-26T04:08] Agents try to claim DONE tasks — PARTIAL FIX: grep READY first
3. [DONE by LEAD_CLAUDE | completed:2026-03-26T04:08] BLOCKED response now tells agent to WORK on existing task
4. [DONE by LEAD_CLAUDE | completed:2026-03-26T04:08] Context overflow — trim history on 400 error
5. [IN_PROGRESS by LEAD_CLAUDE | 70% | started:2026-03-26T04:08] Agents lose task after context trim — PARTIAL: nudge queries API
6. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] Remote agents cant compile/test — agent0 test runner deployed on sys1
7. [DONE by LEAD_CLAUDE | completed:2026-03-26T04:08] GO_PROMPT grep READY head -3 instead of full read
8. [DONE by LEAD_CLAUDE | completed:2026-03-26T04:08] Tool schema 18→10 tools (55% smaller, 525 tokens)
9. [DONE by LEAD_CLAUDE | completed:2026-03-26T04:08] Central task API with flock prevents double-claim
10. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] Stream errors — 1s retry, context trim on 400

## P1: HIGH (agents inefficient)

11. [DONE by LEAD_CLAUDE | completed:2026-03-26T04:08] grep READY | head -3 in GO_PROMPT
12. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] Agents re-read files they already read
13. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No file cache between turns
14. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] push_changes tool deployed, agents dont always call it
15. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] Agent writes code but never calls complete_task
16. [DONE by LEAD_CLAUDE | completed:2026-03-26T04:15] Agent calls ask_cuda_brain for simple questions
17. [DONE by LEAD_CLAUDE | completed:2026-03-26T04:15] Verbose reasoning — system prompt cut to 232 tokens
18. [DONE by LEAD_CLAUDE | completed:2026-03-26T04:15] write_file with large content (>500 lines) always fails JSON parse
19. [DONE by LEAD_CLAUDE | completed:2026-03-26T04:15] Agent doesn't update_progress — task stays at 0% forever
20. [OPEN — next] No way to redirect agent to different task

## P2: MEDIUM (dashboard/monitoring gaps)

21. [DONE by LEAD_CLAUDE | completed:2026-03-26T04:15] Dashboard TPS column shows 0 for sys1 (ggml server logs differently)
22. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] Dashboard Reqs column shows 0 for sys1
23. [OPEN] Dashboard Arch column fails on some models (binary log issue)
24. [DONE by LEAD_CLAUDE | completed:2026-03-26T04:15] Dashboard GPU% uses powermetrics (needs sudo, adds 500ms per node)
25. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] Commander still uses old node names (mlx-0, cuda-1) not sys1, cuda-sys1
26. [OPEN] Commander agent detection shows wrong count in "Agents: 1 running"
27. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] Fleet health checker doesn't detect "agent writing but no files produced"
28. [OPEN] No dashboard column for "last file written" or "last commit"
29. [DONE by LEAD_CLAUDE | completed:2026-03-26T04:15] Task server timestamps accumulate — "| 0% | started:... | 0% | started:..."
30. [DONE by LEAD_CLAUDE | completed:2026-03-26T04:08] Task server BLOCKED regex fixed for brackets

## P3: AGENT LIFECYCLE

31. [DONE by LEAD_CLAUDE | completed:2026-03-26T04:15] No graceful agent shutdown — pkill -9 loses in-flight work
32. [OPEN] Agent doesn't save state between restarts — loses context
33. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] Agent can't resume a partially-completed task after restart
34. [DONE by LEAD_CLAUDE | completed:2026-03-26T04:15] No heartbeat mechanism — can't tell if agent is alive vs thinking
35. [DONE by LEAD_CLAUDE | completed:2026-03-26T04:15] Agent log grows unbounded — no rotation
36. [DONE by LEAD_CLAUDE | completed:2026-03-26T04:15] agent_trace.log has ANSI escape codes making grep/analysis painful
37. [DONE by LEAD_CLAUDE | completed:2026-03-26T04:15] No way to inject a message into a running agent's context
38. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] Agents on macOS use /home/z paths that don't exist (needs /Users/z)
39. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] Agent name patching (sed on OMNIAGENT) is fragile — can corrupt file
40. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] No version tracking — can't tell which code version an agent is running

## P4: TASK MANAGEMENT

41. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] Task queue file gets corrupted by concurrent writes from API + agents
42. [OPEN] Duplicate timestamps accumulate on task entries from multiple claims
43. [DONE by LEAD_CLAUDE | completed:2026-03-26T04:15] No task dependencies — T57 should block T58 but doesn't
44. [DONE by LEAD_CLAUDE | completed:2026-03-26T04:15] No task priority — agents pick random READY tasks, not highest priority
45. [OPEN] No way to assign a task to a specific agent/node
46. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No estimated completion time tracking
47. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No task history — can't see who worked on what and when
48. [DONE by LEAD_CLAUDE | completed:2026-03-26T04:15] claim_task.sh falls back to local sed when API is unreachable — creates split-brain
49. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] complete_task doesn't verify the agent actually did the work
50. [DONE by LEAD_CLAUDE | completed:2026-03-26T04:15] No way to mark a task as BLOCKED with a reason via the API

## P5: CODE QUALITY/TESTING

51. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No automated test suite for OMNIAGENT changes
52. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No CI/CD — changes go live without testing
53. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] agent0 test runner doesn't test new code from remote agents
54. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No regression test when OMNIAGENT is modified
55. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No smoke test after fleet restart
56. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] Vulkan coherency test only runs 5 prompts — need 50+
57. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No load test for task server under concurrent agent pressure
58. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No test for push_changes actually landing on sys1
59. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No test for claim_task race condition prevention
60. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No golden output comparison for different model versions

## P6: INFRASTRUCTURE

61. [DEFERRED — infra/security change needed] No shared filesystem — agents work on stale local copies
62. [DEFERRED — infra/security change needed] SSH auth uses passwords in scripts — should be key-based
63. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No centralized logging — each node has its own logs
64. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No log aggregation or search across fleet
65. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No alerting — watchdog detects issues but no push notifications
66. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No dashboard persistence — state lost on restart
67. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No backup schedule for ~/AGENT directory
68. [DONE by LEAD_CLAUDE | completed:2026-03-26T04:15] No git auto-push — commits stay local until manual push
69. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] task_server.py runs in tmux — should be systemd service
70. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] ggml_server.py runs via sg video hack — should be proper service

## P7: SECURITY

71. [DEFERRED — infra/security change needed] Passwords in git history (sshpass -p z, authpass contents)
72. [DEFERRED — infra/security change needed] GitHub repo is PUBLIC — all code + IPs + hostnames exposed
73. [DEFERRED — infra/security change needed] SSH accepts password auth on all nodes — should be key-only
74. [DEFERRED — infra/security change needed] No firewall rules — all ports open between nodes
75. [DEFERRED — infra/security change needed] Agent can execute arbitrary bash commands on any node
76. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] No audit trail for destructive agent actions (rm, pkill)
77. [DEFERRED — infra/security change needed] No sandboxing — agents have full filesystem access
78. [DEFERRED — infra/security change needed] CUDA cluster settings modifiable by any SSH session
79. [DEFERRED — infra/security change needed] Orchestrator has write access to everything (no read-only mode)
80. [DEFERRED — infra/security change needed] No TLS between nodes — all traffic is plaintext HTTP

## P8: OBSERVABILITY

81. [OPEN] No Prometheus/Grafana for fleet metrics
82. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] No token usage tracking per agent per task
83. [OPEN] No cost tracking for CUDA brain usage
84. [OPEN] No agent efficiency metric (useful output per token)
85. [OPEN] No task completion velocity tracking
86. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] No heatmap of which tools are used most/least
87. [OPEN] No error rate per model/node correlation
88. [OPEN] No TTFT trend over time
89. [OPEN] No disk usage trend alerting
90. [OPEN] No network latency monitoring between nodes

## P9: AGENT INTELLIGENCE

91. [FUTURE — requires architectural redesign] Agents don't learn from previous task completions
92. [FUTURE — requires architectural redesign] No memory between agent sessions (context lost on restart)
93. [FUTURE — requires architectural redesign] Agents can't read other agents' work to avoid duplication
94. [FUTURE — requires architectural redesign] No peer review — agent writes code, nobody checks it
95. [DONE by LEAD_CLAUDE | completed:2026-03-26T04:15] Agents don't consult KNOWLEDGE_BASE.md before starting tasks
96. [FUTURE — requires architectural redesign] No planning phase — agents jump into code without architecture
97. [FUTURE — requires architectural redesign] Agents don't update KNOWLEDGE_BASE.md with findings
98. [FUTURE — requires architectural redesign] No cross-agent communication (agent on sys2 can't ask agent on sys4)
99. [FUTURE — requires architectural redesign] Agents don't adapt prompts based on what worked before
100. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] No abort mechanism — agent keeps working on impossible task forever

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

## P10: ADDITIONAL ISSUES (101-150)

101. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] Agents should write tests for code they produce
102. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] No diff review before push_changes — blind SCP
103. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] Agent should verify compilation succeeds before calling complete_task
104. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No rollback mechanism if agent breaks something
105. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] task_server has no authentication — any node can claim/complete
106. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] Deployer sync uses SCP per-file — should rsync (faster, atomic)
107. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No model version pinning — HF cache auto-updates could break
108. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] Dashboard refresh blocks on slow SSH — should be async
109. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No alert when agent produces gibberish output
110. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] Agent doesn't check if its MLX server is healthy before working
111. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] No rate limit on task claims per minute
112. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] Complete_task doesn't verify files were pushed first
113. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No way to see agent's current conversation context size
114. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] Agent should report estimated time remaining for current task
115. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No cross-agent code review (agent A reviews agent B's output)
116. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] Orchestrator on z4090 not running any agents — wasted capacity
117. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] Dashboard should show agent uptime (time since last restart)
118. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No way to pause an agent without killing it
119. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] Agent should detect when its model server crashes and alert
120. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] No centralized error log aggregating all agent errors
121. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] Task queue file getting corrupted with duplicate brackets ]]]]
122. [OPEN] No visualization of task completion over time
123. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] Agent should batch small file operations instead of one per turn
124. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No way to set max_tokens dynamically per task complexity
125. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] CUDA brain at 42K context but agents rarely need >4K
126. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] Watchdog and fleet_health_check overlap in functionality
127. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No way to see total tokens generated across all agents
128. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] Agent0 test runner doesn't report results to dashboard
129. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] No integration test that verifies full claim→work→push→complete cycle
130. [OPEN] Dashboard doesn't show which phase each agent is working on
131. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] No alert when disk drops below threshold on any node
132. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] Agent should minimize context by summarizing completed work
133. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] No way to share findings between agents (KB not synced to remotes)
134. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] Task server should reject claims for tasks with unmet dependencies
135. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] No health endpoint for the agent process itself
136. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] Dashboard arch column truncates model info with ellipsis
137. [OPEN] No way to see agent error rate over time
138. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] Agent should use sed for small edits instead of rewriting files
139. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No git branch per agent — all work on master
140. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] Push_changes doesn't handle merge conflicts
141. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No way to compare agent productivity across different models
142. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] Agent TTFT spikes when conversation history grows
143. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No way to tell an agent to focus on code quality over speed
144. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] Dashboard should auto-refresh without manual restart
145. [DEFERRED — infra/security change needed] No webhook/Slack notification when task completes
146. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] Agent should validate JSON tool calls before outputting them
147. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] No way to see which tools each agent uses most
148. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] Task queue READY tasks should show estimated effort (time)
149. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:09] No way to mark a task as WONTFIX or DEFERRED
150. [DONE by LEAD_CLAUDE | completed:2026-03-26T09:15] Agent should write a completion report for each task

## TOTAL: 150 issues
