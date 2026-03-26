# AI Coding Agent Quality Baseline Plan
# ProtoAI-Bakari Cluster — Asahi Linux + Vulkan + CUDA Distributed Inference
# Author: Bakari McCoy | Date: 2026-03-25

---

## 0. Purpose

We have seven specialized models serving defined roles in a production swarm. Before we can
route tasks intelligently, swap underperformers, or justify role assignments, we need hard
numbers. This plan defines how to get those numbers, what they mean, and what to do with them.

**Goal:** By end of week 1, have a score for every model on every task category. By end of
week 2, have automated regression that runs nightly and feeds scores back into routing.

---

## 1. The Swarm Roster

| Node      | Model                              | Role         | RAM   |
|-----------|------------------------------------|--------------|-------|
| mlx-2     | Qwen3-235B-A22B-Thinking-4bit      | ARCHITECT    | 192GB |
| mlx-3     | Qwen3.5-122B-A10B-4bit             | ENGINEER     | 192GB |
| mlx-4     | Qwen3-Coder-Next-8bit              | CODER        | 128GB |
| mlx-5     | GLM-4.7-Flash-8bit                 | DESIGNER     | 128GB |
| mlx-6     | Qwen3.5-122B-A10B-4bit             | REVIEWER     | 128GB |
| mlx-7     | Qwen3-Coder-Next-4bit              | FAST CODER   | 128GB |
| cuda-.11  | Qwen3.5-122B-A10B-FP8 (TP8 8x3090)| PRIMARY BRAIN| 320GB |

All models expose OpenAI-compatible `/v1/chat/completions` endpoints. Ports vary per node
(typically 8080 or configurable). All nodes are on the same LAN, reachable from the AGENT
host at 10.255.255.x.

---

## 2. Quality Dimensions and Metrics

Each model output is scored on six dimensions. Every dimension uses a 1-5 integer scale.

### 2.1 Scoring Rubric (applies to all categories)

**Code Correctness (CC)** — Does the code do what was asked?
- 5: Correct, handles edge cases, no bugs found
- 4: Correct for the main case, minor edge case gap
- 3: Mostly correct, one logical bug that can be patched
- 2: Compiles/runs but produces wrong output
- 1: Fundamentally broken or does not address the task

**Compilation / Syntax Success (CS)** — Does the output parse and compile?
- 5: Compiles clean with no warnings (`-Wall -Wextra` for C, `py -m py_compile` for Python)
- 4: Compiles with warnings only (no errors)
- 3: Compiles after trivial fix (missing include, wrong type)
- 2: Multiple compile errors, non-trivial to fix
- 1: Does not compile at all / syntax broken beyond repair

**Test Pass Rate (TP)** — Against a fixed test suite per task:
- 5: 100% pass
- 4: >= 80% pass
- 3: >= 60% pass
- 2: >= 40% pass
- 1: < 40% pass

**Code Review Score (CR)** — Assessed by the REVIEWER role (mlx-6) via a structured
review prompt (see Section 5.3). Also includes manual spot-check by a human.
- 5: Production-quality, would merge without changes
- 4: Minor style or clarity issues, merge with small fixes
- 3: One substantive issue (error handling, memory safety, etc.)
- 2: Multiple substantive issues
- 1: Would not merge, fundamental redesign needed

**Task Completion Rate (TC)** — Did the model complete the full task or stop early/drift?
- 5: Full task delivered, nothing missing
- 4: Core deliverable present, minor omission (e.g., missing comment block)
- 3: ~75% of task done, one major component missing
- 2: Partial attempt, significant scope left undone
- 1: Off-topic, refused, or near-empty response

**Hallucination Rate (HR)** — Did the model fabricate APIs, flags, paths, or facts?
- 5: Zero hallucinations confirmed
- 4: One minor hallucination (wrong flag name, correctable)
- 3: One substantive hallucination (wrong API, wrong system call)
- 2: Multiple hallucinations affecting correctness
- 1: Output is largely fabricated

**Composite Score** = mean(CC, CS, TP, CR, TC, HR). Range 1.0–5.0.

For automated scoring: CC, CS, TP are computed by scripts. CR requires the REVIEWER model.
TC and HR require human review or a dedicated judge prompt (see Section 5.4).

---

## 3. Test Categories and Task Bank

Five categories map to the actual work this swarm does. Each category has a fixed set of
benchmark tasks. Tasks are versioned — do not change them once baselining starts.

### 3.1 Category A: C Code (ggml / Vulkan)

These tasks are the hardest and most domain-specific. Priority: highest.

| Task ID | Description | Expected Output |
|---------|-------------|-----------------|
| C01 | Write a ggml_vk_dispatch_pipeline wrapper that records a compute command buffer, submits to a VkQueue, and waits on a VkFence. No helper libs. | Compilable C with correct Vulkan calls |
| C02 | Implement a ring buffer in C (fixed-size, thread-safe with pthreads, no dynamic alloc after init). | Compilable C, passes 3 test cases |
| C03 | Given this ggml tensor op skeleton [paste ggml_mul_mat stub], fill in the MoE gating logic for 128 experts, 4 active. | Compilable C, correct expert selection |
| C04 | Write a Vulkan pipeline cache save/load function: serialize VkPipelineCache to a file, reload and validate. | Compilable C with correct VkResult checks |
| C05 | Debug this C snippet [paste snippet with use-after-free + integer overflow], identify all bugs, rewrite clean. | Annotated diff + clean rewrite |

Compiler: `clang -Wall -Wextra -Wpedantic -std=c11` on the AGENT host (Asahi Linux, ARM64).
Vulkan headers: `/usr/include/vulkan`. Link: `-lvulkan`.

Test harness for C: compile with clang, run any bundled test cases, check return codes.

### 3.2 Category B: Python Code

| Task ID | Description | Expected Output |
|---------|-------------|-----------------|
| P01 | Write an async Python function that calls an OpenAI-compatible `/v1/chat/completions` endpoint with retry logic (3 retries, exponential backoff, timeout=30s). | Runnable Python 3.10+, passes mock test |
| P02 | Parse a JSON log file where some lines are malformed. Collect all valid entries, report parse errors to stderr, return structured summary. | Runnable, passes 4 test cases |
| P03 | Write a dataclass-based config loader: reads TOML, validates required fields, raises typed exceptions on missing/wrong-type fields. | Runnable, passes 5 test cases |
| P04 | Implement a token bucket rate limiter as a Python class (thread-safe, configurable rate and burst). | Runnable, passes concurrency test |
| P05 | Given this mlx_server.py snippet [paste 50-line real snippet from the project], identify any bugs and refactor for clarity. | Annotated diff |

Test harness for Python: `python -m py_compile` first, then `pytest` against bundled test files.

### 3.3 Category C: Bash Scripts

| Task ID | Description | Expected Output |
|---------|-------------|-----------------|
| B01 | Write a bash script that SSHes to a list of nodes (from a file), runs `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader` or `ioreg -l` (detect platform), collects results, writes a TSV report. | Working bash, passes on Asahi and Darwin |
| B02 | Write a bash health-check script for mlx_server: checks if port is open, sends a test prompt via curl, checks for coherent JSON response, exits 0/1. | Working bash |
| B03 | Write a deployment script that rsync's a Python script to a list of nodes, restarts a systemd service, waits for health check to pass, rolls back on failure. | Working bash with rollback |
| B04 | Write a bash one-liner (max 200 chars) that finds all Python processes using more than 10GB RSS, prints their PIDs and command lines. | Correct one-liner, tested on Linux |
| B05 | Parse the output of `journalctl -u mlx_server --since "1 hour ago"` and extract: error count, warning count, last 3 error messages. | Working bash |

Test harness for bash: `bash -n` (syntax check), then run against a fixture environment or
mock inputs where applicable.

### 3.4 Category D: Architecture Documentation

| Task ID | Description | Expected Output |
|---------|-------------|-----------------|
| D01 | Write a 1-page architecture diagram description (ASCII or Mermaid) for the 7-node inference swarm: data flow, who calls whom, failure modes. | Accurate, no hallucinated endpoints |
| D02 | Write a decision tree for: given a user task (code/review/design/architecture), which swarm role handles it, and when does it escalate? | Logical, consistent with roster |
| D03 | Write a runbook entry: "MLX server returns 500 on first request after restart." Diagnosis steps, fix, prevention. | Accurate, actionable steps |
| D04 | Spec a new API endpoint `/v1/swarm/route` that accepts a task description and returns the recommended model + fallback. Fields, types, error codes. | Correct OpenAPI-style spec |
| D05 | Given this benchmark CSV [paste real benchmark_vulkan.py output], write an analysis: bottlenecks, recommendations, predicted improvement from fixes. | Grounded analysis, no invented numbers |

Scoring for D category: TC and HR are the primary dimensions. CS is replaced by a
"Format Quality" check (valid Mermaid/ASCII, no broken markdown). TP is N/A — replaced by
human accuracy review.

### 3.5 Category E: Code Review

| Task ID | Description | Expected Output |
|---------|-------------|-----------------|
| R01 | Review this C function [paste ggml_vk_dispatch_pipeline, ~80 lines]. Identify bugs, style issues, Vulkan best practice violations. | Structured review with line references |
| R02 | Review this Python async server [paste real mlx_server snippet, ~100 lines]. Thread safety, error handling, resource leaks. | Structured review |
| R03 | Review this bash deployment script [paste B03 task answer from a model]. Correctness, portability, failure handling. | Structured review |
| R04 | Given two implementations of the same ring buffer (from C02, two different models), compare them and recommend one. | Justified recommendation |
| R05 | Review a PR diff [paste a real recent git diff from this repo]. Summary, concerns, approval/rejection with reasoning. | Structured review |

Scoring for R category: CR is self-referential for REVIEWER role. Instead, a human judge
checks the review for accuracy and depth using the same 1-5 rubric applied to the review
itself.

---

## 4. Benchmark Methodology

### 4.1 Sending Tasks

All tasks are sent via the OpenAI-compatible API. System prompt is standardized:

```
You are a senior systems engineer specializing in C, Python, and bash on Asahi Linux with
Vulkan compute. Be direct. Output only the requested artifact. No meta-commentary.
```

Temperature: 0.2 (low, for reproducibility). Max tokens: 4096. No streaming during baseline
collection (collect full response for deterministic scoring).

Each task is sent three times per model to account for non-determinism. Scores are averaged
across three runs. Standard deviation is recorded — high variance is itself a quality signal.

### 4.2 Isolation

- Flush model KV cache between runs (send a reset request or restart the server if needed)
- Record wall-clock time per request (latency matters for role fitness)
- One model at a time during baseline to avoid cross-node interference
- Record model load time separately from inference time

### 4.3 Ground Truth

For C, Python, and bash tasks: a human-written reference solution exists for each task,
stored in `/home/z/AGENT/quality_baseline/reference/`. These are not shown to models.

The automated test suite runs both the model output and the reference solution against the
same test cases. The reference solution must score 5/5 on CC, CS, TP to be valid.

### 4.4 Latency as a Quality Dimension

For production routing, latency matters alongside correctness. Record:
- Time-to-first-token (TTFT) in ms
- Total response time in seconds
- Tokens per second (TPS)

Latency scores feed into routing decisions separately from quality scores. A model that
scores 4.8/5.0 quality but takes 180 seconds per response cannot be used for synchronous
coding tasks.

---

## 5. Automated Testing Scripts

### 5.1 Directory Structure

```
/home/z/AGENT/quality_baseline/
  tasks/
    C01.json  C02.json ... C05.json
    P01.json  P02.json ... P05.json
    B01.json  B02.json ... B05.json
    D01.json  D02.json ... D05.json
    R01.json  R02.json ... R05.json
  reference/
    C01_ref.c  C02_ref.c ...
    P01_ref.py ...
    B01_ref.sh ...
  tests/
    C01_test.c  C02_test.c ...   # compiled and run against model output
    P01_test.py ...
    B01_test.sh ...
  results/
    YYYY-MM-DD/
      mlx-2_results.json
      mlx-3_results.json
      ...
  run_baseline.py          # main orchestration script
  score_output.py          # scoring engine
  compile_and_test.sh      # C/Python/bash test runner
  judge_prompt.py          # sends output to REVIEWER for CR score
  report.py                # generates summary report
```

### 5.2 Task JSON Format

```json
{
  "task_id": "C01",
  "category": "C",
  "description": "Write a ggml_vk_dispatch_pipeline wrapper...",
  "system_prompt": "You are a senior systems engineer...",
  "user_prompt": "...",
  "expected_artifact": "c_file",
  "compile_cmd": "clang -Wall -Wextra -std=c11 {output} -lvulkan -o {binary}",
  "test_cmd": "./tests/C01_test.sh {output}",
  "rubric_weights": {"CC": 2, "CS": 1, "TP": 2, "CR": 1, "TC": 1, "HR": 1}
}
```

Weights allow per-task emphasis. C tasks weight CC and TP higher. D tasks weight TC and HR.

### 5.3 run_baseline.py Core Logic

```python
# Pseudocode — do not modify actual files
for node in NODES:
    for task in TASKS:
        for run in range(3):
            response = call_api(node.endpoint, task.system_prompt, task.user_prompt)
            save_response(response, node, task, run)
            cs_score = compile_and_syntax_check(response, task)
            tp_score = run_tests(response, task)
            tc_score = check_task_completion(response, task)
            hr_score = check_hallucinations(response, task)
            cr_score = get_reviewer_score(response, task)  # calls mlx-6
            cc_score = human_review_queue(response, task)  # queued for human
            save_scores(node, task, run, all_scores)
```

Human review queue: tasks where CC requires human judgment are written to
`results/YYYY-MM-DD/human_review_queue.json`. Human reviews them and enters scores via
a simple CLI: `python score_output.py --review`.

### 5.4 Judge Prompt for Hallucination and Task Completion

Send model output to cuda-.11 (PRIMARY BRAIN, highest capability) with this prompt:

```
You are a code quality judge. Given the task specification and the model output below,
score on two dimensions using integer 1-5:

TC (Task Completion): Did the model fully complete the requested task?
HR (Hallucination): Did the model fabricate any APIs, functions, flags, paths, or facts?

Task: {task_description}
Output: {model_output}

Respond ONLY with JSON: {"TC": <int>, "HR": <int>, "TC_reason": "<one sentence>", "HR_reason": "<one sentence>"}
```

This uses the strongest available model as judge. Log judge responses for audit.

### 5.5 Reviewer Prompt for CR Score

Send model output to mlx-6 (REVIEWER role) with:

```
You are a senior code reviewer. Review the following {language} code produced in response
to this task: {task_description}

Code:
{model_output}

Score on a 1-5 scale:
5 = production-quality, would merge without changes
4 = minor issues, merge with small fixes
3 = one substantive issue (safety, correctness, error handling)
2 = multiple substantive issues
1 = would not merge

Respond ONLY with JSON: {"CR": <int>, "issues": ["<issue1>", "<issue2>"], "recommendation": "<one sentence>"}
```

### 5.6 Compile and Test Script (compile_and_test.sh)

For C:
```bash
clang -Wall -Wextra -std=c11 "$OUTPUT_FILE" -lvulkan -o /tmp/qa_binary 2>/tmp/compile_errors
if [ $? -eq 0 ]; then CS=5; elif grep -c "error:" /tmp/compile_errors == 0; then CS=4; else CS=1; fi
/tmp/qa_binary < tests/C01_input.txt > /tmp/actual_output
diff /tmp/actual_output tests/C01_expected.txt && TP=5 || TP=$(compute_partial_score)
```

For Python:
```bash
python -m py_compile "$OUTPUT_FILE" 2>/tmp/syntax_errors
python -m pytest tests/P01_test.py --tb=short -q 2>/tmp/test_results
```

For bash:
```bash
bash -n "$OUTPUT_FILE" 2>/tmp/syntax_errors
bash tests/B01_test.sh "$OUTPUT_FILE" 2>/tmp/test_results
```

---

## 6. Timeline: What to Build First

### Week 1 — Manual Baseline (Days 1-3)

**Day 1:**
- Create `/home/z/AGENT/quality_baseline/` directory structure
- Write all 25 task JSON files (5 per category)
- Write human reference solutions for all C and Python tasks
- Write test cases for C01-C05, P01-P05

**Day 2:**
- Manually send C category tasks to mlx-4 (CODER) and mlx-7 (FAST CODER) — these are the
  primary code producers, baseline them first
- Compile and test all outputs by hand, record scores in a spreadsheet
- Identify any tasks that are too easy or too hard (no discrimination value)

**Day 3:**
- Manually send C and Python tasks to remaining models
- Complete manual scoring for Week 1 sample
- Identify top 3 surprises (model that outperforms its role, model that underperforms)

### Week 1 — Automation (Days 4-5)

**Day 4:**
- Write `run_baseline.py`: API calls, response saving, CS/TP automated scoring
- Write `compile_and_test.sh`
- Test against mlx-4 for C category only

**Day 5:**
- Add judge_prompt.py (TC, HR via cuda-.11)
- Add reviewer prompt (CR via mlx-6)
- Run full automated baseline for all nodes on C and Python categories
- Compare automated scores to Day 2-3 manual scores — calibrate

### Week 2 — Regression and Integration

**Day 6:**
- Add bash, D, and R category tasks and tests
- Run full baseline across all 7 nodes and all 25 tasks
- Generate first `report.py` summary

**Day 7:**
- Wire results into a routing weight file (see Section 8)
- Set up cron job: `run_baseline.py --category C,P --quick` nightly (3 tasks per category,
  1 run each — takes ~20 min)
- Full baseline weekly (all 25 tasks, 3 runs each)

**Day 8-10:**
- Review first week of nightly results
- Identify any models drifting (score changes after server restart or model reload)
- Document findings in `QUALITY_BASELINE_RESULTS.md`

---

## 7. Decision Framework: When to Act on Scores

### 7.1 Role Fitness Thresholds

A model is considered fit for its role if it meets these minimums on tasks relevant to
that role:

| Role         | Primary Categories | Minimum Composite | Minimum CC | Min TPS |
|--------------|--------------------|-------------------|------------|---------|
| ARCHITECT    | D, R               | 3.5               | 3.0        | 5       |
| ENGINEER     | C, P, D            | 3.8               | 3.5        | 8       |
| CODER        | C, P               | 4.0               | 4.0        | 10      |
| DESIGNER     | D, R               | 3.5               | 3.0        | 15      |
| REVIEWER     | R                  | 4.0               | 3.5        | 8       |
| FAST CODER   | C, P, B            | 3.5               | 3.5        | 30      |
| PRIMARY BRAIN| All                | 4.0               | 4.0        | 5       |

FAST CODER has a lower composite threshold but higher TPS requirement — it exists for
latency-sensitive tasks where a slightly lower quality answer in 5s beats a perfect answer
in 60s.

### 7.2 Swap Triggers

**Immediate swap (act within 24h):**
- Any category score drops below 2.5 on 2 consecutive nightly runs
- Hallucination rate (HR) drops below 3.0 on code tasks (fabricated APIs in production code
  is an active risk)
- Compilation success (CS) below 3.0 on C category (the model cannot produce valid C)

**Investigation trigger (act within 72h):**
- Composite score drops more than 0.5 points from baseline
- Latency (TTFT) increases more than 2x from baseline
- Standard deviation across 3 runs exceeds 1.0 on any dimension (inconsistent model)

**Role reassignment consideration:**
- A model consistently scores higher on a different category than its assigned role
- Example: if mlx-5 (DESIGNER) scores 4.5 on C tasks but 2.8 on D tasks, reconsider its role

### 7.3 Swap Options

When a model underperforms, options in order of preference:

1. **Server restart**: if the model was running for >72h, restart the mlx_server process and
   re-run baseline. Model quality can degrade due to KV cache state, memory fragmentation.

2. **Quantization change**: if the 4-bit model is underperforming on correctness, try 8-bit
   if the node has headroom. Example: mlx-7 (Coder-Next-4bit) vs mlx-4 (Coder-Next-8bit)
   is a natural comparison pair.

3. **Model swap**: load a different model checkpoint. Maintain a candidate list:
   - Backup CODER: Qwen2.5-Coder-32B-Instruct (fits 128GB at 4bit)
   - Backup REVIEWER: any of the 122B models with a different system prompt
   - Backup FAST CODER: any flash/turbo model with high TPS

4. **Role reassignment**: reassign the underperforming model to a role better matched to
   its actual scores, and promote a better-performing model to the critical role.

### 7.4 Do Not Swap When

- Score drop is on D category only and the model's primary role is CODER — architecture docs
  are not in its critical path
- Score drop is on a single run with high standard deviation — wait for trend confirmation
- The swap candidate is not available (node offline, model not loaded) — document and wait

---

## 8. Integration with the Swarm: Routing Weight Feedback

### 8.1 Routing Weight File

Maintain `/home/z/AGENT/quality_baseline/routing_weights.json`:

```json
{
  "updated": "2026-03-25T00:00:00Z",
  "models": {
    "mlx-2": {
      "role": "ARCHITECT",
      "composite": 4.1,
      "by_category": {"C": 3.2, "P": 3.8, "B": 3.5, "D": 4.8, "R": 4.3},
      "latency_ttft_ms": 2400,
      "tps": 6,
      "available": true,
      "last_baseline": "2026-03-25"
    },
    "mlx-4": {
      "role": "CODER",
      "composite": 4.4,
      "by_category": {"C": 4.7, "P": 4.5, "B": 4.1, "D": 3.2, "R": 3.8},
      "latency_ttft_ms": 800,
      "tps": 18,
      "available": true,
      "last_baseline": "2026-03-25"
    }
  }
}
```

### 8.2 Task Router Logic

When a task comes in, the router selects models based on:

```
score = quality_weight * category_score[task_category] + latency_weight * normalized_tps
```

Default weights: `quality_weight=0.7, latency_weight=0.3`

For latency-sensitive tasks (user waiting synchronously): flip to `0.4/0.6`.
For critical tasks (production C code, architecture decisions): `0.9/0.1`.

The router reads `routing_weights.json` at startup and reloads it hourly. When nightly
baseline updates scores, the router picks up changes on next reload without restart.

### 8.3 Escalation Chain

If the primary model for a task scores below threshold at routing time (based on stored
scores), escalate:

```
CODER task fails → ENGINEER → PRIMARY BRAIN → human queue
REVIEWER task fails → ENGINEER → PRIMARY BRAIN → human queue
ARCHITECT task fails → PRIMARY BRAIN → human queue
```

The escalation chain is hardcoded per role, not dynamically computed — keep it simple and
auditable.

### 8.4 Score Decay

Scores older than 7 days are penalized: multiply composite by 0.95 per week stale.
This ensures that if nightly baseline stops running (node down, script broken), the router
naturally becomes more conservative and escalates more, rather than blindly trusting stale
scores.

---

## 9. Data to Collect in the First Week

Priority order — collect this even if automation is not ready:

1. **mlx-4 vs mlx-7 on C category**: these are the two CODER-role models. The 8-bit vs
   4-bit comparison on actual Vulkan/ggml tasks is the most operationally relevant data point.
   Run C01-C05 on both, score manually. Time: 2 hours.

2. **cuda-.11 on all categories**: the PRIMARY BRAIN should set the ceiling. Any model that
   scores higher than cuda-.11 on any category is a remarkable finding (or a test design flaw).
   Time: 3 hours.

3. **mlx-6 as REVIEWER on R01-R05**: the REVIEWER role is only valuable if it produces
   accurate, actionable reviews. Score its reviews manually — does it catch real bugs? Does
   it hallucinate issues? Time: 1.5 hours.

4. **mlx-5 (GLM-4.7-Flash) on D category**: GLM is an unusual choice for DESIGNER. Baseline
   it on D tasks to confirm it belongs in this role vs. being a latency-optimized fallback.
   Time: 1 hour.

5. **Latency for all nodes on a standard prompt** (P01): collect TTFT and TPS from all 7
   nodes on the same prompt in the same hour. This gives a latency ranking that feeds
   immediately into routing decisions. Time: 30 minutes.

---

## 10. Reporting

### 10.1 Nightly Report Format

`report.py` generates `/home/z/AGENT/quality_baseline/results/YYYY-MM-DD/report.md`:

```
# Quality Baseline Report — 2026-03-25

## Summary Table
| Node   | Role       | Composite | C    | P    | B    | D    | R    | TPS  |
|--------|------------|-----------|------|------|------|------|------|------|
| mlx-4  | CODER      | 4.4       | 4.7  | 4.5  | 4.1  | 3.2  | 3.8  | 18   |
| ...    |            |           |      |      |      |      |      |      |

## Alerts
- mlx-3 C category score dropped from 3.9 to 3.1 (threshold: 3.5) — INVESTIGATE
- mlx-7 TPS degraded from 32 to 18 — CHECK SERVER LOAD

## Trend (last 7 days)
[ascii chart or table]

## Recommended Actions
1. Restart mlx-3 server and re-run C baseline
2. Check mlx-7 node for competing processes
```

### 10.2 Weekly Summary

Human reads the weekly report. Key questions to answer:
- Is every model fit for its role? (above threshold on primary categories)
- Did any model improve or degrade significantly?
- Did any task in the test bank become too easy (all models scoring 5/5)?
  If so, replace with a harder variant.
- Are the routing weights being used, and do task outcomes match predictions?

---

## 11. Known Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Test tasks become stale (models see similar tasks in training) | Rotate 20% of task bank monthly with new real-world problems from the project |
| Judge model (cuda-.11) is itself wrong | Human spot-check 10% of judge scores weekly |
| Baseline run interferes with production traffic | Run nightly at 02:00 local time, send one task at a time per node |
| A model's quality varies by prompt wording | Standardize prompts strictly, never paraphrase between runs |
| Vulkan/clang not available on AGENT host | All C compilation and testing runs on AGENT host (Asahi Linux ARM64) — confirmed available |
| mlx-6 REVIEWER gives inflated scores to other mlx models | Cross-check mlx-6 reviews against cuda-.11 judge scores for 10% of tasks |

---

## 12. First Actions (Do This Now)

1. `mkdir -p /home/z/AGENT/quality_baseline/{tasks,reference,tests,results}`
2. Write C01 task JSON and C01 reference solution
3. Send C01 to mlx-4 and mlx-7 manually, compile both outputs, record scores
4. Send P01 to all 7 nodes, record latency (TTFT, TPS) — takes 30 minutes
5. Review results. The numbers from steps 3-4 will immediately tell you whether the
   rubric is calibrated correctly and whether the tasks have discrimination value.

Do not wait for full automation before collecting data. The first 5 data points from
manual testing are worth more than a week of script-writing with no data.

---

*Plan version 1.0 — Bakari McCoy / ProtoAI-Bakari — 2026-03-25*
*Review and update after first week of data collection.*
