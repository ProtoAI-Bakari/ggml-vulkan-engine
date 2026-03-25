# Agent Intelligence Improvements

## Why the Current Agent (v44_GPU) Fails

1. **No persistent memory between turns** - rediscovers the same bugs every session
2. **Can't follow multi-step instructions** - given "do A then B then C", it does A then goes off-rails
3. **Benchmark addiction** - defaults to testing instead of coding when uncertain
4. **No error dedup** - hits the same error 5x without recognizing it saw it before
5. **Poor JSON formatting** - Qwen3.5-122B frequently produces malformed tool_call JSON
6. **Ignores bridge messages** - doesn't prioritize architect directives over its own plan
7. **Server spam** - launches multiple servers without killing old ones
8. **No self-assessment** - can't tell when it's going in circles

## Fixes for v55+ Agent

### Architecture Changes
1. **Persistent scratchpad file** (`~/AGENT/scratchpad.md`) - agent reads at start of EVERY turn, writes findings after EVERY tool result. Survives restarts.
2. **Error registry** (`~/AGENT/error_registry.json`) - logs every error + what fixed it. Before attempting a fix, check if this error was seen before.
3. **Action plan queue** - instead of free-form, agent maintains a TODO list. Each turn picks next item. Architect can modify the queue.
4. **Mandatory knowledge base read** - system prompt forces agent to read VULKAN_KNOWLEDGE.md + scratchpad BEFORE doing anything.

### Prompt Engineering
5. **Few-shot examples in system prompt** - show the agent EXACTLY what good tool use looks like (read file -> patch with replace -> test -> report)
6. **Negative examples** - "NEVER do: sleep 30, pkill -f python, launch server without killing old one, benchmark without being asked"
7. **Role clarity** - "You are a CODE WRITER. 80% of your turns should produce code changes. 20% reading/analyzing. 0% benchmarking unless asked."

### Tool Improvements
8. **Smarter execute_bash** - detect and block dangerous patterns (pkill python, rm -rf, multiple nohup without kill)
9. **patch_file tool** - dedicated tool that does Python replace() with validation. Prevents write_file disasters.
10. **server_manage tool** - handles kill/launch/poll as ONE atomic operation. No more "launch then sleep 30 then check".

### Multi-Agent Collaboration
11. **Role-based agents** - Architect (plans, reviews), Builder (writes code), Tester (benchmarks), Debugger (reads logs/traces)
12. **Shared state via files** - all agents read/write to shared scratchpad, knowledge base, error registry
13. **Turn-taking protocol** - agents don't work simultaneously on same file. Lock mechanism.
14. **Escalation chain** - Builder stuck 3 turns -> Debugger takes over -> still stuck -> Architect reviews -> still stuck -> call Claude API

### Model Selection
15. **Bigger model for planning** - 70B+ for architecture decisions (M2 Ultra 192GB can run this)
16. **Fast model for execution** - 8B-30B for repetitive tool use (each M1 Ultra 128GB)
17. **Claude API for deep reasoning** - already implemented, use more aggressively
