# Agent Signal Protocol
# Real-time coordination between LEAD and WORKER agents on Sys0

## How it works
Each agent runs a background watcher on the other's signal file.
When agent A needs to notify agent B, it writes to the signal file.
Agent B's watcher fires instantly via inotify.

## Signal files
- ~/AGENT/.signals/LEAD_TO_WORKER — LEAD writes, WORKER watches
- ~/AGENT/.signals/WORKER_TO_LEAD — WORKER writes, LEAD watches

## Message format (one line, overwritten each time)
TIMESTAMP|ACTION|MESSAGE
Example: 2026-03-25T01:50:00|PAUSE|Running llama-bench, hold compute

## Actions
- PAUSE — stop compute-heavy work
- RESUME — OK to run benchmarks again
- DATA — new data posted to bridge, go read it
- CLAIM — I'm about to use the GPU, wait
- DONE — I'm done with GPU, you can go
- URGENT — stop everything and read bridge NOW

## Start watcher (each agent runs at session start)
```bash
# LEAD watches for worker signals:
inotifywait -m -e modify ~/AGENT/.signals/WORKER_TO_LEAD &

# WORKER watches for lead signals:
inotifywait -m -e modify ~/AGENT/.signals/LEAD_TO_WORKER &
```

## Benchmark coordination
Before ANY benchmark/inference run:
1. Write CLAIM to your signal file
2. Wait 2 seconds for the other agent to see it
3. Run your benchmark
4. Write DONE when finished
