# Mission: T11 - Graph Topology Fingerprinting
# Agent: Sub-Agent (Parallel Worker)
# Priority: HIGH
# Time Budget: 4 hours

---

## OBJECTIVE
Implement graph topology fingerprinting to detect when compute graph is unchanged between tokens (99%+ of decode steps).

## SUCCESS CRITERIA
- [ ] Hash function: node count + op types + tensor shapes
- [ ] Fingerprint matches consecutive decode tokens
- [ ] Detects topology changes (context size, batch size)
- [ ] Enables graph caching in T12

## WORKING DIRECTORY
**CRITICAL:** Work in `/home/z/AGENT` and `/home/z/GITDEV/vllm-vulkan`

## KEY FILES TO READ FIRST
1. `/home/z/AGENT/ggml_vulkan_engine.py` — Main Vulkan engine
2. `/home/z/AGENT/VULKAN_DEEP_RESEARCH/ggml_vulkan_internals.md` — Vulkan internals
3. `/home/z/GITDEV/vllm-vulkan/vllm/platforms/vulkan.py` — Vulkan platform layer
4. `/home/z/GITDEV/llama.cpp/ggml/src/ggml-vulkan/` — llama.cpp Vulkan backend (reference)

## TASKS
1. **Explore codebase first:**
   ```bash
   cd /home/z/AGENT
   cat ggml_vulkan_engine.py  # Understand current graph handling
   ls -la ~/GITDEV/llama.cpp/ggml/src/ggml-vulkan/  # Reference implementation
   ```

2. **Add fingerprinting to ggml Vulkan backend:**
   - Check `/home/z/GITDEV/llama.cpp/ggml/src/ggml-vulkan/` for existing graph code
   - Implement fingerprint function in Python wrapper first (faster iteration)
   - Hash function: node_count + op_types[] + tensor_shapes[]
   - Use xxHash or FNV-1a for speed

3. **Store fingerprint at each token:**
   - First token: compute and store
   - Subsequent tokens: compare with previous
   - If match: skip graph rebuild, use cached command buffer

4. **Log fingerprint changes:**
   - When does topology change? (context thresholds, batch changes)
   - How often during typical generation?

5. **Integrate with command buffer template recording (T13)**

## DELIVERABLES
- `fingerprint_graph()` function in Python (or C if needed)
- Logging of fingerprint changes
- Test showing 99%+ match rate during decode
- Documentation of when topology changes

## NOTES
- Focus on decode phase (batch=1, context growing)
- Topology should be stable after first token
- Pre-compute fingerprint for common graph patterns
- Check llama.cpp ggml-vulkan implementation for reference
- ggml graph structure: `struct ggml_cgraph` with nodes, ops, shapes

## COORDINATION
- Read `agent-comms-bridge.md` before starting
- Append results to bridge after completion
- Work in parallel with T04 benchmarking
