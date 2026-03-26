# M1 Ultra Memory Bandwidth — Real-World Analysis

## Apple's Claim: 800 GB/s
## Reality: 320-340 GB/s usable for inference

### The Math

Each M1 Max die:
- 4 LPDDR5 channels × 32-bit = 128-bit bus
- 6400 MT/s × 128-bit / 8 = 204.8 GB/s theoretical per die
- Real-world sustained: ~170 GB/s (cache misses, refresh, ECC)

Two dies (Ultra):
- 409.6 GB/s theoretical
- ~340 GB/s real-world

### Why Apple Says 800 GB/s
- Counts READ + WRITE separately: 204.8 read + 204.8 write = 409.6 "per die"
- × 2 dies = 819.2 ≈ "800 GB/s"
- LLM inference is READ-DOMINANT — writes are negligible
- Real usable: 2 × 204.8 × 0.82 efficiency = ~336 GB/s

### The 30GB Cliff
- Model < 30GB: layers fit in SLC (96MB on Ultra) → compute-bound → FAST
- Model > 30GB: SLC thrashing, every layer evicts previous → bandwidth-bound → SLOW
- MoE 60GB but 6GB active: active experts fit in SLC → behaves small → FAST

### TPS Predictions
| Model Size (active) | Bandwidth Needed | Predicted TPS | Measured |
|---------------------|-----------------|---------------|----------|
| 6GB (oss-120b active) | 6GB/tok | 340/6 = 56 | 45-60 ✅ |
| 18GB (35B-4bit) | 18GB/tok | 340/18 = 19 (but SLC helps) | 80+ est |
| 35GB (Coder-Next-4b) | ~8GB active | 340/8 = 42 | 51 ✅ |
| 76GB (Coder-Next-8b) | ~16GB active | 340/16 = 21 | 30 ✅ |

### CUDA Comparison (8x3090 TP8)
- Per-GPU bandwidth: 936 GB/s × 8 = 7.5 TB/s aggregate
- But cross-node 100G RDMA: ~12 GB/s per link
- Real bottleneck: NCCL all-reduce over 100G RoCEv2
- GDR upgrades: 50 → 65 → 75 → 95 → 105-111+ TPS (each RDMA optimization added 10-15%)
