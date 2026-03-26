#!/usr/bin/env python3
"""
T20: Memory bandwidth profiler for Apple Silicon.
Minimal version for quick measurement.
"""

import time
import numpy as np
import json

def measure_bandwidth(size_mb: float = 512.0, num_iterations: int = 20) -> float:
    """Measure memory bandwidth with small buffer."""
    buffer_size_bytes = int(size_mb * 1024 * 1024)
    num_elements = buffer_size_bytes // 4
    
    input_data = np.random.randn(num_elements).astype(np.float32)
    output_data = np.zeros(num_elements, dtype=np.float32)
    
    # Warmup
    output_data[:] = input_data
    
    # Measure
    start = time.perf_counter()
    for i in range(num_iterations):
        output_data[:] = input_data * 1.0001
        input_data[:] = output_data
    elapsed = time.perf_counter() - start
    
    total_bytes = buffer_size_bytes * 2 * num_iterations
    bandwidth_gb_s = (total_bytes / 1024**3) / elapsed
    return bandwidth_gb_s

def main():
    print("Memory Bandwidth Profiler")
    print("Theoretical peak: 800 GB/s")
    
    # Quick test
    bw = measure_bandwidth(512, 20)
    print(f"512 MB buffer: {bw:.1f} GB/s ({bw/800*100:.1f}% of peak)")
    
    # llama.cpp analysis
    print("\n=== llama.cpp Analysis ===")
    print("8B Q4_K_M: 4.6 GB per token")
    print(f"22 TPS → {4.6*22:.1f} GB/s ({4.6*22/800*100:.1f}%)")
    print(f"30 TPS → {4.6*30:.1f} GB/s ({4.6*30/800*100:.1f}%)")
    
    # Save
    output = {
        "theoretical_peak_gb_s": 800,
        "measured_bandwidth_gb_s": bw,
        "utilization_percent": bw/800*100,
        "model": "Apple M1 Ultra",
        "tps_22_bandwidth_gb_s": 4.6*22,
        "tps_30_bandwidth_gb_s": 4.6*30,
    }
    
    with open("bandwidth_profile.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to bandwidth_profile.json")

if __name__ == "__main__":
    main()
