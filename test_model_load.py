#!/usr/bin/env python3
"""
Quick model load test with surgical layer migration.

This test attempts to load a small model and measures load time.
It should complete in under 30 seconds to avoid Ghost Load.
"""

import sys
import time

sys.path.insert(0, '/home/z/GITDEV/vllm')

from vllm import LLM

def test_small_model_load():
    """Test loading a small model with surgical migration."""
    print("=" * 60)
    print("Model Load Test with Surgical Migration")
    print("=" * 60)
    
    # Use a very small model for quick testing
    model_name = "facebook/opt-125m"
    
    print(f"\nAttempting to load: {model_name}")
    print("This test measures load time to detect Ghost Loads...")
    
    start_time = time.time()
    
    try:
        print("\nInitializing LLM engine...")
        llm = LLM(
            model=model_name,
            enforce_eager=True,  # Disable compilation for faster startup
            gpu_memory_utilization=0.5,
            max_model_len=128,
            dtype="float16",
            disable_log_stats=True,
        )
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 60)
        print(f"Model loaded successfully in {elapsed:.2f} seconds")
        print("=" * 60)
        
        # Check against Ghost Load threshold
        if elapsed < 30:
            print(f"✓ PASS: Load time ({elapsed:.2f}s) is within Ghost Load threshold (30s)")
            return True
        else:
            print(f"✗ FAIL: Load time ({elapsed:.2f}s) exceeds Ghost Load threshold (30s)")
            return False
            
    except Exception as e:
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"✗ Model load failed after {elapsed:.2f} seconds")
        print(f"Error: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_small_model_load()
    sys.exit(0 if success else 1)