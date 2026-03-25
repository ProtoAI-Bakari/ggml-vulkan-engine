#!/usr/bin/env python3
"""
T01: Verify standalone engine coherency with 12 diverse sequential requests
FIXED: Removed n_batch parameter that doesn't exist in GgmlLLM.__init__
Success: 12/12 coherent, no crash, consistent TPS
"""

import sys
import time
import json
from datetime import datetime

# Add current directory to path
sys.path.insert(0, '/home/z/AGENT')

try:
    from ggml_vllm_backend import GgmlLLM, SamplingParams
    print("✓ Successfully imported GgmlLLM")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# 12 diverse prompts covering different categories
TEST_PROMPTS = [
    # Math (3 prompts)
    {
        "category": "math",
        "prompt": "What is 237 + 489? Show your work.",
        "expected_pattern": "726"
    },
    {
        "category": "math",
        "prompt": "If a train travels 60 mph for 2.5 hours, how far does it go?",
        "expected_pattern": "150"
    },
    {
        "category": "math",
        "prompt": "What is the square root of 144?",
        "expected_pattern": "12"
    },
    
    # Factual (3 prompts)
    {
        "category": "factual",
        "prompt": "What is the capital of France?",
        "expected_pattern": "Paris"
    },
    {
        "category": "factual",
        "prompt": "Who wrote Romeo and Juliet?",
        "expected_pattern": "Shakespeare"
    },
    {
        "category": "factual",
        "prompt": "What year did World War II end?",
        "expected_pattern": "1945"
    },
    
    # Creative (3 prompts)
    {
        "category": "creative",
        "prompt": "Write a haiku about autumn leaves.",
        "expected_pattern": None  # Just check it generates something
    },
    {
        "category": "creative",
        "prompt": "Write a short story about a robot learning to paint.",
        "expected_pattern": None
    },
    {
        "category": "creative",
        "prompt": "Generate Python code to calculate fibonacci numbers.",
        "expected_pattern": "def"
    },
    
    # Edge cases (3 prompts)
    {
        "category": "edge",
        "prompt": "Hello",
        "expected_pattern": None  # Short prompt
    },
    {
        "category": "edge",
        "prompt": "Explain quantum entanglement in simple terms for a 10-year-old.",
        "expected_pattern": None  # Long prompt
    },
    {
        "category": "edge",
        "prompt": "What is 2+2?",
        "expected_pattern": "4"
    }
]

def test_coherence():
    """Run all 12 prompts and verify coherence"""
    
    print("\n" + "="*70)
    print("T01: STANDALONE ENGINE COHERENCY TEST")
    print("="*70)
    print(f"Started: {datetime.now()}")
    print(f"Model: Llama-3.1-8B-Instruct-Q4_K_M")
    print(f"Total prompts: {len(TEST_PROMPTS)}")
    print("="*70 + "\n")
    
    # Load model
    print("Loading model...")
    start_load = time.time()
    try:
        # FIXED: Removed n_batch parameter
        llm = GgmlLLM(
            model_path="/home/z/models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            n_ctx=2048
        )
        load_time = time.time() - start_load
        print(f"✓ Model loaded in {load_time:.1f}s\n")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Run tests
    results = []
    coherent_count = 0
    crash_count = 0
    tps_values = []
    
    for i, test in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}] {test['category'].upper()}: {test['prompt'][:50]}...")
        print("-" * 70)
        
        try:
            start_time = time.time()
            response = llm.generate(
                test['prompt'],
                params=SamplingParams(
                    temperature=0.7,
                    max_tokens=100,
                    top_p=0.9
                )
            )
            elapsed = time.time() - start_time
            
            tps = response.tps if hasattr(response, 'tps') else 0
            tps_values.append(tps)
            
            output_text = response.text if hasattr(response, 'text') else str(response)
            
            # Check coherence
            is_coherent = True
            reason = "OK"
            
            if test['expected_pattern']:
                if test['expected_pattern'].lower() in output_text.lower():
                    reason = f"Pattern '{test['expected_pattern']}' found"
                else:
                    is_coherent = False
                    reason = f"Pattern '{test['expected_pattern']}' NOT found"
            
            # Check for basic output
            if len(output_text.strip()) < 5:
                is_coherent = False
                reason = "Output too short"
            
            if is_coherent:
                coherent_count += 1
                status = "✓ COHERENT"
            else:
                status = f"✗ INCOHERENT: {reason}"
            
            result = {
                "index": i,
                "category": test['category'],
                "prompt": test['prompt'],
                "output": output_text[:200],  # Truncate for logging
                "tps": tps,
                "elapsed_ms": elapsed * 1000,
                "coherent": is_coherent,
                "reason": reason
            }
            results.append(result)
            
            print(f"  Output: {output_text[:150]}...")
            print(f"  TPS: {tps:.1f}, Time: {elapsed:.2f}s")
            print(f"  Status: {status}")
            
        except Exception as e:
            crash_count += 1
            result = {
                "index": i,
                "category": test['category'],
                "prompt": test['prompt'],
                "error": str(e),
                "coherent": False,
                "reason": f"CRASH: {e}"
            }
            results.append(result)
            print(f"  ✗ CRASH: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("T01 TEST SUMMARY")
    print("="*70)
    print(f"Total prompts: {len(TEST_PROMPTS)}")
    print(f"Coherent: {coherent_count}/{len(TEST_PROMPTS)} ({100*coherent_count/len(TEST_PROMPTS):.1f}%)")
    print(f"Crashes: {crash_count}")
    
    if tps_values:
        avg_tps = sum(tps_values) / len(tps_values)
        min_tps = min(tps_values)
        max_tps = max(tps_values)
        print(f"\nTPS Statistics:")
        print(f"  Average: {avg_tps:.1f}")
        print(f"  Min: {min_tps:.1f}")
        print(f"  Max: {max_tps:.1f}")
        print(f"  StdDev: {(sum((t - avg_tps)**2 for t in tps_values) / len(tps_values))**0.5:.1f}")
    
    # Success criteria
    success = coherent_count >= 12 and crash_count == 0
    print(f"\nSuccess Criteria: {'✓ MET' if success else '✗ NOT MET'}")
    print(f"  Required: 12/12 coherent, 0 crashes")
    print(f"  Actual: {coherent_count}/{len(TEST_PROMPTS)} coherent, {crash_count} crashes")
    print("="*70)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"T01_results_v4_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "test": "T01 Standalone Coherency",
            "timestamp": datetime.now().isoformat(),
            "model": "Llama-3.1-8B-Instruct-Q4_K_M",
            "total_prompts": len(TEST_PROMPTS),
            "coherent_count": coherent_count,
            "crash_count": crash_count,
            "tps_stats": {
                "avg": sum(tps_values) / len(tps_values) if tps_values else 0,
                "min": min(tps_values) if tps_values else 0,
                "max": max(tps_values) if tps_values else 0
            },
            "results": results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return success

if __name__ == "__main__":
    success = test_coherence()
    sys.exit(0 if success else 1)
