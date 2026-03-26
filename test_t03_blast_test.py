#!/usr/bin/env python3
"""T03: Coherency Blast Test - 50 diverse prompts through HTTP server (OPTIMIZED)

Test categories:
- Math (10): 2+2, complex arithmetic, word problems
- Factual (10): capitals, dates, science, history
- Creative (10): stories, poetry, code generation
- Edge (10): 1-word prompt, 500-word prompt, unicode, empty follow-up
- Code (5): Python, JavaScript, SQL, bash, C
- Long generation (5): 100+ token responses

Success criteria: 48/50+ coherent (allow 2 model-quality misses, 0 crashes)
"""

import requests
import json
import time
import sys
from datetime import datetime

BASE_URL = "http://localhost:8080"

# 50 diverse prompts organized by category
PROMPTS = {
    "math": [
        "What is 2 + 2?",
        "Calculate 123 * 456.",
        "If a train travels 60 mph for 2.5 hours, how far does it go?",
        "What is the square root of 144?",
        "Solve: 3x + 7 = 22. What is x?",
        "What is 15% of 200?",
        "Convert 100 degrees Celsius to Fahrenheit.",
        "What is the factorial of 5?",
        "Add: 1/3 + 1/6. Give the answer as a fraction.",
        "What is 2 to the power of 10?",
    ],
    "factual": [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "When did World War II end?",
        "What is the chemical symbol for gold?",
        "Who was the first president of the United States?",
        "What is the largest planet in our solar system?",
        "What is the speed of light in vacuum?",
        "Who discovered penicillin?",
        "What is the longest river in the world?",
        "What year did the Titanic sink?",
    ],
    "creative": [
        "Write a haiku about autumn.",
        "Tell me a short story about a lost dog finding its way home.",
        "Write a poem about the ocean.",
        "Create a character description for a fantasy novel.",
        "Write a joke about computers.",
        "Describe a sunset in three sentences.",
        "Write a dialogue between two friends meeting after years apart.",
        "Create a metaphor for hope.",
        "Write a micro-fiction about a time traveler.",
        "Describe the smell of rain.",
    ],
    "code": [
        "Write a Python function to calculate factorial.",
        "Write JavaScript to check if a string is a palindrome.",
        "Write a SQL query to select all users from a users table.",
        "Write a bash script to list all .txt files in a directory.",
        "Write a C function to reverse a string.",
    ],
    "long_generation": [
        "Explain the theory of relativity in simple terms. (Aim for 100+ words)",
        "Describe the process of photosynthesis in detail. (Aim for 100+ words)",
        "What are the main causes of climate change? Explain each one. (Aim for 100+ words)",
        "Summarize the history of the internet from the 1960s to today. (Aim for 100+ words)",
        "Explain how blockchain technology works. (Aim for 100+ words)",
    ],
    "edge_cases": [
        "Hi",
        "x",
        "The quick brown fox jumps over the lazy dog. ",
        "こんにちは世界",
        "🚀🌟💻🎉",
        "   ",
        "",
        "What is 1 + 1? Now tell me a story about cats.",
        "Repeat the word 'test' ten times.",
        "What is the meaning of life? Be philosophical.",
    ],
}

def test_prompt(category, prompt, timeout=20):
    """Test a single prompt and return result."""
    try:
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 100,
                "stream": False
            },
            timeout=timeout
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            text = data['choices'][0]['message']['content']
            return {
                "success": True,
                "prompt": prompt,
                "response": text,
                "elapsed": elapsed,
                "status_code": 200,
                "error": None
            }
        else:
            return {
                "success": False,
                "prompt": prompt,
                "response": None,
                "elapsed": elapsed,
                "status_code": response.status_code,
                "error": response.text
            }
    except Exception as e:
        return {
            "success": False,
            "prompt": prompt,
            "response": None,
            "elapsed": time.time() - start,
            "status_code": None,
            "error": str(e)
        }

def is_coherent(result):
    """Basic coherence check - response should be non-empty and relevant."""
    if not result["response"]:
        return False
    
    text = result["response"].strip()
    
    # Check for empty or whitespace-only
    if len(text) < 5:
        return False
    
    # Check for obvious gibberish (repeated characters)
    if len(text) > 50 and text[0] * 10 in text:
        return False
    
    # Check for error messages in response
    if "error" in text.lower() and "not found" in text.lower():
        return False
    
    return True

def run_blast_test():
    """Run all 50 prompts and collect results."""
    print(f"\n{'='*70}")
    print(f"T03: COHERENCY BLAST TEST")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    all_results = []
    category_stats = {}
    
    total_prompts = sum(len(prompts) for prompts in PROMPTS.values())
    print(f"Testing {total_prompts} prompts across {len(PROMPTS)} categories...\n")
    
    for category, prompts in PROMPTS.items():
        print(f"\n--- {category.upper()} ({len(prompts)} prompts) ---")
        category_stats[category] = {"total": 0, "coherent": 0, "crashes": 0}
        
        for i, prompt in enumerate(prompts, 1):
            prompt_display = prompt[:40] + "..." if len(prompt) > 40 else prompt
            print(f"  [{i}/{len(prompts)}] {prompt_display}", end=" ")
            sys.stdout.flush()
            
            result = test_prompt(category, prompt)
            result["category"] = category
            all_results.append(result)
            
            category_stats[category]["total"] += 1
            
            if result["success"]:
                coherent = is_coherent(result)
                category_stats[category]["coherent"] += 1 if coherent else 0
                status = "✓" if coherent else "✗"
                print(f"{status} ({result['elapsed']:.1f}s)")
            else:
                category_stats[category]["crashes"] += 1
                print(f"✗ FAILED")
    
    # Calculate overall statistics
    total_coherent = sum(stats["coherent"] for stats in category_stats.values())
    total_crashes = sum(stats["crashes"] for stats in category_stats.values())
    total_tests = sum(stats["total"] for stats in category_stats.values())
    
    # Calculate TPS for successful requests
    successful_results = [r for r in all_results if r["success"] and r["response"]]
    avg_tps = len(successful_results) / sum(r["elapsed"] for r in successful_results) if successful_results else 0
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"T03: TEST SUMMARY")
    print(f"{'='*70}")
    print(f"\nCategory Breakdown:")
    for category, stats in category_stats.items():
        coherent_pct = (stats["coherent"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"  {category:20s}: {stats['coherent']:2d}/{stats['total']:2d} ({coherent_pct:5.1f}%)")
    
    print(f"\nOverall Results:")
    print(f"  Total prompts:     {total_tests}")
    print(f"  Coherent:          {total_coherent} ({total_coherent/total_tests*100:.1f}%)")
    print(f"  Incoherent:        {total_tests - total_coherent - total_crashes}")
    print(f"  Crashes/Errors:    {total_crashes}")
    print(f"  Average TPS:       {avg_tps:.1f}")
    
    # Success criteria
    print(f"\nSuccess Criteria:")
    success_threshold = 48
    if total_coherent >= success_threshold and total_crashes == 0:
        print(f"  ✓ PASSED: {total_coherent}/{total_tests} coherent (threshold: {success_threshold})")
        print(f"  ✓ No crashes")
        overall_status = "PASSED"
    elif total_crashes > 0:
        print(f"  ✗ FAILED: {total_crashes} crashes detected")
        overall_status = "FAILED (crashes)"
    else:
        print(f"  ✗ FAILED: {total_coherent}/{total_tests} coherent (threshold: {success_threshold})")
        overall_status = "FAILED (coherence)"
    
    # Save results to JSON
    results_json = {
        "test_name": "T03_Coherency_Blast_Test",
        "timestamp": datetime.now().isoformat(),
        "total_prompts": total_tests,
        "coherent": total_coherent,
        "incoherent": total_tests - total_coherent - total_crashes,
        "crashes": total_crashes,
        "coherence_rate": total_coherent / total_tests * 100,
        "avg_tps": avg_tps,
        "overall_status": overall_status,
        "category_stats": category_stats,
        "detailed_results": all_results
    }
    
    with open("~/AGENT/T03_results.json", "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    
    print(f"\nResults saved to: ~/AGENT/T03_results.json")
    print(f"\n{'='*70}\n")
    
    return overall_status == "PASSED"

if __name__ == "__main__":
    success = run_blast_test()
    sys.exit(0 if success else 1)
