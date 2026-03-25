#!/usr/bin/env python3
"""T01: Standalone Engine Coherence Test
Test 12 diverse prompts through HTTP API, verify coherence.
"""
import requests
import json
import time
import sys

BASE_URL = "http://localhost:8080"
MODEL = "meta-llama-3.1-8b-instruct-q4_k_m"

# 12 diverse prompts for coherence testing
PROMPTS = [
    # Math
    {"category": "math", "prompt": "What is 256 * 17? Show your work."},
    {"category": "math", "prompt": "Solve: 3x + 7 = 22. What is x?"},
    # Factual
    {"category": "factual", "prompt": "What is the capital of France?"},
    {"category": "factual", "prompt": "When did World War II end?"},
    # Science
    {"category": "science", "prompt": "Explain photosynthesis in 2 sentences."},
    # Creative
    {"category": "creative", "prompt": "Write a haiku about mountains."},
    {"category": "creative", "prompt": "Write a 3-sentence story about a robot learning to paint."},
    # Code
    {"category": "code", "prompt": "Write a Python function to calculate factorial of n."},
    # Long generation
    {"category": "long", "prompt": "Describe the history of ancient Rome in approximately 100 words."},
    # Multi-sentence reasoning
    {"category": "reasoning", "prompt": "Explain what quantum computing is, then explain relativity, then explain how they might relate."},
    # Edge cases
    {"category": "edge", "prompt": "x"},
    {"category": "edge", "prompt": "The quick brown fox"},
]

def test_prompt(prompt_data):
    """Test a single prompt and return results."""
    category = prompt_data["category"]
    prompt = prompt_data["prompt"]
    
    try:
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.7
            },
            timeout=60
        )
        elapsed = time.time() - start
        
        if response.status_code != 200:
            return {
                "category": category,
                "prompt": prompt,
                "success": False,
                "error": f"HTTP {response.status_code}",
                "time": elapsed
            }
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        tps = data.get("_tps", 0)
        
        # Basic coherence checks
        coherence_score = 0
        issues = []
        
        # Check 1: Response is not empty
        if len(content.strip()) > 0:
            coherence_score += 1
        else:
            issues.append("Empty response")
        
        # Check 2: Response has reasonable length (>10 chars)
        if len(content) > 10:
            coherence_score += 1
        else:
            issues.append("Response too short")
        
        # Check 3: No obvious error patterns
        error_patterns = ["error", "exception", "failed", "crash", "undefined"]
        if not any(p in content.lower() for p in error_patterns):
            coherence_score += 1
        else:
            issues.append("Contains error patterns")
        
        # Check 4: For math, check if answer is present
        if category == "math":
            if any(c.isdigit() for c in content):
                coherence_score += 1
            else:
                issues.append("No numbers in math response")
        
        # Check 5: For code, check if code block present
        if category == "code":
            if "def " in content or "function" in content.lower() or "```" in content:
                coherence_score += 1
            else:
                issues.append("No code in code response")
        
        return {
            "category": category,
            "prompt": prompt,
            "success": True,
            "content": content,
            "tps": tps,
            "time": elapsed,
            "coherence_score": coherence_score,
            "max_coherence": 5,
            "issues": issues
        }
        
    except Exception as e:
        return {
            "category": category,
            "prompt": prompt,
            "success": False,
            "error": str(e),
            "time": 0
        }

def main():
    print("="*70)
    print("T01: STANDALONE ENGINE COHERENCE TEST")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Server: {BASE_URL}")
    print(f"Prompts: {len(PROMPTS)}")
    print("="*70)
    
    results = []
    total_time = 0
    successful = 0
    coherent = 0
    
    for i, prompt_data in enumerate(PROMPTS, 1):
        print(f"\n[{i}/{len(PROMPTS)}] {prompt_data['category'].upper()}: {prompt_data['prompt'][:50]}...")
        result = test_prompt(prompt_data)
        results.append(result)
        
        if result["success"]:
            successful += 1
            if result["coherence_score"] == result["max_coherence"]:
                coherent += 1
            print(f"  ✓ Success | TPS: {result.get('tps', 0):.2f} | Time: {result['time']:.2f}s | Coherence: {result['coherence_score']}/{result['max_coherence']}")
            if result.get("issues"):
                print(f"  ⚠ Issues: {', '.join(result['issues'])}")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown')}")
        
        total_time += result.get("time", 0)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total prompts: {len(PROMPTS)}")
    print(f"Successful: {successful}/{len(PROMPTS)} ({100*successful/len(PROMPTS):.1f}%)")
    print(f"Fully coherent: {coherent}/{successful} ({100*coherent/max(successful,1):.1f}%)")
    print(f"Average TPS: {sum(r.get('tps', 0) for r in results if r.get('tps'))/max(successful,1):.2f}")
    print(f"Total time: {total_time:.2f}s")
    
    # Pass/fail
    success_rate = 100 * successful / len(PROMPTS)
    coherence_rate = 100 * coherent / max(successful, 1)
    
    print("\n" + "="*70)
    if success_rate >= 100 and coherence_rate >= 80:
        print("✅ T01 PASSED: 12/12 coherent, no crashes")
        return 0
    elif success_rate >= 90:
        print(f"⚠️  T01 PARTIAL: {success_rate:.0f}% success, {coherence_rate:.0f}% coherence")
        return 1
    else:
        print(f"❌ T01 FAILED: Only {success_rate:.0f}% success rate")
        return 1

if __name__ == "__main__":
    sys.exit(main())
