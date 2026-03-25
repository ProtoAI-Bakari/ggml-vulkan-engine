#!/usr/bin/env python3
"""T01: Standalone Engine Coherence Test v2
Fixed coherence scoring - less strict error pattern detection.
"""
import requests
import json
import time
import sys

BASE_URL = "http://localhost:8080"
MODEL = "meta-llama-3.1-8b-instruct-q4_k_m"

PROMPTS = [
    {"category": "math", "prompt": "What is 256 * 17? Show your work."},
    {"category": "math", "prompt": "Solve: 3x + 7 = 22. What is x?"},
    {"category": "factual", "prompt": "What is the capital of France?"},
    {"category": "factual", "prompt": "When did World War II end?"},
    {"category": "science", "prompt": "Explain photosynthesis in 2 sentences."},
    {"category": "creative", "prompt": "Write a haiku about mountains."},
    {"category": "creative", "prompt": "Write a 3-sentence story about a robot learning to paint."},
    {"category": "code", "prompt": "Write a Python function to calculate factorial of n."},
    {"category": "long", "prompt": "Describe the history of ancient Rome in approximately 100 words."},
    {"category": "reasoning", "prompt": "Explain what quantum computing is, then explain relativity, then explain how they might relate."},
    {"category": "edge", "prompt": "x"},
    {"category": "edge", "prompt": "The quick brown fox"},
]

def test_prompt(prompt_data):
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
            return {"category": category, "prompt": prompt, "success": False, "error": f"HTTP {response.status_code}", "time": elapsed}
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        tps = data.get("_tps", 0)
        
        # Coherence checks
        coherence_score = 0
        issues = []
        
        if len(content.strip()) > 0:
            coherence_score += 1
        else:
            issues.append("Empty response")
        
        if len(content) > 10:
            coherence_score += 1
        else:
            issues.append("Response too short")
        
        # Check for ACTUAL errors (not just the word "error")
        if "error:" in content.lower() or "exception:" in content.lower() or "traceback" in content.lower():
            issues.append("Contains error output")
        else:
            coherence_score += 1
        
        if category == "math" and any(c.isdigit() for c in content):
            coherence_score += 1
        elif category != "math":
            coherence_score += 1  # N/A for non-math
        
        if category == "code" and ("def " in content or "function" in content.lower() or "```" in content):
            coherence_score += 1
        elif category != "code":
            coherence_score += 1  # N/A for non-code
        
        return {
            "category": category, "prompt": prompt, "success": True,
            "content": content, "tps": tps, "time": elapsed,
            "coherence_score": coherence_score, "max_coherence": 5, "issues": issues
        }
        
    except Exception as e:
        return {"category": category, "prompt": prompt, "success": False, "error": str(e), "time": 0}

def main():
    print("="*70)
    print("T01: STANDALONE ENGINE COHERENCE TEST (v2)")
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
            if result["coherence_score"] >= 4:  # 4/5 or better = coherent
                coherent += 1
            print(f"  ✓ Success | TPS: {result.get('tps', 0):.2f} | Time: {result['time']:.2f}s | Coherence: {result['coherence_score']}/{result['max_coherence']}")
            if result.get("issues"):
                print(f"  ⚠ Issues: {', '.join(result['issues'])}")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown')}")
        
        total_time += result.get("time", 0)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total prompts: {len(PROMPTS)}")
    print(f"Successful: {successful}/{len(PROMPTS)} ({100*successful/len(PROMPTS):.1f}%)")
    print(f"Coherent (4+/5): {coherent}/{successful} ({100*coherent/max(successful,1):.1f}%)")
    print(f"Average TPS: {sum(r.get('tps', 0) for r in results if r.get('tps'))/max(successful,1):.2f}")
    print(f"Total time: {total_time:.2f}s")
    
    print("\n" + "="*70)
    if successful == len(PROMPTS) and coherent >= 10:
        print("✅ T01 PASSED: 12/12 successful, 10+ coherent")
        return 0
    elif successful >= 10:
        print(f"⚠️  T01 PARTIAL: {successful}/12 successful")
        return 1
    else:
        print(f"❌ T01 FAILED: Only {successful}/12 successful")
        return 1

if __name__ == "__main__":
    sys.exit(main())
