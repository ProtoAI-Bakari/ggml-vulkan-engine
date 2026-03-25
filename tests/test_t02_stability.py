#!/usr/bin/env python3
"""T02: Server Startup + Stability Test
Goal: 50 sequential HTTP requests without crash, all coherent.
"""
import requests
import json
import time
import sys
import subprocess

BASE_URL = "http://localhost:8080"
MODEL = "meta-llama-3.1-8b-instruct-q4_k_m"

# 50 diverse prompts for stability testing
PROMPTS = [
    # Math (10)
    "What is 256 * 17?",
    "Solve: 3x + 7 = 22. What is x?",
    "What is the square root of 144?",
    "Calculate 15% of 200.",
    "What is 2^10?",
    "Add 1234 + 5678.",
    "What is 100 / 4?",
    "Multiply 12 * 13.",
    "What is 5! (factorial)?",
    "Solve: x^2 = 81. What is x?",
    # Factual (10)
    "What is the capital of France?",
    "When did World War II end?",
    "Who wrote Romeo and Juliet?",
    "What is the largest planet?",
    "What is the speed of light?",
    "Who was the first president of the US?",
    "What is the chemical symbol for gold?",
    "How many continents are there?",
    "What is the boiling point of water?",
    "Who painted the Mona Lisa?",
    # Science (10)
    "Explain photosynthesis in 2 sentences.",
    "What is gravity?",
    "What is DNA?",
    "Explain the water cycle.",
    "What is an atom?",
    "What is evolution?",
    "Explain Newton's first law.",
    "What is the periodic table?",
    "What is a black hole?",
    "Explain how vaccines work.",
    # Creative (10)
    "Write a haiku about mountains.",
    "Write a 3-sentence story about a robot.",
    "Write a poem about the ocean.",
    "Describe a sunset in 3 sentences.",
    "Write a short joke.",
    "Create a character name for a fantasy novel.",
    "Write a 2-line dialogue between friends.",
    "Describe your favorite color.",
    "Write a haiku about technology.",
    "Tell me a one-sentence mystery.",
    # Code (5)
    "Write a Python function to calculate factorial.",
    "Write a JavaScript function to reverse a string.",
    "Write SQL to select all users from table.",
    "Write a bash script to list files.",
    "Write HTML for a simple button.",
    # Edge cases (5)
    "x",
    "The quick brown fox",
    "1",
    "Hello",
    "..."
]

def test_prompt(prompt, idx):
    """Test a single prompt."""
    try:
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 50,
                "temperature": 0.7
            },
            timeout=60
        )
        elapsed = time.time() - start
        
        if response.status_code != 200:
            return {"idx": idx, "success": False, "error": f"HTTP {response.status_code}", "time": elapsed}
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        tps = data.get("_tps", 0)
        
        # Basic coherence
        coherent = len(content.strip()) > 5 and "error:" not in content.lower()
        
        return {
            "idx": idx, "success": True, "content": content[:50],
            "tps": tps, "time": elapsed, "coherent": coherent
        }
        
    except Exception as e:
        return {"idx": idx, "success": False, "error": str(e), "time": 0}

def main():
    print("="*70)
    print("T02: SERVER STARTUP + STABILITY TEST")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Server: {BASE_URL}")
    print(f"Requests: {len(PROMPTS)}")
    print("="*70)
    
    # Check server health first
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"\n✓ Server health: {health.json()}")
    except Exception as e:
        print(f"\n✗ Server health check failed: {e}")
        return 1
    
    results = []
    total_time = 0
    successful = 0
    coherent = 0
    tps_values = []
    
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n[{i}/{len(PROMPTS)}] {prompt[:40]}...", end=" ")
        result = test_prompt(prompt, i)
        results.append(result)
        
        if result["success"]:
            successful += 1
            if result["coherent"]:
                coherent += 1
            tps_values.append(result["tps"])
            print(f"✓ | TPS: {result['tps']:.1f} | {result['time']:.1f}s")
        else:
            print(f"✗ | {result.get('error', 'Unknown')}")
        
        total_time += result.get("time", 0)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total requests: {len(PROMPTS)}")
    print(f"Successful: {successful}/{len(PROMPTS)} ({100*successful/len(PROMPTS):.1f}%)")
    print(f"Coherent: {coherent}/{successful} ({100*coherent/max(successful,1):.1f}%)")
    print(f"Average TPS: {sum(tps_values)/max(len(tps_values),1):.2f}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Avg time/request: {total_time/len(PROMPTS):.2f}s")
    
    print("\n" + "="*70)
    if successful == len(PROMPTS) and coherent >= 48:
        print("✅ T02 PASSED: 50/50 successful, 48+ coherent")
        return 0
    elif successful >= 45:
        print(f"⚠️  T02 PARTIAL: {successful}/50 successful")
        return 1
    else:
        print(f"❌ T02 FAILED: Only {successful}/50 successful")
        return 1

if __name__ == "__main__":
    sys.exit(main())
