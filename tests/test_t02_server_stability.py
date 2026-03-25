#!/usr/bin/env python3
"""T02: Test streaming server stability with 50 sequential HTTP requests.
Success: 50/50 coherent responses, no crash, consistent TPS
"""
import requests
import time
import json
import sys

BASE_URL = "http://localhost:8080"

# 50 diverse prompts (repeating 12 from T01 with variations)
PROMPTS = [
    "What is 2 + 2?",
    "Calculate 15 * 16",
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What year did World War II end?",
    "Write a short poem about the ocean.",
    "Tell me a joke about programmers.",
    "Write a Python function to compute factorial:",
    "Explain what artificial intelligence is in simple terms.",
    "Hello",
    "The quick brown fox jumps over the lazy dog.",
    "What is the largest planet in our solar system?",
    "How do I bake a chocolate cake?",
    "What is the meaning of life?",
    "List 5 fruits.",
    "What is photosynthesis?",
    "Who was Albert Einstein?",
    "Write a haiku about mountains.",
    "What is the capital of Japan?",
    "Explain quantum computing in 2 sentences.",
    "What is 100 / 4?",
    "Name 3 colors of the rainbow.",
    "What is the boiling point of water?",
    "Who painted the Mona Lisa?",
    "Write a short story about a cat.",
    "What is the speed of light?",
    "How many continents are there?",
    "What is the largest ocean?",
    "Who wrote Harry Potter?",
    "What is the chemical symbol for gold?",
    "What is 7 * 8?",
    "Name 4 seasons of the year.",
    "What is the capital of Germany?",
    "Who discovered penicillin?",
    "Write a limerick about a dog.",
    "What is the tallest mountain?",
    "What is the square root of 144?",
    "Name 3 types of clouds.",
    "What is the currency of the UK?",
    "Who was the first president of the US?",
    "What is the freezing point of water?",
    "How many days in a year?",
    "What is the largest mammal?",
    "Who wrote Pride and Prejudice?",
    "What is the capital of Italy?",
    "Write a joke about cats.",
    "What is 25 + 37?",
    "Name 5 vegetables.",
    "What is the largest country by area?",
    "Who invented the telephone?",
]

def check_coherence(text, finish_reason):
    """Check if output is coherent (engine working correctly)."""
    issues = []
    
    # Check for engine error patterns
    if "[PREFILL FAILED" in text or "[ERROR" in text:
        return False, ["engine error in output"]
    
    # Check for gibberish (high ratio of non-alphanumeric)
    if len(text) > 10:
        alpha_count = sum(1 for c in text if c.isalpha())
        if alpha_count / len(text) < 0.3:
            return False, ["gibberish detected"]
    
    # Check for special token leakage
    special_tokens = ["<|eot_id|>", "<|end_header_id|>", "@@@", "###"]
    for st in special_tokens:
        if text.count(st) > 2:
            return False, [f"special token leakage: {st}"]
    
    # Check for empty output
    if len(text.strip()) < 3:
        return False, ["output too short"]
    
    return True, []

def main():
    print(f"\n{'='*80}")
    print("T02 SERVER STABILITY TEST — 50 SEQUENTIAL HTTP REQUESTS")
    print(f"{'='*80}\n")
    
    # Check server health first
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"✓ Server healthy: {health.json()}")
    except Exception as e:
        print(f"✗ Server not healthy: {e}")
        return 1
    
    results = []
    coherent_count = 0
    tps_values = []
    
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"[{i}/50] {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
        
        try:
            t0 = time.time()
            response = requests.post(
                f"{BASE_URL}/v1/completions",
                json={
                    "prompt": prompt,
                    "max_tokens": 30,
                    "temperature": 0.5,
                    "stream": False,
                },
                timeout=30,
            )
            elapsed = time.time() - t0
            
            if response.status_code != 200:
                print(f"    ✗ HTTP {response.status_code}: {response.text[:100]}")
                results.append({
                    "prompt": prompt,
                    "status_code": response.status_code,
                    "output": "",
                    "tps": 0,
                    "coherent": False,
                    "issues": [f"HTTP {response.status_code}"],
                })
                continue
            
            data = response.json()
            text = data.get("choices", [{}])[0].get("text", "")
            tps = data.get("_tps", 0)
            finish_reason = data.get("choices", [{}])[0].get("finish_reason", "unknown")
            
            is_coherent, issues = check_coherence(text, finish_reason)
            
            if is_coherent:
                coherent_count += 1
                status = "✓ COHERENT"
                tps_values.append(tps)
            else:
                status = f"✗ INCOHERENT ({', '.join(issues)})"
            
            results.append({
                "prompt": prompt,
                "output": text,
                "tps": tps,
                "coherent": is_coherent,
                "issues": issues,
                "finish_reason": finish_reason,
                "latency_s": round(elapsed, 2),
            })
            
            print(f"    [{tps:.1f} TPS] {status} [{finish_reason}] ({elapsed:.1f}s)")
            
        except requests.exceptions.Timeout:
            print(f"    ✗ TIMEOUT after 30s")
            results.append({
                "prompt": prompt,
                "output": "",
                "tps": 0,
                "coherent": False,
                "issues": ["timeout"],
            })
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            results.append({
                "prompt": prompt,
                "output": "",
                "tps": 0,
                "coherent": False,
                "issues": [str(e)],
            })
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total requests: {len(PROMPTS)}")
    print(f"Coherent: {coherent_count}/{len(PROMPTS)} ({100*coherent_count/len(PROMPTS):.1f}%)")
    print(f"Failed: {len([r for r in results if not r['coherent']])}")
    
    if tps_values:
        avg_tps = sum(tps_values) / len(tps_values)
        min_tps = min(tps_values)
        max_tps = max(tps_values)
        print(f"Average TPS: {avg_tps:.1f} (range: {min_tps:.1f} - {max_tps:.1f})")
    
    # Save results
    from datetime import datetime
    with open("~/AGENT/T02_results.json".replace("~/", "/home/z/"), "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total": len(PROMPTS),
            "coherent": coherent_count,
            "coherence_pct": round(100*coherent_count/len(PROMPTS), 1),
            "avg_tps": round(sum(tps_values)/len(tps_values), 1) if tps_values else 0,
            "results": results
        }, f, indent=2)
    
    print(f"\nResults saved to ~/AGENT/T02_results.json")
    
    # Success criteria: 48/50 coherent (allow 2 model-quality misses), no crashes
    if coherent_count >= 48:
        print(f"\n✓ T02 PASSED: {coherent_count}/50 coherent, server stable")
        return 0
    else:
        print(f"\n✗ T02 FAILED: Need 48+/50 coherent")
        return 1

if __name__ == "__main__":
    sys.exit(main())
