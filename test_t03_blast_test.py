#!/usr/bin/env python3
"""T03: Coherency blast test — 50 diverse prompts through HTTP server.
Includes edge cases: math, factual, creative, long-gen, multi-sentence, unicode, empty follow-up.
Success: 48/50+ coherent (allow 2 model-quality misses, 0 crashes)
"""
import requests
import time
import json
import sys

BASE_URL = "http://localhost:8080"

# 50 diverse prompts with edge cases
PROMPTS = [
    # Math (10)
    ("2 + 2 =", "Greedy math"),
    ("15 * 16 =", "Greedy arithmetic"),
    ("sqrt(144) =", "Greedy root"),
    ("100 / 4 =", "Greedy division"),
    ("7 * 8 =", "Greedy multiplication"),
    ("What is 25 + 37?", "Greedy addition"),
    ("Calculate 123 * 456", "Greedy large mult"),
    ("10! =", "Greedy factorial"),
    ("What is 2^10?", "Greedy power"),
    ("1000 - 377 =", "Greedy subtraction"),
    
    # Factual (10)
    ("Capital of France?", "Geography"),
    ("Capital of Japan?", "Geography"),
    ("Capital of Germany?", "Geography"),
    ("Capital of Italy?", "Geography"),
    ("Who wrote Hamlet?", "Literature"),
    ("Who wrote Pride and Prejudice?", "Literature"),
    ("Who wrote Harry Potter?", "Literature"),
    ("When did WWII end?", "History"),
    ("Who discovered penicillin?", "Science"),
    ("What is the speed of light?", "Physics"),
    
    # Creative (10)
    ("Write a poem about rain.", "Temp=0.7"),
    ("Write a poem about stars.", "Temp=0.7"),
    ("Write a haiku about winter.", "Temp=0.7"),
    ("Tell me a joke about AI.", "Temp=0.8"),
    ("Tell me a joke about food.", "Temp=0.8"),
    ("Write a short story about a robot.", "Temp=0.6"),
    ("Write a short story about a forest.", "Temp=0.6"),
    ("Describe a beautiful sunset.", "Temp=0.7"),
    ("Describe a busy city street.", "Temp=0.7"),
    ("Write a limerick about a cat.", "Temp=0.8"),
    
    # Code (5)
    ("def hello():", "Python function"),
    ("function add(a, b) {", "JavaScript function"),
    ("Write a SQL query to select all:", "SQL"),
    ("import numpy as np\narr = np.array([", "NumPy"),
    ("CREATE TABLE users (", "SQL schema"),
    
    # Long generation (5)
    ("Explain machine learning in simple terms.", "Temp=0.5, long"),
    ("What is the history of the internet?", "Temp=0.5, long"),
    ("Describe the process of photosynthesis.", "Temp=0.5, long"),
    ("What are the causes of climate change?", "Temp=0.5, long"),
    ("Explain how a computer works.", "Temp=0.5, long"),
    
    # Edge cases (10)
    ("Hi", "Very short"),
    ("?", "Single char"),
    ("Hello world", "Simple"),
    ("The quick brown fox jumps over the lazy dog. Now write another sentence.", "Medium"),
    ("\u4e2d\u6587\u6d4b\u8bd5", "Unicode Chinese"),
    ("\u00e9\u00e0\u00fc\u00f1", "Unicode accented"),
    ("", "Empty prompt"),
    ("   ", "Whitespace only"),
    ("\n\n\n", "Newlines only"),
    ("Test\t\t\ttabs", "Tabs"),
]

def check_coherence(text, finish_reason, prompt):
    """Check if output is coherent (engine working correctly)."""
    issues = []
    
    # Check for engine error patterns
    if "[PREFILL FAILED" in text or "[ERROR" in text:
        return False, ["engine error in output"]
    
    # Check for gibberish (high ratio of non-alphanumeric for non-empty prompts)
    if len(prompt.strip()) > 0 and len(text) > 10:
        alpha_count = sum(1 for c in text if c.isalpha() or c.isdigit() or ord(c) > 127)
        if alpha_count / len(text) < 0.3:
            return False, ["gibberish detected"]
    
    # Check for special token leakage
    special_tokens = ["<|eot_id|>", "<|end_header_id|>", "@@@", "###"]
    for st in special_tokens:
        if text.count(st) > 2:
            return False, [f"special token leakage: {st}"]
    
    # For empty prompts, allow any output (model can be creative)
    # For non-empty prompts, require at least some output
    if len(prompt.strip()) > 0 and len(text.strip()) < 3:
        return False, ["output too short"]
    
    return True, []

def main():
    print(f"\n{'='*80}")
    print("T03 COHERENCY BLAST TEST — 50 DIVERSE PROMPTS (HTTP)")
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
    category_stats = {}
    
    for i, (prompt, category) in enumerate(PROMPTS, 1):
        print(f"[{i}/50] {category}: {prompt[:40]}{'...' if len(prompt) > 40 else ''}")
        
        # Set sampling params based on category
        if "Greedy" in category:
            params = {"temperature": 0, "max_tokens": 20}
        elif "long" in category:
            params = {"temperature": 0.5, "max_tokens": 60, "top_p": 0.9}
        elif "Temp=0.8" in category:
            params = {"temperature": 0.8, "max_tokens": 30, "top_k": 40}
        elif "Temp=0.7" in category:
            params = {"temperature": 0.7, "max_tokens": 30, "top_k": 40}
        elif "Temp=0.6" in category:
            params = {"temperature": 0.6, "max_tokens": 40}
        else:
            params = {"temperature": 0.5, "max_tokens": 30}
        
        try:
            t0 = time.time()
            response = requests.post(
                f"{BASE_URL}/v1/completions",
                json={
                    "prompt": prompt,
                    "stream": False,
                    **params
                },
                timeout=30,
            )
            elapsed = time.time() - t0
            
            if response.status_code != 200:
                print(f"    ✗ HTTP {response.status_code}")
                results.append({
                    "prompt": prompt,
                    "category": category,
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
            
            is_coherent, issues = check_coherence(text, finish_reason, prompt)
            
            # Track category stats
            if category not in category_stats:
                category_stats[category] = {"total": 0, "coherent": 0}
            category_stats[category]["total"] += 1
            if is_coherent:
                category_stats[category]["coherent"] += 1
            
            if is_coherent:
                coherent_count += 1
                status = "✓"
                tps_values.append(tps)
            else:
                status = f"✗({', '.join(issues)[:10]})"
            
            results.append({
                "prompt": prompt,
                "category": category,
                "output": text,
                "tps": tps,
                "coherent": is_coherent,
                "issues": issues,
                "finish_reason": finish_reason,
                "latency_s": round(elapsed, 2),
            })
            
            print(f"    [{tps:.1f} TPS] {status} [{finish_reason}]")
            
        except requests.exceptions.Timeout:
            print(f"    ✗ TIMEOUT")
            results.append({
                "prompt": prompt,
                "category": category,
                "output": "",
                "tps": 0,
                "coherent": False,
                "issues": ["timeout"],
            })
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            results.append({
                "prompt": prompt,
                "category": category,
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
    
    # Category breakdown
    print(f"\nCategory breakdown:")
    for cat, stats in sorted(category_stats.items()):
        pct = 100 * stats["coherent"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {cat}: {stats['coherent']}/{stats['total']} ({pct:.0f}%)")
    
    # Save results
    from datetime import datetime
    with open("/home/z/AGENT/T03_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total": len(PROMPTS),
            "coherent": coherent_count,
            "coherence_pct": round(100*coherent_count/len(PROMPTS), 1),
            "avg_tps": round(sum(tps_values)/len(tps_values), 1) if tps_values else 0,
            "category_stats": category_stats,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to ~/AGENT/T03_results.json")
    
    # Success criteria: 48/50 coherent (allow 2 model-quality misses)
    if coherent_count >= 48:
        print(f"\n✓ T03 PASSED: {coherent_count}/50 coherent")
        return 0
    else:
        print(f"\n✗ T03 FAILED: Need 48+/50 coherent")
        return 1

if __name__ == "__main__":
    sys.exit(main())
