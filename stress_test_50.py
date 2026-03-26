#!/usr/bin/env python3
"""50-request stress test for ggml_server.py - NO DELAY"""
import requests
import json
import time
import sys

BASE_URL = "http://localhost:8080"

PROMPTS = [
    "What is the capital of France?", "Explain quantum computing simply.",
    "Write a haiku about autumn.", "Pizza ingredients?", "Who wrote Romeo and Juliet?",
    "Calculate 15% of 240.", "Boiling point of water?", "Three programming languages?",
    "WWII end year?", "Photosynthesis briefly.", "Largest planet?", "Ocean poem.",
    "Square root of 144?", "Mona Lisa painter?", "Water chemical formula?",
    "Seven continents.", "What is machine learning?", "Scrambled eggs recipe.",
    "Speed of light?", "First US president?", "What is gravity?", "Computer joke.",
    "Tallest mountain?", "Rainbow colors.", "What is an API?", "Japan currency?",
    "Penicillin discoverer?", "Largest ocean?", "Robot short story.", "Australia capital?",
    "Days in leap year?", "What is DNA?", "Pride and Prejudice author?",
    "Water freezing Celsius?", "Three primary colors.", "What is the internet?",
    "Telephone inventor?", "Smallest prime number?", "Germany capital?",
    "Cat limerick.", "Largest human organ?", "Titanic sinking year?",
    "Main atmospheric gas?", "Moonlight Sonata composer?", "Pi to 4 decimals?",
    "Largest mammal?", "Happiness metaphor.", "Brazil capital?",
    "Adult human bones count?", "What is AI?",
]

def main():
    print(f"50-request stress test starting at {time.strftime('%H:%M:%S')}")
    print(f"Target: {BASE_URL}")
    print("=" * 60)
    
    results = []
    start_time = time.time()
    
    for i, prompt in enumerate(PROMPTS, 1):
        try:
            response = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                json={"model": "llama-8b", "messages": [{"role": "user", "content": prompt}], "max_tokens": 30, "temperature": 0.7},
                timeout=45
            )
            response.raise_for_status()
            data = response.json()
            text = data['choices'][0]['message']['content'].strip()
            results.append({'success': True, 'idx': i, 'len': len(text)})
            print(f"[{i:2d}/50] ✓ OK ({len(text):3d} chars)")
        except Exception as e:
            results.append({'success': False, 'idx': i, 'error': str(e)})
            print(f"[{i:2d}/50] ✗ FAIL: {e}")
    
    elapsed = time.time() - start_time
    successes = [r for r in results if r['success']]
    failures = [r for r in results if not r['success']]
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(successes)}/50 passed, {len(failures)} failed")
    print(f"Total time: {elapsed:.1f}s ({elapsed/len(PROMPTS):.1f}s/req)")
    
    if failures:
        print("\nFAILURES:")
        for r in failures:
            print(f"  [{r['idx']}] {r['error']}")
    
    return 0 if len(successes) == 50 else 1

if __name__ == "__main__":
    sys.exit(main())
