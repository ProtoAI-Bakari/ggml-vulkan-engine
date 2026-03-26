#!/usr/bin/env python3
"""T02: Test 50 sequential HTTP requests to standalone server."""

import requests
import time
import sys

BASE_URL = "http://localhost:8080"

PROMPTS = [
    "What is 123 * 456?",
    "Who wrote Pride and Prejudice?",
    "Write a short poem about the ocean.",
    "What is the capital of Japan?",
    "Explain photosynthesis in one sentence.",
    "What year did World War II end?",
    "Write a haiku about autumn.",
    "What is the chemical formula for water?",
    "Who painted the Mona Lisa?",
    "What is the speed of light?",
    "Write a joke about computers.",
    "What is the largest planet?",
    "Who discovered penicillin?",
    "What is the boiling point of water?",
    "Write a short story about a robot.",
    "What is the square root of 144?",
    "Who wrote Romeo and Juliet?",
    "What is the currency of Germany?",
    "What is the tallest mountain?",
    "Write a limerick about a cat.",
    "What is the atomic number of carbon?",
    "Who invented the telephone?",
    "What is the capital of France?",
    "What is the largest ocean?",
    "Write a poem about stars.",
    "What is the freezing point of water?",
    "Who wrote To Kill a Mockingbird?",
    "What is the chemical symbol for iron?",
    "What is the smallest prime number?",
    "Who painted Starry Night?",
    "What is the capital of Italy?",
    "Write a joke about AI.",
    "What is the speed of sound?",
    "Who discovered gravity?",
    "What is the largest mammal?",
    "What is the chemical formula for salt?",
    "Write a haiku about winter.",
    "What is the capital of Spain?",
    "Who wrote 1984?",
    "What is the atomic mass of hydrogen?",
    "What is the longest river?",
    "Write a short story about a dragon.",
    "What is the square of 15?",
    "Who painted the Sistine Chapel?",
    "What is the capital of Russia?",
    "What is the smallest continent?",
    "Write a joke about math.",
    "Who invented the light bulb?",
    "What is the chemical symbol for gold?",
    "What is the largest desert?",
]

def test_request(prompt, i):
    """Send one request and check response."""
    try:
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "temperature": 0.0,
                "max_tokens": 50
            },
            timeout=30
        )
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}"
        
        data = resp.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        if not text or len(text.strip()) < 5:
            return False, "Empty or too short response"
        
        # Check for repetition
        words = text.lower().split()
        if len(words) > 20:
            for j in range(len(words) - 5):
                phrase = ' '.join(words[j:j+5])
                if phrase in text[j*10:]:
                    return False, "Repetition detected"
        
        return True, text[:100]
        
    except Exception as e:
        return False, str(e)

def main():
    print("[T02] Testing server stability with 50 sequential requests...")
    
    success_count = 0
    failed_requests = []
    
    for i, prompt in enumerate(PROMPTS, 1):
        start = time.time()
        success, result = test_request(prompt, i)
        elapsed = time.time() - start
        
        if success:
            success_count += 1
            print(f"[{i}/50] ✓ ({elapsed:.1f}s) {result[:60]}...")
        else:
            failed_requests.append((i, prompt, result))
            print(f"[{i}/50] ✗ ({elapsed:.1f}s) {result}")
    
    # Summary
    print("\n" + "="*60)
    print(f"T02 SERVER STABILITY TEST RESULTS")
    print(f"="*60)
    print(f"Total requests: {len(PROMPTS)}")
    print(f"Successful: {success_count}/{len(PROMPTS)}")
    print(f"Failed: {len(failed_requests)}")
    print(f"Success rate: {success_count/len(PROMPTS)*100:.0f}%")
    
    if failed_requests:
        print("\nFailed requests:")
        for i, prompt, reason in failed_requests:
            print(f"  #{i}: {prompt[:40]}... -> {reason}")
    
    if success_count >= 48:
        print("\n✓ T02 PASSED: 48+/50 requests successful")
        return 0
    else:
        print(f"\n✗ T02 FAILED: Only {success_count}/50 successful")
        return 1

if __name__ == "__main__":
    sys.exit(main())
