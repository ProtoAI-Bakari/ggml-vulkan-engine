#!/usr/bin/env python3
"""Test chat completion through full vLLM pipeline."""

import requests
import json
import time

def test_chat_completion():
    """Send a chat completion request and verify coherent response."""
    url = "http://localhost:8080/v1/chat/completions"
    
    payload = {
        "model": "Llama-3.1-8B",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is the capital of France? Answer in one word."}
        ],
        "temperature": 0.0,
        "max_tokens": 10
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print("Sending chat completion request...")
    start_time = time.time()
    
    response = requests.post(url, headers=headers, json=payload)
    
    elapsed = time.time() - start_time
    print(f"Response received in {elapsed:.3f}s")
    
    if response.status_code != 200:
        print(f"ERROR: Status code {response.status_code}")
        print(f"Response: {response.text}")
        return False
    
    result = response.json()
    print(f"\nFull response:")
    print(json.dumps(result, indent=2))
    
    # Verify coherent response
    content = result["choices"][0]["message"]["content"]
    print(f"\nExtracted content: '{content}'")
    
    # Check for coherent response (contains 'Paris' or similar)
    if "paris" in content.lower() or "parise" in content.lower():
        print("✓ Response is coherent (contains Paris)")
        return True
    else:
        print(f"✗ Response may be incoherent: '{content}'")
        return False

if __name__ == "__main__":
    success = test_chat_completion()
    exit(0 if success else 1)
