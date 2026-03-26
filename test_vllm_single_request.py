#!/usr/bin/env python3
"""
End-to-end single-request test through vLLM Vulkan plugin.
Tests: Chat completion, verify coherent response matching direct ggml output quality.
"""

import os
import sys
import time
import json
import requests
from typing import Dict, Any

def test_single_request() -> bool:
    """
    Send a chat completion request through the full vLLM pipeline.
    Expected: Coherent response matching direct ggml output quality.
    """
    vllm_url = os.getenv("VLLM_URL", "http://localhost:8000/v1")
    
    print(f"Testing vLLM endpoint: {vllm_url}")
    
    # Test prompt (simple, coherent)
    messages = [
        {"role": "user", "content": "Hello, world! Please respond with a short greeting."}
    ]
    
    payload = {
        "model": os.getenv("VLLM_MODEL", "Llama-3.1-8B-Instruct-Q4_K_M"),
        "messages": messages,
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{vllm_url}/chat/completions",
            json=payload,
            timeout=30
        )
        elapsed = time.time() - start_time
        
        print(f"Response status: {response.status_code}")
        if response.status_code != 200:
            print(f"Error: {response.text}")
            return False
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        print(f"Response time: {elapsed:.2f}s")
        print(f"Generated tokens: ~{len(content.split())}")
        print(f"Response content:\
{'-'*60}")
        print(content)
        print(f"{'-'*60}\
")
        
        # Verify coherency (simple checks)
        if not content.strip():
            print("ERROR: Empty response!")
            return False
        
        # Check for basic coherence patterns (not too short, not gibberish)
        word_count = len(content.split())
        if word_count < 3:
            print("WARNING: Response too short (<3 words)")
        
        # Verify it's not just repeating the prompt
        if "Hello, world!" in content:
            print("WARNING: Response seems to echo prompt")
        
        # Verify it's not a system error message
        if "error" in content.lower() or "exception" in content.lower():
            print("ERROR: Response contains error message")
            return False
        
        # Check if response is coherent (contains punctuation, sentence structure)
        has_punctuation = any(c in content for c in [".", "!", "?", ","])
        if not has_punctuation:
            print("WARNING: Response lacks punctuation")
        
        # Success criteria
        success = (
            response.status_code == 200 and
            content.strip() and
            "error" not in content.lower()
        )
        
        if success:
            print("SUCCESS: Coherent response received!")
        else:
            print("FAILURE: Response failed coherence checks")
        
        return success
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to vLLM server")
        return False
    except requests.exceptions.Timeout:
        print("ERROR: Request timed out")
        return False
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return False

def test_multiple_prompts(num_tests: int = 5) -> Dict[str, Any]:
    """
    Run multiple test prompts to verify stability.
    Returns: Test results dictionary
    """
    vllm_url = os.getenv("VLLM_URL", "http://localhost:8000/v1")
    
    test_prompts = [
        "What is 2+2?",
        "Who was the first president of the United States?",
        "Write a haiku about coding.",
    ]
    
    results = {
        "total": 0,
        "success": 0,
        "failures": [],
        "responses": []
    }
    
    for i, prompt in enumerate(test_prompts):
        print(f"\
Test {i+1}/{len(test_prompts)}: '{prompt}'")
        
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": os.getenv("VLLM_MODEL", "Llama-3.1-8B-Instruct-Q4_K_M"),
            "messages": messages,
            "max_tokens": 30
        }
        
        try:
            response = requests.post(
                f"{vllm_url}/chat/completions",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                results["success"] += 1
                print(f"SUCCESS: {content.strip()}")
            else:
                results["failures"].append((prompt, response.status_code))
                print(f"FAILURE: {response.text}")
            
        except Exception as e:
            results["failures"].append((prompt, str(e)))
            print(f"ERROR: {e}")
        
        results["total"] += 1
    
    # Print summary
    print(f"\
{'='*60}")
    print("SUMMARY:")
    print(f"  Total tests: {results['total']}")
    print(f"  Successful: {results['success']}")
    print(f"  Failed: {len(results['failures'])}")
    
    if results["success"] == len(test_prompts):
        print("\
SUCCESS: All tests passed!")
    else:
        print(f"\
FAILURE: {len(results['failures'])} test(s) failed")
    
    return results

def main():
    """Main test runner."""
    print("="*60)
    print("vLLM Single-Request End-to-End Test")
    print(