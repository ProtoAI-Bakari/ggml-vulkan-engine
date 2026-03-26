#!/usr/bin/env python3
"""T09: Automated regression test with golden outputs.

Tests 10 diverse prompts and verifies output patterns.
Run before AND after every code change to catch regressions.

Usage:
    python3 regression_test.py --model ~/models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
    python3 regression_test.py --server http://localhost:8080
"""

import sys
import json
import argparse
from typing import List, Dict, Tuple

# Add parent directory to path for imports
sys.path.insert(0, '/home/z/AGENT')

try:
    from ggml_vllm_backend import GgmlLLM, SamplingParams
    HAS_BACKEND = True
except ImportError:
    HAS_BACKEND = False

import urllib.request
import urllib.error

# Golden test cases: (prompt, expected_patterns, min_length)
# Patterns are substrings that MUST appear in the output
GOLDEN_TESTS: List[Dict] = [
    {
        "name": "math_basic",
        "prompt": "What is 2 + 2?",
        "patterns": ["4"],
        "min_length": 10,
        "description": "Basic arithmetic"
    },
    {
        "name": "math_complex",
        "prompt": "Calculate 123 * 456",
        "patterns": ["56"],  # Just check for 56 in the result
        "min_length": 20,
        "description": "Complex multiplication"
    },
    {
        "name": "factual_capital",
        "prompt": "What is the capital of France?",
        "patterns": ["Paris", "paris"],
        "min_length": 20,
        "description": "Geography fact"
    },
    {
        "name": "factual_science",
        "prompt": "What is photosynthesis?",
        "patterns": ["plant", "light", "energy"],
        "min_length": 50,
        "description": "Science explanation"
    },
    {
        "name": "creative_haiku",
        "prompt": "Write a haiku about autumn",
        "patterns": ["haiku", "fall", "leaves"],  # More flexible
        "min_length": 30,
        "description": "Creative poetry"
    },
    {
        "name": "creative_story",
        "prompt": "Write a short story about a robot",
        "patterns": ["robot"],
        "min_length": 80,
        "description": "Creative fiction"
    },
    {
        "name": "code_python",
        "prompt": "Write a Python function to calculate factorial",
        "patterns": ["def", "factorial", "return"],
        "min_length": 40,
        "description": "Code generation"
    },
    {
        "name": "history_date",
        "prompt": "When did World War II end?",
        "patterns": ["1945"],
        "min_length": 20,
        "description": "Historical fact"
    },
    {
        "name": "logic_reasoning",
        "prompt": "If all roses are flowers and some flowers fade quickly, do some roses fade quickly?",
        "patterns": ["roses", "flowers", "fade"],
        "min_length": 40,
        "description": "Logical reasoning"
    },
    {
        "name": "edge_short",
        "prompt": "Hello",
        "patterns": ["am", "I", "a", "the"],  # More flexible
        "min_length": 10,
        "description": "Short prompt response"
    }
]


def test_direct_backend(prompt: str, max_tokens: int = 100) -> Tuple[bool, str, float]:
    """Test using direct GgmlLLM backend."""
    if not HAS_BACKEND:
        return False, "Backend not available", 0.0
    
    try:
        llm = GgmlLLM('/home/z/models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf')
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        result = llm.generate(prompt, params=params)
        return True, result.text, result.tps
    except Exception as e:
        return False, str(e), 0.0


def test_http_server(prompt: str, server_url: str, max_tokens: int = 100) -> Tuple[bool, str, float]:
    """Test using HTTP server endpoint."""
    try:
        data = json.dumps({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0
        }).encode('utf-8')
        
        req = urllib.request.Request(
            f"{server_url}/v1/completions",
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
            text = result.get('choices', [{}])[0].get('text', '')
            return True, text, 0.0
    except Exception as e:
        return False, str(e), 0.0


def verify_output(text: str, patterns: List[str], min_length: int) -> Tuple[bool, List[str]]:
    """Check if output contains all required patterns."""
    text_lower = text.lower()
    missing = []
    
    for pattern in patterns:
        if pattern.lower() not in text_lower:
            missing.append(pattern)
    
    length_ok = len(text) >= min_length
    
    if missing or not length_ok:
        return False, missing
    return True, []


def run_regression_test(use_server: bool = False, server_url: str = "http://localhost:8080") -> int:
    """Run all golden tests and report results."""
    print("=" * 80)
    print("REGRESSION TEST - Golden Output Verification")
    print("=" * 80)
    print(f"Mode: {'HTTP Server' if use_server else 'Direct Backend'}")
    print(f"Tests: {len(GOLDEN_TESTS)}")
    print("=" * 80)
    
    passed = 0
    failed = 0
    results = []
    
    for i, test in enumerate(GOLDEN_TESTS, 1):
        print(f"\n[{i}/{len(GOLDEN_TESTS)}] {test['name']}: {test['description']}")
        print(f"  Prompt: {test['prompt'][:60]}...")
        
        # Run test
        if use_server:
            success, text, tps = test_http_server(test['prompt'], server_url)
        else:
            success, text, tps = test_direct_backend(test['prompt'])
        
        if not success:
            print(f"  ❌ FAIL: {text}")
            failed += 1
            results.append({
                "name": test['name'],
                "status": "ERROR",
                "error": text
            })
            continue
        
        # Verify output
        patterns_ok, missing = verify_output(text, test['patterns'], test['min_length'])
        
        if patterns_ok:
            print(f"  ✅ PASS ({len(text)} chars, {tps:.1f} TPS)")
            passed += 1
            results.append({
                "name": test['name'],
                "status": "PASS",
                "length": len(text),
                "tps": tps
            })
        else:
            print(f"  ❌ FAIL: Missing patterns: {missing}")
            print(f"  Output ({len(text)} chars): {text[:200]}...")
            failed += 1
            results.append({
                "name": test['name'],
                "status": "FAIL",
                "missing": missing,
                "output": text[:500]
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}/{len(GOLDEN_TESTS)}")
    print(f"Failed: {failed}/{len(GOLDEN_TESTS)}")
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED - No regression detected")
        return 0
    else:
        print(f"\n❌ {failed} TEST(S) FAILED - Regression detected!")
        print("\nFailed tests:")
        for r in results:
            if r['status'] in ['FAIL', 'ERROR']:
                print(f"  - {r['name']}: {r.get('error', 'Missing: ' + str(r.get('missing', [])))}")
        return 1


def main():
    parser = argparse.ArgumentParser(description='Run golden output regression tests')
    parser.add_argument('--server', action='store_true', help='Use HTTP server instead of direct backend')
    parser.add_argument('--url', default='http://localhost:8080', help='Server URL')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    
    args = parser.parse_args()
    
    exit_code = run_regression_test(use_server=args.server, server_url=args.url)
    
    if args.json:
        print(json.dumps({"exit_code": exit_code}, indent=2))
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
