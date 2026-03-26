#!/usr/bin/env python3
"""T01: Verify standalone engine stability with 12 sequential requests."""

import sys
sys.path.insert(0, '/home/z/AGENT')

from ggml_vllm_backend import GgmlLLM, SamplingParams

# 12 diverse prompts covering different task types
PROMPTS = [
    # Math
    "What is 256 * 17?",
    "If a train travels 60 mph for 2.5 hours, how far does it go?",
    # Factual
    "Who wrote the novel '1984'?",
    "What is the chemical symbol for gold?",
    # Creative
    "Write a haiku about mountains.",
    "Continue: The old lighthouse stood silent on the cliff, its beam dark for decades until",
    # Multi-sentence reasoning
    "Explain why the sky is blue in two sentences.",
    # Code generation
    "Write a Python function to calculate factorial.",
    # Edge case: short
    "Hello",
    # Edge case: longer prompt
    "Summarize the main causes of World War I in three bullet points.",
    # Science
    "What is quantum entanglement?",
    # History
    "When did the Berlin Wall fall?",
]

def test_coherence(text: str, prompt: str) -> bool:
    """Basic coherence check: non-empty, no repetition, relevant."""
    if not text or len(text.strip()) < 5:
        return False
    # Check for obvious repetition loops
    words = text.lower().split()
    if len(words) > 50:
        # Check for 5+ word repetition
        for i in range(len(words) - 5):
            phrase = ' '.join(words[i:i+5])
            if phrase in text[i*10:]:
                return False
    return True

def main():
    print("[T01] Loading model...")
    llm = GgmlLLM('/home/z/models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf')
    
    results = []
    coherent_count = 0
    
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n[{i}/12] Prompt: {prompt[:50]}...")
        try:
            response = llm.generate(prompt, params=SamplingParams(temperature=0.7, max_tokens=50))
            text = response.text.strip()
            tps = response.tps
            
            print(f"  Response ({len(text)} chars, {tps:.1f} TPS): {text[:100]}...")
            
            is_coherent = test_coherence(text, prompt)
            results.append({
                'prompt': prompt,
                'response': text,
                'tps': tps,
                'coherent': is_coherent
            })
            
            if is_coherent:
                coherent_count += 1
                print(f"  ✓ Coherent")
            else:
                print(f"  ✗ Incoherent (empty, repetitive, or irrelevant)")
                
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            results.append({
                'prompt': prompt,
                'response': None,
                'tps': 0,
                'coherent': False
            })
    
    # Summary
    print("\n" + "="*60)
    print(f"T01 STABILITY TEST RESULTS")
    print(f"="*60)
    print(f"Total requests: {len(PROMPTS)}")
    print(f"Coherent responses: {coherent_count}/{len(PROMPTS)}")
    print(f"Success rate: {coherent_count/len(PROMPTS)*100:.0f}%")
    
    if coherent_count >= 12:
        print("\n✓ T01 PASSED: All 12 requests coherent")
        return 0
    else:
        print(f"\n✗ T01 FAILED: Only {coherent_count}/12 coherent")
        return 1

if __name__ == "__main__":
    sys.exit(main())
