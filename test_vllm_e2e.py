#!/usr/bin/env python3
"""End-to-end test for vLLM Vulkan plugin - T38."""
import asyncio
import json
from typing import List, Dict

class VLLMTestClient:
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
    
    async def chat_completion(self, messages: List[Dict[str, str]],
                             temperature: float = 0.7,
                             max_tokens: int = 100) -> Dict:
        """Send chat completion request to vLLM."""
        url = f"http://{self.host}:{self.port}/v1/chat/completions"
        payload = {
            "model": "llama-3.1-8b-vulkan",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                return await resp.json()
    
    async def simple_generate(self, prompt: str,
                             temperature: float = 0.7) -> Dict:
        """Simple text generation request."""
        return await self.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
    
    def verify_coherent(self, response: Dict) -> bool:
        """Verify the response is coherent (not crash, has content)."""
        if "error" in response:
            print(f"ERROR: {response['error']}")
            return False
        if "choices" not in response:
            print(f"ERROR: No choices in response")
            return False
        if len(response["choices"]) == 0:
            print(f"ERROR: No choices returned")
            return False
        content = response["choices"][0]["message"]["content"]
        if not content or len(content.strip()) == 0:
            print(f"ERROR: Empty response")
            return False
        
        # Check for common failure patterns
        if "<error>" in content or "ERROR" in content.upper():
            print(f"WARNING: Response contains error markers")
        
        return True
    
    async def run_tests(self):
        """Run end-to-end tests."""
        print("="*60)
        print("T38: End-to-end single-request test through vLLM plugin")
        print("="*60)
        
        tests = [
            ("Math: 2+2", "What is 2 plus 2? Give me the answer.", True),
            ("Factual: Capital", "What is the capital of France? Just say Paris.", True),
            ("Creative: Story", "Write a 3-sentence story about a robot learning to paint.", True),
            ("Code: Python", "Write a function that sorts a list in ascending order.", True),
            ("Long-gen: Poetry", "Write a short poem about the ocean. Include at least 10 lines.", True),
        ]
        
        results = []
        for name, prompt, expected_success in tests:
            print(f"\nTest: {name}")
            try:
                response = await self.simple_generate(prompt)
                success = self.verify_coherent(response) and (response["choices"][0]["finish_reason"] == "stop")
                results.append((name, success))
                
                if expected_success:
                    content = response["choices"][0]["message"]["content"][:100]
                    print(f"  ✓ SUCCESS: {repr(content)}...")
                else:
                    if success:
                        print(f"  ✓ UNEXPECTED SUCCESS")
                    else:
                        print(f"  ✗ FAILED as expected: {response.get('choices', [{}])[0].get('message', {}).get('content', 'N/A')}")
                    
            except Exception as e:
                print(f"  ✗ EXCEPTION: {e}")
                results.append((name, False))
        
        print("\n" + "="*60)
        passed = sum(1 for _, s in results if s)
        total = len(results)
        print(f"Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("✓ T38 COMPLETE: All coherent responses!")
        else:
            print(f"✗ T38 INCOMPLETE: {total - passed} failures")
        
        return all(s for _, s in results)

async def main():
    client = VLLMTestClient(host="localhost", port=8000)
    success = await client.run_tests()
    exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())
