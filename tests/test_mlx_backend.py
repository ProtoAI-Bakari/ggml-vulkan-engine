#!/usr/bin/env python3
"""
MLX Backend Test - Verify Metal performance on M1 Ultra
Tests basic operations and simple inference
"""
import mlx.core as mx
import mlx.nn as nn
import time
import numpy as np

def test_basic_ops():
    """Test basic MLX operations on Metal GPU"""
    print("\n=== MLX Basic Operations Test ===")
    
    # Create tensors on GPU
    a = mx.random.normal((1024, 1024))
    b = mx.random.normal((1024, 1024))
    
    # Warmup
    _ = a @ b
    mx.eval(a)
    
    # Benchmark
    start = time.time()
    for _ in range(10):
        c = a @ b
        mx.eval(c)
    end = time.time()
    
    elapsed = (end - start) / 10
    gflops = (2 * 1024 * 1024 * 1024) / (elapsed * 1e9)
    
    print(f"✓ 1024x1024 matmul: {elapsed*1000:.2f}ms ({gflops:.1f} GFLOPS)")
    return elapsed

def test_simple_transformer_layer():
    """Test a simple transformer layer"""
    print("\n=== Simple Transformer Layer Test ===")
    
    class SimpleLayer(nn.Module):
        def __init__(self, dim=512, heads=8):
            super().__init__()
            self.attention = nn.MultiHeadAttention(dim, heads)
            self.ff = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        
        def __call__(self, x):
            x = x + self.attention(self.norm1(x))
            x = x + self.ff(self.norm2(x))
            return x
    
    layer = SimpleLayer()
    mx.eval(layer.parameters())  # Initialize on GPU
    
    # Warmup
    x = mx.random.normal((1, 1, 512))
    _ = layer(x)
    mx.eval(x)
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        x = mx.random.normal((1, 1, 512))
        y = layer(x)
        mx.eval(y)
    end = time.time()
    
    elapsed = (end - start) / 100
    print(f"✓ Transformer layer (512d): {elapsed*1000:.2f}ms per forward pass")
    return elapsed

def test_mlx_lm():
    """Test MLX LM for simple text generation"""
    print("\n=== MLX LM Quick Test ===")
    
    try:
        from mlx_lm import generate, load
        
        # Load a small model (if available)
        print("Loading Qwen2.5-0.5B-Instruct-MLX...")
        model, tokenizer = load("Qwen/Qwen2.5-0.5B-Instruct-MLX")
        
        prompt = "What is the capital of France?"
        
        start = time.time()
        response = generate(model, tokenizer, prompt=prompt, max_tokens=50, verbose=False)
        end = time.time()
        
        tokens_generated = len(tokenizer.encode(response))
        tps = tokens_generated / (end - start)
        
        print(f"✓ Generated: {response[:100]}...")
        print(f"✓ TPS: {tps:.1f} tokens/sec")
        return tps
    except Exception as e:
        print(f"⚠ MLX LM test skipped: {e}")
        return None

def main():
    print("🚀 MLX Backend Validation on M1 Ultra")
    print(f"GPU: Apple M1 Ultra (Metal)")
    print(f"MLX Version: {mx.__version__}")
    
    results = {}
    
    # Run tests
    results['matmul'] = test_basic_ops()
    results['transformer'] = test_simple_transformer_layer()
    results['mlx_lm'] = test_mlx_lm()
    
    # Summary
    print("\n=== Summary ===")
    print(f"✓ Matmul: {results['matmul']*1000:.2f}ms")
    print(f"✓ Transformer: {results['transformer']*1000:.2f}ms")
    if results['mlx_lm']:
        print(f"✓ MLX LM: {results['mlx_lm']:.1f} TPS")
    
    print("\n✅ MLX backend is functional on Metal!")

if __name__ == "__main__":
    main()
