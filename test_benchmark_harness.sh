#!/bin/bash
# Quick test of benchmark_vulkan.py
# Tests basic functionality without running full suite

set -e

echo "========================================"
echo "Benchmark Harness Quick Test"
echo "========================================"
echo ""

# Check if benchmark script exists
if [ ! -f "$HOME/AGENT/benchmark_vulkan.py" ]; then
    echo "ERROR: benchmark_vulkan.py not found"
    exit 1
fi

echo "✓ benchmark_vulkan.py exists"

# Check if run_benchmarks.sh exists
if [ ! -f "$HOME/AGENT/run_benchmarks.sh" ]; then
    echo "ERROR: run_benchmarks.sh not found"
    exit 1
fi

echo "✓ run_benchmarks.sh exists"

# Check if model exists
MODEL_PATH="$HOME/models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
if [ -f "$MODEL_PATH" ]; then
    echo "✓ Model found: $MODEL_PATH"
    MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    echo "  Size: $MODEL_SIZE"
else
    echo "⚠ Model not found: $MODEL_PATH"
    echo "  Benchmark will fail if model doesn't exist"
fi

# Check Vulkan availability
echo ""
echo "Checking Vulkan availability..."
if command -v vulkaninfo &> /dev/null; then
    echo "✓ vulkaninfo found"
    vulkaninfo --summary 2>/dev/null | grep -E "deviceName|apiVersion" | head -2 || true
else
    echo "⚠ vulkaninfo not found"
fi

# Check Python dependencies
echo ""
echo "Checking Python dependencies..."
python3 -c "import numpy" 2>/dev/null && echo "✓ numpy installed" || echo "⚠ numpy not installed"
python3 -c "import requests" 2>/dev/null && echo "✓ requests installed" || echo "⚠ requests not installed"

# Test benchmark script help
echo ""
echo "Testing benchmark script help..."
python3 "$HOME/AGENT/benchmark_vulkan.py" --help > /dev/null 2>&1 && echo "✓ benchmark_vulkan.py --help works" || echo "✗ benchmark_vulkan.py --help failed"

# Test shell script help
echo ""
echo "Testing shell script help..."
bash "$HOME/AGENT/run_benchmarks.sh" --help > /dev/null 2>&1 && echo "✓ run_benchmarks.sh --help works" || echo "✗ run_benchmarks.sh --help failed"

# Create sample results directory
echo ""
echo "Creating results directory..."
mkdir -p "$HOME/AGENT/results"
echo "✓ Results directory created"

# Summary
echo ""
echo "========================================"
echo "Quick Test Complete"
echo "========================================"
echo ""
echo "To run a quick benchmark:"
echo "  bash $HOME/AGENT/run_benchmarks.sh --quick"
echo ""
echo "To run full suite:"
echo "  bash $HOME/AGENT/run_benchmarks.sh --all"
echo ""
echo "To benchmark specific model:"
echo "  bash $HOME/AGENT/run_benchmarks.sh --model <path> <tokens> <runs>"
echo ""
echo "========================================"
