#!/bin/bash
# Vulkan LLM Benchmark Runner
# Usage: ./run_benchmarks.sh [--all-models] [--tokens N] [--runs N] [--output FILE]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_SCRIPT="${SCRIPT_DIR}/benchmark_vulkan.py"

# Default parameters
TOKENS=100
RUNS=10
OUTPUT="benchmark_results.csv"
ALL_MODELS=false
MODEL_PATH=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all-models)
            ALL_MODELS=true
            shift
            ;;
        --tokens)
            TOKENS="$2"
            shift 2
            ;;
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --all-models    Run benchmark on all configured models"
            echo "  --model PATH    Run benchmark on specific model"
            echo "  --tokens N      Number of decode tokens (default: 100)"
            echo "  --runs N        Number of benchmark runs (default: 10)"
            echo "  --output FILE   Output CSV file (default: benchmark_results.csv)"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if Python3 is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 is not installed or not in PATH"
    exit 1
fi

# Check if benchmark script exists
if [[ ! -f "$BENCHMARK_SCRIPT" ]]; then
    echo "ERROR: Benchmark script not found at $BENCHMARK_SCRIPT"
    exit 1
fi

# Create output directory if needed
OUTPUT_DIR=$(dirname "$OUTPUT")
if [[ ! -d "$OUTPUT_DIR" ]]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Created output directory: $OUTPUT_DIR"
fi

echo "========================================"
echo "Vulkan LLM Benchmark Runner"
echo "========================================"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Script: $BENCHMARK_SCRIPT"
echo "Tokens: $TOKENS"
echo "Runs: $RUNS"
echo "Output: $OUTPUT"
echo "========================================"
echo ""

# Run benchmark
if [[ "$ALL_MODELS" == true ]]; then
    echo "Running comprehensive benchmark on all models..."
    python3 "$BENCHMARK_SCRIPT" --all-models --tokens "$TOKENS" --runs "$RUNS" --output "$OUTPUT"
elif [[ -n "$MODEL_PATH" ]]; then
    echo "Running benchmark on specific model: $MODEL_PATH"
    python3 "$BENCHMARK_SCRIPT" --model "$MODEL_PATH" --tokens "$TOKENS" --runs "$RUNS" --output "$OUTPUT"
else
    echo "ERROR: Must specify either --all-models or --model PATH"
    echo ""
    echo "Usage:"
    echo "  $0 --all-models --tokens 100 --runs 10 --output results.csv"
    echo "  $0 --model ~/models/gguf/llama-3.1-8b-q4_k_m.gguf --tokens 100 --runs 10"
    exit 1
fi

echo ""
echo "========================================"
echo "Benchmark Complete!"
echo "========================================"
echo "Results saved to: $OUTPUT"
if [[ -f "${OUTPUT%.csv}.json" ]]; then
    echo "Full JSON results: ${OUTPUT%.csv}.json"
fi
echo "========================================"
