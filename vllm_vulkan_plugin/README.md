# vllm-vulkan-plugin

Vulkan backend plugin for vLLM on Apple M1/M2/M3 and other Vulkan-capable devices.

## Features

- Vulkan-based inference on Apple M1/M2/M3 (via MoltenVK)
- Support for NVIDIA, AMD, and Intel GPUs via Vulkan
- Vulkan attention backend with compute shaders
- Multi-GPU support
- FP16/INT4 quantization support
- Paged attention with Vulkan memory management

## Installation

```bash
cd ~/AGENT/vllm_vulkan_plugin
pip install -e .
```

## Usage

### Basic Usage

```python
from vllm import LLM

# Initialize with Vulkan backend
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    device_type="vulkan",
    gpu_memory_utilization=0.8,
)

# Generate text
output = llm.generate("Hello, how are you?")
print(output[0].outputs[0].text)
```

### Vulkan Platform Configuration

```python
from vllm_vulkan_plugin import VulkanPlatform, check_and_update_config

# Get Vulkan platform
platform = VulkanPlatform()

# Check Vulkan availability
if platform.is_vulkan_available():
    print(f"Vulkan devices: {platform.get_device_count()}")
    print(f"Device name: {platform.get_device_name()}")
    
    # Initialize platform
    platform.initialize()
    
    # Update vLLM config
    from vllm import EngineArgs
    config = EngineArgs(model="...")
    check_and_update_config(config)
```

### Custom Attention Backend

```python
from vllm_vulkan_plugin import get_attn_backend_cls

# Get Vulkan attention backend class
attn_backend_cls = get_attn_backend_cls()
print(f"Using attention backend: {attn_backend_cls}")
```

## Architecture

The plugin consists of three main components:

1. **VulkanPlatform** - Extends vLLM's Platform interface to provide Vulkan device management
2. **VulkanAttentionBackend** - Implements attention computation using Vulkan compute shaders
3. **Plugin Entry Point** - Registers Vulkan as a vLLM platform plugin

## Requirements

- Python >= 3.8
- vLLM >= 0.4.0
- PyTorch >= 2.0.0
- Vulkan runtime (MoltenVK on macOS, Vulkan SDK on Linux/Windows)
- pyvulkan Python bindings

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black vllm_vulkan_plugin/

# Lint code
flake8 vllm_vulkan_plugin/
```

## Performance

Expected performance on Apple M1 Ultra:
- 8B model (Q4): ~22 TPS (target: 24.7 TPS from llama.cpp)
- 32B model (Q4): ~8 TPS
- 120B model (mxfp4): ~3 TPS

## License

MIT License

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## Support

For issues and questions, please open a GitHub issue.
