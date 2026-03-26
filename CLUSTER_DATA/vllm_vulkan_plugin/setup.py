"""Setup script for vllm-vulkan-plugin.

This package provides Vulkan backend support for vLLM on Apple M1/M2/M3
and other Vulkan-capable devices.
"""

from setuptools import setup, find_packages

setup(
    name="vllm-vulkan-plugin",
    version="0.1.0",
    author="OmniAgent v4",
    author_email="omni@agent.local",
    description="Vulkan backend plugin for vLLM on Apple M1/M2/M3",
    long_description="""Vulkan backend plugin for vLLM

This package provides Vulkan-based inference support for vLLM on:
- Apple M1/M2/M3 (via MoltenVK)
- NVIDIA GPUs (via Vulkan CUDA interop)
- AMD GPUs (via Vulkan)
- Intel GPUs (via Vulkan)

Features:
- Vulkan-based attention kernels
- Unified memory management
- Multi-GPU support
- FP16/INT4 quantization support
""",
    long_description_content_type="text/markdown",
    url="https://github.com/omni-agent/vllm-vulkan-plugin",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "vllm>=0.4.0",
        "torch>=2.0.0",
        "pyvulkan>=0.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "vllm.platform_plugins": [
            "vulkan = vllm_vulkan_plugin.vulkan_platform:VulkanPlatform",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="vllm vulkan inference ml ai apple m1 m2 m3",
)
