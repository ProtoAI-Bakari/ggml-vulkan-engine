"""Vulkan platform plugin for vLLM.

This package provides Vulkan support for vLLM inference on Apple M1/M2/M3
and other Vulkan-capable devices.
"""

from .vulkan_platform import VulkanPlatform, get_vulkan_platform, check_and_update_config, get_attn_backend_cls

__version__ = "0.1.0"
__all__ = [
    "VulkanPlatform",
    "get_vulkan_platform",
    "check_and_update_config",
    "get_attn_backend_cls",
]
