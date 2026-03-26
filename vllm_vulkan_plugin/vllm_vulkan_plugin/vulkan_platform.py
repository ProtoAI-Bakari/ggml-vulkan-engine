"""Vulkan platform support for vLLM.

This module implements the VulkanPlatform class that extends vLLM's Platform
interface to support Vulkan-based inference on Apple M1/M2/M3 and other
Vulkan-capable devices.
"""

import logging
from typing import Optional, Tuple

import torch

from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum

logger = logging.getLogger(__name__)


class VulkanPlatform(Platform):
    """Vulkan platform implementation for vLLM.
    
    Supports Vulkan-based inference on:
    - Apple M1/M2/M3 (via MoltenVK)
    - NVIDIA GPUs (via Vulkan CUDA interop)
    - AMD GPUs (via Vulkan)
    - Intel GPUs (via Vulkan)
    """
    
    _enum = PlatformEnum.VULKAN
    
    def __init__(self):
        """Initialize Vulkan platform.
        
        Raises:
            RuntimeError: If Vulkan is not available on this system.
        """
        self._vulkan_available = self._check_vulkan_availability()
        if not self._vulkan_available:
            raise RuntimeError("Vulkan is not available on this system")
        
        # Vulkan instance and device handles (lazy initialization)
        self._vk_instance = None
        self._vk_physical_device = None
        self._vk_device = None
        self._vk_queue = None
        self._device_count = 0
        
        # Device properties (populated on first access)
        self._device_name = None
        self._device_capability = None
        self._memory_properties = None
    
    def _check_vulkan_availability(self) -> bool:
        """Check if Vulkan runtime is available.
        
        Returns:
            True if Vulkan is available, False otherwise.
        """
        try:
            # Try to import Vulkan bindings
            import vk  # type: ignore
            
            # Try to create a Vulkan instance
            vk.init()
            vk.shutdown()
            
            return True
        except (ImportError, RuntimeError, OSError) as e:
            logger.debug(f"Vulkan not available: {e}")
            return False
    
    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> Optional[DeviceCapability]:
        """Get Vulkan device capability.
        
        Args:
            device_id: Device index (default 0).
            
        Returns:
            DeviceCapability tuple (major, minor) or None if unavailable.
        """
        # For Vulkan, we report API version as capability
        # This is a placeholder - actual implementation would query
        # vkGetPhysicalDeviceProperties
        try:
            import vk  # type: ignore
            
            # Get physical device count
            physical_devices = vk.enumeratePhysicalDevices()
            if device_id >= len(physical_devices):
                return None
            
            # Get device properties
            props = vk.getPhysicalDeviceProperties(physical_devices[device_id])
            
            # Map Vulkan API version to DeviceCapability
            # Vulkan 1.0 = (1, 0), Vulkan 1.1 = (1, 1), etc.
            api_version = props["apiVersion"]
            major = (api_version >> 22) & 0x3FF
            minor = (api_version >> 12) & 0x3FF
            
            return DeviceCapability(major=major, minor=minor)
        except Exception as e:
            logger.warning(f"Failed to get device capability: {e}")
            return None
    
    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Get Vulkan device name.
        
        Args:
            device_id: Device index (default 0).
            
        Returns:
            Human-readable device name (e.g., "Apple M1 Ultra").
        """
        try:
            import vk  # type: ignore
            
            physical_devices = vk.enumeratePhysicalDevices()
            if device_id >= len(physical_devices):
                return "Unknown Vulkan Device"
            
            props = vk.getPhysicalDeviceProperties(physical_devices[device_id])
            return props.get("deviceName", "Unknown Vulkan Device")
        except Exception as e:
            logger.warning(f"Failed to get device name: {e}")
            return "Unknown Vulkan Device"
    
    @classmethod
    def get_device_count(cls) -> int:
        """Get number of Vulkan-capable devices.
        
        Returns:
            Number of Vulkan devices, or 0 if unavailable.
        """
        try:
            import vk  # type: ignore
            
            return len(vk.enumeratePhysicalDevices())
        except Exception as e:
            logger.warning(f"Failed to get device count: {e}")
            return 0
    
    @classmethod
    def is_vulkan_available(cls) -> bool:
        """Check if Vulkan is available on this system.
        
        Returns:
            True if Vulkan is available, False otherwise.
        """
        try:
            import vk  # type: ignore
            
            vk.init()
            vk.shutdown()
            return True
        except Exception:
            return False
    
    @classmethod
    def get_device_memory_info(cls, device_id: int = 0) -> Tuple[int, int]:
        """Get device memory information.
        
        Args:
            device_id: Device index (default 0).
            
        Returns:
            Tuple of (total_memory_bytes, free_memory_bytes).
        """
        try:
            import vk  # type: ignore
            
            physical_devices = vk.enumeratePhysicalDevices()
            if device_id >= len(physical_devices):
                return (0, 0)
            
            mem_props = vk.getPhysicalDeviceMemoryProperties(physical_devices[device_id])
            
            # Sum up all device-local memory heaps
            total_memory = 0
            for heap in mem_props.get("memoryHeaps", []):
                if heap["flags"] & vk.MemoryHeapFlagBits.DEVICE_LOCAL:
                    total_memory += heap["size"]
            
            # For now, assume all memory is free (will be tracked by vLLM)
            return (total_memory, total_memory)
        except Exception as e:
            logger.warning(f"Failed to get device memory info: {e}")
            return (0, 0)
    
    @classmethod
    def inference_mode(cls):
        """Return inference mode context manager.
        
        Vulkan doesn't use torch.inference_mode, but we return it for
        compatibility with vLLM's execution model.
        
        Returns:
            torch.inference_mode context manager.
        """
        return torch.inference_mode(mode=True)
    
    def initialize(self) -> None:
        """Initialize Vulkan instance and device.
        
        This should be called before any Vulkan operations.
        """
        if self._vk_instance is not None:
            return  # Already initialized
        
        try:
            import vk  # type: ignore
            
            # Create Vulkan instance
            self._vk_instance = vk.createInstance(
                app_name="vllm-vulkan",
                app_version=1,
                engine_name="vllm",
                engine_version=1,
                api_version=vk.make_api_version(1, 0, 0),
                enabled_extensions=[
                    "VK_KHR_surface",
                    "VK_KHR_get_physical_device_properties2",
                ],
            )
            
            # Enumerate physical devices
            physical_devices = vk.enumeratePhysicalDevices(self._vk_instance)
            if not physical_devices:
                raise RuntimeError("No Vulkan physical devices found")
            
            self._vk_physical_device = physical_devices[0]
            self._device_count = len(physical_devices)
            
            # Create logical device
            self._vk_device = vk.createDevice(
                self._vk_physical_device,
                enabled_features=[],
                enabled_extensions=[
                    "VK_KHR_swapchain",
                    "VK_KHR_push_descriptor",
                ],
            )
            
            # Get queue (graphics/compute queue)
            queue_family_indices = vk.getQueueFamilyProperties(self._vk_physical_device)
            for i, props in enumerate(queue_family_indices):
                if props["queueFlags"] & vk.QueueFlagBits.COMPUTE:
                    self._vk_queue = vk.getDeviceQueue(
                        self._vk_device, i, 0
                    )
                    break
            
            if self._vk_queue is None:
                raise RuntimeError("No compute queue found")
            
            logger.info(f"Vulkan initialized: {self.get_device_name()}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vulkan: {e}")
            raise
    
    def shutdown(self) -> None:
        """Shutdown Vulkan instance and device."""
        if self._vk_instance is not None:
            try:
                import vk  # type: ignore
                
                if self._vk_device is not None:
                    vk.destroyDevice(self._vk_device)
                    self._vk_device = None
                
                if self._vk_instance is not None:
                    vk.destroyInstance(self._vk_instance)
                    self._vk_instance = None
                
                self._vk_physical_device = None
                self._vk_queue = None
                
                logger.info("Vulkan shutdown complete")
                
            except Exception as e:
                logger.warning(f"Error during Vulkan shutdown: {e}")


# Global platform instance
_vulkan_platform = None


def get_vulkan_platform() -> VulkanPlatform:
    """Get or create the global Vulkan platform instance.
    
    Returns:
        VulkanPlatform instance.
    """
    global _vulkan_platform
    
    if _vulkan_platform is None:
        _vulkan_platform = VulkanPlatform()
    
    return _vulkan_platform


def check_and_update_config(config) -> None:
    """Check Vulkan compatibility and update vLLM config.
    
    Args:
        config: vLLM configuration object.
    """
    platform = get_vulkan_platform()
    
    # Set device type
    config.device_type = "vulkan"
    
    # Update GPU count
    config.gpu_device_count = platform.get_device_count()
    
    # Set memory limits based on device memory
    total_mem, _ = platform.get_device_memory_info()
    if total_mem > 0:
        # Use 80% of available memory for KV cache
        config.max_model_len = min(config.max_model_len, int(total_mem * 0.8) // 1024)
    
    logger.info(f"Vulkan config updated: {config.gpu_device_count} devices, "
                f"{total_mem / (1024**3):.1f} GB total memory")


def get_attn_backend_cls() -> str:
    """Get the attention backend class name for Vulkan.
    
    Returns:
        Full class path for Vulkan attention backend.
    """
    return "vllm_vulkan_plugin.vulkan_attn.VulkanAttentionBackend"
