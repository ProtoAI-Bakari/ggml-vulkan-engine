# vLLM Vulkan Plugin Interface Specification

**Date**: 2026-03-25  
**Agent**: OmniAgent v4  
**Target**: Apple M1 Ultra Vulkan backend integration  

---

## Architecture Overview

vLLM uses a layered architecture with platform-specific implementations:

```
┌─────────────────────────────────────────────────┐
│              vLLM Core (Python)                 │
│  - LLMEngine, Scheduler, Tokenizer              │
└─────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐ ┌─────▼──────┐ ┌──────▼──────┐
│ VulkanPlugin │ │ CudaPlugin │ │  CPUPlugin  │
│ (Our Target) │ │ (Template) │ │  (Ref)      │
└───────┬──────┘ └─────┬──────┘ └──────┬──────┘
        │               │               │
┌───────▼───────────────▼───────────────▼──────┐
│          WorkerBase (Abstract)               │
│  - init_device()                             │
│  - determine_num_available_blocks()          │
│  - initialize_cache()                        │
│  - execute_model()                           │
└──────────────────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────┐
│        ModelRunnerBase (Abstract)            │
│  - prepare_model_input()                     │
│  - execute_model()                           │
│  - profile_run()                             │
└──────────────────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────┐
│         Vulkan-specific Implementation       │
│  - VulkanPlatform                            │
│  - VulkanWorker                              │
│  - VulkanModelRunner                         │
│  - VulkanAttentionBackend                    │
└──────────────────────────────────────────────┘
```

---

## Required Components

### 1. VulkanPlatform (extends Platform)

**File**: `vllm/platforms/vulkan.py`

**Purpose**: Platform-specific metadata and capabilities

**Interface**:
```python
class VulkanPlatform(Platform):
    _enum = PlatformEnum.VULKAN  # Need to add to PlatformEnum

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        """Return Vulkan physical device properties."""
        # Query vkEnumeratePhysicalDevices
        # Return device type, API version, extensions
        pass

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Return device name (e.g., 'Apple M1 Ultra')."""
        pass

    @classmethod
    def get_device_count(cls) -> int:
        """Return number of Vulkan-capable devices."""
        # vkEnumeratePhysicalDevices
        pass

    @classmethod
    def is_vulkan_available(cls) -> bool:
        """Check if Vulkan is available on this system."""
        # Check vkCreateInstance success
        pass

    @classmethod
    def inference_mode(cls):
        """Vulkan doesn't use torch.inference_mode, return context manager."""
        return torch.inference_mode(mode=True)  # Or no-op
```

**Key Methods to Implement**:
- `get_device_capability()`: Query Vulkan physical device properties
- `get_device_name()`: Return human-readable device name
- `get_device_count()`: Number of Vulkan GPUs
- `is_vulkan_available()`: Check Vulkan runtime presence

---

### 2. VulkanWorker (extends WorkerBase)

**File**: `vllm/worker/vulkan_worker.py`

**Purpose**: Device-specific model execution orchestration

**Interface**:
```python
class VulkanWorker(WorkerBase):
    """GPU worker for Vulkan backend."""

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        **kwargs,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.load_config = load_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method

        # Vulkan-specific
        self.vulkan_device = None  # vkDevice handle
        self.vulkan_queue = None   # vkQueue handle
        self.model_runner = None   # VulkanModelRunner instance

    def init_device(self) -> None:
        """Initialize Vulkan device and load model."""
        # 1. vkCreateInstance
        # 2. vkEnumeratePhysicalDevices
        # 3. vkCreateDevice
        # 4. vkGetDeviceQueue
        # 5. Initialize VulkanModelRunner
        # 6. Load GGUF model via ggml_llama_gguf.so
        pass

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine KV cache block capacity."""
        # Query Vulkan physical device memory properties
        # vkGetPhysicalDeviceMemoryProperties
        # Calculate: total_memory / block_size
        # Return (num_gpu_blocks, 0)  # No CPU swap for now
        pass

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        """Allocate KV cache memory on Vulkan device."""
        # Allocate Vulkan buffer for KV cache
        # vkCreateBuffer + vkAllocateMemory
        # Pass to VulkanModelRunner
        pass

    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> Optional[List[SamplerOutput]]:
        """Execute model inference on Vulkan."""
        # 1. Prepare input tensors
        # 2. Submit Vulkan command buffers
        # 3. vkQueueSubmit + vkQueueWaitIdle
        # 4. Copy logits back to CPU
        # 5. Return SamplerOutput
        pass

    def get_cache_block_size_bytes(self) -> int:
        """Return size of single KV cache block in bytes."""
        # block_size * num_layers * 2 (K+V) * dtype_size
        pass

    # LoRA methods (optional - can raise NotImplementedError)
    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError("Vulkan backend doesn't support LoRA yet")

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError("Vulkan backend doesn't support LoRA yet")

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError("Vulkan backend doesn't support LoRA yet")

    def list_loras(self) -> Set[int]:
        return set()
```

**Key Methods to Implement**:
- `init_device()`: Initialize Vulkan instance, device, queue
- `determine_num_available_blocks()`: Calculate KV cache capacity
- `initialize_cache()`: Allocate Vulkan buffers for KV cache
- `execute_model()`: Run inference via Vulkan command buffers
- `get_cache_block_size_bytes()`: Calculate block size

---

### 3. VulkanModelRunner (extends ModelRunnerBase)

**File**: `vllm/worker/vulkan_model_runner.py`

**Purpose**: Model execution on Vulkan device

**Interface**:
```python
class VulkanModelRunner(ModelRunnerBase):
    """Model runner for Vulkan backend."""

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.load_config = load_config
        self.lora_config = lora_config

        # Vulkan-specific
        self.vulkan_device = None
        self.model = None  # ggml_llama_gguf handle
        self.kv_cache_blocks = None  # Vulkan buffer
        self.block_table = None  # Block table data structure

    def load_model(self) -> None:
        """Load GGUF model into Vulkan memory."""
        # 1. Load GGUF file
        # 2. Allocate Vulkan buffers for model weights
        # 3. Copy weights from CPU to Vulkan
        # 4. Initialize ggml_llama_gguf context
        pass

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SeqGroupMetadata]
    ) -> VulkanModelInput:
        """Prepare model inputs from vLLM sequence metadata."""
        # 1. Build block table from seq_group_metadata
        # 2. Allocate input tensors (input_ids, positions, etc.)
        # 3. Copy to Vulkan device
        # 4. Return VulkanModelInput
        pass

    def execute_model(
        self,
        model_input: VulkanModelInput,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> SamplerOutput:
        """Execute model on Vulkan."""
        # 1. Submit Vulkan command buffer with model graph
        # 2. vkQueueSubmit + vkQueueWaitIdle
        # 3. Copy logits back to CPU
        # 4. Return SamplerOutput
        pass

    def profile_run(self) -> None:
        """Profile memory usage for model."""
        # Run dummy inference to measure memory footprint
        pass

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        """Save model state to disk."""
        # Copy Vulkan buffers to CPU and save
        pass

    def load_sharded_state(
        self,
        sharded_state: ShardedState,
        max_shard_size: Optional[int] = None,
    ) -> None:
        """Load model state from disk."""
        # Load from CPU and copy to Vulkan
        pass
```

**Key Methods to Implement**:
- `load_model()`: Load GGUF weights into Vulkan memory
- `prepare_model_input()`: Convert vLLM metadata to Vulkan inputs
- `execute_model()`: Run Vulkan compute graph
- `profile_run()`: Measure memory usage

---

### 4. VulkanAttentionBackend

**File**: `vllm/attention/backends/vulkan_attn.py`

**Purpose**: Vulkan-specific attention implementation

**Interface**:
```python
class VulkanAttentionBackend:
    """Vulkan implementation of attention."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        block_size: int,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.block_size = block_size

        # Vulkan-specific
        self.vulkan_pipeline_layout = None
        self.vulkan_compute_pipeline = None
        self.vulkan_descriptor_sets = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: VulkanAttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        """Execute attention on Vulkan."""
        # 1. Update KV cache (reshape_and_cache)
        # 2. Submit attention compute pipeline
        # 3. vkQueueSubmit + vkQueueWaitIdle
        # 4. Return output tensor
        pass

    def make_attention_metadata(
        self,
        seq_lens: List[int],
        query_start_loc: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_tables: torch.Tensor,
    ) -> VulkanAttentionMetadata:
        """Create attention metadata for Vulkan."""
        # Pack metadata into Vulkan-friendly format
        pass
```

**Key Methods to Implement**:
- `forward()`: Execute attention via Vulkan compute shaders
- `make_attention_metadata()`: Prepare metadata for Vulkan

---

## Integration Points

### A. Plugin Registration

**File**: `setup.py` or `pyproject.toml`

```python
# In setup.py
setup(
    name="vllm-vulkan",
    version="0.1.0",
    entry_points={
        'vllm.general_plugins': [
            'vulkan = vllm_vulkan_plugin:register',
        ],
    },
)

# In vllm_vulkan_plugin/__init__.py
def register():
    from vllm.platforms import current_platform
    from vllm.platforms.vulkan import VulkanPlatform
    
    # Register Vulkan platform
    current_platform._enum = PlatformEnum.VULKAN
```

### B. Executor Selection

**File**: `vllm/executor/vulkan_executor.py`

```python
from vllm.executor.executor_base import ExecutorBase
from vllm.executor.gpu_executor import GPUExecutor

class VulkanExecutor(ExecutorBase):
    """Vulkan-specific executor."""
    
    uses_ray = False  # Single-GPU for now
    
    def _init_executor(self) -> None:
        from vllm.worker.vulkan_worker import VulkanWorker
        
        self.worker = VulkanWorker(
            model_config=self.model_config,
            cache_config=self.cache_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            device_config=self.device_config,
            load_config=self.load_config,
            lora_config=self.lora_config,
            local_rank=0,
            rank=0,
            distributed_init_method="env://",
        )
```

### C. Attention Backend Selection

**File**: `vllm/attention/layer.py`

```python
def get_attention_backend():
    from vllm.platforms import current_platform
    
    if current_platform._enum == PlatformEnum.VULKAN:
        from vllm.attention.backends.vulkan_attn import VulkanAttentionBackend
        return VulkanAttentionBackend
    elif current_platform.is_cuda():
        from vllm.attention.backends.flash_attn import FlashAttentionBackend
        return FlashAttentionBackend
    # ... other backends
```

---

## Mapping to Existing ggml Vulkan Backend

### Current ggml Implementation (~/AGENT/ggml_llama_gguf.c)

| ggml Component | vLLM Equivalent | Status |
|----------------|-----------------|--------|
| `VulkanPlatform` | `VulkanPlatform` | Need to create |
| `VulkanWorker` | `VulkanWorker` | Need to create |
| `ggml_llama_gguf` | `VulkanModelRunner` | Wrap existing |
| `ggml_flash_attn_ext()` | `VulkanAttentionBackend` | Wrap existing |
| `vkCreateBuffer()` | KV cache allocation | Reuse |
| `vkQueueSubmit()` | `execute_model()` | Reuse |

### Key Integration Steps

1. **Wrap ggml engine in VulkanModelRunner**
   - Load GGUF via `ggml_llama_gguf.so`
   - Execute via existing Vulkan command buffer system
   - Return logits to vLLM

2. **Implement paged KV cache**
   - Use existing `block_table` data structure
   - Allocate Vulkan buffers for blocks
   - Implement `reshape_and_cache()` for Vulkan

3. **Attention backend**
   - Use existing `ggml_flash_attn_ext()` for FA_SCALAR path
   - Wrap in `VulkanAttentionBackend.forward()`

4. **Plugin registration**
   - Create `vllm-vulkan` pip package
   - Register via `entry_points`

---

## Success Criteria

- [ ] `VulkanPlatform` registered and detectable
- [ ] `VulkanWorker` can initialize device and load model
- [ ] `VulkanModelRunner` executes single inference step
- [ ] `VulkanAttentionBackend` computes attention correctly
- [ ] End-to-end: `LLMEngine` → `VulkanExecutor` → `VulkanWorker` → output
- [ ] 22+ TPS on 8B Q4_K_M (matching standalone ggml)
- [ ] Support for 120B model (MoE routing)

---

## References

- vLLM source: `/home/z/vllm/vllm/`
- CudaPlatform template: `/home/z/vllm/vllm/platforms/cuda.py`
- WorkerBase template: `/home/z/vllm/vllm/worker/worker_base.py`
- GPUExecutor template: `/home/z/vllm/vllm/executor/gpu_executor.py`
- FlashAttention backend: `/home/z/vllm/vllm/attention/backends/flash_attn.py`
- Our ggml Vulkan backend: `~/AGENT/ggml_llama_gguf.c`

---

*Generated by OmniAgent v4 on 2026-03-25*
