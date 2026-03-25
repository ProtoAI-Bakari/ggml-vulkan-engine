# Comprehensive Vulkan operator fallback for vLLM

# Patch the linear layer to use CPU fallback
import torch
import torch.nn.functional as F

# Read and patch utils.py
with open('/home/z/GITDEV/vllm_0.17.1/vllm/model_executor/layers/utils.py', 'r') as f:
    content = f.read()

# Update the default_unquantized_gemm function with comprehensive fallback
old_func = '''def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    # VULKAN: Use CPU for linear operations to avoid version_counter errors
    if current_platform.__class__.__name__ == 'VulkanPlatform':
        x_cpu = x.to('cpu')
        weight_cpu = weight.to('cpu')
        bias_cpu = bias.to('cpu') if bias is not None else None
        output_cpu = torch.nn.functional.linear(x_cpu, weight_cpu, bias_cpu)
        return output_cpu.to(x.device)
    return torch.nn.functional.linear(x, weight, bias)'''

new_func = '''def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    # VULKAN: Use CPU for all linear operations
    if current_platform.__class__.__name__ == 'VulkanPlatform':
        try:
            x_cpu = x.to('cpu')
            weight_cpu = weight.to('cpu')
            bias_cpu = bias.to('cpu') if bias is not None else None
            output_cpu = torch.nn.functional.linear(x_cpu, weight_cpu, bias_cpu)
            return output_cpu.to(x.device)
        except Exception:
            # Fallback to original if CPU transfer fails
            pass
    return torch.nn.functional.linear(x, weight, bias)'''

if old_func in content:
    content = content.replace(old_func, new_func)
    with open('/home/z/GITDEV/vllm_0.17.1/vllm/model_executor/layers/utils.py', 'w') as f:
        f.write(content)
    print("✓ Patched utils.py")
else:
    print("⚠ utils.py already patched or pattern not found")

# Patch the RMSNorm layer to use native implementation
with open('/home/z/GITDEV/vllm_0.17.1/vllm/model_executor/layers/layernorm.py', 'r') as f:
    content = f.read()

# Update forward_cuda to use forward_native for Vulkan
old_cuda = '''    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:'''

new_cuda = '''    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # VULKAN: Use native implementation for Vulkan backend
        if current_platform.__class__.__name__ == 'VulkanPlatform':
            return self.forward_native(x, residual)'''

if old_cuda in content:
    content = content.replace(old_cuda, new_cuda)
    with open('/home/z/GITDEV/vllm_0.17.1/vllm/model_executor/layers/layernorm.py', 'w') as f:
        f.write(content)
    print("✓ Patched layernorm.py")
else:
    print("⚠ layernorm.py already patched or pattern not found")

print("Vulkan fallback patches applied!")
