# Read the file
with open('/home/z/GITDEV/vllm_0.17.1/vllm/model_executor/layers/utils.py', 'r') as f:
    content = f.read()

# Find and replace the default_unquantized_gemm function
old_code = '''def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    return torch.nn.functional.linear(x, weight, bias)'''

new_code = '''def default_unquantized_gemm(
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

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('/home/z/GITDEV/vllm_0.17.1/vllm/model_executor/layers/utils.py', 'w') as f:
        f.write(content)
    print("Patched utils.py for Vulkan linear operations")
else:
    print("Could not find the code to patch")
