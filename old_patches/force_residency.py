import os
path = os.path.expanduser("~/GITDEV/vllm_0.17.1/vllm/model_executor/model_loader/default_loader.py")
with open(path, "r") as f:
    lines = f.readlines()

# Locate the interceptor loop and force the .to("vulkan")
new_lines = []
for line in lines:
    if 'tensor = tensor.cpu()' in line:
        new_lines.append(line.replace('tensor.cpu()', 'tensor.to("vulkan")'))
    elif 'logger.debug(f"Vulkan Interceptor: Keeping {name} on CPU")' in line:
        new_lines.append(line.replace('Keeping {name} on CPU', 'Forcing {name} to VULKAN'))
    else:
        new_lines.append(line)

with open(path, "w") as f:
    f.writelines(new_lines)
print("✅ SURGERY COMPLETE: Full GPU Residency forced in default_loader.py")
