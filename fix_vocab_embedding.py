import re

# Read the file
with open('/home/z/GITDEV/vllm_0.17.1/vllm/model_executor/layers/vocab_parallel_embedding.py', 'r') as f:
    content = f.read()

# Find and replace the problematic code
old_code = '''            # Perform embedding lookup on CPU
            input_cpu = input_.to('cpu')
            # Only move weight to CPU if it's not already there
            weight_cpu = layer.weight if layer.weight.device.type == 'cpu' else layer.weight.to('cpu')
            output_cpu = F.embedding(input_cpu, weight_cpu)'''

new_code = '''            # Perform embedding lookup on CPU
            input_cpu = input_.to('cpu')
            # Copy weight to CPU - use detach().cpu() to avoid VMA errors
            if layer.weight.device.type == 'cpu':
                weight_cpu = layer.weight
            else:
                # Use detach() to avoid gradient tracking, then copy to CPU
                weight_cpu = layer.weight.detach().cpu()
            output_cpu = F.embedding(input_cpu, weight_cpu)'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('/home/z/GITDEV/vllm_0.17.1/vllm/model_executor/layers/vocab_parallel_embedding.py', 'w') as f:
        f.write(content)
    print("Patched vocab_parallel_embedding.py")
else:
    print("Could not find the code to patch")
