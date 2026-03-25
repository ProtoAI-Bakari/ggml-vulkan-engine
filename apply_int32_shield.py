#!/usr/bin/env python3
"""
VULKAN INT32 SCALAR TYPE SHIELD - PATCH SCRIPT
Fixes int64 bottleneck on Vulkan by converting metadata tensors to int32.

Patches:
1. gpu/block_table.py: slot_mappings dtype int64 -> int32
2. gpu_model_runner.py: logits_indices, target_logits_indices, bonus_logits_indices -> int32
3. v1/utils.py: copy_slice function casts integer tensors to int32 before Vulkan transfer
"""

import os
import re

VLLM_ROOT = "/home/z/GITDEV/vllm_0.17.1"

def patch_block_table():
    """Patch slot_mappings dtype from int64 to int32 in gpu/block_table.py"""
    file_path = os.path.join(VLLM_ROOT, "vllm/v1/worker/gpu/block_table.py")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find and replace slot_mappings dtype
    old_pattern = r'''self\.slot_mappings = torch\.zeros\(
            self\.num_kv_cache_groups,
            self\.max_num_batched_tokens,
            dtype=torch\.int64,
            device=self\.device,
        \)'''
    
    new_code = '''self.slot_mappings = torch.zeros(
            self.num_kv_cache_groups,
            self.max_num_batched_tokens,
            dtype=torch.int32,  # VULKAN: int64 not supported
            device=self.device,
        )'''
    
    if re.search(old_pattern, content, re.MULTILINE | re.DOTALL):
        content = re.sub(old_pattern, new_code, content)
        print(f"✓ Patched: {file_path} - slot_mappings dtype int64 -> int32")
    else:
        print(f"⚠ Warning: Could not find slot_mappings pattern in {file_path}")
    
    with open(file_path, 'w') as f:
        f.write(content)

def patch_model_runner():
    """Patch logits_indices tensors to use int32 in gpu_model_runner.py"""
    file_path = os.path.join(VLLM_ROOT, "vllm/v1/worker/gpu_model_runner.py")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Patch logits_indices
    old_logits = '''logits_indices = torch.from_numpy(logits_indices).to(
            self.device, non_blocking=True
        )'''
    
    new_logits = '''# VULKAN: Cast to int32 for Vulkan compatibility
        logits_indices = torch.from_numpy(logits_indices).to(
            self.device, non_blocking=True
        ).to(torch.int32)'''
    
    if old_logits in content:
        content = content.replace(old_logits, new_logits)
        print(f"✓ Patched: {file_path} - logits_indices -> int32")
    else:
        print(f"⚠ Warning: Could not find logits_indices pattern in {file_path}")
    
    # Patch target_logits_indices
    old_target = '''target_logits_indices = torch.from_numpy(target_logits_indices).to(
            self.device, non_blocking=True
        )'''
    
    new_target = '''# VULKAN: Cast to int32 for Vulkan compatibility
        target_logits_indices = torch.from_numpy(target_logits_indices).to(
            self.device, non_blocking=True
        ).to(torch.int32)'''
    
    if old_target in content:
        content = content.replace(old_target, new_target)
        print(f"✓ Patched: {file_path} - target_logits_indices -> int32")
    else:
        print(f"⚠ Warning: Could not find target_logits_indices pattern in {file_path}")
    
    # Patch bonus_logits_indices
    old_bonus = '''bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(
            self.device, non_blocking=True
        )'''
    
    new_bonus = '''# VULKAN: Cast to int32 for Vulkan compatibility
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(
            self.device, non_blocking=True
        ).to(torch.int32)'''
    
    if old_bonus in content:
        content = content.replace(old_bonus, new_bonus)
        print(f"✓ Patched: {file_path} - bonus_logits_indices -> int32")
    else:
        print(f"⚠ Warning: Could not find bonus_logits_indices pattern in {file_path}")
    
    with open(file_path, 'w') as f:
        f.write(content)

def patch_utils():
    """Patch copy_slice to cast integer tensors to int32 in v1/utils.py"""
    file_path = os.path.join(VLLM_ROOT, "vllm/v1/utils.py")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the copy_slice function and add int32 casting
    old_func = '''def copy_slice(
    from_tensor: torch.Tensor, to_tensor: torch.Tensor, length: int
) -> torch.Tensor:
    """
    Copy the first length elements of a tensor into another tensor in a
    non-blocking manner.

    Used to copy pinned CPU tensor data to pre-allocated GPU tensors.

    Returns the sliced target tensor.
    """
    with torch.inference_mode(False):
        return to_tensor[:length].copy_(from_tensor[:length], non_blocking=True)'''
    
    new_func = '''def copy_slice(
    from_tensor: torch.Tensor, to_tensor: torch.Tensor, length: int
) -> torch.Tensor:
    """
    Copy the first length elements of a tensor into another tensor in a
    non-blocking manner.

    Used to copy pinned CPU tensor data to pre-allocated GPU tensors.

    VULKAN FIX: Cast integer tensors to int32 before transfer (Vulkan int64 unsupported)

    Returns the sliced target tensor.
    """
    with torch.inference_mode(False):
        # VULKAN: Cast integer source tensors to int32 before transfer
        src = from_tensor[:length]
        if src.dtype in (torch.int64, torch.int32):
            src = src.to(torch.int32)
        return to_tensor[:length].copy_(src, non_blocking=True)'''
    
    if old_func in content:
        content = content.replace(old_func, new_func)
        print(f"✓ Patched: {file_path} - copy_slice int32 casting")
    else:
        print(f"⚠ Warning: Could not find copy_slice function in {file_path}")
    
    with open(file_path, 'w') as f:
        f.write(content)

def create_backup():
    """Create backup of all patched files"""
    import shutil
    backup_dir = os.path.join(os.path.expanduser("~"), "AGENT", "INT32_FIX_BACKUP")
    os.makedirs(backup_dir, exist_ok=True)
    
    files_to_backup = [
        "vllm/v1/worker/gpu/block_table.py",
        "vllm/v1/worker/gpu_model_runner.py",
        "vllm/v1/utils.py"
    ]
    
    for file_path in files_to_backup:
        src = os.path.join(VLLM_ROOT, file_path)
        dst = os.path.join(backup_dir, file_path.replace("/", "_"))
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"✓ Backed up: {file_path}")

def main():
    print("=" * 60)
    print("VULKAN INT32 SCALAR TYPE SHIELD - PATCH SCRIPT")
    print("=" * 60)
    print()
    
    # Create backups first
    print("Creating backups...")
    create_backup()
    print()
    
    # Apply patches
    print("Applying patches...")
    print()
    patch_block_table()
    patch_model_runner()
    patch_utils()
    print()
    
    print("=" * 60)
    print("PATCHING COMPLETE")
    print("=" * 60)
    print()
    print("Summary:")
    print("  1. slot_mappings: int64 → int32 (gpu/block_table.py)")
    print("  2. logits_indices: int32 cast (gpu_model_runner.py)")
    print("  3. target_logits_indices: int32 cast (gpu_model_runner.py)")
    print("  4. bonus_logits_indices: int32 cast (gpu_model_runner.py)")
    print("  5. copy_slice: int32 casting for Vulkan (v1/utils.py)")
    print()
    print("Backups saved to: ~/AGENT/INT32_FIX_BACKUP/")
    print()
    print("Next: Restart vLLM server and test token generation")
    print("=" * 60)

if __name__ == "__main__":
    main()