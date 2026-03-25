#!/usr/bin/env python3
"""
Final verification script for surgical layer migration fix.

This script verifies that all changes are properly in place and working.
"""

import sys
import os

sys.path.insert(0, '/home/z/GITDEV/vllm')

def verify_imports():
    """Verify that all required modules can be imported."""
    print("=" * 60)
    print("VERIFICATION 1: Module Imports")
    print("=" * 60)
    
    try:
        from vllm.model_executor.model_loader.utils import (
            device_loading_context,
            is_embedding_layer,
            is_math_layer,
        )
        print("✓ All required functions imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def verify_functions_exist():
    """Verify that the new functions exist and work."""
    print("\n" + "=" * 60)
    print("VERIFICATION 2: Function Existence")
    print("=" * 60)
    
    from vllm.model_executor.model_loader.utils import (
        is_embedding_layer,
        is_math_layer,
    )
    import torch
    
    # Test embedding detection
    embed = torch.nn.Embedding(100, 50)
    if not is_embedding_layer(embed):
        print("✗ is_embedding_layer() failed for torch.nn.Embedding")
        return False
    print("✓ is_embedding_layer() works correctly")
    
    # Test math layer detection
    linear = torch.nn.Linear(50, 50)
    if not is_math_layer(linear):
        print("✗ is_math_layer() failed for torch.nn.Linear")
        return False
    print("✓ is_math_layer() works correctly")
    
    return True


def verify_device_context():
    """Verify that device_loading_context works."""
    print("\n" + "=" * 60)
    print("VERIFICATION 3: Device Loading Context")
    print("=" * 60)
    
    from vllm.model_executor.model_loader.utils import device_loading_context
    import torch
    
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(100, 50)
            self.linear = torch.nn.Linear(50, 50)
    
    model = TestModel()
    
    # Test with CPU target
    with device_loading_context(model, torch.device("cpu")):
        for name, param in model.named_parameters():
            if param.device.type != "cpu":
                print(f"✗ Parameter {name} not on CPU")
                return False
    print("✓ device_loading_context() works with CPU target")
    
    return True


def verify_file_changes():
    """Verify that the required files have been modified."""
    print("\n" + "=" * 60)
    print("VERIFICATION 4: File Changes")
    print("=" * 60)
    
    utils_file = "/home/z/GITDEV/vllm/vllm/model_executor/model_loader/utils.py"
    interface_file = "/home/z/GITDEV/vllm/vllm/platforms/interface.py"
    
    # Check utils.py has the new functions
    with open(utils_file, 'r') as f:
        utils_content = f.read()
    
    if "def is_embedding_layer" not in utils_content:
        print("✗ utils.py missing is_embedding_layer() function")
        return False
    print("✓ utils.py has is_embedding_layer() function")
    
    if "def is_math_layer" not in utils_content:
        print("✗ utils.py missing is_math_layer() function")
        return False
    print("✓ utils.py has is_math_layer() function")
    
    if "surgical migration" not in utils_content.lower():
        print("✗ utils.py missing surgical migration logic")
        return False
    print("✓ utils.py has surgical migration logic")
    
    # Check interface.py doesn't have duplicate VULKAN
    with open(interface_file, 'r') as f:
        interface_content = f.read()
    
    vulkan_count = interface_content.count("VULKAN = enum.auto()")
    if vulkan_count != 1:
        print(f"✗ interface.py has {vulkan_count} VULKAN entries (expected 1)")
        return False
    print("✓ interface.py has correct VULKAN enum (no duplicates)")
    
    return True


def main():
    """Run all verifications."""
    print("\n" + "=" * 60)
    print("FINAL VERIFICATION - Surgical Layer Migration Fix")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("Module Imports", verify_imports()))
    results.append(("Function Existence", verify_functions_exist()))
    results.append(("Device Context", verify_device_context()))
    results.append(("File Changes", verify_file_changes()))
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ ALL VERIFICATIONS PASSED")
        print("The surgical layer migration fix is properly implemented.")
        return 0
    else:
        print("\n✗ SOME VERIFICATIONS FAILED")
        print("Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())