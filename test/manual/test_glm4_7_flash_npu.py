#!/usr/bin/env python3
"""
Quick test script to verify GLM-4.7-Flash model loading and basic inference on NPU.
"""

import os
import sys

# Add local repository to PYTHONPATH for running on current repo
_repo_root = os.path.join(os.path.dirname(__file__), "../../python")
sys.path.insert(0, _repo_root)
os.environ["PYTHONPATH"] = _repo_root + ":" + os.environ.get("PYTHONPATH", "")

# Set NPU environment variables
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ASCEND_MF_STORE_URL"] = "tcp://127.0.0.1:24666"
os.environ["HCCL_BUFFSIZE"] = "200"
os.environ["HCCL_EXEC_TIMEOUT"] = "200"
os.environ["STREAMS_PER_DEVICE"] = "32"

import torch
import torch_npu

# Check NPU availability
print(f"torch_npu available: {hasattr(torch, 'npu')}")
if hasattr(torch, 'npu'):
    print(f"NPU device count: {torch.npu.device_count()}")
    if torch.npu.device_count() > 0:
        print(f"NPU device name: {torch.npu.get_device_name(0)}")

# Test model loading
MODEL_PATH = "/home/trae/testCode/weight/GLM4-7-flash"

print(f"\nLoading model from: {MODEL_PATH}")

try:
    from sglang import Engine, RuntimeEndpoint

    # Create engine with NPU backend
    engine = Engine(
        model_path=MODEL_PATH,
        trust_remote_code=True,
        attention_backend="ascend",
        disable_cuda_graph=True,
        mem_fraction_static=0.7,
    )
    
    print("Model loaded successfully!")
    
    # Test basic inference
    prompt = "The capital of France is"
    print(f"\nTesting inference with prompt: '{prompt}'")
    
    result = engine.generate(
        prompt,
        max_new_tokens=32,
        temperature=0.0,
    )
    
    print(f"Generated text: {result}")
    print("\nBasic inference test passed!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nAll tests passed!")
