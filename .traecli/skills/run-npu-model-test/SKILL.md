---
name: run-npu-model-test
description: "Run NPU model tests with TP/DP-Attention/EP parallelism and ACLGraph on Ascend. Invoke when testing NPU inference, verifying parallelism configs, debugging NPU model issues, or running model tests on Ascend hardware."
---

# NPU Model Test Skill

This skill runs model tests on Ascend NPU devices with various parallelism configurations and graph optimization features.

## When to Use

Invoke this skill when:
- User wants to test model inference on Ascend NPU
- User needs to verify TP/DP/EP parallelism configurations
- User wants to debug NPU-specific model issues
- User asks about ACLGraph or NPU graph optimization

## Features Tested

| Feature | Description |
|---------|-------------|
| **TP (Tensor Parallelism)** | Distributes model across multiple NPUs |
| **DP-Attention** | Data parallel attention for improved throughput |
| **EP (Expert Parallelism)** | For MoE models with expert distribution |
| **ACLGraph** | NPU graph optimization (equivalent to CUDA Graph) |

## Prerequisites

- Ascend NPU device available
- CANN toolkit installed
- Model weights accessible (local path or HuggingFace/ModelScope ID)
- SGLang installed from current project repo

## Execution Steps

### 1. Environment Setup

Set PYTHONPATH to use local SGLang code:
```bash
export PYTHONPATH=<project-root>/python:$PYTHONPATH
```

### 2. Locate or Create Test Script

Search for existing NPU test scripts in the project:
```bash
find . -name "*npu*test*.py" -o -name "*test*npu*.py"
```

If no test script exists, create one following the standard pattern (see Test Script Template below).

### 3. Configure Test Parameters

Key parameters to configure based on user requirements:

| Parameter | Description | Constraint |
|-----------|-------------|------------|
| `--model-path` | Model weights path or HF ID | Must be accessible |
| `--tp-size` | Tensor parallelism size | TP <= available NPUs |
| `--dp-size` | Data parallelism size | TP * DP <= total NPUs |
| `--ep-size` | Expert parallelism size | For MoE models only |
| `--enable-dp-attention` | Enable DP-Attention | Requires DP > 1 |
| `--disable-graph` | Disable ACLGraph | Use for debugging |

### 4. Run Test

Execute the test with appropriate parameters:
```bash
python <test-script> --model-path <model-path> --tp-size <n> [other-args]
```

### 5. Verify Results

Check for:
- Server startup success (within reasonable time)
- Inference returns valid output
- No errors in logs

## Test Script Template

If no existing test script is found, create one:

```python
import requests
import subprocess
import time

def test_npu_model(
    model_path: str,
    tp_size: int = 1,
    enable_dp_attention: bool = False,
    dp_size: int = 1,
    disable_graph: bool = False,
):
    args = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--tp-size", str(tp_size),
    ]
    if enable_dp_attention:
        args.extend(["--enable-dp-attention", "--dp-size", str(dp_size)])
    if disable_graph:
        args.append("--disable-cuda-graph")
    
    # Launch server, wait for readiness, run inference test
    # ... implementation details
```

## Common Test Scenarios

### Basic Inference Test
```bash
python <test-script> --model-path <model-path> --tp-size 1 --disable-graph
```

### TP Test with Graph Optimization
```bash
python <test-script> --model-path <model-path> --tp-size 2
```

### DP-Attention Test
```bash
python <test-script> --model-path <model-path> --tp-size 2 --enable-dp-attention --dp-size 2
```

## Troubleshooting

| Issue | Diagnostic | Solution |
|-------|------------|----------|
| Server fails to start | `npu-smi info` | Check NPU availability and CANN installation |
| Inference fails | Check memory usage | Reduce `--mem-fraction-static` or batch size |
| TP/DP issues | Check NPU count | Ensure TP * DP <= total NPUs |
| Graph capture fails | Check error logs | Try `--disable-graph` for debugging |

## Quality Gate

Before reporting success:
- [ ] Server starts successfully
- [ ] At least one inference request returns valid output
- [ ] No critical errors in logs
- [ ] Parallelism configuration matches user requirements
