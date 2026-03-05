---
name: run_npu_model_test
description: Run NPU model tests with TP (Tensor Parallelism), DP-Attention (Data Parallel Attention), EP (Expert Parallelism), and ACLGraph features on Ascend NPU devices. Use this skill when users want to test NPU model inference, verify parallelism configurations, or run model tests on Ascend hardware.
---

# NPU Model Test Skill

This skill runs model tests on Ascend NPU devices with various parallelism configurations and graph optimization features.

## Features Tested

- **TP (Tensor Parallelism)**: Distributes model across multiple NPUs
- **DP-Attention**: Data parallel attention for improved throughput
- **EP (Expert Parallelism)**: For MoE models with expert distribution
- **ACLGraph**: NPU graph optimization (equivalent to CUDA Graph on NVIDIA)

## Prerequisites

- Ascend NPU device (A2 or A3 series)
- CANN toolkit installed
- Model weights available locally

## Test Script Location

The test script is located at: `test_npu_model.py` in the project root.

## Usage

### Basic Test (TP=1, no graph)
```bash
export PYTHONPATH=/home/trae/testCode/sglang/python:$PYTHONPATH
python test_npu_model.py --tp-size 1 --disable-graph
```

### Test with TP=2
```bash
python test_npu_model.py --tp-size 2 --disable-graph
```

### Test with TP=2 and ACLGraph
```bash
python test_npu_model.py --tp-size 2
```

### Test with DP-Attention
```bash
python test_npu_model.py --tp-size 2 --enable-dp-attention --dp-size 2
```

### Full Test (TP, DP-Attention, ACLGraph)
```bash
python test_npu_model.py --tp-size 2 --enable-dp-attention --dp-size 2
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | `/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B` | Model path |
| `--tp-size` | 2 | Tensor parallelism size |
| `--dp-size` | 2 | Data parallelism size (for DP-Attention) |
| `--ep-size` | 1 | Expert parallelism size |
| `--enable-dp-attention` | False | Enable DP-Attention |
| `--disable-graph` | False | Disable ACLGraph/NPU Graph |
| `--quick` | False | Run quick test with minimal questions |

## Test Output

The test will:
1. Launch the SGLang server with specified configuration
2. Wait for server to be ready
3. Run basic inference test
4. Report success or failure

## Expected Results

- Server should start within 60 seconds
- Basic inference should return valid output
- All parallelism configurations should work correctly

## Troubleshooting

### Server fails to start
- Check NPU availability: `npu-smi info`
- Verify CANN installation
- Check model path exists

### Inference fails
- Check memory allocation (`--mem-fraction-static`)
- Reduce batch size (`--cuda-graph-max-bs`)
- Try with `--disable-graph` first

### TP/DP issues
- Ensure enough NPUs available (TP * DP <= total NPUs)
- Check HCCL communication between NPUs
