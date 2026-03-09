---
name: npu-testing-workflow
description: "Use when testing models on Ascend NPU devices, performing NPU-related performance evaluation, or generating test reports. Invoke after model adaptation is complete or when user needs comprehensive NPU testing."
---

# NPU Testing Workflow Skill

## Overview

Systematic NPU testing workflow for service deployment, client benchmarking, and test report generation on Ascend devices.

## Input/Output Specification

### Required Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | string | Path to the model checkpoint |
| `hardware_type` | string | `Atlas 800I A2` or `Atlas 800I A3` |
| `deployment_mode` | string | `PD Mixed` or `PD Separation` |

### Optional Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `performance_targets` | string | `balanced` | `latency`, `throughput`, or `balanced` |
| `test_datasets` | list | `[random]` | Test datasets (random, sharegpt, sonnet) |
| `accuracy_benchmarks` | list | `[GSM8K]` | Accuracy benchmarks |
| `baseline_values` | dict | `{}` | Reference values for comparison |

### Outputs

| Artifact | Description |
|----------|-------------|
| Deployment script | Executable service deployment script |
| Benchmark script | Client benchmarking script |
| Test report | Structured test report (Markdown) |

## Prerequisites

### From Model Adapter

If transitioning from `sglang-model-adapter`, expect:
- `model_path`: Model checkpoint path
- `working_command`: Validated server startup command
- `known_limitations`: Known issues or constraints
- `basic_test_results`: Health check, text inference, VL inference status

### Documentation Review

Review from `./docs/platforms/`:
- `ascend_npu.md` - Installation and basic configuration
- `ascend_npu_best_practice.md` - Best practices
- `ascend_npu_environment_variables.md` - Environment variables
- `ascend_npu_support_features.md` - Supported features
- `ascend_npu_support_models.md` - Supported models

## Task 1: Service Deployment Script

### Environment Variables

See [shared/npu_common_reference.md](../shared/npu_common_reference.md) for complete reference.

**Essential setup**:
```bash
# System
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
export SGLANG_SET_CPU_AFFINITY=1

# Memory
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

# Communication
export HCCL_BUFFSIZE=1600
export HCCL_SOCKET_IFNAME=lo
```

### Deployment Modes

#### PD Mixed Mode (Single Instance)

```bash
python -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --tp-size 16 --attention-backend ascend --device npu \
    --host 127.0.0.1 --port 8000 \
    --mem-fraction-static 0.8 --max-running-requests 256 \
    --cuda-graph-bs 8 16 24 32
```

#### PD Separation Mode

**Prefill Server**:
```bash
export ASCEND_MF_STORE_URL="tcp://PREFILL_IP:PORT"
python -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --disaggregation-mode prefill \
    --host ${PREFILL_IP} --port 8000 \
    --disaggregation-bootstrap-port 8995 \
    --tp-size 16 --attention-backend ascend --device npu
```

**Decode Server**:
```bash
export ASCEND_MF_STORE_URL="tcp://PREFILL_IP:PORT"
python -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --disaggregation-mode decode \
    --host ${DECODE_IP} --port 8001 \
    --tp-size 32 --dp-size 32 \
    --deepep-mode low_latency \
    --attention-backend ascend --device npu
```

**Router**:
```bash
python -m sglang_router.launch_router \
    --pd-disaggregation --policy cache_aware \
    --prefill http://PREFILL_IP:8000 8995 \
    --decode http://DECODE_IP:8001 \
    --host 127.0.0.1 --port 6688
```

### Key Server Arguments

| Argument | Description | Typical Values |
|----------|-------------|----------------|
| `--tp-size` | Tensor parallelism | 1, 2, 4, 8, 16, 32 |
| `--dp-size` | Data parallelism | 1, 4, 8, 16, 32 |
| `--mem-fraction-static` | Memory fraction | 0.6-0.86 |
| `--max-running-requests` | Max concurrent | 32-832 |
| `--cuda-graph-bs` | Graph batch sizes | `8 16 24 32` |
| `--moe-a2a-backend` | MoE backend | `deepep` |
| `--deepep-mode` | DeepEP mode | `normal`, `low_latency`, `auto` |

## Task 2: Client Benchmarking

### Performance Testing

```bash
python -m sglang.bench_serving \
    --dataset-name random --backend sglang \
    --host 127.0.0.1 --port 6688 \
    --max-concurrency 256 \
    --random-input-len 3500 --random-output-len 1500 \
    --num-prompts 1024
```

### Accuracy Testing (GSM8K)

```python
from types import SimpleNamespace
from sglang.test.few_shot_gsm8k import run_eval

args = SimpleNamespace(
    num_shots=5, num_questions=200, max_new_tokens=512,
    parallel=32, host="http://127.0.0.1", port=6688
)
metrics = run_eval(args)
print(f"Accuracy: {metrics['accuracy']}")
```

### Performance Metrics

| Metric | Description | Source |
|--------|-------------|--------|
| TTFT | Time to First Token | bench_serving |
| TPOT | Time Per Output Token | bench_serving |
| Throughput | Output tokens/sec | bench_serving |
| ITL | Inter-Token Latency | bench_serving |
| Accuracy | Benchmark score | GSM8K/etc. |

## Task 3: Test Report Generation

### Report Structure

```markdown
# NPU Model Testing Report

## 1. Test Environment
- Hardware: [Atlas 800I A2/A3]
- CANN Version: [version]
- Model: [model_name]
- Deployment Mode: [PD Mixed/Separation]
- TP/DP Size: [tp_size]/[dp_size]

## 2. Functional Verification
| Test Case | Expected | Actual | Status |
|-----------|----------|--------|--------|
| Basic Inference | 200 | [actual] | PASS/FAIL |
| Streaming Output | 200 | [actual] | PASS/FAIL |

## 3. Performance Metrics
| Metric | Value | Baseline | Comparison |
|--------|-------|----------|------------|
| TTFT (ms) | [value] | [baseline] | +/- X% |
| TPOT (ms) | [value] | [baseline] | +/- X% |
| Throughput (tok/s) | [value] | [baseline] | +/- X% |

## 4. Accuracy Results
| Benchmark | Score | Baseline | Status |
|-----------|-------|----------|--------|
| GSM8K | [score] | [baseline] | PASS/FAIL |

## 5. Issues and Suggestions
- [Issue]: [Description and suggestion]

## 6. Conclusion
- Overall: PASS/FAIL
- Key Findings: [Summary]
- Next Steps: [Action items]
```

### Baseline Reference Values

**DeepSeek-R1 (Atlas 800I A3)**:
| Cards | Mode | Input+Output | TPOT Target |
|-------|------|--------------|-------------|
| 32 | Separation | 3.5K+1.5K | 20ms (low latency) / 50ms (high throughput) |
| 8 | Mixed | 2K+2K | 50ms |

**Qwen3-32B (Atlas 800I A3)**:
| Cards | Mode | Input+Output | TPOT Target |
|-------|------|--------------|-------------|
| 4 | Mixed | 6K+1.5K | 18ms |
| 4 | Mixed | 4K+1.5K | 11ms |

## Execution Workflow

### Step 1: Gather Requirements

1. Identify target model and hardware (A2/A3)
2. Determine deployment mode (PD Mixed/Separation)
3. Define performance targets (latency vs throughput)
4. Specify test datasets and accuracy benchmarks

### Step 2: Create Deployment Script

1. Generate environment variable configuration
2. Configure server arguments based on model type
3. Set up parallel strategy (TP/DP/EP)
4. Create startup script

### Step 3: Develop Benchmark Scripts

1. Create performance test script
2. Create accuracy test script
3. Configure test parameters

### Step 4: Execute Tests

1. Start model service
2. Wait for service readiness
3. Run performance benchmarks
4. Run accuracy benchmarks
5. Collect all metrics

### Step 5: Generate Report

1. Compile test environment details
2. Summarize functional verification
3. Present performance metrics
4. Document accuracy results
5. List issues and suggestions

### Step 6: Feedback Loop

If testing reveals code issues:

1. **Document the issue**: Error message, reproduction steps, expected vs actual
2. **Return to `sglang-model-adapter`** if: architecture support incomplete, NPU code bugs, weight mapping incorrect
3. **Provide feedback**: failed test case, relevant logs, suggested fix

## Model-Specific Configurations

### DeepSeek-R1 / DeepSeek-V3

```bash
# Key settings
--quantization modelslim
--moe-a2a-backend deepep
--enable-dp-attention
--speculative-algorithm NEXTN  # optional
```

### Qwen3 Series

```bash
# Standard
--attention-backend ascend

# MoE
--moe-a2a-backend deepep

# Speculative
--speculative-algorithm EAGLE3
```

## Common Issues

See [shared/npu_common_reference.md](../shared/npu_common_reference.md) for quick fixes.

| Issue | Solution |
|-------|----------|
| OOM Error | Reduce `--mem-fraction-static` by 0.05 |
| Slow TTFT | Increase `--max-prefill-tokens` |
| Low Throughput | Adjust `--cuda-graph-bs` |
| Communication Timeout | Increase `HCCL_BUFFSIZE` |
| Accuracy Drop | Check quantization config |

## Hardware Notes

| Hardware | Devices | Best For |
|----------|---------|----------|
| Atlas 800I A2 | 8 NPU | Small/Medium models |
| Atlas 800I A3 | 16 NPU | Large models, High throughput |
