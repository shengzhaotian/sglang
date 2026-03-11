# NPU Testing Quick Reference Guide

Quick reference for common NPU testing configurations. For complete reference, see [shared/npu_common_reference.md](../shared/npu_common_reference.md).

## Hard Constraints

- **Never modify system environment** (e.g., `apt install`, `pip install`, system config changes) without explicit user approval.
- **Always set PYTHONPATH before launching sglang services**:
  ```bash
  export PYTHONPATH=${PWD}/python:$PYTHONPATH
  ```

## Hardware Reference

| Hardware | Devices | Memory | Best For |
|----------|---------|--------|----------|
| Atlas 800I A2 | 8 NPU | 64GB/device | Small/Medium models |
| Atlas 800I A3 | 16 NPU | 64GB/device | Large models, High throughput |

## Model Configuration Quick Reference

### DeepSeek-R1 / DeepSeek-V3

#### Low Latency (20ms TPOT) - 32 Cards Separation
```bash
export PYTHONPATH=${PWD}/python:$PYTHONPATH
export HCCL_BUFFSIZE=650
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=78

# Prefill: --tp-size 16 --dp-size 2 --deepep-mode normal --speculative-algorithm NEXTN
# Decode: --tp-size 32 --dp-size 32 --deepep-mode low_latency --cuda-graph-bs 12 14 16 18 20 22 24 26
```

#### High Throughput (50ms TPOT) - 8 Cards Mixed
```bash
export PYTHONPATH=${PWD}/python:$PYTHONPATH
export HCCL_BUFFSIZE=1600
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64

# --tp 16 --enable-dp-attention --dp-size 16 --deepep-mode auto --speculative-algorithm NEXTN
```

### Qwen3-32B

#### Low Latency (11-18ms TPOT) - 4 Cards
```bash
export PYTHONPATH=${PWD}/python:$PYTHONPATH
export HCCL_BUFFSIZE=400
export HCCL_OP_EXPANSION_MODE=AIV

# --tp-size 8 --cuda-graph-bs 8 16 24 32 --speculative-algorithm EAGLE3
```

#### High Throughput (50ms TPOT) - 2 Cards
```bash
export PYTHONPATH=${PWD}/python:$PYTHONPATH
# --tp-size 4 --quantization modelslim --cuda-graph-bs 16 32 64 68 72 78 --speculative-algorithm EAGLE3
```

### Qwen3-235B-A22B (MoE)

```bash
export PYTHONPATH=${PWD}/python:$PYTHONPATH
export HCCL_BUFFSIZE=1600
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16

# --tp-size 16 --moe-a2a-backend deepep --deepep-mode auto --enable-dp-attention --dp-size 16
```

## Environment Variables Quick Reference

See [shared/npu_common_reference.md](../shared/npu_common_reference.md) for complete list.

### Essential
| Variable | Purpose | Value |
|----------|---------|-------|
| `ASCEND_USE_FIA` | FIA attention | `1` |
| `HCCL_BUFFSIZE` | Communication buffer | `400-1600` |

### Model-Specific
| Variable | When to Use |
|----------|-------------|
| `SGLANG_NPU_USE_MLAPO` | DeepSeek models |
| `SGLANG_USE_FIA_NZ` | With MLAPO |
| `DEEP_NORMAL_MODE_USE_INT8_QUANT` | MoE models |
| `SGLANG_ENABLE_SPEC_V2` | Speculative decoding |

## Benchmark Commands

### Performance Testing
```bash
# Standard
export PYTHONPATH=${PWD}/python:$PYTHONPATH
python -m sglang.bench_serving --dataset-name random --backend sglang \
    --host 127.0.0.1 --port 6688 --num-prompts 1024 \
    --random-input-len 3500 --random-output-len 1500 --max-concurrency 256

# Low latency
export PYTHONPATH=${PWD}/python:$PYTHONPATH
python -m sglang.bench_serving --dataset-name random --backend sglang \
    --host 127.0.0.1 --port 6688 --num-prompts 32 \
    --random-input-len 6000 --random-output-len 1600 --max-concurrency 32

# High throughput
export PYTHONPATH=${PWD}/python:$PYTHONPATH
python -m sglang.bench_serving --dataset-name random --backend sglang \
    --host 127.0.0.1 --port 6688 --num-prompts 3072 \
    --random-input-len 3500 --random-output-len 1500 --max-concurrency 768 --request-rate 16
```

### Accuracy Testing (GSM8K)
```python
from types import SimpleNamespace
from sglang.test.few_shot_gsm8k import run_eval

args = SimpleNamespace(num_shots=5, num_questions=200, max_new_tokens=512,
    parallel=32, host="http://127.0.0.1", port=6688)
metrics = run_eval(args)
```

## Performance Targets

### DeepSeek-R1 (Atlas 800I A3)
| Cards | Mode | Input+Output | TPOT | Throughput |
|-------|------|--------------|------|------------|
| 32 | Separation | 3.5K+1.5K | 20ms | ~800 tok/s |
| 32 | Separation | 3.5K+1.5K | 50ms | ~1500 tok/s |
| 8 | Mixed | 2K+2K | 50ms | ~800 tok/s |

### Qwen3-32B (Atlas 800I A3)
| Cards | Mode | Input+Output | TPOT | Throughput |
|-------|------|--------------|------|------------|
| 8 | Mixed | 18K+4K | 12ms | ~80 tok/s |
| 4 | Mixed | 6K+1.5K | 18ms | ~55 tok/s |
| 4 | Mixed | 4K+1.5K | 11ms | ~90 tok/s |

## Troubleshooting

See [shared/npu_common_reference.md](../shared/npu_common_reference.md) for error codes.

| Issue | Solution |
|-------|----------|
| OOM | Reduce `--mem-fraction-static` by 0.05 |
| Slow TTFT | Increase `--max-prefill-tokens` |
| Low Throughput | Adjust `--cuda-graph-bs` |
| Timeout | Increase `HCCL_BUFFSIZE` |
| High ITL Variance | Set `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` |
