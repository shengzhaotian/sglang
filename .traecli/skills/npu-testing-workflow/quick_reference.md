# NPU Testing Quick Reference Guide

This guide provides quick reference configurations for common NPU testing scenarios.

## Hardware Reference

| Hardware | Device Count | Memory | Best For |
|----------|-------------|--------|----------|
| Atlas 800I A2 | 8 NPU | 64GB/device | Small/Medium models |
| Atlas 800I A3 | 16 NPU | 64GB/device | Large models, High throughput |

## Model Configuration Quick Reference

### DeepSeek-R1 / DeepSeek-V3

#### Low Latency Configuration (20ms TPOT)
```bash
# Hardware: Atlas 800I A3 x 2 (32 cards)
# Dataset: 3.5K input + 1.5K output

export HCCL_BUFFSIZE=650
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=78

# Prefill Server
python -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --disaggregation-mode prefill \
    --tp-size 16 --dp-size 2 \
    --mem-fraction-static 0.81 \
    --max-running-requests 8 \
    --deepep-mode normal \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 1 \
    --speculative-num-draft-tokens 2

# Decode Server
python -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --disaggregation-mode decode \
    --tp-size 32 --dp-size 32 \
    --mem-fraction-static 0.815 \
    --max-running-requests 832 \
    --cuda-graph-bs 12 14 16 18 20 22 24 26 \
    --deepep-mode low_latency \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 2 \
    --speculative-num-draft-tokens 3
```

#### High Throughput Configuration (50ms TPOT)
```bash
# Hardware: Atlas 800I A3 x 1 (8 cards)
# Dataset: 2K input + 2K output

export HCCL_BUFFSIZE=1600
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64

python -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --tp 16 \
    --mem-fraction-static 0.74 \
    --max-running-requests 256 \
    --cuda-graph-bs 4 8 16 \
    --deepep-mode auto \
    --enable-dp-attention \
    --dp-size 16 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-num-draft-tokens 4
```

### Qwen3-32B

#### Low Latency Configuration (11-18ms TPOT)
```bash
# Hardware: Atlas 800I A3 x 1 (4 cards)
# Dataset: 4K-6K input + 1.5K output

export HCCL_BUFFSIZE=400
export HCCL_OP_EXPANSION_MODE=AIV

python -m sglang.launch_server \
    --model-path Qwen/Qwen3-32B \
    --tp-size 8 \
    --mem-fraction-static 0.72 \
    --max-running-requests 32 \
    --cuda-graph-bs 8 16 24 32 \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path ${DRAFT_MODEL_PATH} \
    --speculative-num-steps 4 \
    --speculative-num-draft-tokens 5
```

#### High Throughput Configuration (50ms TPOT)
```bash
# Hardware: Atlas 800I A3 x 1 (2 cards)
# Dataset: 3.5K input + 1.5K output

python -m sglang.launch_server \
    --model-path Qwen/Qwen3-32B \
    --tp-size 4 \
    --quantization modelslim \
    --mem-fraction-static 0.72 \
    --max-running-requests 78 \
    --cuda-graph-bs 16 32 64 68 72 78 \
    --speculative-algorithm EAGLE3 \
    --speculative-num-steps 3 \
    --speculative-num-draft-tokens 4
```

### Qwen3-235B-A22B (MoE)

```bash
# Hardware: Atlas 800I A3 x 1 (8 cards)

export HCCL_BUFFSIZE=1600
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16

python -m sglang.launch_server \
    --model-path Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --tp-size 16 \
    --mem-fraction-static 0.8 \
    --max-running-requests 272 \
    --cuda-graph-bs 3 4 6 8 10 12 13 14 15 16 17 \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --enable-dp-attention \
    --dp-size 16 \
    --enable-dp-lm-head \
    --speculative-algorithm EAGLE3 \
    --speculative-num-steps 3 \
    --speculative-num-draft-tokens 4
```

## Environment Variables Quick Reference

### Essential Variables
| Variable | Purpose | Typical Values |
|----------|---------|----------------|
| `SGLANG_SET_CPU_AFFINITY` | CPU affinity | `1` |
| `PYTORCH_NPU_ALLOC_CONF` | Memory allocation | `expandable_segments:True` |
| `STREAMS_PER_DEVICE` | Stream pool size | `32` |
| `HCCL_BUFFSIZE` | Communication buffer (MB) | `400-1600` |

### NPU Optimization Variables
| Variable | Purpose | When to Use |
|----------|---------|-------------|
| `SGLANG_NPU_USE_MLAPO` | MLA attention fusion | DeepSeek models |
| `SGLANG_USE_FIA_NZ` | KV Cache reshape | With MLAPO |
| `SGLANG_ENABLE_OVERLAP_PLAN_STREAM` | Overlap optimization | Decode servers |
| `SGLANG_ENABLE_SPEC_V2` | Speculative decoding v2 | With speculative decoding |

### DeepEP Variables
| Variable | Purpose | Typical Values |
|----------|---------|----------------|
| `DEEP_NORMAL_MODE_USE_INT8_QUANT` | INT8 quantization | `1` |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | Dispatch tokens | `12-96` |
| `DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS` | Long sequence tokens | `512-1024` |
| `TASK_QUEUE_ENABLE` | Task queue optimization | `1` (decode), `2` (prefill) |

## Benchmark Commands Quick Reference

### Performance Testing
```bash
# Basic benchmark
python -m sglang.bench_serving \
    --dataset-name random \
    --backend sglang \
    --host 127.0.0.1 --port 6688 \
    --num-prompts 1024 \
    --random-input-len 3500 \
    --random-output-len 1500 \
    --max-concurrency 256

# Low latency test
python -m sglang.bench_serving \
    --dataset-name random \
    --backend sglang \
    --host 127.0.0.1 --port 6688 \
    --num-prompts 32 \
    --random-input-len 6000 \
    --random-output-len 1600 \
    --max-concurrency 32

# High throughput test
python -m sglang.bench_serving \
    --dataset-name random \
    --backend sglang \
    --host 127.0.0.1 --port 6688 \
    --num-prompts 3072 \
    --random-input-len 3500 \
    --random-output-len 1500 \
    --max-concurrency 768 \
    --request-rate 16
```

### Accuracy Testing (GSM8K)
```python
from types import SimpleNamespace
from sglang.test.few_shot_gsm8k import run_eval

args = SimpleNamespace(
    num_shots=5,
    data_path=None,
    num_questions=200,
    max_new_tokens=512,
    parallel=32,
    host="http://127.0.0.1",
    port=6688,
)
metrics = run_eval(args)
print(f"Accuracy: {metrics['accuracy']}")
```

## Troubleshooting Quick Reference

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| OOM Error | Memory fraction too high | Reduce `--mem-fraction-static` by 0.05 |
| Slow TTFT | Prefill bottleneck | Increase `--max-prefill-tokens` |
| Low Throughput | Suboptimal batch size | Adjust `--cuda-graph-bs` |
| Communication Timeout | HCCL buffer too small | Increase `HCCL_BUFFSIZE` |
| High ITL Variance | Memory fragmentation | Set `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` |
| Speculative Decoding Slow | Draft model mismatch | Check draft model compatibility |

## Performance Targets Reference

### DeepSeek-R1 (Atlas 800I A3)
| Cards | Mode | Input+Output | TPOT Target | Throughput Target |
|-------|------|--------------|-------------|-------------------|
| 32 | Separation | 3.5K+1.5K | 20ms | ~800 tok/s |
| 32 | Separation | 3.5K+1.5K | 50ms | ~1500 tok/s |
| 16 | Separation | 2K+2K | 50ms | ~1200 tok/s |
| 8 | Mixed | 2K+2K | 50ms | ~800 tok/s |

### Qwen3-32B (Atlas 800I A3)
| Cards | Mode | Input+Output | TPOT Target | Throughput Target |
|-------|------|--------------|-------------|-------------------|
| 8 | Mixed | 18K+4K | 12ms | ~80 tok/s |
| 4 | Mixed | 6K+1.5K | 18ms | ~55 tok/s |
| 4 | Mixed | 4K+1.5K | 11ms | ~90 tok/s |
| 2 | Mixed | 3.5K+1.5K | 50ms | ~30 tok/s |

## File Locations

| File | Purpose |
|------|---------|
| `SKILL.md` | Main skill documentation |
| `deployment_script_template.sh` | Service deployment script template |
| `benchmark_script_template.py` | Client benchmarking script template |
| `report_generator.py` | Test report generator |
| `quick_reference.md` | This quick reference guide |
