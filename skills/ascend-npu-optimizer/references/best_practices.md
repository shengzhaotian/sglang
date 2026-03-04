# Best Practices for LLM on Ascend NPU

This document contains model-specific best practices from SGLang documentation for Ascend NPU deployment.

## Hardware Reference

| Hardware | Chip | HBM | NPUs per Server |
|----------|------|-----|-----------------|
| Atlas 800I A2 | 910B/910B3 | 64 GB per NPU | 8 |
| Atlas 800I A3 | 910A | 64 GB per NPU | 8 |

---

## Parallelism Strategy by Architecture

### Standard Models (Dense)

| Model | Recommended | Reason |
|-------|-------------|--------|
| Qwen3-32B | TP | All weights benefit from TP sharding |
| Llama-3.1-70B | TP | Standard transformer architecture |
| Qwen3-30B-A3B | TP | Dense MoE (small expert count) |

### MoE Models

| Model | Recommended | Reason |
|-------|-------------|--------|
| Qwen3-235B-A22B | **EP + TP** | EP has lower communication for experts |
| Mixtral-8x7B | **EP + TP** | MoE architecture benefits from EP |
| Qwen3-Coder-480B-A35B | **EP + TP** | Large MoE model |

### MLA + MoE Models

| Model | Recommended | Reason |
|-------|-------------|--------|
| DeepSeek-V3 | **EP + TP + MLA optimizations** | Combined architecture |
| DeepSeek-R1 | **EP + TP + MLA optimizations** | Combined architecture |

---

## Detailed Model Configurations

### Qwen3-32B

#### Atlas 800I A3

| Scenario | Input+Output | TPOT | Cards | Configuration |
|----------|--------------|------|-------|---------------|
| Low Latency | 4K+1.5K | 11ms | 4 | BF16, TP=8, EAGLE3 |
| Low Latency | 6K+1.5K | 18ms | 4 | BF16, TP=8 |
| Long Context | 18K+4K | 12ms | 8 | BF16, TP=16, EAGLE3 |
| High Throughput | 3.5K+1.5K | 50ms | 2 | W8A8 INT8, TP=4, EAGLE3 |
| High Throughput | 2K+2K | 50ms | 2 | W8A8 INT8, TP=4, EAGLE3 |

#### Atlas 800I A2

| Scenario | Input+Output | TPOT | Cards | Configuration |
|----------|--------------|------|-------|---------------|
| Low Latency | 4K+1.5K | 11ms | 8 | BF16, TP=8, EAGLE3 |
| Low Latency | 6K+1.5K | 18ms | 8 | W8A8 INT8, TP=8 |
| High Throughput | 3.5K+1.5K | 50ms | 8 | W8A8 INT8, TP=8 |
| High Throughput | 2K+2K | 50ms | 8 | W8A8 INT8, TP=8 |

---

### Qwen3-235B-A22B

#### Atlas 800I A3

| Scenario | Input+Output | TPOT | Cards | Configuration |
|----------|--------------|------|-------|---------------|
| Low Latency | 11K+1K | 10ms | 8 | BF16, TP=16, EAGLE3 |
| High Throughput | 3.5K+1.5K | 50ms | 8 | W8A8 INT8, **TP=16, DP=16**, EAGLE3 |
| High Throughput | 2K+2K | 50ms | 8 | W8A8 INT8, **TP=16, DP=16**, EAGLE3 |
| High Throughput | 2K+2K | 100ms | 8 | W8A8 INT8, **TP=16, DP=16**, EAGLE3 |
| High Throughput | 3.5K+1.5K | 50ms | 24 | W8A8 INT8, PD Separation |
| High Throughput | 3.5K+1.5K | 50ms | 16 | W8A8 INT8, **TP=32, DP=32**, EAGLE3 |

**Key Insight**: For Qwen3-235B-A22B (MoE), **DP-Attention with DP=TP is recommended** for better throughput.

---

### DeepSeek-R1

#### Atlas 800I A3

| Scenario | Input+Output | TPOT | Cards | Deploy Mode | Configuration |
|----------|--------------|------|-------|-------------|---------------|
| Low Latency | 6K+1.6K | 20ms | 32 | PD Separation | W8A8 INT8, TP=32, DP=16 |
| Low Latency | 3.9K+1K | 20ms | 32 | PD Separation | W8A8 INT8, TP=32, DP=16 |
| Low Latency | 3.5K+1.5K | 20ms | 32 | PD Separation | W8A8 INT8, TP=32, DP=32 |
| Low Latency | 3.5K+1K | 20ms | 32 | PD Separation | W8A8 INT8, TP=32, DP=16 |
| High Throughput | 3.5K+1.5K | 50ms | 32 | PD Separation | W8A8 INT8, TP=32, DP=32 |
| High Throughput | 2K+2K | 50ms | 8 | PD Mixed | W4A8 INT8, TP=16, DP=16 |
| High Throughput | 2K+2K | 50ms | 16 | PD Separation | W4A8 INT8, TP=16, DP=16 |
| High Throughput | 3.5K+1.5K | 50ms | 8 | PD Mixed | W4A8 INT8, TP=16, DP=4 |
| High Throughput | 3.5K+1.5K | 50ms | 16 | PD Separation | W4A8 INT8, TP=16, DP=16 |

---

### DeepSeek-V3.2-Exp

#### Atlas 800I A3

| Scenario | Input+Output | TPOT | Cards | Deploy Mode | Configuration |
|----------|--------------|------|-------|-------------|---------------|
| Long Context | 64K+3K | 30ms | 32 | PD Separation | W8A8 INT8, TP=32, DP=8, NSA CP |

---

### Qwen3-Coder-480B-A35B-Instruct

#### Atlas 800I A3

| Scenario | Input+Output | TPOT | Cards | Deploy Mode | Configuration |
|----------|--------------|------|-------|-------------|---------------|
| High Throughput | 3.5K+1.5K | 50ms | 24 | PD Separation | W8A8 INT8, TP=32, DP=32 |
| High Throughput | 3.5K+1.5K | 50ms | 16 | PD Mixed | W8A8 INT8, TP=32, DP=32 |
| High Throughput | 3.5K+1.5K | 50ms | 8 | PD Mixed | W8A8 INT8, TP=16, DP=16 |

---

### Qwen3-30B-A3B

#### Atlas 800I A3

| Scenario | Input+Output | TPOT | Cards | Deploy Mode | Configuration |
|----------|--------------|------|-------|-------------|---------------|
| High Throughput | 3.5K+1.5K | 50ms | 1 | PD Mixed | W8A8 INT8, TP=2 |

---

### Qwen3-Next-80B-A3B-Instruct

#### Atlas 800I A3

| Scenario | Input+Output | TPOT | Cards | Deploy Mode | Configuration |
|----------|--------------|------|-------|-------------|---------------|
| High Throughput | 3.5K+1.5K | 50ms | 2 | PD Mixed | W8A8 INT8 |

---

## Key Configuration Patterns

### EP for MoE (Recommended)

```bash
# For MoE models, use EP for experts
--tp-size 8 \
--ep-size 8 \
--enable-dp-attention \
--enable-dp-lm-head \
--moe-a2a-backend deepep \
--deepep-mode auto
```

### DP-Attention for MLA Models

```bash
# For MLA models (DeepSeek), use DP-Attention
--tp-size <tp> \
--dp-size <dp> \
--enable-dp-attention \
--enable-dp-lm-head \
--moe-dense-tp-size 1
```

### EAGLE3 Speculative Decoding

```bash
--speculative-algorithm EAGLE3 \
--speculative-draft-model-path <draft_model_path> \
--speculative-num-steps 4 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 5 \
--speculative-draft-model-quantization unquant
```

### NEXTN Speculative Decoding (DeepSeek)

```bash
--speculative-algorithm NEXTN \
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4
```

### INT8 Quantization (ModelSlim)

```bash
--quantization modelslim \
--dtype bfloat16
```

### PD Separation Mode

**Prefill Server:**
```bash
--disaggregation-mode prefill \
--disaggregation-transfer-backend ascend \
--disaggregation-bootstrap-port 8995 \
--tp-size 16 \
--dp-size 2 \
--enable-dp-attention \
--moe-a2a-backend deepep \
--deepep-mode normal
```

**Decode Server:**
```bash
--disaggregation-mode decode \
--disaggregation-transfer-backend ascend \
--tp-size 32 \
--dp-size 32 \
--enable-dp-attention \
--enable-dp-lm-head \
--moe-dense-tp-size 1 \
--moe-a2a-backend deepep \
--deepep-mode low_latency
```

**Router:**
```bash
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://P_IP:8000 8995 \
    --decode http://D_IP:8001 \
    --host 127.0.0.1 \
    --port 6688 \
    --mini-lb
```

---

## Environment Variables Reference

### SGLang-Specific Variables

| Variable | Description | Default | Typical Values |
|----------|-------------|---------|----------------|
| `SGLANG_NPU_USE_MLAPO` | Enable MLAPO fusion operator in attention preprocessing for MLA models | `false` | `1` for MLA models |
| `SGLANG_USE_FIA_NZ` | Reshape KV Cache for FIA NZ format (must enable with MLAPO) | `false` | `1` with MLAPO |
| `SGLANG_NPU_USE_MULTI_STREAM` | Enable dual-stream for shared/routing experts and NSA Indexer | `false` | `1` for DeepSeek |
| `SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT` | Disable casting weight to NPU ACL format | `false` | `1` |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | Max dispatched tokens per rank | `128` | `8-96` based on scenario |
| `SGLANG_SET_CPU_AFFINITY` | Enable CPU pinning for performance | `false` | `1` (recommended) |
| `SGLANG_ENABLE_OVERLAP_PLAN_STREAM` | Enable overlap optimization | `false` | `1` |
| `SGLANG_ENABLE_SPEC_V2` | Enable speculative decoding v2 | `false` | `1` |
| `SGLANG_SCHEDULER_SKIP_ALL_GATHER` | Skip all-gather in scheduler | `false` | `1` for decode |
| `SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE` | Decrease prefill idle time | `false` | `1` or `2` |
| `SGLANG_DP_ROUND_ROBIN` | Enable DP round robin routing | `false` | `1` |

### DeepEP Variables

| Variable | Description | Default | Typical Values |
|----------|-------------|---------|----------------|
| `DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS` | Tokens per round in dispatch (ant-moving) | `8192` | `512-1024` |
| `DEEPEP_NORMAL_LONG_SEQ_ROUND` | Number of rounds in dispatch | `1` | `5-16` |
| `DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ` | Enable ant-moving in combine | `0` | `0` or `1` |
| `MOE_ENABLE_TOPK_NEG_ONE` | Enable when expert ID contains -1 | `0` | `0` or `1` |
| `DEEP_NORMAL_MODE_USE_INT8_QUANT` | Quantize x to int8 in dispatch | `0` | `1` for INT8 models |

### HCCL Communication Variables

| Variable | Description | Default | Typical Values |
|----------|-------------|---------|----------------|
| `HCCL_BUFFSIZE` | Buffer size for NPU communication (MB) | `200` | `400-4300` based on model |
| `HCCL_SOCKET_IFNAME` | Network card for HCCL | - | `lo` for single node, `data0.3001` for multi-node |
| `GLOO_SOCKET_IFNAME` | Network card for GLOO | - | Same as HCCL_SOCKET_IFNAME |
| `HCCL_OP_EXPANSION_MODE` | Communication algorithm scheduling | - | `AIV` |

### System Variables

| Variable | Description | Default | Typical Values |
|----------|-------------|---------|----------------|
| `PYTORCH_NPU_ALLOC_CONF` | Memory allocator config | - | `expandable_segments:True` |
| `STREAMS_PER_DEVICE` | Max streams for stream pool | `32` | `32` |
| `TASK_QUEUE_ENABLE` | Dispatch queue optimization level | `1` | `0`, `1`, or `2` |
| `ASCEND_MF_STORE_URL` | MemFabric store URL for PD separation | - | `tcp://IP:PORT` |
| `ASCEND_LAUNCH_BLOCKING` | Synchronous operator execution | `0` | `0` (unset recommended) |

---

## mem-fraction-static Guidelines

| Memory Utilization | Recommended Value | Scenario |
|-------------------|-------------------|----------|
| < 30% | 0.50 - 0.60 | Small models, large HBM |
| 30-50% | 0.60 - 0.70 | Balanced |
| 50-70% | 0.70 - 0.80 | Moderate pressure |
| 70-85% | 0.80 - 0.86 | High utilization |
| > 85% | 0.86+ | Maximum utilization (risky) |

### Model-Specific mem-fraction-static Values

| Model | Cards | Mode | mem-fraction-static |
|-------|-------|------|---------------------|
| DeepSeek-R1 | 32 | Prefill | 0.81 |
| DeepSeek-R1 | 32 | Decode | 0.75 - 0.815 |
| DeepSeek-R1 | 8 | Mixed | 0.71 - 0.74 |
| Qwen3-235B-A22B | 8 | Mixed | 0.75 - 0.81 |
| Qwen3-32B | 4 | Mixed | 0.72 |
| Qwen3-30B-A3B | 1 | Mixed | 0.86 |

---

## cuda-graph-bs Configuration

### Guidelines

- Include batch sizes that cover expected concurrency levels
- Start from small batch sizes (1-4) for low latency
- Include medium batch sizes (16-32) for balanced workloads
- Include large batch sizes (64+) for high throughput

### Model-Specific Examples

| Model | Cards | Scenario | cuda-graph-bs |
|-------|-------|----------|---------------|
| DeepSeek-R1 | 32 | Low Latency | 2 4 6 |
| DeepSeek-R1 | 32 | High Throughput | 12 14 16 18 20 22 24 26 |
| DeepSeek-R1 | 16 | High Throughput | 8 10 12 14 16 18 20 22 24 |
| Qwen3-235B-A22B | 8 | High Throughput | 6 8 10 12 15 18 28 30 |
| Qwen3-32B | 4 | Low Latency | 8 16 24 32 |
| Qwen3-32B | 2 | High Throughput | 54 60 66 72 78 84 90 108 114 120 |

---

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| OOM during loading | Reduce `mem-fraction-static` or use INT8 quantization |
| Slow token generation | Add EAGLE3 speculative decoding |
| Low throughput for MoE | Use DP-Attention with `--enable-dp-attention` |
| HCCL timeout | Increase `HCCL_BUFFSIZE` or check network |
| KV cache OOM | Reduce `max-running-requests` or use KV cache quantization |
| Low latency requirement | Use smaller `cuda-graph-bs`, enable EAGLE3 |
| High throughput requirement | Use larger `cuda-graph-bs`, increase `max-running-requests` |

---

## EP vs TP for MoE: Performance Comparison

| Metric | TP for MoE | EP for MoE |
|--------|------------|------------|
| Communication Pattern | All-reduce every layer | All-to-all for dispatch/combine |
| Communication Volume | O(hidden_size × num_layers) | O(batch_size × hidden_size) |
| Memory per Device | Same | Same |
| Throughput | Baseline | **1.5-2x higher** |
| Latency | Higher | **Lower** |
| Scalability | Limited by all-reduce | **Better with more NPUs** |

**Recommendation**: Always use EP for MoE models when possible.

---

## System Optimization Commands

```bash
# CPU performance mode
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable NUMA balancing
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

# Set CPU affinity
export SGLANG_SET_CPU_AFFINITY=1

# Unset blocking mode
unset ASCEND_LAUNCH_BLOCKING
```

---

## Quick Reference: Launch Commands

### DeepSeek-R1 (32 cards, PD Separation, Low Latency)

```bash
# Prefill (on prefill nodes)
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1
export HCCL_BUFFSIZE=1536
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export TASK_QUEUE_ENABLE=2

python -m sglang.launch_server --model-path ${MODEL_PATH} \
    --disaggregation-mode prefill \
    --tp-size 16 --dp-size 2 \
    --enable-dp-attention \
    --attention-backend ascend --device npu \
    --quantization modelslim \
    --moe-a2a-backend deepep --deepep-mode normal \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 1 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 2 \
    --mem-fraction-static 0.81

# Decode (on decode nodes)
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export HCCL_BUFFSIZE=650
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=12
export TASK_QUEUE_ENABLE=1
export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1

python -m sglang.launch_server --model-path ${MODEL_PATH} \
    --disaggregation-mode decode \
    --tp-size 32 --dp-size 16 \
    --enable-dp-attention --enable-dp-lm-head \
    --moe-dense-tp-size 1 \
    --attention-backend ascend --device npu \
    --quantization modelslim \
    --moe-a2a-backend deepep --deepep-mode low_latency \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --cuda-graph-bs 2 4 6 \
    --mem-fraction-static 0.75
```

### Qwen3-235B-A22B (8 cards, Mixed Mode, High Throughput)

```bash
export HCCL_BUFFSIZE=2100
export HCCL_OP_EXPANSION_MODE="AIV"
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1

python -m sglang.launch_server --model-path $MODEL_PATH \
    --tp 16 --dp-size 16 \
    --enable-dp-attention --enable-dp-lm-head \
    --attention-backend ascend --device npu \
    --quantization modelslim \
    --max-running-requests 480 \
    --moe-a2a-backend deepep --deepep-mode auto \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path <draft_path> \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --cuda-graph-bs 6 8 10 12 15 18 28 30 \
    --mem-fraction-static 0.75
```
