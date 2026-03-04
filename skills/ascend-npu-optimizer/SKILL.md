---
name: ascend-npu-optimizer
description: |
  Optimize LLM deployment on Ascend NPU hardware (Atlas 800I A2/A3). Use this skill when the user wants to run, optimize, or configure models on Huawei Ascend NPUs. Triggers when user mentions: Ascend, NPU, Atlas 800I, 910B, A2, A3, Huawei AI, CANN, torch_npu, or asks about running models on NPUs. Also triggers when user asks about model optimization, parallelism configuration, or deployment setup for NPU environments.
---

# Ascend NPU Model Optimization Skill

This skill guides you through optimizing LLM deployment on Huawei Ascend NPU hardware. It covers hardware detection, parallelism selection, and combined optimization strategies.

**IMPORTANT**: This skill focuses on NPU-specific optimization. For memory calculation, use the `memory-calculation` skill first to understand model memory requirements.

## Workflow Overview

1. **Detect Hardware**: Identify NPU type, count, and memory
2. **Gather Requirements**: Input/output length, concurrency, latency/throughput goals
3. **Calculate Memory**: Use `memory-calculation` skill to analyze model memory
4. **Select Parallelism**: Choose optimal TP/EP/DP configuration based on architecture
5. **Configure Environment**: Set environment variables for optimal performance
6. **Recommend Configuration**: Generate launch command with optimal settings

---

## Step 1: Hardware Detection

Run these commands to identify the hardware:

```bash
# Check NPU info
npu-smi info

# Check CANN version
cat /usr/local/Ascend/version.info

# List NPU devices
ls -la /dev/davinci*
```

**Hardware Types:**
| Chip | Hardware | HBM per NPU | NPUs per Server |
|------|----------|-------------|-----------------|
| 910B3 | Atlas 800I A2 | 64 GB | 8 |
| 910B | Atlas 800I A2 | 64 GB | 8 |
| 910A | Atlas 800I A3 | 64 GB | 8 |

---

## Step 2: Gather Requirements

Ask the user for:

1. **Model path**: Local path to model weights
2. **Max input length**: Tokens (e.g., 4096)
3. **Max output length**: Tokens (e.g., 2048)
4. **Max running requests**: Concurrency (e.g., 256)
5. **Optimization goal**: Low latency or high throughput
6. **Available resources**: EAGLE3 draft model? Quantized weights? Docker container?

---

## Step 3: Calculate Memory

**CRITICAL**: Before selecting parallelism, use the `memory-calculation` skill to:

1. Read the model implementation files to understand architecture
2. Calculate weight memory based on actual model structure
3. Calculate KV cache memory based on attention mechanism
4. Determine memory per device with different parallelism configurations

```
Use the memory-calculation skill to get:
- Model architecture type (Standard, MoE, MLA, MLA+MoE)
- Total weight memory
- KV cache per token
- Memory per device with different TP/EP configurations
```

---

## Step 4: Select Parallelism Strategy

### 4.1 Architecture-Based Parallelism Selection

**CRITICAL**: Choose parallelism based on model architecture:

| Architecture | Recommended Strategy | Reason |
|--------------|---------------------|--------|
| **Standard (Dense)** | TP | All weights benefit from TP sharding |
| **MoE** | **DP-Attention (TP=DP)** | Lower communication overhead for experts |
| **MLA** | **DP-Attention** | Eliminates KV cache duplication |
| **MLA + MoE** | **DP-Attention + EP** | Best of both worlds |

### 4.2 Why DP-Attention is Better for MoE/MLA

For MoE and MLA models, **DP-Attention is generally more efficient than pure TP**:

1. **No KV cache duplication**: Each DP replica has its own KV cache
2. **Larger batch sizes**: Memory savings enable larger batch sizes
3. **Independent forward modes**: Each DP replica can be in different modes
4. **Better throughput**: DP-Attention typically achieves 1.5-2x throughput for MLA models

### 4.3 Parallelism Configuration Matrix

#### For Standard Models (e.g., Qwen3-32B, Llama-70B)

```bash
# Use TP to shard all weights
--tp-size <num_npus>
```

#### For MoE Models (e.g., Qwen3-235B, Mixtral)

```bash
# RECOMMENDED: DP-Attention with TP=DP
--tp-size <num_npus> \
--dp-size <num_npus> \
--enable-dp-attention \
--enable-dp-lm-head \
--moe-a2a-backend deepep \
--deepep-mode auto
```

#### For MLA + MoE Models (e.g., DeepSeek-V3, DeepSeek-R1)

```bash
# DP-Attention with EP for routed experts
--tp-size <num_npus> \
--dp-size <dp_size> \
--ep-size <num_npus> \
--moe-dense-tp-size 1 \
--enable-dp-attention \
--enable-dp-lm-head \
--moe-a2a-backend deepep \
--deepep-mode low_latency
```

### 4.4 Decision Flowchart

```
Is model MoE or MLA?
  |
  +-- No --> Use TP (standard parallelism)
  |
  +-- Yes --> Is model MLA + MoE?
                |
                +-- No (Pure MoE or MLA) --> Use DP-Attention
                |                               --tp-size N --dp-size N --enable-dp-attention
                |
                +-- Yes (MLA + MoE) --> Use DP-Attention + EP
                                        --tp-size N --dp-size D --ep-size N --moe-dense-tp-size 1
```

---

## Step 5: Configure Environment Variables

### 5.1 Essential Environment Variables

```bash
# CPU affinity (always recommended)
export SGLANG_SET_CPU_AFFINITY=1

# Memory allocation
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Concurrency
export STREAMS_PER_DEVICE=32
```

### 5.2 Model-Specific Environment Variables

#### For MLA Models (DeepSeek)

```bash
# MLA optimizations
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1

# Multi-stream for shared/routing experts
export SGLANG_NPU_USE_MULTI_STREAM=1
```

#### For Speculative Decoding

```bash
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
```

#### For DeepEP with MoE

```bash
# DeepEP tuning
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export DEEPEP_NORMAL_LONG_SEQ_ROUND=10
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=512
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1  # For INT8 models
```

### 5.3 Communication Variables

```bash
# HCCL buffer size (varies by model and scenario)
export HCCL_BUFFSIZE=400  # Small for low latency
# or
export HCCL_BUFFSIZE=1600  # Large for high throughput

# Network interface
export HCCL_SOCKET_IFNAME=lo  # Single node
# or
export HCCL_SOCKET_IFNAME=data0.3001  # Multi-node

export GLOO_SOCKET_IFNAME=$HCCL_SOCKET_IFNAME

# Optimization mode
export HCCL_OP_EXPANSION_MODE="AIV"
```

### 5.4 PD Separation Variables

```bash
# MemFabric store URL (same for all servers)
export ASCEND_MF_STORE_URL="tcp://<prefill_ip>:24667"

# For A2 hardware with RDMA
export ASCEND_MF_TRANSFER_PROTOCOL="device_rdma"
```

### 5.5 Task Queue Configuration

| Mode | TASK_QUEUE_ENABLE | Use Case |
|------|-------------------|----------|
| Prefill | 2 | Optimize for prefill throughput |
| Decode | 1 | Optimize for decode latency |
| Mixed | 1 or 2 | Based on workload |

---

## Step 6: Optimization Options

### 6.1 Quantization

| Quantization | Memory Reduction | Latency | Throughput | Command |
|--------------|-----------------|---------|------------|---------|
| **W8A8 INT8** | 50% weights | Similar | +50% | `--quantization modelslim` |
| W4A8 INT4 | 75% weights | Similar | +30% | `--quantization modelslim` (requires W4A8 model) |
| FP8 | 50% weights | Similar | +40% | `--quantization fp8` |

### 6.2 Speculative Decoding

| Optimization | Memory Impact | Latency | Throughput | Requirements |
|--------------|---------------|---------|------------|--------------|
| **EAGLE3** | +10% (draft model) | **2-3x faster** | **2-3x higher** | Draft model |
| NEXTN (DeepSeek) | +5% | **2x faster** | **2x higher** | Built-in MTP |

### 6.3 Combined Optimizations

| Combination | Memory/NPU | TPOT | Throughput | Best For |
|-------------|------------|------|------------|----------|
| INT8 + EAGLE3 | Lowest | 11-18ms | **2-3x** | Low latency |
| INT8 + DP-Attention | Low | 20-50ms | **1.5-2x** | High throughput (MoE/MLA) |
| INT8 + EAGLE3 + DP | Moderate | 11-18ms | **4-6x** | Maximum throughput |

---

## Step 7: Configuration Recommendation

### 7.1 Determine mem-fraction-static

```python
# From memory-calculation skill output
required_mem = weight_per_device + kv_cache_per_device + buffer

# mem-fraction-static with headroom
mem_fraction = min(0.86, max(0.50, required_mem / hbm_per_device + 0.10))
```

**Model-Specific Guidelines:**
| Model | Cards | Mode | mem-fraction-static |
|-------|-------|------|---------------------|
| DeepSeek-R1 | 32 | Prefill | 0.81 |
| DeepSeek-R1 | 32 | Decode | 0.75 - 0.815 |
| Qwen3-235B-A22B | 8 | Mixed | 0.75 - 0.81 |
| Qwen3-32B | 4 | Mixed | 0.72 |

### 7.2 Determine Max Running Requests

```python
# Available memory for KV cache
available_kv = hbm_per_device * mem_fraction - weight_per_device - runtime_overhead

# Max tokens
tokens_per_request = input_length + output_length
max_tokens = available_kv / kv_bytes_per_token

# Max requests
max_running_requests = int(max_tokens / tokens_per_request)
```

### 7.3 Determine cuda-graph-bs

**Guidelines:**
- Low latency: Small batch sizes (1-6)
- Balanced: Medium batch sizes (8-24)
- High throughput: Large batch sizes (16-64+)

**Model-Specific Examples:**
| Model | Cards | Scenario | cuda-graph-bs |
|-------|-------|----------|---------------|
| DeepSeek-R1 | 32 | Low Latency | 2 4 6 |
| DeepSeek-R1 | 32 | High Throughput | 12 14 16 18 20 22 24 26 |
| Qwen3-235B-A22B | 8 | High Throughput | 6 8 10 12 15 18 28 30 |

### 7.4 MoE Backend Selection

| Backend | Best For | Command |
|---------|----------|---------|
| **deepep** | Low latency, DP > 1 | `--moe-a2a-backend deepep --deepep-mode low_latency` |
| **deepep** | High throughput | `--moe-a2a-backend deepep --deepep-mode auto` |
| **ascend_fuseep** | High throughput, EP > 1 | `--moe-a2a-backend ascend_fuseep` |

---

## Step 8: Launch Command Generation

### 8.1 Standard Model (e.g., Qwen3-32B)

```bash
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=400
export HCCL_OP_EXPANSION_MODE="AIV"
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1

python -m sglang.launch_server \
    --model-path <model_path> \
    --tp-size 8 \
    --attention-backend ascend \
    --device npu \
    --quantization modelslim \
    --mem-fraction-static 0.72 \
    --max-running-requests 128 \
    --cuda-graph-bs 8 16 24 32 \
    --dtype bfloat16 \
    --trust-remote-code
```

### 8.2 MoE Model with DP-Attention (e.g., Qwen3-235B-A22B)

```bash
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=2100
export HCCL_OP_EXPANSION_MODE="AIV"
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1

python -m sglang.launch_server \
    --model-path <model_path> \
    --tp-size 16 \
    --dp-size 16 \
    --enable-dp-attention \
    --enable-dp-lm-head \
    --attention-backend ascend \
    --device npu \
    --quantization modelslim \
    --mem-fraction-static 0.75 \
    --max-running-requests 480 \
    --cuda-graph-bs 6 8 10 12 15 18 28 30 \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path <draft_model_path> \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --speculative-draft-model-quantization unquant \
    --dtype bfloat16 \
    --trust-remote-code
```

### 8.3 MLA + MoE Model PD Separation (e.g., DeepSeek-R1)

**Prefill Server:**
```bash
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1
export HCCL_BUFFSIZE=1536
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export TASK_QUEUE_ENABLE=2
export ASCEND_MF_STORE_URL="tcp://<prefill_ip>:24667"

python -m sglang.launch_server \
    --model-path <model_path> \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend ascend \
    --disaggregation-bootstrap-port 8995 \
    --tp-size 16 \
    --dp-size 2 \
    --enable-dp-attention \
    --attention-backend ascend \
    --device npu \
    --quantization modelslim \
    --mem-fraction-static 0.81 \
    --max-running-requests 8 \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 1 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 2 \
    --dtype bfloat16 \
    --trust-remote-code
```

**Decode Server:**
```bash
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export HCCL_BUFFSIZE=650
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=12
export TASK_QUEUE_ENABLE=1
export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1
export ASCEND_MF_STORE_URL="tcp://<prefill_ip>:24667"

python -m sglang.launch_server \
    --model-path <model_path> \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend ascend \
    --tp-size 32 \
    --dp-size 16 \
    --enable-dp-attention \
    --enable-dp-lm-head \
    --moe-dense-tp-size 1 \
    --attention-backend ascend \
    --device npu \
    --quantization modelslim \
    --mem-fraction-static 0.75 \
    --max-running-requests 32 \
    --cuda-graph-bs 2 4 6 \
    --moe-a2a-backend deepep \
    --deepep-mode low_latency \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --dtype bfloat16 \
    --trust-remote-code
```

**Router:**
```bash
export SGLANG_DP_ROUND_ROBIN=1

python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://<prefill_ip>:8000 8995 \
    --decode http://<decode_ip>:8001 \
    --host 127.0.0.1 \
    --port 6688 \
    --mini-lb
```

---

## Best Practice Reference

### Qwen3-32B on Atlas 800I A2/A3

| Scenario | Input+Output | TPOT | Cards | Config |
|----------|--------------|------|-------|--------|
| Low Latency | 4K+1.5K | 11ms | 4 | BF16 + EAGLE3, TP=8 |
| Balanced | 6K+1.5K | 18ms | 4 | BF16, TP=8 |
| High Throughput | 3.5K+1.5K | 50ms | 2 | W8A8 INT8, TP=4, EAGLE3 |

### Qwen3-235B-A22B on Atlas 800I A3

| Scenario | Input+Output | TPOT | Cards | Config |
|----------|--------------|------|-------|--------|
| Low Latency | 11K+1K | 10ms | 8 | BF16 + EAGLE3, TP=16 |
| High Throughput | 3.5K+1.5K | 50ms | 8 | W8A8 INT8, **TP=16, DP=16**, EAGLE3 |
| High Throughput | 2K+2K | 50ms | 8 | W8A8 INT8, **TP=16, DP=16**, EAGLE3 |

**Note**: For Qwen3-235B-A22B, DP-Attention with DP=TP is recommended for better throughput.

### DeepSeek-R1 on Atlas 800I A3

| Scenario | Input+Output | TPOT | Cards | Deploy Mode | Config |
|----------|--------------|------|-------|-------------|--------|
| Low Latency | 6K+1.6K | 20ms | 32 | PD Separation | W8A8 INT8, TP=32, DP=16 |
| High Throughput | 3.5K+1.5K | 50ms | 32 | PD Separation | W8A8 INT8, TP=32, DP=32 |
| High Throughput | 2K+2K | 50ms | 8 | PD Mixed | W4A8 INT8, TP=16, DP=16 |

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **Out of Memory** | Reduce `max-running-requests` or increase `mem-fraction-static` |
| **Slow Inference** | Add EAGLE3/NEXTN speculative decoding |
| **Low Throughput for MoE/MLA** | Use DP-Attention with `--enable-dp-attention` |
| **HCCL Errors** | Check `HCCL_SOCKET_IFNAME` matches network interface |
| **KV cache OOM** | Reduce `max-running-requests` or use KV cache quantization |
| **Low latency requirement** | Use smaller `cuda-graph-bs`, enable speculative decoding |
| **High throughput requirement** | Use larger `cuda-graph-bs`, increase `max-running-requests` |

### Debug Commands

```bash
# Check NPU status
npu-smi info

# Check memory usage
cat /proc/meminfo

# Check HCCL status
export ASCEND_LAUNCH_BLOCKING=1  # For debugging only
```

---

## Reference Files

- `references/best_practices.md` - Detailed model-specific best practices from SGLang docs
- **`memory-calculation` skill** - Use this skill for detailed memory analysis
