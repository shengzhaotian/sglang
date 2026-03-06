# NPU Parallel Strategies Reference

## Overview

SGLang supports multiple parallel strategies to scale large model inference. This document details parallel strategy implementation and configuration on NPU.

## Core Files

```
python/sglang/srt/distributed/
├── parallel_state.py              # Parallel state management
├── communication_op.py            # Communication operations
└── device_communicators/
    ├── npu_communicator.py        # NPU communicator
    └── custom_all_reduce.py       # Custom AllReduce

python/sglang/srt/layers/
├── dp_attention.py                # DP Attention implementation
└── moe/                           # MoE-related layers
```

## Parallel Strategy Types

### 1. Tensor Parallelism (TP)

**Purpose**: Shard model weights across multiple devices

**Parameter**: `--tp-size`

**How It Works**:
```
Original weights: [hidden_size, hidden_size]
After TP=4: Each device holds [hidden_size, hidden_size/4]

Computation Flow:
Input → QKV_proj (sharded) → AllReduce → Output_proj (sharded) → AllReduce
```

**NPU Implementation**:
```python
# In npu_communicator.py
class NpuCommunicator:
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(x, group=self.group)
        return x
```

**Configuration Example**:
```bash
python -m sglang.launch_server \
    --model-path /models/llama-70b \
    --tp-size 8 \
    --device npu
```

### 2. Data Parallelism (DP)

**Purpose**: Replicate model across multiple devices, process different requests in parallel

**Parameter**: `--dp-size`

**How It Works**:
```
Request batch is split across multiple DP replicas
┌─────────┐  ┌─────────┐  ┌─────────┐
│  DP 0   │  │  DP 1   │  │  DP 2   │
│ Model   │  │ Model   │  │ Model   │
│ Req 0-2 │  │ Req 3-5 │  │ Req 6-8 │
└─────────┘  └─────────┘  └─────────┘
```

**Configuration Example**:
```bash
python -m sglang.launch_server \
    --model-path /models/llama-7b \
    --dp-size 4 \
    --device npu
```

### 3. DP Attention

**Purpose**: Use data parallelism in attention layers to reduce communication overhead

**Parameter**: `--enable-dp-attention`

**How It Works**:
```
Standard TP Attention:
  Q, K, V sharded → Compute → AllReduce

DP Attention:
  Each DP rank independently computes full attention
  AllReduce only in non-attention layers
```

**Advantages**:
- Reduces communication in attention layers
- Better performance for long sequences
- Suitable for decode phase

**NPU Special Handling**:
```python
# In dp_attention.py
class DpPaddingMode(IntEnum):
    MAX_LEN = auto()   # Pad to max length
    SUM_LEN = auto()   # Pad to total length
```

**Configuration Example**:
```bash
python -m sglang.launch_server \
    --model-path /models/deepseek-v3 \
    --tp-size 16 \
    --enable-dp-attention \
    --device npu
```

### 4. Expert Parallelism (EP)

**Purpose**: Shard MoE experts across multiple devices

**Parameter**: `--ep-size`

**How It Works**:
```
MoE Layer Structure:
┌─────────────────────────────────────┐
│           Router                    │
├─────────┬─────────┬─────────┬───────┤
│ Expert 0│ Expert 1│ Expert 2│ ...   │  ← EP sharding
│ (Dev 0) │ (Dev 1) │ (Dev 2) │       │
└─────────┴─────────┴─────────┴───────┘
         ↓ All-to-All communication
```

**Key Operators**:
- `all-to-all`: Expert dispatch and gather
- `DeepEP`: Efficient MoE communication backend

**Configuration Example**:
```bash
python -m sglang.launch_server \
    --model-path /models/deepseek-v3 \
    --tp-size 8 \
    --ep-size 8 \
    --moe-a2a-backend deepep \
    --device npu
```

## Parallel Strategy Combinations

### Common Combination Patterns

| Pattern | Configuration | Use Case |
|---------|---------------|----------|
| Pure TP | `--tp-size 8` | Single node, large model |
| Pure DP | `--dp-size 8` | Single node, small model, high throughput |
| TP + DP | `--tp-size 4 --dp-size 2` | Single node, balance latency and throughput |
| TP + EP | `--tp-size 8 --ep-size 8` | MoE models |
| TP + DP Attention | `--tp-size 16 --enable-dp-attention` | Long sequences, low latency |

### DeepSeek-V3 Recommended Configuration

```bash
# High throughput configuration
python -m sglang.launch_server \
    --model-path /models/deepseek-v3 \
    --tp-size 32 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --device npu

# Low latency configuration
python -m sglang.launch_server \
    --model-path /models/deepseek-v3 \
    --tp-size 32 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --deepep-mode low_latency \
    --device npu
```

## Parallel Group Management

### GroupCoordinator Class

```python
class GroupCoordinator:
    """Manage communication within a parallel group"""
    
    def __init__(self, group: ProcessGroup, ...):
        self.group = group
        self.world_size = dist.get_world_size(group)
    
    def all_reduce(self, tensor: torch.Tensor):
        """In-group AllReduce"""
        
    def all_gather(self, tensor: torch.Tensor, dim: int):
        """In-group AllGather"""
```

### Parallel Group Initialization

```python
# In parallel_state.py
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    data_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    ...
):
    """Initialize all parallel groups"""
```

### Getting Parallel Information

```python
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,    # TP rank
    get_tensor_model_parallel_world_size,  # TP size
    get_attention_dp_size,             # Attention DP size
    get_tp_group,                      # TP group
    get_attn_tp_group,                 # Attention TP group
)
```

## NPU Communication Optimization

### HCCL Configuration

```bash
# HCCL buffer size
export HCCL_BUFFSIZE=1600

# Socket interface (single node)
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

# Operator expansion mode
export HCCL_OP_EXPANSION_MODE=AIV
```

### DeepEP Configuration (MoE)

```bash
# DeepEP INT8 quantization
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

# Max dispatch tokens per rank
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32

# Long sequence configuration
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
export DEEPEP_NORMAL_LONG_SEQ_ROUND=16
```

## Common Issue Troubleshooting

### 1. TP Communication Timeout

**Symptom**: `RuntimeError: NCCL/HCCL timeout`

**Checkpoints**:
- Is network configuration correct
- Is HCCL_BUFFSIZE sufficient
- Check for deadlocks

**Solution**:
```bash
export HCCL_BUFFSIZE=2000
export NCCL_TIMEOUT=1800  # 30 minutes
```

### 2. DP Attention Dimension Error

**Symptom**: `RuntimeError: shape mismatch in all_gather`

**Checkpoints**:
- Is `global_num_tokens` correct
- Does padding mode match
- Check `DpPaddingMode` selection

### 3. EP Load Imbalance

**Symptom**: Low utilization on some devices

**Checkpoints**:
- Is expert distribution uniform
- Does `--ep-size` match expert count
- Check router output distribution

### 4. Out of Memory

**Symptom**: OOM errors

**Checkpoints**:
- Is parallelism too high
- `--mem-fraction-static` setting
- Check KV Cache allocation

**Calculation Formula**:
```python
# Memory required per GPU
memory_per_gpu = (
    model_params / tp_size +      # Model weights
    kv_cache_size / dp_size +     # KV Cache
    activation_memory             # Activations
)
```

## Performance Tuning Suggestions

### 1. Choosing Appropriate Parallelism

| Model Size | Recommended TP | Recommended DP |
|------------|----------------|----------------|
| 7B | 1-2 | 4-8 |
| 70B | 4-8 | 1-2 |
| 600B+ | 16-32 | 1 |

### 2. DP Attention Scenarios

**Recommended to Enable**:
- Long sequences (>4K)
- Low latency requirements
- MoE models

**Not Recommended to Enable**:
- Short sequences
- High throughput scenarios
- Small models

### 3. EP Configuration

```bash
# Number of experts = ep_size * experts_per_device
# Example: 256 experts, ep_size=16 → 16 experts per device

--ep-size 16  # Suitable for 256-expert models
--moe-a2a-backend deepep  # Use DeepEP backend
```

## Relationship with Other Modules

```
Parallel Strategies
├── Initialization: parallel_state.py
├── Communication: npu_communicator.py, HCCL
├── Attention: dp_attention.py
├── MoE: ep_fusion, DeepEP
└── Scheduling: scheduler (considers DP distribution)
```
