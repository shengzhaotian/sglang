# NPU Parallel Strategies Reference

## 概述

SGLang 支持多种并行策略来扩展大模型推理。本文档详细介绍 NPU 上的并行策略实现和配置。

## 核心文件

```
python/sglang/srt/distributed/
├── parallel_state.py              # 并行状态管理
├── communication_op.py            # 通信操作
└── device_communicators/
    ├── npu_communicator.py        # NPU 通信器
    └── custom_all_reduce.py       # 自定义 AllReduce

python/sglang/srt/layers/
├── dp_attention.py                # DP Attention 实现
└── moe/                           # MoE 相关层
```

## 并行策略类型

### 1. Tensor Parallelism (TP)

**用途**: 将模型权重分片到多个设备

**参数**: `--tp-size`

**工作原理**:
```
原始权重: [hidden_size, hidden_size]
TP=4 后: 每个设备持有 [hidden_size, hidden_size/4]

计算流程:
Input → QKV_proj (分片) → AllReduce → Output_proj (分片) → AllReduce
```

**NPU 实现**:
```python
# 在 npu_communicator.py 中
class NpuCommunicator:
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(x, group=self.group)
        return x
```

**配置示例**:
```bash
python -m sglang.launch_server \
    --model-path /models/llama-70b \
    --tp-size 8 \
    --device npu
```

### 2. Data Parallelism (DP)

**用途**: 复制模型到多个设备，并行处理不同请求

**参数**: `--dp-size`

**工作原理**:
```
请求批次被分割到多个 DP 副本
┌─────────┐  ┌─────────┐  ┌─────────┐
│  DP 0   │  │  DP 1   │  │  DP 2   │
│ Model   │  │ Model   │  │ Model   │
│ Req 0-2 │  │ Req 3-5 │  │ Req 6-8 │
└─────────┘  └─────────┘  └─────────┘
```

**配置示例**:
```bash
python -m sglang.launch_server \
    --model-path /models/llama-7b \
    --dp-size 4 \
    --device npu
```

### 3. DP Attention

**用途**: 在 Attention 层使用数据并行，减少通信开销

**参数**: `--enable-dp-attention`

**工作原理**:
```
标准 TP Attention:
  Q, K, V 分片 → 计算 → AllReduce

DP Attention:
  每个 DP rank 独立计算完整 Attention
  只在非 Attention 层进行 AllReduce
```

**优势**:
- 减少 Attention 层的通信
- 更好的长序列性能
- 适合 decode 阶段

**NPU 特殊处理**:
```python
# 在 dp_attention.py 中
class DpPaddingMode(IntEnum):
    MAX_LEN = auto()   # Padding 到最大长度
    SUM_LEN = auto()   # Padding 到总长度
```

**配置示例**:
```bash
python -m sglang.launch_server \
    --model-path /models/deepseek-v3 \
    --tp-size 16 \
    --enable-dp-attention \
    --device npu
```

### 4. Expert Parallelism (EP)

**用途**: 将 MoE 专家分片到多个设备

**参数**: `--ep-size`

**工作原理**:
```
MoE 层结构:
┌─────────────────────────────────────┐
│           Router                    │
├─────────┬─────────┬─────────┬───────┤
│ Expert 0│ Expert 1│ Expert 2│ ...   │  ← EP 分片
│ (Dev 0) │ (Dev 1) │ (Dev 2) │       │
└─────────┴─────────┴─────────┴───────┘
         ↓ All-to-All 通信
```

**关键算子**:
- `all-to-all`: 专家分发和收集
- `DeepEP`: 高效 MoE 通信后端

**配置示例**:
```bash
python -m sglang.launch_server \
    --model-path /models/deepseek-v3 \
    --tp-size 8 \
    --ep-size 8 \
    --moe-a2a-backend deepep \
    --device npu
```

## 并行策略组合

### 常见组合模式

| 模式 | 配置 | 适用场景 |
|------|------|----------|
| 纯 TP | `--tp-size 8` | 单节点，模型较大 |
| 纯 DP | `--dp-size 8` | 单节点，模型较小，高吞吐 |
| TP + DP | `--tp-size 4 --dp-size 2` | 单节点，平衡延迟和吞吐 |
| TP + EP | `--tp-size 8 --ep-size 8` | MoE 模型 |
| TP + DP Attention | `--tp-size 16 --enable-dp-attention` | 长序列，低延迟 |

### DeepSeek-V3 推荐配置

```bash
# 高吞吐配置
python -m sglang.launch_server \
    --model-path /models/deepseek-v3 \
    --tp-size 32 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --device npu

# 低延迟配置
python -m sglang.launch_server \
    --model-path /models/deepseek-v3 \
    --tp-size 32 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --deepep-mode low_latency \
    --device npu
```

## 并行组管理

### GroupCoordinator 类

```python
class GroupCoordinator:
    """管理一个并行组的通信"""
    
    def __init__(self, group: ProcessGroup, ...):
        self.group = group
        self.world_size = dist.get_world_size(group)
    
    def all_reduce(self, tensor: torch.Tensor):
        """组内 AllReduce"""
        
    def all_gather(self, tensor: torch.Tensor, dim: int):
        """组内 AllGather"""
```

### 并行组初始化

```python
# 在 parallel_state.py 中
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    data_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    ...
):
    """初始化所有并行组"""
```

### 获取并行信息

```python
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,    # TP rank
    get_tensor_model_parallel_world_size,  # TP size
    get_attention_dp_size,             # Attention DP size
    get_tp_group,                      # TP group
    get_attn_tp_group,                 # Attention TP group
)
```

## NPU 通信优化

### HCCL 配置

```bash
# HCCL 缓冲区大小
export HCCL_BUFFSIZE=1600

# Socket 接口 (单机)
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

# 算子扩展模式
export HCCL_OP_EXPANSION_MODE=AIV
```

### DeepEP 配置 (MoE)

```bash
# DeepEP INT8 量化
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

# 每秩最大分发 token 数
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32

# 长序列配置
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
export DEEPEP_NORMAL_LONG_SEQ_ROUND=16
```

## 常见问题排查

### 1. TP 通信超时

**症状**: `RuntimeError: NCCL/HCCL timeout`

**检查点**:
- 网络配置是否正确
- HCCL_BUFFSIZE 是否足够
- 检查是否有死锁

**解决方案**:
```bash
export HCCL_BUFFSIZE=2000
export NCCL_TIMEOUT=1800  # 30 分钟
```

### 2. DP Attention 维度错误

**症状**: `RuntimeError: shape mismatch in all_gather`

**检查点**:
- `global_num_tokens` 是否正确
- padding mode 是否匹配
- 检查 `DpPaddingMode` 选择

### 3. EP 负载不均衡

**症状**: 部分设备利用率低

**检查点**:
- 专家分配是否均匀
- `--ep-size` 是否与专家数匹配
- 检查 router 输出分布

### 4. 内存不足

**症状**: OOM 错误

**检查点**:
- 并行度是否过高
- `--mem-fraction-static` 设置
- 检查 KV Cache 分配

**计算公式**:
```python
# 每 GPU 内存需求
memory_per_gpu = (
    model_params / tp_size +      # 模型权重
    kv_cache_size / dp_size +     # KV Cache
    activation_memory             # 激活值
)
```

## 性能调优建议

### 1. 选择合适的并行度

| 模型规模 | 推荐 TP | 推荐 DP |
|----------|---------|---------|
| 7B | 1-2 | 4-8 |
| 70B | 4-8 | 1-2 |
| 600B+ | 16-32 | 1 |

### 2. DP Attention 场景

**推荐启用**:
- 长序列 (>4K)
- 低延迟要求
- MoE 模型

**不推荐启用**:
- 短序列
- 高吞吐场景
- 小模型

### 3. EP 配置

```bash
# 专家数 = ep_size * 每设备专家数
# 例如: 256 专家, ep_size=16 → 每设备 16 专家

--ep-size 16  # 适合 256 专家模型
--moe-a2a-backend deepep  # 使用 DeepEP 后端
```

## 与其他模块的关系

```
Parallel Strategies
├── 初始化: parallel_state.py
├── 通信: npu_communicator.py, HCCL
├── Attention: dp_attention.py
├── MoE: ep_fusion, DeepEP
└── 调度: scheduler (考虑 DP 分发)
```
