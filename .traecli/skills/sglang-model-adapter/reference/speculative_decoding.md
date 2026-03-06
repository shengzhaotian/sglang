# NPU Speculative Decoding Reference

## 概述

投机解码 (Speculative Decoding) 是一种加速 LLM 推理的技术，通过使用小型 draft 模型预测多个 token，然后由 target 模型验证，从而减少解码延迟。本文档介绍 NPU 上的投机解码实现。

## 核心文件

```
python/sglang/srt/speculative/
├── spec_info.py                   # 投机算法定义
├── spec_utils.py                  # 工具函数
├── eagle_worker.py                # EAGLE Worker (非重叠)
├── eagle_worker_v2.py             # EAGLE Worker V2 (重叠)
├── eagle_info.py                  # EAGLE 输入/输出定义
├── eagle_info_v2.py               # EAGLE V2 输入/输出定义
├── eagle_utils.py                 # EAGLE 工具函数
├── standalone_worker.py           # Standalone Worker
├── ngram_worker.py                # NGRAM Worker
├── multi_layer_eagle_worker.py    # 多层 EAGLE Worker
└── draft_utils.py                 # Draft 工具

python/sglang/srt/hardware_backend/npu/graph_runner/
├── eagle_draft_npu_graph_runner.py      # NPU EAGLE Draft Graph
└── eagle_draft_extend_npu_graph_runner.py  # NPU EAGLE Extend Graph
```

## 投机算法类型

### SpeculativeAlgorithm 枚举

```python
class SpeculativeAlgorithm(Enum):
    EAGLE = auto()       # EAGLE 算法
    EAGLE3 = auto()      # EAGLE3 算法 (改进版)
    STANDALONE = auto()  # 独立 Draft 模型
    NGRAM = auto()       # NGRAM 自回归
    NONE = auto()        # 无投机解码
```

### 算法对比

| 算法 | Draft 模型 | 特点 | 适用场景 |
|------|-----------|------|----------|
| EAGLE | 轻量级扩展 | 与 Target 共享权重 | EAGLE 训练过的模型 |
| EAGLE3 | 改进版 EAGLE | 更高接受率 | EAGLE3 训练过的模型 |
| STANDALONE | 独立小模型 | 灵活选择 | 通用场景 |
| NGRAM | 无需模型 | 零额外开销 | 简单场景 |

## EAGLE 投机解码

### 工作原理

```
┌─────────────────────────────────────────────────────────┐
│                    EAGLE 投机解码流程                     │
├─────────────────────────────────────────────────────────┤
│  1. Draft 阶段:                                         │
│     Input → Draft Model → 生成 N 个候选 token          │
│                                                         │
│  2. Verify 阶段:                                        │
│     候选 token → Target Model → 并行验证               │
│                                                         │
│  3. 接受/拒绝:                                          │
│     根据概率决定接受哪些 token                          │
│     接受的 token 加入输出序列                           │
└─────────────────────────────────────────────────────────┘
```

### EAGLE Worker 结构

```python
class EagleDraftWorker(BaseDraftWorker):
    def __init__(self, server_args, ...):
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        
        # Draft worker 使用共享的内存池
        self.req_to_token_pool = target_worker.get_memory_pool()
        
        # 初始化 draft model
        self.draft_worker = TpModelWorker(
            server_args=server_args,
            is_draft_worker=True,
            ...
        )
```

### 关键参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--speculative-algorithm` | 投机算法 | None |
| `--speculative-draft-model-path` | Draft 模型路径 | None |
| `--speculative-num-steps` | Draft 步数 | 5 |
| `--speculative-eagle-topk` | EAGLE top-k | 8 |
| `--speculative-num-draft-tokens` | Draft token 数 | 64 |
| `--speculative-accept-threshold-single` | 单 token 接受阈值 | 1.0 |
| `--speculative-accept-threshold-acc` | 累积接受阈值 | 1.0 |

### NPU 特殊实现

#### Graph Runner

```python
# 在 eagle_draft_npu_graph_runner.py 中
class EAGLEDraftNpuGraphRunner:
    """NPU 上的 EAGLE Draft Graph Runner"""
    
    def __init__(self, ...):
        # 初始化 NPU graph
        self.graph = torch.npu.CUDAGraph()
        
    def capture(self, ...):
        # 捕获 NPU graph
        with torch.npu.graph(self.graph):
            output = self.model(...)
```

#### Attention Backend

```python
# 在 ascend_backend.py 中
def forward_mtp(self, q, k, v, layer, forward_batch, ...):
    """处理 MTP (Multi-Token Prediction) 推测解码"""
    
    if self.use_mla:
        # MLA 模型的 MTP 处理
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        
        # 使用 FIA 进行批量 attention
        attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            query, k_cache, v_cache,
            block_table=self.forward_metadata.block_tables,
            ...
        )
```

## 配置示例

### EAGLE 配置

```bash
python -m sglang.launch_server \
    --model-path /models/llama-70b \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 8 \
    --speculative-num-draft-tokens 64 \
    --device npu
```

### EAGLE3 配置

```bash
python -m sglang.launch_server \
    --model-path /models/llama-70b \
    --speculative-algorithm EAGLE3 \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 4 \
    --device npu
```

### STANDALONE 配置

```bash
python -m sglang.launch_server \
    --model-path /models/llama-70b \
    --speculative-algorithm STANDALONE \
    --speculative-draft-model-path /models/llama-7b \
    --speculative-num-steps 5 \
    --device npu
```

### NGRAM 配置

```bash
python -m sglang.launch_server \
    --model-path /models/llama-70b \
    --speculative-algorithm NGRAM \
    --speculative-ngram-min-match-window-size 1 \
    --speculative-ngram-max-match-window-size 12 \
    --device npu
```

## 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `SGLANG_ENABLE_SPEC_V2` | 启用 Spec V2 | 0 |
| `SGLANG_ENABLE_OVERLAP_PLAN_STREAM` | 启用流重叠 | 0 |

```bash
# 启用投机解码 V2 和流重叠
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
```

## 推理流程详解

### Draft 阶段

```python
def forward_draft(self, forward_batch: ForwardBatch):
    """Draft 模型前向传播"""
    
    # 1. 准备输入
    hidden_states = forward_batch.hidden_states
    
    # 2. Draft 模型推理 (多步)
    for step in range(self.speculative_num_steps):
        # 获取 draft token
        draft_logits = self.draft_worker.forward(hidden_states)
        draft_tokens = sample(draft_logits, topk=self.topk)
        
        # 更新 hidden states
        hidden_states = embed(draft_tokens)
    
    # 3. 返回 draft tokens
    return draft_tokens
```

### Verify 阶段

```python
def forward_verify(self, forward_batch: ForwardBatch):
    """Target 模型验证"""
    
    # 1. 准备 tree attention mask
    tree_mask = build_tree_mask(draft_tokens)
    
    # 2. Target 模型并行验证所有 draft tokens
    target_logits = self.target_worker.forward(
        forward_batch,
        attention_mask=tree_mask
    )
    
    # 3. 接受/拒绝决策
    accepted_tokens = verify_tokens(
        draft_tokens, target_logits,
        threshold=self.accept_threshold
    )
    
    return accepted_tokens
```

### Tree Attention

```python
# 在 eagle_utils.py 中
def build_tree_kernel_efficient(
    draft_tokens: torch.Tensor,
    topk: int,
    ...
) -> Tuple[torch.Tensor, torch.Tensor]:
    """构建 tree attention 的 mask 和位置信息"""
    
    # Tree 结构:
    #       [root]
    #      /  |  \
    #    [t1] [t2] [t3]  ← top-k 候选
    #    /|    |    |\
    #  ...    ...  ... ...
```

## 与 DP Attention 的交互

```python
# 在 eagle_worker_v2.py 中
if server_args.enable_dp_attention and self.speculative_algorithm.is_eagle3():
    # EAGLE3 + DP Attention 需要特殊的 TP context
    ctx = draft_tp_context(get_attention_tp_group())
else:
    ctx = empty_context()

with ctx:
    self.draft_worker = TpModelWorker(...)
```

## 常见问题排查

### 1. Draft 模型加载失败

**症状**: `RuntimeError: Failed to load draft model`

**检查点**:
- `--speculative-draft-model-path` 是否正确
- Draft 模型是否与 Target 模型兼容
- 内存是否足够

### 2. 接受率过低

**症状**: 投机解码加速效果不明显

**检查点**:
- `--speculative-accept-threshold-single` 设置
- Draft 模型质量
- `--speculative-eagle-topk` 是否合适

**调优建议**:
```bash
# 降低接受阈值
--speculative-accept-threshold-single 0.9
--speculative-accept-threshold-acc 0.95

# 调整 top-k
--speculative-eagle-topk 4
```

### 3. Graph 捕获失败

**症状**: `RuntimeError: NPU graph capture failed`

**检查点**:
- 是否有动态控制流
- 检查 tensor 形状是否固定
- 检查 `--cuda-graph-bs` 设置

### 4. 内存不足

**症状**: OOM 错误

**检查点**:
- Draft 模型额外内存
- KV Cache 双倍需求
- `--mem-fraction-static` 设置

**解决方案**:
```bash
# 减少内存占用
--mem-fraction-static 0.75
--speculative-num-draft-tokens 32
```

## 性能调优建议

### 1. 选择合适的算法

| 场景 | 推荐算法 |
|------|----------|
| 有 EAGLE 训练的模型 | EAGLE3 |
| 通用加速 | STANDALONE |
| 无额外模型 | NGRAM |
| 最高加速比 | EAGLE3 + DP Attention |

### 2. 参数调优

```bash
# 高接受率配置
--speculative-num-steps 5
--speculative-eagle-topk 4
--speculative-num-draft-tokens 32

# 高吞吐配置
--speculative-num-steps 8
--speculative-eagle-topk 8
--speculative-num-draft-tokens 64
```

### 3. 与其他特性组合

```bash
# EAGLE3 + DP Attention + DeepEP (推荐)
python -m sglang.launch_server \
    --model-path /models/deepseek-v3 \
    --speculative-algorithm EAGLE3 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --device npu
```

## 与其他模块的关系

```
Speculative Decoding
├── Draft 模型: TpModelWorker (is_draft_worker=True)
├── Target 模型: TpModelWorker
├── Attention: AscendAttnBackend.forward_mtp()
├── Graph: EAGLEDraftNpuGraphRunner
├── 调度: Scheduler (overlap schedule)
└── 内存: 共享 req_to_token_pool
```

## 调试建议

### 1. 打印接受率

```python
# 在 verify 阶段添加日志
accepted_count = sum(accepted_tokens)
total_count = len(draft_tokens)
accept_rate = accepted_count / total_count
logger.info(f"Accept rate: {accept_rate:.2%}")
```

### 2. 检查 Draft 输出

```python
# 在 draft 阶段添加日志
logger.info(f"Draft tokens: {draft_tokens}")
logger.info(f"Draft logits shape: {draft_logits.shape}")
```

### 3. 验证 Tree Mask

```python
# 检查 tree mask 是否正确
assert tree_mask.shape == (num_draft_tokens, num_draft_tokens)
assert tree_mask.dtype == torch.bool
```
