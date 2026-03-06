# NPU RoPE (Rotary Position Embedding) Reference

## 概述

旋转位置编码 (RoPE) 是现代 LLM 的核心组件。在 NPU 上，RoPE 的实现有其特殊性，需要特别注意算子选择和数据布局。

## 核心实现位置

```
python/sglang/srt/layers/rotary_embedding.py       # 通用 RoPE 实现
python/sglang/srt/hardware_backend/npu/attention/
├── ascend_backend.py                              # RoPE 在 attention 中的使用
└── mla_preprocess.py                              # MLA 模型的 RoPE 预处理
```

## NPU RoPE 算子

### npu_interleave_rope

**用途**: NPU 专用的交错 RoPE 算子

**函数签名**:
```python
torch.ops.npu.npu_interleave_rope(
    x,           # [Batch, Num_heads, Seq_len, Head_dim] 或 [B*S, N, 1, D]
    cos,         # 余弦值
    sin,         # 正弦值
)
```

**特点**:
- 适用于 MLA 模型的 RoPE 部分
- 输入需要 reshape 为 4D tensor
- `cos` 和 `sin` 需要预先计算并缓存

**使用示例** (来自 `mla_preprocess.py`):
```python
q_pe = q_pe.view(-1, self.num_local_heads, 1, self.qk_rope_head_dim)
cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)
q_pe = torch.ops.npu.npu_interleave_rope(q_pe, cos, sin)
```

### npu_kv_rmsnorm_rope_cache

**用途**: 融合 RMSNorm + RoPE + KV Cache 写入

**函数签名**:
```python
torch.ops.npu.npu_kv_rmsnorm_rope_cache(
    latent_cache,      # [B*S, 1, 1, kv_lora_rank + qk_rope_head_dim]
    rmsnorm_weight,    # LayerNorm 权重
    cos,               # 余弦值
    sin,               # 正弦值
    slot_mapping,      # Cache 写入位置
    k_rope_cache,      # K RoPE cache 输出
    c_kv_cache,        # C KV cache 输出
    epsilon,           # RMSNorm epsilon
    cache_mode,        # "PA_BNSD" 或 "PA_NZ"
    is_output_kv,      # 是否输出 KV
)
```

**返回**: `(k_pe, k_nope, ...)` 或根据参数返回多个值

**使用场景**: MLA 模型的 KV cache 预处理

## RoPE 维度说明

### 标准 RoPE 模型

| 模型 | RoPE 维度 | 总 Head Dim |
|------|-----------|-------------|
| LLaMA | 128 | 128 |
| Qwen2 | 128 | 128 |
| Mistral | 128 | 128 |

### MLA 模型 (DeepSeek-V2/V3)

| 组件 | 维度 | 说明 |
|------|------|------|
| `qk_rope_head_dim` | 64 | RoPE 部分维度 |
| `qk_nope_head_dim` | 128 | 非 RoPE 部分维度 |
| `qk_head_dim` | 192 | 总维度 |

**关键区别**: MLA 模型将 Q 和 K 分为 RoPE 和非 RoPE 两部分，只有 RoPE 部分应用位置编码。

## RoPE 计算流程

### 非 MLA 模型

```
positions → rotary_emb.get_cos_sin_cache() → cos, sin
                                              ↓
q, k → rotary_emb(positions, q, k) → q_rotated, k_rotated
```

### MLA 模型

```
positions → get_sin_cos() → cos, sin
                              ↓
q_pe → npu_interleave_rope(q_pe, cos, sin) → q_pe_rotated
                              ↓
latent_cache → npu_kv_rmsnorm_rope_cache(...) → k_pe, k_nope
```

## DeepSeek Yarn RoPE

DeepSeek 模型使用特殊的 Yarn RoPE 扩展：

```python
# 在 deepseek_v2_attention_mla_npu.py 中
if m.use_deepseek_yarn_rope:
    cos, sin = m.rotary_emb.get_cos_sin_cache(positions, dtype, offsets=None)
    q_pe = torch_npu.npu_interleave_rope(
        q_pe.reshape(B, -1, S, m.qk_rope_head_dim),
        cos, sin,
    )
```

## Cos/Sin Cache 管理

### 标准实现

```python
class RotaryEmbedding:
    def __init__(self, ...):
        self.cos_cached = None
        self.sin_cached = None
    
    def get_cos_sin_cache(self, positions, dtype, offsets=None):
        # 计算或返回缓存的 cos/sin
```

### NPU 优化实现

```python
# 在 mla_preprocess.py 中
def get_sin_cos(self, positions):
    cos_sin = self.rotary_emb.cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.repeat(1, 2)  # 扩展维度
    sin = sin.repeat(1, 2)
    return cos, sin
```

## 常见问题排查

### 1. 维度不匹配

**症状**: `RuntimeError: shape mismatch in npu_interleave_rope`

**检查点**:
- 输入是否为 4D tensor
- `qk_rope_head_dim` 是否正确
- `cos` 和 `sin` 的维度是否与 `q_pe` 匹配

**解决方案**:
```python
# 确保正确的 reshape
q_pe = q_pe.view(-1, num_heads, 1, qk_rope_head_dim)
cos = cos.view(-1, 1, 1, qk_rope_head_dim)
sin = sin.view(-1, 1, 1, qk_rope_head_dim)
```

### 2. 位置编码错误

**症状**: 模型输出乱码或精度下降

**检查点**:
- `positions` tensor 是否正确 (从 0 开始)
- `cos_sin_cache` 是否正确初始化
- 是否使用了正确的 RoPE 类型 (标准 vs Yarn)

### 3. Cache 未正确更新

**症状**: 长序列推理出错

**检查点**:
- `slot_mapping` 是否正确映射到 cache 位置
- `cache_mode` 是否与实际 cache 布局匹配
- 检查 `npu_kv_rmsnorm_rope_cache` 的输出是否正确写入

### 4. MLA 模型 RoPE 分离问题

**症状**: `q_nope` 和 `q_pe` 分离错误

**检查点**:
```python
# 正确的分离方式
q_nope, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)

# 错误方式 (维度顺序错误)
q_pe, q_nope = q.split([qk_rope_head_dim, qk_nope_head_dim], dim=-1)
```

## 调试建议

### 1. 打印中间结果

```python
def forward(self, positions, hidden_states, ...):
    cos, sin = self.get_sin_cos(positions)
    print(f"positions: {positions.shape}, cos: {cos.shape}, sin: {sin.shape}")
    print(f"q_pe before rope: {q_pe.shape}")
    q_pe = torch.ops.npu.npu_interleave_rope(q_pe, cos, sin)
    print(f"q_pe after rope: {q_pe.shape}")
```

### 2. 验证 cos/sin 值

```python
# 检查 cos/sin 是否在合理范围
assert cos.min() >= -1.0 and cos.max() <= 1.0
assert sin.min() >= -1.0 and sin.max() <= 1.0
```

### 3. 对比 CPU 实现

```python
# 使用 PyTorch 标准 RoPE 实现对比
def reference_rope(x, cos, sin):
    # 标准 RoPE 实现
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
```

## 与其他模块的关系

```
RoPE
├── 输入: positions (来自 tokenizer)
├── 依赖: rotary_emb (模型初始化时创建)
├── 输出: q_pe, k_pe (传递给 attention backend)
└── 与 KV Cache 交互: npu_kv_rmsnorm_rope_cache
```
