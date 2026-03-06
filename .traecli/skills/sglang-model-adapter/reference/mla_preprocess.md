# NPU MLA Preprocess Reference

## 概述

MLA (Multi-head Latent Attention) 预处理模块是 DeepSeek-V2/V3 等 MLA 架构模型在 NPU 上运行的关键组件。它负责将压缩的 latent representation 转换为可计算的形式。

## 核心文件

```
python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py
```

## 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `SGLANG_NPU_USE_MLAPO` | 启用 MLAPO 优化 | False |
| `SGLANG_USE_FIA_NZ` | 启用 NZ 格式 (需配合 MLAPO) | False |

## NPUFusedMLAPreprocess 类

### 初始化参数

```python
NPUFusedMLAPreprocess(
    fused_qkv_a_proj_with_mqa,  # 融合的 QKV 投影层
    q_a_layernorm,              # Q 的 LayerNorm
    kv_a_layernorm,             # KV 的 LayerNorm
    q_b_proj,                   # Q 的第二投影层
    w_kc,                       # Key 压缩权重
    rotary_emb,                 # 旋转位置编码
    layer_id,                   # 层 ID
    num_local_heads,            # 本地头数 (TP 后)
    qk_nope_head_dim,           # 非 RoPE 维度 (如 128)
    qk_rope_head_dim,           # RoPE 维度 (如 64)
    v_head_dim,                 # Value 维度
    quant_config,               # 量化配置
)
```

### 关键维度说明

| 维度 | DeepSeek-V2/V3 典型值 | 说明 |
|------|----------------------|------|
| `q_lora_rank` | 1536 | Q 的 LoRA 压缩秩 |
| `kv_lora_rank` | 512 | KV 的 LoRA 压缩秩 |
| `qk_nope_head_dim` | 128 | Query/Key 非 RoPE 部分 |
| `qk_rope_head_dim` | 64 | Query/Key RoPE 部分 |
| `qk_head_dim` | 192 | qk_nope + qk_rope |
| `v_head_dim` | 128 | Value 头维度 |

### 三种预处理模式

#### 1. forward_mlapo (W8A8 量化模式)

**触发条件**: `qkv_a_proj` 使用 modelslim 量化

**流程**:
```
hidden_states
    ↓
qkv_a_proj (量化矩阵乘)
    ↓
split → q_lowrank, latent_cache
    ↓
q_a_layernorm → q_b_proj → q
    ↓
split → q_nope, q_pe
    ↓
matmul(q_nope, w_kc) → q_nope_out
    ↓
npu_interleave_rope(q_pe) → q_rope_out
    ↓
npu_kv_rmsnorm_rope_cache(latent_cache) → k_rope, k_nope
```

**关键算子**: `torch.ops.npu.mla_preprocess`

**权重预处理** (`preprocess_weights` 方法):
- 将权重转换为 NZ 格式
- 重新排列 RoPE 维度 (`trans_rope_weight`)
- 准备 dequant_scale 和 quant_bias

#### 2. forward_mlaprolog (MLA Prolog 模式)

**触发条件**: `quant_config.ignore` 包含 `kv_b_proj`

**流程**:
```
hidden_states
    ↓
npu_mla_prolog_v3(
    token_x, weight_dq, weight_uq_qr,
    weight_uk, weight_dkv_kr,
    rmsnorm_gamma_cq, rmsnorm_gamma_ckv,
    rope_sin, rope_cos,
    kv_cache, kr_cache, cache_index
)
    ↓
q_nope, q_pe, dequant_scale_q_nope, qr, dequant_q_norm
```

**特点**: 使用 CANN 的 MLA Prolog 算子，更高效

#### 3. forward_absorb_prepare_npu_rms_norm_cache (非量化模式)

**触发条件**: 非量化模型

**流程**:
```
hidden_states
    ↓
qkv_a_proj → q_lowrank, latent_cache
    ↓
q_a_layernorm → q_b_proj → q
    ↓
split → q_nope, q_pe
    ↓
matmul(q_nope, w_kc) → q_nope_out
    ↓
rotary_emb(q_pe, k_pe) → q_pe, k_pe
    ↓
npu_kv_rmsnorm_rope_cache → k_rope, k_nope
```

## 权重变换详解

### trans_rope_weight 函数

**用途**: 重新排列 RoPE 维度以适应 NPU 算子

```python
def trans_rope_weight(weight, rope_dim):
    # 将 RoPE 维度的奇偶位置重新排列
    weight_1 = weight[..., -rope_dim::2, :]  # 偶数位置
    weight_2 = weight[..., -rope_dim + 1 :: 2, :]  # 奇数位置
    weight[..., -rope_dim:, :] = torch.cat([weight_1, weight_2], dim=-2)
    return weight
```

### transdata 函数

**用途**: 将矩阵转换为 NZ (Block Major) 格式

```python
def transdata(nd_mat, block_size=(16, 16)):
    # 将矩阵分块并重排为 NPU 优化的格式
    # 输入: [M, N]
    # 输出: [N//16, M*16, 16] (NZ 格式)
```

## KV Cache 操作

### get_kv_cache_and_cache_idx

```python
def get_kv_cache_and_cache_idx(self, forward_batch):
    k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(self.layer_id)
    slot_mapping = forward_batch.out_cache_loc.to(dtype=torch.int32)
    return k_cache, v_cache, slot_mapping
```

**注意**: MLA 模型中:
- `k_cache` 存储 `kv_c` (压缩的 KV latent)
- `v_cache` 存储 `k_pe` (Key 的 RoPE 部分)

### Cache 模式

| 模式 | 说明 | 触发条件 |
|------|------|----------|
| `PA_BNSD` | 标准 Paged Attention 布局 | 默认 |
| `PA_NZ` | NZ 格式 | `SGLANG_USE_FIA_NZ=1` |
| `krope_ctkv` | K RoPE + CTKV | MLAPO 模式 |
| `nzcache` | NZ Cache | FIA NZ 模式 |

## 与 Attention Backend 的交互

### 数据流

```
NPUFusedMLAPreprocess.forward()
    ↓
(q_rope, k_rope, q_nope, k_nope, ...)
    ↓
AscendAttnBackend.forward_mla() 或 forward_sparse()
    ↓
npu_fused_infer_attention_score 或 npu_ring_mla
```

### 关键接口

```python
# 在 deepseek_v2_attention_mla_npu.py 中
def forward_mla_prepare_npu(m, positions, hidden_states, forward_batch, ...):
    if is_mla_preprocess_enabled():
        if not hasattr(m, "mla_preprocess"):
            m.mla_preprocess = NPUFusedMLAPreprocess(...)
        (q_pe, k_pe, q_nope_out, k_nope, ...) = m.mla_preprocess.forward(...)
```

## 常见问题排查

### 1. 维度错误

**症状**: `RuntimeError: shape mismatch`

**检查点**:
- `kv_lora_rank` 是否与模型配置一致
- `qk_rope_head_dim` 是否正确
- `num_local_heads` 是否考虑了 TP 分片

### 2. 量化相关问题

**症状**: 精度下降或 NaN

**检查点**:
- `deq_scale` 和 `quant_bias` 是否正确加载
- 权重是否正确转换为 NZ 格式
- 检查 `input_scale` 和 `input_offset`

### 3. Cache 格式问题

**症状**: FIA 算子报错

**检查点**:
- `SGLANG_USE_FIA_NZ` 是否与实际 cache 格式匹配
- `cache_mode` 参数是否正确
- `slot_mapping` 类型是否为 `int32` 或 `int64`

### 4. RoPE 相关问题

**症状**: 位置编码错误

**检查点**:
- `cos` 和 `sin` 是否正确计算
- `npu_interleave_rope` vs `rotary_emb` 的选择
- 检查 `positions` tensor 是否正确

## 调试建议

1. **添加日志**: 在 `forward` 方法开始处打印 tensor shapes
2. **检查权重**: 确保 `preprocess_weights` 只执行一次
3. **验证 cache**: 检查 `slot_mapping` 的值是否在有效范围内
4. **对比路径**: 对于同一模型，对比三种模式的输出
