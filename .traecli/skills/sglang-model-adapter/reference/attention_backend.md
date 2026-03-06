# NPU Attention Backend Reference

## 概述

NPU Attention 后端是 SGLang 在华为昇腾 NPU 上运行的核心组件，负责处理所有注意力计算。

## 核心文件

```
python/sglang/srt/hardware_backend/npu/attention/
├── ascend_backend.py           # 主后端实现
├── ascend_torch_native_backend.py  # Torch Native SDPA 后端
└── mla_preprocess.py           # MLA 预处理模块
```

## AscendAttnBackend 类结构

### 初始化参数

| 参数 | 来源 | 说明 |
|------|------|------|
| `device` | `model_runner.device` | NPU 设备 |
| `page_size` | `model_runner.page_size` | KV cache 页大小 |
| `use_mla` | `model_config.attention_arch == AttentionArch.MLA` | 是否使用 MLA 架构 |
| `kv_lora_rank` | `model_config.kv_lora_rank` | MLA 的 KV LoRA 秩 |
| `qk_rope_head_dim` | `model_config.qk_rope_head_dim` | RoPE 维度 |
| `qk_nope_head_dim` | `model_config.qk_nope_head_dim` | 非 RoPE 维度 |
| `use_fia` | `ASCEND_USE_FIA` 环境变量 | 是否使用 FIA 后端 |

### 核心方法

#### 1. forward_extend (扩展阶段)

**用途**: 处理 prefill 和 extend 请求

**关键路径选择**:

```
forward_extend()
├── topk_indices != None? → forward_sparse() (NSA/稀疏注意力)
├── is_target_verify() or is_draft_extend()? → forward_mtp() (推测解码)
├── use_mla == False?
│   ├── use_fia == True? → npu_fused_infer_attention_score (FIA路径)
│   ├── qk_head_dim <= 128 and causal? → _npu_flash_attention_qlens
│   └── else → native_attn.run_sdpa_forward_extend
└── use_mla == True?
    ├── prefix_lens > 0? → npu_ring_mla (两阶段计算)
    └── else → FIA 或 native SDPA
```

**关键代码位置**: `ascend_backend.py:714-1087`

#### 2. forward_decode (解码阶段)

**用途**: 处理 decode 请求

**关键路径选择**:

```
forward_decode()
├── graph_mode and not enable_torch_compile? → forward_decode_graph()
├── use_mla == False?
│   ├── use_fia == True? → npu_fused_infer_attention_score
│   ├── encoder_lens is None and logit_cap == 0? → _npu_paged_attention
│   └── else → native_attn.run_sdpa_forward_decode
└── use_mla == True?
    ├── use_fia and GQA >= 8? → npu_fused_infer_attention_score
    └── else → _npu_paged_attention_mla
```

**关键代码位置**: `ascend_backend.py:1427-1677`

#### 3. forward_mtp (推测解码)

**用途**: 处理 MTP (Multi-Token Prediction) 推测解码

**关键代码位置**: `ascend_backend.py:1088-1267`

## NPU 算子参考

### npu_fused_infer_attention_score (FIA)

**用途**: 华为 CANN 提供的融合注意力算子

**关键参数**:

| 参数 | 说明 | 常用值 |
|------|------|--------|
| `input_layout` | 输入布局 | "TND", "BSND", "BSH" |
| `sparse_mode` | 稀疏模式 | 0 (无mask), 3 (因果mask) |
| `num_heads` | Query 头数 | - |
| `num_key_value_heads` | KV 头数 | - |
| `scale` | 缩放因子 | `1/sqrt(head_dim)` |
| `block_table` | KV cache 页表 | - |
| `block_size` | 页大小 | - |
| `actual_seq_lengths` | Query 实际序列长度 | - |
| `actual_seq_lengths_kv` | KV 实际序列长度 | - |

**布局说明**:
- `TND`: [Token, Num_heads, Head_dim] - 适用于变长序列
- `BSND`: [Batch, Seq, Num_heads, Head_dim] - 适用于固定批次
- `BSH`: [Batch, Seq, Hidden] - 适用于单头或融合布局

### _npu_flash_attention_qlens

**用途**: 支持变长序列的 Flash Attention

**关键参数**:
- `query`: [Total_tokens, Hidden]
- `key_cache`: KV cache 键
- `value_cache`: KV cache 值
- `mask`: 注意力掩码
- `block_table`: 页表
- `seq_len`: 扩展长度
- `context_lens`: 上下文长度

### _npu_paged_attention

**用途**: 解码阶段的 Paged Attention

**关键参数**:
- `query`: [Num_tokens, Num_heads, Head_dim]
- `key_cache`: KV cache
- `value_cache`: KV cache
- `block_table`: 页表
- `context_lens`: 上下文长度

### npu_ring_mla

**用途**: MLA 架构的环形注意力计算

**关键参数**:
- `q_nope`, `q_rope`: Query 的非 RoPE 和 RoPE 部分
- `k_nope`, `k_rope`: Key 的非 RoPE 和 RoPE 部分
- `value`: Value
- `mask`: 注意力掩码
- `seqlen`: 序列长度
- `kernel_type`: "kernel_type_high_precision"
- `mask_type`: "mask_type_triu" 或 "no_mask"
- `calc_type`: "calc_type_first_ring" 或 "calc_type_default"

## MLA vs 非 MLA 差异

### 非 MLA 模型
- K 和 V 分别存储
- 标准 QKV 投影
- 使用 `_npu_paged_attention` 或 FIA

### MLA 模型 (DeepSeek-V2/V3)
- K 压缩为 latent representation
- 分离 `q_nope` 和 `q_pe` (RoPE 部分)
- 使用 `npu_ring_mla` 或 `_npu_paged_attention_mla`
- 需要额外的 `kv_b_proj` 解压缩

## 常见问题排查

### 1. 维度不匹配
- 检查 `qk_head_dim` vs `v_head_dim`
- MLA 模型中 `qk_head_dim != v_head_dim` 是正常的
- 确认 `num_heads` 和 `num_kv_heads` 的 GQA 比例

### 2. 布局错误
- FIA 路径: 检查 `input_layout` 参数
- 非 FIA 路径: 检查 tensor reshape 是否正确
- NZ 格式: 检查 `SGLANG_USE_FIA_NZ` 环境变量

### 3. 掩码问题
- `sparse_mode=3` 用于因果注意力
- `sparse_mode=0` 用于无掩码 (decode 阶段)
- 检查 `atten_mask` 是否正确初始化

### 4. 性能问题
- 检查是否启用了 FIA (`ASCEND_USE_FIA=1`)
- MLA 模型检查 `SGLANG_NPU_USE_MLAPO`
- 检查 graph mode 是否正确启用
