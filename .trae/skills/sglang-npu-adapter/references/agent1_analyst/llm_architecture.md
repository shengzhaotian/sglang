# LLM架构识别知识

## 1. LLM架构概述

大语言模型(LLM)通常采用 **Decoder-only Transformer** 架构，由以下层次结构组成：

```
Embedding层 → [Decoder Layer × N] → LayerNorm → LM Head
                  │
                  ├── Self-Attention
                  ├── MLP (Feed-Forward)
                  └── LayerNorm × 2
```

### 核心设计理念
- **自回归生成**：模型逐token预测下一个token
- **因果注意力**：每个位置只能看到之前位置的信息
- **KV Cache**：缓存历史token的K、V，避免重复计算

---

## 2. Attention机制详解

### 2.1 注意力计算基础

标准注意力计算公式：
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

其中：
- Q (Query): 查询向量，形状 `[batch, seq_len, num_heads, head_dim]`
- K (Key): 键向量，形状 `[batch, seq_len, num_kv_heads, head_dim]`
- V (Value): 值向量，形状 `[batch, seq_len, num_kv_heads, head_dim]`
- d_k: head_dim，注意力缩放因子

### 2.2 Attention变体对比

| 类型 | KV头数 | KV Cache大小 | 特点 | 代表模型 |
|------|--------|-------------|------|----------|
| **MHA** | = num_heads | 100% | 每个Query头有独立K、V | GPT-2, BERT |
| **MQA** | = 1 | ~3% | 所有Query头共享一组K、V | PaLM |
| **GQA** | 介于1和num_heads之间 | 比例可调 | Query头分组共享K、V | Llama2, Qwen2, Mistral |
| **MLA** | 压缩表示 | ~5-10% | KV被压缩到低维空间 | DeepSeek-V2/V3 |

### 2.3 KV Cache形态详解

#### MHA的KV Cache
```
K Cache: [batch, num_heads, seq_len, head_dim]
V Cache: [batch, num_heads, seq_len, head_dim]
总大小: 2 × batch × num_heads × seq_len × head_dim × dtype_size
```

#### GQA的KV Cache
```
K Cache: [batch, num_kv_heads, seq_len, head_dim]
V Cache: [batch, num_kv_heads, seq_len, head_dim]
总大小: 2 × batch × num_kv_heads × seq_len × head_dim × dtype_size
```
- 当 `num_kv_heads = num_heads / 4` 时，KV Cache减少75%

#### MLA的KV Cache（DeepSeek特有）
```
压缩KV Cache: [batch, seq_len, kv_lora_rank]
总大小: batch × seq_len × kv_lora_rank × dtype_size
```
- `kv_lora_rank` 通常远小于 `num_heads × head_dim`
- 例如：DeepSeek-V2的 `kv_lora_rank = 512`，而 `num_heads × head_dim = 128 × 128 = 16384`
- KV Cache减少约97%

### 2.4 MLA架构详解

MLA (Multi-Head Latent Attention) 是DeepSeek提出的创新注意力机制：

**核心思想**：将KV压缩到低维潜在空间，在attention时解压缩

**计算流程**：
```
1. Query压缩: hidden → [q_lora_rank] → [num_heads, head_dim]
2. KV压缩: hidden → [kv_lora_rank] (缓存这个)
3. KV解压缩: [kv_lora_rank] → [num_heads, head_dim] (在线计算)
4. 标准注意力计算
```

**关键配置字段**：
- `kv_lora_rank`: KV压缩后的维度（如512）
- `q_lora_rank`: Query压缩后的维度（如1536）
- `qk_rope_head_dim`: RoPE应用的维度（如64）

**MLA的优势**：
- 大幅减少KV Cache内存占用
- 减少内存带宽压力
- 适合长序列推理

---

## 3. MLP变体详解

### 3.1 MLP的作用
MLP层负责模型非线性变换和特征提取，通常占模型参数量的2/3。

### 3.2 MLP类型对比

| 类型 | 结构 | 激活函数 | 参数量 | 代表模型 |
|------|------|----------|--------|----------|
| **标准MLP** | up_proj → act → down_proj | GELU/ReLU | 2 × H × I | GPT-2, BERT |
| **Gated MLP** | gate_proj × up_proj → act → down_proj | SiLU (Swish) | 3 × H × I | Llama, Qwen, Mistral |

其中 H = hidden_size, I = intermediate_size

### 3.3 Gated MLP原理
```
output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
```
- `gate_proj`: 产生门控信号
- `up_proj`: 产生值信号
- `SiLU`: 平滑的非线性激活
- 门控机制允许模型选择性传递信息

### 3.4 intermediate_size的选择
- 通常为 hidden_size 的 2.7-4 倍
- Llama: `intermediate_size = 2.7 × hidden_size` (近似)
- Qwen2: `intermediate_size = 2.7 × hidden_size`

---

## 4. 归一化层详解

### 4.1 LayerNorm vs RMSNorm

| 特性 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 计算公式 | (x - μ) / σ × γ + β | x / RMS(x) × γ |
| 均值中心化 | ✓ | ✗ |
| 偏置项 | β + γ | 仅γ |
| 计算复杂度 | 较高 | 较低 |
| 代表模型 | GPT-2, BERT | Llama, Qwen, Mistral |

### 4.2 RMSNorm优势
- 计算更简单，无需计算均值
- 在LLM中效果与LayerNorm相当
- 更适合大规模训练

### 4.3 归一化位置
- **Pre-Norm**: 在attention/mlp之前归一化（主流）
- **Post-Norm**: 在attention/mlp之后归一化（原始Transformer）

---

## 5. 位置编码详解

### 5.1 RoPE原理
旋转位置编码(RoPE)通过旋转向量来编码位置信息：
```
f(x, m) = x * e^{imθ}
```
- m: 位置索引
- θ: 频率，θ_i = 10000^{-2i/d}

### 5.2 RoPE Scaling方法

当需要扩展上下文长度时，使用RoPE Scaling：

| 方法 | 原理 | 效果 |
|------|------|------|
| **Linear** | 线性插值位置索引 | 简单有效，可能损失精度 |
| **Dynamic NTK** | 动态调整频率 | 更好的外推能力 |
| **Yarn** | 结合多种策略 | 最佳外推效果 |
| **LongRoPE** | 优化的插值策略 | 支持超长上下文 |

### 5.3 格式转换注意事项

HuggingFace和SGLang的rope_scaling格式可能不同：
- HF: `{"rope_type": "linear", "factor": 2.0}`
- SGLang: `{"type": "linear", "factor": 2.0}`

适配时需要检查并转换格式。

---

## 6. 配置文件关键字段

### 6.1 必要字段
| 字段 | 说明 | 典型值范围 |
|------|------|-----------|
| hidden_size | 隐藏层维度 | 2048-8192 |
| intermediate_size | MLP中间层维度 | 5504-22016 |
| num_hidden_layers | 层数 | 24-80 |
| num_attention_heads | 注意力头数 | 16-64 |
| vocab_size | 词表大小 | 32000-152064 |
| max_position_embeddings | 最大序列长度 | 2048-131072 |

### 6.2 GQA相关字段
| 字段 | 说明 |
|------|------|
| num_key_value_heads | KV头数，小于num_heads表示GQA |

### 6.3 MLA相关字段（DeepSeek）
| 字段 | 说明 |
|------|------|
| kv_lora_rank | KV压缩维度 |
| q_lora_rank | Query压缩维度 |
| qk_rope_head_dim | RoPE维度 |

### 6.4 RoPE相关字段
| 字段 | 说明 |
|------|------|
| rope_theta | RoPE基频，通常为10000或1000000 |
| rope_scaling | 缩放配置（可选） |
| head_dim | 注意力头维度（可选） |

---

## 7. 架构识别决策树

```
开始
  │
  ├─ 检查 kv_lora_rank 存在？
  │     ├─ 是 → MLA架构 (DeepSeek-V2/V3)
  │     └─ 否 → 继续
  │
  ├─ 检查 num_experts 存在？
  │     ├─ 是 → MoE架构
  │     └─ 否 → Dense架构
  │
  ├─ 检查 num_key_value_heads
  │     ├─ = num_heads → MHA
  │     ├─ = 1 → MQA
  │     └─ 其他 → GQA
  │
  ├─ 检查 gate_proj 存在？
  │     ├─ 是 → Gated MLP
  │     └─ 否 → 标准MLP
  │
  └─ 检查 rms_norm_eps 存在？
        ├─ 是 → RMSNorm
        └─ 否 → LayerNorm
```

---

## 8. 参考代码

- `python/sglang/srt/models/qwen2.py` - GQA + Gated MLP + RMSNorm
- `python/sglang/srt/models/llama.py` - Llama架构参考
- `python/sglang/srt/models/deepseek_v2.py` - MLA架构参考
- `python/sglang/srt/models/mistral.py` - GQA架构参考
