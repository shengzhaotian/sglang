# NPU Reference Documentation Index

## 概述

本目录包含 SGLang NPU 实现的详细参考文档，帮助理解和调试 NPU 相关的代码。

## 文档列表

### 1. [Attention Backend](./attention_backend.md)

**适用场景**:
- 理解 NPU 注意力计算流程
- 调试 attention 相关错误
- 选择正确的 attention 路径

**核心内容**:
- `AscendAttnBackend` 类结构
- `forward_extend`, `forward_decode`, `forward_mtp` 方法
- NPU 算子参考 (`npu_fused_infer_attention_score`, `_npu_paged_attention` 等)
- MLA vs 非 MLA 差异

### 2. [MLA Preprocess](./mla_preprocess.md)

**适用场景**:
- 适配 DeepSeek-V2/V3 等 MLA 模型
- 调试 MLA 预处理错误
- 理解权重变换流程

**核心内容**:
- `NPUFusedMLAPreprocess` 类
- 三种预处理模式 (MLAPO, MLAProlog, 非量化)
- 权重变换 (`trans_rope_weight`, `transdata`)
- KV Cache 操作

### 3. [RoPE Embedding](./rope_embedding.md)

**适用场景**:
- 调试位置编码错误
- 理解 NPU RoPE 算子
- 处理 MLA 模型的 RoPE 分离

**核心内容**:
- `npu_interleave_rope` 算子
- `npu_kv_rmsnorm_rope_cache` 算子
- RoPE 维度说明
- Cos/Sin Cache 管理

### 4. [KV Cache](./kv_cache.md)

**适用场景**:
- 理解 KV Cache 结构
- 调试 cache 相关错误
- 优化内存使用

**核心内容**:
- KV Cache 架构 (标准 vs MLA)
- 内存池管理
- Page Table 管理
- Cache 格式 (PA_BNSD vs PA_NZ)
- Slot Mapping

### 5. [Environment Variables](./environment_variables.md)

**适用场景**:
- 配置 NPU 运行环境
- 性能调优
- 问题排查

**核心内容**:
- 核心环境变量
- 内存/通信相关变量
- 变量组合推荐
- 常见问题

### 6. [Parallel Strategies](./parallel_strategies.md)

**适用场景**:
- 配置多卡并行
- 理解 TP/DP/EP 策略
- 调试通信问题

**核心内容**:
- Tensor Parallelism (TP)
- Data Parallelism (DP)
- DP Attention
- Expert Parallelism (EP)
- 并行策略组合
- NPU 通信优化 (HCCL, DeepEP)

### 7. [Speculative Decoding](./speculative_decoding.md)

**适用场景**:
- 配置投机解码加速
- 理解 EAGLE/EAGLE3 算法
- 调试 draft/verify 流程

**核心内容**:
- EAGLE, EAGLE3, STANDALONE, NGRAM 算法
- Draft 和 Verify 阶段
- Tree Attention
- NPU Graph Runner
- 与 DP Attention 的交互

### 8. [ACLGraph](./aclgraph.md)

**适用场景**:
- 配置图捕获优化
- 理解 NPU Graph 原理
- 调试 Graph 捕获/重放问题

**核心内容**:
- NPUGraphRunner 类结构
- Graph 捕获和重放流程
- Piecewise CUDA Graph
- 内存管理和优化
- 与 Attention Backend 的交互

### 9. [Quantization](./quantization.md)

**适用场景**:
- 配置量化模型推理
- 理解 ModelSlim 量化方案
- 调试量化相关问题

**核心内容**:
- W8A8/W4A4 量化类型
- 静态量化 vs 动态量化
- MoE 量化实现
- 关键 NPU 量化算子
- RMSNorm 量化适配

## 快速查找指南

### 按错误类型查找

| 错误类型 | 参考文档 |
|----------|----------|
| 维度不匹配 | Attention Backend, MLA Preprocess |
| Attention 报错 | Attention Backend |
| RoPE 相关错误 | RoPE Embedding, MLA Preprocess |
| Cache 错误 | KV Cache |
| OOM 错误 | KV Cache, Environment Variables, ACLGraph |
| 性能问题 | Environment Variables, Attention Backend, ACLGraph |
| 通信超时 | Parallel Strategies |
| 投机解码错误 | Speculative Decoding |
| Graph 捕获失败 | ACLGraph |
| 量化相关错误 | Quantization |

### 按模型类型查找

| 模型类型 | 推荐阅读顺序 |
|----------|--------------|
| 标准 LLM (LLaMA, Qwen) | Attention Backend → KV Cache → Environment Variables |
| MLA 模型 (DeepSeek-V2/V3) | MLA Preprocess → Attention Backend → RoPE Embedding → KV Cache |
| MoE 模型 | Attention Backend → Parallel Strategies (EP) → Environment Variables (DeepEP) |
| 大模型多卡 | Parallel Strategies → Environment Variables |
| 投机解码模型 | Speculative Decoding → Attention Backend (forward_mtp) |

### 按调试阶段查找

| 阶段 | 参考文档 |
|------|----------|
| 模型加载失败 | MLA Preprocess (权重变换) |
| Prefill 错误 | Attention Backend (forward_extend) |
| Decode 错误 | Attention Backend (forward_decode), KV Cache |
| 精度问题 | RoPE Embedding, MLA Preprocess |
| 性能调优 | Environment Variables, Parallel Strategies |
| 多卡通信问题 | Parallel Strategies |
| 投机解码问题 | Speculative Decoding |

## 代码文件映射

| 参考文档 | 对应代码文件 |
|----------|--------------|
| Attention Backend | `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` |
| MLA Preprocess | `python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py` |
| RoPE Embedding | `python/sglang/srt/layers/rotary_embedding.py`, `mla_preprocess.py` |
| KV Cache | `python/sglang/srt/hardware_backend/npu/memory_pool_npu.py` |
| Parallel Strategies | `python/sglang/srt/distributed/parallel_state.py`, `dp_attention.py` |
| Speculative Decoding | `python/sglang/srt/speculative/eagle_worker_v2.py`, `spec_info.py` |
| ACLGraph | `python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py` |
| Quantization | `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `linear_method_npu.py` |

## 使用建议

1. **首次阅读**: 建议按顺序阅读所有文档，建立整体理解
2. **问题排查**: 根据错误类型快速定位相关文档
3. **代码调试**: 结合参考文档和代码文件一起阅读
4. **性能优化**: 重点关注 Environment Variables 和 Attention Backend

## 特性兼容性矩阵

### 模型架构与特性兼容性

| 特性 | 标准 LLM | MLA 模型 | MoE 模型 | VLM 模型 |
|------|----------|----------|----------|----------|
| ACLGraph | ✅ | ✅ | ✅ | ✅ (部分) |
| DeepEP | N/A | N/A | ✅ | N/A |
| DP Attention | ✅ | ✅ | ✅ | ⚠️ |
| Speculative Decoding | ✅ | ✅ | ✅ | ⚠️ |
| LoRA | ✅ | ⚠️ | ⚠️ | ⚠️ |

### 特性组合兼容性

| 组合 | 兼容性 | 说明 |
|------|--------|------|
| ACLGraph + TP | ✅ | 推荐 |
| ACLGraph + DP | ✅ | 推荐 |
| ACLGraph + EP | ❌ | 自动禁用 Graph |
| ACLGraph + LoRA | ❌ | 自动禁用 Graph |
| ACLGraph + Speculative | ✅ | 需要特殊配置 |
| DP Attention + Speculative (EAGLE3) | ✅ | 需要特殊 TP context |
| DeepEP + MoE | ✅ | 推荐 |
| MLAPO + FIA_NZ | ✅ | 需同时启用 |

### 环境变量依赖关系

| 变量 | 依赖条件 |
|------|----------|
| `SGLANG_USE_FIA_NZ` | 需要 `SGLANG_NPU_USE_MLAPO=1` |
| `SGLANG_ENABLE_SPEC_V2` | 需要 `speculative_algorithm` 设置 |
| `DEEP_NORMAL_MODE_USE_INT8_QUANT` | 需要 `--moe-a2a-backend deepep` |

## 常见错误速查表

### 启动阶段错误

| 错误信息 | 可能原因 | 参考文档 |
|----------|----------|----------|
| `Architecture not found` | 模型架构未注册 | SKILL.md (步骤 3) |
| `Weight key mismatch` | 权重映射错误 | mla_preprocess.md |
| `OOM during loading` | 内存不足 | kv_cache.md, environment_variables.md |
| `HCCL init failed` | 通信初始化失败 | parallel_strategies.md |

### 推理阶段错误

| 错误信息 | 可能原因 | 参考文档 |
|----------|----------|----------|
| `Dimension mismatch in attention` | 头数/维度配置错误 | attention_backend.md |
| `Invalid cache slot` | Slot mapping 错误 | kv_cache.md |
| `Graph capture failed` | 动态控制流/不支持的算子 | aclgraph.md |
| `RoPE position error` | 位置编码配置错误 | rope_embedding.md |
| `Speculative accept rate low` | Draft 模型配置不当 | speculative_decoding.md |

### 性能问题

| 症状 | 可能原因 | 参考文档 |
|------|----------|----------|
| TTFT 过高 | Prefill 瓶颈 | environment_variables.md, attention_backend.md |
| TPOT 过高 | Decode 瓶颈 | aclgraph.md, environment_variables.md |
| 内存利用率低 | 配置不当 | kv_cache.md, parallel_strategies.md |
| 通信开销大 | 并行策略不当 | parallel_strategies.md |

## 更新说明

本文档随 SGLang NPU 实现更新。如有疑问或发现文档过时，请参考最新代码实现。
