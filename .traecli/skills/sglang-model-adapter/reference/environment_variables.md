# NPU Environment Variables Reference

## 概述

NPU 推理依赖多个环境变量来控制行为和优化性能。本文档列出所有相关环境变量及其用途。

## 核心环境变量

### SGLang 通用变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `SGLANG_SET_CPU_AFFINITY` | 0 | 设置 CPU 亲和性 |
| `SGLANG_ENABLE_TORCH_COMPILE` | true | 启用 torch.compile |
| `SGLANG_IS_FLASHINFER_AVAILABLE` | true | FlashInfer 可用性检查 |

### NPU 专用变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `ASCEND_USE_FIA` | False | 启用 FIA (Fused Infer Attention) 后端 |
| `SGLANG_NPU_USE_MLAPO` | False | 启用 MLAPO 预处理优化 |
| `SGLANG_USE_FIA_NZ` | False | 启用 NZ 格式 (需配合 MLAPO) |
| `SGLANG_ENABLE_OVERLAP_PLAN_STREAM` | 0 | 启用流重叠优化 |
| `SGLANG_ENABLE_SPEC_V2` | 0 | 启用推测解码 V2 |

## 内存相关变量

| 变量名 | 说明 | 推荐值 |
|--------|------|--------|
| `PYTORCH_NPU_ALLOC_CONF` | NPU 内存分配配置 | `expandable_segments:True` |
| `STREAMS_PER_DEVICE` | 每设备流数量 | 32 |

## 通信相关变量 (HCCL)

| 变量名 | 说明 | 推荐值 |
|--------|------|--------|
| `HCCL_BUFFSIZE` | HCCL 缓冲区大小 | 1600 |
| `HCCL_SOCKET_IFNAME` | HCCL Socket 接口 | `lo` (单机) |
| `GLOO_SOCKET_IFNAME` | Gloo Socket 接口 | `lo` (单机) |
| `HCCL_OP_EXPANSION_MODE` | HCCL 算子扩展模式 | `AIV` |

## DeepEP 相关变量 (MoE 模型)

| 变量名 | 说明 | 推荐值 |
|--------|------|--------|
| `DEEP_NORMAL_MODE_USE_INT8_QUANT` | DeepEP INT8 量化 | 1 |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | 每秩最大分发 token 数 | 32 |
| `DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS` | 长序列每轮 token 数 | 1024 |
| `DEEPEP_NORMAL_LONG_SEQ_ROUND` | 长序列轮数 | 16 |

## 系统性能变量

```bash
# CPU 性能模式
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 内存设置
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
```

## CANN 环境设置

```bash
# CANN 工具链
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 编译器路径
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH
```

## 环境变量组合推荐

### 标准 LLM 模型 (LLaMA, Qwen 等)

```bash
export ASCEND_USE_FIA=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_SET_CPU_AFFINITY=1
```

### MLA 模型 (DeepSeek-V2/V3)

```bash
export ASCEND_USE_FIA=1
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
```

### MoE 模型 (DeepSeek-V3, Qwen3-MoE)

```bash
export ASCEND_USE_FIA=1
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
export DEEPEP_NORMAL_LONG_SEQ_ROUND=16
```

### 推测解码

```bash
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
```

## 变量影响分析

### ASCEND_USE_FIA

**启用时**:
- 使用 `npu_fused_infer_attention_score` 算子
- 更好的性能，特别是长序列
- 需要特定的输入布局

**禁用时**:
- 使用 `_npu_flash_attention_qlens` 或 `_npu_paged_attention`
- 兼容性更好，但性能可能较低

### SGLANG_NPU_USE_MLAPO

**启用时**:
- 使用融合的 MLA 预处理算子
- 减少 kernel launch 开销
- 仅适用于 MLA 模型 (DeepSeek-V2/V3)

**禁用时**:
- 使用分离的预处理步骤
- 更容易调试

### SGLANG_USE_FIA_NZ

**启用时**:
- KV Cache 使用 NZ (Block Major) 格式
- 更好的 NPU 内存访问模式
- 必须配合 `SGLANG_NPU_USE_MLAPO=1`

**禁用时**:
- 使用标准 BNSD 格式
- 更容易理解和调试

## 调试环境变量

| 变量名 | 说明 | 用途 |
|--------|------|------|
| `SGLANG_LOG_LEVEL` | 日志级别 | 调试输出 |
| `ASCEND_GLOBAL_LOG_LEVEL` | CANN 日志级别 | 底层调试 |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | 打印到标准输出 | 日志重定向 |

## 常见问题

### 1. FIA 不可用

**症状**: 报错 "FIA not supported"

**检查**:
- CANN 版本是否支持 FIA
- `ASCEND_USE_FIA` 是否设置

### 2. NZ 格式错误

**症状**: 维度不匹配错误

**检查**:
- `SGLANG_USE_FIA_NZ` 是否与代码逻辑一致
- 是否同时设置了 `SGLANG_NPU_USE_MLAPO`

### 3. 内存不足

**症状**: OOM 错误

**检查**:
- `PYTORCH_NPU_ALLOC_CONF` 设置
- `mem_fraction_static` 参数
- NPU 内存使用情况

## 获取当前环境变量

```python
from sglang.srt.utils import get_bool_env_var

# 检查布尔环境变量
use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
use_mlapo = get_bool_env_var("SGLANG_NPU_USE_MLAPO", "False")
use_fia_nz = get_bool_env_var("SGLANG_USE_FIA_NZ", "False")
```
