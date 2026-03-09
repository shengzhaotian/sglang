# NPU Environment Variables Reference

## Overview

NPU inference relies on multiple environment variables to control behavior and optimize performance. This document lists all relevant environment variables and their purposes.

## Core Environment Variables

### SGLang General Variables

| Variable Name | Default | Description |
|---------------|---------|-------------|
| `SGLANG_SET_CPU_AFFINITY` | 0 | Set CPU affinity |
| `SGLANG_ENABLE_TORCH_COMPILE` | true | Enable torch.compile |
| `SGLANG_IS_FLASHINFER_AVAILABLE` | true | FlashInfer availability check |

### NPU-Specific Variables

| Variable Name | Default | Description |
|---------------|---------|-------------|
| `ASCEND_USE_FIA` | False | Enable FIA (Fused Infer Attention) backend |
| `SGLANG_NPU_USE_MLAPO` | False | Enable MLAPO preprocessing optimization |
| `SGLANG_USE_FIA_NZ` | False | Enable NZ format (requires MLAPO) |
| `SGLANG_ENABLE_OVERLAP_PLAN_STREAM` | 0 | Enable stream overlap optimization |
| `SGLANG_ENABLE_SPEC_V2` | 0 | Enable speculative decoding V2 |

## Memory-Related Variables

| Variable Name | Description | Recommended Value |
|---------------|-------------|-------------------|
| `PYTORCH_NPU_ALLOC_CONF` | NPU memory allocation config | `expandable_segments:True` |
| `STREAMS_PER_DEVICE` | Streams per device | 32 |

## Communication-Related Variables (HCCL)

| Variable Name | Description | Recommended Value |
|---------------|-------------|-------------------|
| `HCCL_BUFFSIZE` | HCCL buffer size | 1600 |
| `HCCL_SOCKET_IFNAME` | HCCL Socket interface | `lo` (single node) |
| `GLOO_SOCKET_IFNAME` | Gloo Socket interface | `lo` (single node) |
| `HCCL_OP_EXPANSION_MODE` | HCCL operator expansion mode | `AIV` |

## DeepEP-Related Variables (MoE Models)

| Variable Name | Description | Recommended Value |
|---------------|-------------|-------------------|
| `DEEP_NORMAL_MODE_USE_INT8_QUANT` | DeepEP INT8 quantization | 1 |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | Max dispatch tokens per rank | 32 |
| `DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS` | Long sequence tokens per round | 1024 |
| `DEEPEP_NORMAL_LONG_SEQ_ROUND` | Long sequence rounds | 16 |

## System Performance Variables

```bash
# CPU performance mode
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Memory settings
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
```

## CANN Environment Setup

```bash
# CANN toolkit
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# Compiler path
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH
```

## Environment Variable Combination Recommendations

### Standard LLM Models (LLaMA, Qwen, etc.)

```bash
export ASCEND_USE_FIA=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_SET_CPU_AFFINITY=1
```

### MLA Models (DeepSeek-V2/V3)

```bash
export ASCEND_USE_FIA=1
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
```

### MoE Models (DeepSeek-V3, Qwen3-MoE)

```bash
export ASCEND_USE_FIA=1
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
export DEEPEP_NORMAL_LONG_SEQ_ROUND=16
```

### Speculative Decoding

```bash
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
```

## Variable Impact Analysis

### ASCEND_USE_FIA

**When Enabled**:
- Uses `npu_fused_infer_attention_score` operator
- Better performance, especially for long sequences
- Requires specific input layout

**When Disabled**:
- Uses `_npu_flash_attention_qlens` or `_npu_paged_attention`
- Better compatibility, but potentially lower performance

### SGLANG_NPU_USE_MLAPO

**When Enabled**:
- Uses fused MLA preprocessing operator
- Reduces kernel launch overhead
- Only applicable to MLA models (DeepSeek-V2/V3)

**When Disabled**:
- Uses separate preprocessing steps
- Easier to debug

### SGLANG_USE_FIA_NZ

**When Enabled**:
- KV Cache uses NZ (Block Major) format
- Better NPU memory access pattern
- Must be combined with `SGLANG_NPU_USE_MLAPO=1`

**When Disabled**:
- Uses standard BNSD format
- Easier to understand and debug

## Debugging Environment Variables

| Variable Name | Description | Purpose |
|---------------|-------------|---------|
| `SGLANG_LOG_LEVEL` | Log level | Debug output |
| `ASCEND_GLOBAL_LOG_LEVEL` | CANN log level | Low-level debugging |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | Print to stdout | Log redirection |

## Common Issues

### 1. FIA Not Available

**Symptom**: Error "FIA not supported"

**Check**:
- Does CANN version support FIA
- Is `ASCEND_USE_FIA` set

### 2. NZ Format Error

**Symptom**: Dimension mismatch error

**Check**:
- Does `SGLANG_USE_FIA_NZ` match code logic
- Is `SGLANG_NPU_USE_MLAPO` also set

### 3. Out of Memory

**Symptom**: OOM errors

**Check**:
- `PYTORCH_NPU_ALLOC_CONF` setting
- `mem_fraction_static` parameter
- NPU memory usage

## Getting Current Environment Variables

```python
from sglang.srt.utils import get_bool_env_var

# Check boolean environment variables
use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
use_mlapo = get_bool_env_var("SGLANG_NPU_USE_MLAPO", "False")
use_fia_nz = get_bool_env_var("SGLANG_USE_FIA_NZ", "False")
```
