# NPU Common Reference

Shared reference for NPU-related skills. Both `sglang-model-adapter` and `npu-testing-workflow` reference this file.

## Installation

```bash
# Prerequisites: CANN toolkit and torch_npu
pip install torch-npu

# Install SGLang from source
pip install -e "python"
```

## Pre-flight Checks

Before any NPU operation, verify environment:

```bash
# NPU Hardware
python -c "import torch; print(f'NPU available: {torch.npu.is_available()}')"
python -c "import torch; print(f'Device count: {torch.npu.device_count()}')"

# CANN Version
npu-smi info

# torch_npu Version
python -c "import torch_npu; print(f'torch_npu: {torch_npu.__version__}')"

# SGLang Installation
python -c "import sglang; print(f'sglang: {sglang.__version__}')"
```

## Environment Variables Quick Reference

### Essential Variables

| Variable | Purpose | Typical Value |
|----------|---------|---------------|
| `ASCEND_USE_FIA` | Enable FIA attention backend | `1` |
| `SGLANG_SET_CPU_AFFINITY` | CPU affinity | `1` |
| `PYTORCH_NPU_ALLOC_CONF` | Memory allocation | `expandable_segments:True` |
| `STREAMS_PER_DEVICE` | Stream pool size | `32` |
| `HCCL_BUFFSIZE` | Communication buffer (MB) | `400-1600` |

### MLA Model Variables

| Variable | Purpose | When to Use |
|----------|---------|-------------|
| `SGLANG_NPU_USE_MLAPO` | MLA preprocessing optimization | DeepSeek-V2/V3 |
| `SGLANG_USE_FIA_NZ` | NZ format KV Cache | With MLAPO |

### MoE Model Variables

| Variable | Purpose | Typical Value |
|----------|---------|---------------|
| `DEEP_NORMAL_MODE_USE_INT8_QUANT` | DeepEP INT8 quantization | `1` |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | Max dispatch tokens | `16-96` |

### Speculative Decoding Variables

| Variable | Purpose | When to Use |
|----------|---------|-------------|
| `SGLANG_ENABLE_SPEC_V2` | Speculative decoding v2 | With EAGLE/EAGLE3/NEXTN |
| `SGLANG_ENABLE_OVERLAP_PLAN_STREAM` | Stream overlap optimization | Decode servers |

## Error Code Reference

| Error Pattern | Category | First Action | Reference |
|---------------|----------|--------------|-----------|
| `Architecture not found` | Registration | Check `registry.py` | adapter:Step 3 |
| `Weight key mismatch` | Loading | Check weight mapping | `mla_preprocess.md` |
| `Dimension mismatch` | Attention | Check head dimensions | `attention_backend.md` |
| `HCCL init failed` | Communication | Check network config | `parallel_strategies.md` |
| `OOM` | Memory | Reduce `--mem-fraction-static` | `environment_variables.md` |
| `Graph capture failed` | ACLGraph | Check dynamic ops | `aclgraph.md` |
| `RoPE position error` | Position | Check position encoding | `rope_embedding.md` |
| `Cache slot error` | KV Cache | Check slot mapping | `kv_cache.md` |
| `Speculative accept rate low` | Speculative | Check draft model config | `speculative_decoding.md` |
| `Quantization error` | Quantization | Check quant config | `quantization.md` |

## Common Issues Quick Fix

| Issue | Solution |
|-------|----------|
| OOM Error | Reduce `--mem-fraction-static` by 0.05 |
| Slow TTFT | Increase `--max-prefill-tokens` |
| Low Throughput | Adjust `--cuda-graph-bs` |
| Communication Timeout | Increase `HCCL_BUFFSIZE` |
| High ITL Variance | Set `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` |

## Hardware Reference

| Hardware | Devices | Memory | Best For |
|----------|---------|--------|----------|
| Atlas 800I A2 | 8 NPU | 64GB/device | Small/Medium models |
| Atlas 800I A3 | 16 NPU | 64GB/device | Large models, High throughput |

## Server Arguments Quick Reference

| Argument | Description | Typical Values |
|----------|-------------|----------------|
| `--tp-size` | Tensor parallelism | 1, 2, 4, 8, 16, 32 |
| `--dp-size` | Data parallelism | 1, 4, 8, 16, 32 |
| `--ep-size` | Expert parallelism (MoE) | 8, 16 |
| `--mem-fraction-static` | Memory fraction | 0.6-0.86 |
| `--max-running-requests` | Max concurrent requests | 32-832 |
| `--cuda-graph-bs` | Graph batch sizes | `8 16 24 32` |
| `--attention-backend` | Attention backend | `ascend` |
| `--quantization` | Quantization method | `modelslim` |
| `--moe-a2a-backend` | MoE backend | `deepep` |
| `--deepep-mode` | DeepEP mode | `normal`, `low_latency`, `auto` |

## Feature Compatibility Matrix

| Feature | Standard LLM | MLA Models | MoE Models | VLM Models |
|---------|--------------|------------|------------|------------|
| ACLGraph | ✅ | ✅ | ✅ | ✅ (partial) |
| DeepEP | N/A | N/A | ✅ | N/A |
| DP Attention | ✅ | ✅ | ✅ | ⚠️ |
| Speculative Decoding | ✅ | ✅ | ✅ | ⚠️ |

## Skill Transition Protocol

### adapter → testing-workflow

**Trigger**: Basic functional tests pass

**Handoff Data**:
- `model_path`: Model checkpoint path
- `working_command`: Validated server startup command
- `known_limitations`: Known issues or constraints
- `basic_test_results`: Health check, text inference, VL inference status

### testing-workflow → adapter

**Trigger**: Testing reveals code issues

**Handoff Data**:
- `failed_test`: Test case that failed
- `error_message`: Specific error details
- `suggested_fix`: Proposed solution (if known)

## NPU Test Commands

```bash
# Run NPU-specific tests
python3 test/srt/test_srt_endpoint.py --device npu

# Run with NPU hardware flag
python3 test/run_suite.py --hw npu --suite stage-b-test-small-1-gpu
```

## Code Conventions for NPU

### Branch Isolation

Use `is_npu()` for NPU-specific paths:

```python
from sglang.srt.utils import is_npu
_is_npu = is_npu()

if _is_npu:
    # NPU-specific implementation
    pass
else:
    # CUDA/general implementation
    pass
```

### Key Principles

| Principle | Description |
|-----------|-------------|
| **Code Location** | NPU-specific implementations go in `hardware_backend/npu/` |
| **Reuse First** | Reuse existing model classes when possible, add NPU branches |
| **Testing** | Always verify existing CUDA path still works after NPU changes |
