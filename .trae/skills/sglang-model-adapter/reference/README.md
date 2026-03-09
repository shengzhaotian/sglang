# NPU Reference Documentation Index

## Overview

This directory contains detailed reference documentation for SGLang NPU implementation, helping to understand and debug NPU-related code.

**For common reference (environment variables, error codes, troubleshooting)**, see [shared/npu_common_reference.md](../../shared/npu_common_reference.md).

## Document List

### 1. [Attention Backend](./attention_backend.md)

**Applicable Scenarios**:
- Understanding NPU attention computation flow
- Debugging attention-related errors
- Selecting the correct attention path

**Core Content**:
- `AscendAttnBackend` class structure
- `forward_extend`, `forward_decode`, `forward_mtp` methods
- NPU operator reference (`npu_fused_infer_attention_score`, `_npu_paged_attention`, etc.)
- MLA vs non-MLA differences

### 2. [MLA Preprocess](./mla_preprocess.md)

**Applicable Scenarios**:
- Adapting MLA models (DeepSeek-V2/V3)
- Debugging MLA preprocessing errors
- Understanding weight transformation flow

**Core Content**:
- `NPUFusedMLAPreprocess` class
- Three preprocessing modes (MLAPO, MLAProlog, non-quantized)
- Weight transformation (`trans_rope_weight`, `transdata`)
- KV Cache operations

### 3. [RoPE Embedding](./rope_embedding.md)

**Applicable Scenarios**:
- Debugging position encoding errors
- Understanding NPU RoPE operators
- Handling RoPE separation for MLA models

**Core Content**:
- `npu_interleave_rope` operator
- `npu_kv_rmsnorm_rope_cache` operator
- RoPE dimension specifications
- Cos/Sin Cache management

### 4. [KV Cache](./kv_cache.md)

**Applicable Scenarios**:
- Understanding KV Cache structure
- Debugging cache-related errors
- Optimizing memory usage

**Core Content**:
- KV Cache architecture (standard vs MLA)
- Memory pool management
- Page Table management
- Cache formats (PA_BNSD vs PA_NZ)
- Slot Mapping

### 5. [Environment Variables](./environment_variables.md)

**Applicable Scenarios**:
- Configuring NPU runtime environment
- Performance tuning
- Troubleshooting

**Core Content**:
- Core environment variables
- Memory/communication-related variables
- Variable combination recommendations
- Common issues

### 6. [Parallel Strategies](./parallel_strategies.md)

**Applicable Scenarios**:
- Configuring multi-GPU parallelism
- Understanding TP/DP/EP strategies
- Debugging communication issues

**Core Content**:
- Tensor Parallelism (TP)
- Data Parallelism (DP)
- DP Attention
- Expert Parallelism (EP)
- Parallel strategy combinations
- NPU communication optimization (HCCL, DeepEP)

### 7. [Speculative Decoding](./speculative_decoding.md)

**Applicable Scenarios**:
- Configuring speculative decoding acceleration
- Understanding EAGLE/EAGLE3 algorithms
- Debugging draft/verify flow

**Core Content**:
- EAGLE, EAGLE3, STANDALONE, NGRAM algorithms
- Draft and Verify phases
- Tree Attention
- NPU Graph Runner
- Interaction with DP Attention

### 8. [ACLGraph](./aclgraph.md)

**Applicable Scenarios**:
- Configuring graph capture optimization
- Understanding NPU Graph principles
- Debugging Graph capture/replay issues

**Core Content**:
- NPUGraphRunner class structure
- Graph capture and replay flow
- Piecewise CUDA Graph
- Memory management and optimization
- Interaction with Attention Backend

### 9. [Quantization](./quantization.md)

**Applicable Scenarios**:
- Configuring quantized model inference
- Understanding ModelSlim quantization schemes
- Debugging quantization-related issues

**Core Content**:
- W8A8/W4A4 quantization types
- Static vs dynamic quantization
- MoE quantization implementation
- Key NPU quantization operators
- RMSNorm quantization adaptation

### 10. [Resource Assessment](./resource_assessment.md)

**Applicable Scenarios**:
- Pre-flight memory estimation before model loading
- Parallel strategy recommendation
- Hardware resource detection

**Core Content**:
- Memory estimation formulas (weights, activation, KV cache)
- NPU hardware detection
- Parallel strategy decision logic (single card vs TP)

### 11. [Quick Start Commands](./quick_start_commands.md)

**Applicable Scenarios**:
- Quick reference for model startup commands
- Accuracy test commands
- Performance benchmark commands

**Core Content**:
- Standard LLM / MLA / MoE / Speculative startup commands
- Accuracy test suite (math/knowledge/logic)
- Benchmark commands

### 12. [Common Pitfalls](./common_pitfalls.md)

**Applicable Scenarios**:
- Avoiding common mistakes during adaptation
- Quick troubleshooting reference

**Core Content**:
- CUDA vs NPU compatibility issues
- Accuracy validation importance
- Code modification best practices

## Quick Reference Guide

### By Error Type

| Error Type | Reference Document |
|------------|-------------------|
| Dimension mismatch | Attention Backend, MLA Preprocess |
| Attention errors | Attention Backend |
| RoPE-related errors | RoPE Embedding, MLA Preprocess |
| Cache errors | KV Cache |
| OOM errors | Resource Assessment, KV Cache, Environment Variables |
| Performance issues | Environment Variables, Attention Backend, ACLGraph |
| Communication timeout | Parallel Strategies |
| Speculative decoding errors | Speculative Decoding |
| Graph capture errors | ACLGraph |
| Quantization errors | Quantization |

### By Model Type

| Model Type | Recommended Reading Order |
|------------|---------------------------|
| Standard LLM (LLaMA, Qwen) | Attention Backend → KV Cache → Environment Variables |
| MLA Models (DeepSeek-V2/V3) | MLA Preprocess → Attention Backend → RoPE Embedding → KV Cache |
| MoE Models | Attention Backend → Parallel Strategies (EP) → Environment Variables (DeepEP) |
| Large-scale Multi-GPU | Parallel Strategies → Environment Variables |
| Speculative Decoding Models | Speculative Decoding → Attention Backend (forward_mtp) |

### By Debugging Phase

| Phase | Reference Document |
|-------|-------------------|
| Pre-flight assessment | Resource Assessment |
| Model loading failure | MLA Preprocess (weight transformation), Resource Assessment |
| Prefill errors | Attention Backend (forward_extend) |
| Decode errors | Attention Backend (forward_decode), KV Cache |
| Accuracy issues | RoPE Embedding, MLA Preprocess |
| Performance tuning | Environment Variables, Parallel Strategies |
| Multi-GPU communication issues | Parallel Strategies |
| Speculative decoding issues | Speculative Decoding |

## Feature Compatibility Matrix

### Model Architecture vs Feature Compatibility

| Feature | Standard LLM | MLA Models | MoE Models | VLM Models |
|---------|--------------|------------|------------|------------|
| ACLGraph | ✅ | ✅ | ✅ | ✅ (partial) |
| DeepEP | N/A | N/A | ✅ | N/A |
| DP Attention | ✅ | ✅ | ✅ | ⚠️ |
| Speculative Decoding | ✅ | ✅ | ✅ | ⚠️ |
| LoRA | ✅ | ⚠️ | ⚠️ | ⚠️ |

### Feature Combination Compatibility

| Combination | Compatibility | Notes |
|-------------|---------------|-------|
| ACLGraph + TP | ✅ | Recommended |
| ACLGraph + DP | ✅ | Recommended |
| ACLGraph + EP | ❌ | Graph auto-disabled |
| ACLGraph + LoRA | ❌ | Graph auto-disabled |
| ACLGraph + Speculative | ✅ | Requires special configuration |
| DP Attention + Speculative (EAGLE3) | ✅ | Requires special TP context |
| DeepEP + MoE | ✅ | Recommended |
| MLAPO + FIA_NZ | ✅ | Must be enabled together |

### Environment Variable Dependencies

| Variable | Dependency Condition |
|----------|---------------------|
| `SGLANG_USE_FIA_NZ` | Requires `SGLANG_NPU_USE_MLAPO=1` |
| `SGLANG_ENABLE_SPEC_V2` | Requires `speculative_algorithm` to be set |
| `DEEP_NORMAL_MODE_USE_INT8_QUANT` | Requires `--moe-a2a-backend deepep` |

## Common Error Quick Reference

### Startup Phase Errors

| Error Message | Possible Cause | Reference Document |
|---------------|----------------|-------------------|
| `Architecture not found` | Model architecture not registered | SKILL.md (Step 3) |
| `Weight key mismatch` | Weight mapping error | mla_preprocess.md |
| `OOM during loading` | Insufficient memory | kv_cache.md, environment_variables.md |
| `HCCL init failed` | Communication initialization failed | parallel_strategies.md |

### Inference Phase Errors

| Error Message | Possible Cause | Reference Document |
|---------------|----------------|-------------------|
| `Dimension mismatch in attention` | Head count/dimension configuration error | attention_backend.md |
| `Invalid cache slot` | Slot mapping error | kv_cache.md |
| `Graph capture failed` | Dynamic control flow/unsupported operator | aclgraph.md |
| `RoPE position error` | Position encoding configuration error | rope_embedding.md |
| `Speculative accept rate low` | Draft model misconfiguration | speculative_decoding.md |

### Performance Issues

| Symptom | Possible Cause | Reference Document |
|---------|----------------|-------------------|
| High TTFT | Prefill bottleneck | environment_variables.md, attention_backend.md |
| High TPOT | Decode bottleneck | aclgraph.md, environment_variables.md |
| Low memory utilization | Misconfiguration | kv_cache.md, parallel_strategies.md |
| High communication overhead | Suboptimal parallel strategy | parallel_strategies.md |

## Code File Mapping

| Reference Document | Corresponding Code Files |
|-------------------|-------------------------|
| Attention Backend | `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` |
| MLA Preprocess | `python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py` |
| RoPE Embedding | `python/sglang/srt/layers/rotary_embedding.py`, `mla_preprocess.py` |
| KV Cache | `python/sglang/srt/hardware_backend/npu/memory_pool_npu.py` |
| Parallel Strategies | `python/sglang/srt/distributed/parallel_state.py`, `dp_attention.py` |
| Speculative Decoding | `python/sglang/srt/speculative/eagle_worker_v2.py`, `spec_info.py` |
| ACLGraph | `python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py` |
| Quantization | `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `linear_method_npu.py` |

## Usage Recommendations

1. **First Reading**: Read all documents in order to build a comprehensive understanding
2. **Troubleshooting**: Quickly locate relevant documents based on error type
3. **Code Debugging**: Read reference documents together with code files
4. **Performance Optimization**: Focus on Environment Variables and Attention Backend

## Update Notes

This document is updated along with SGLang NPU implementation. If you have questions or find documentation outdated, please refer to the latest code implementation.
