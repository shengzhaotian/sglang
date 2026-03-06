---
name: sglang-model-adapter
description: "Adapt and debug existing or new models for sglang on Ascend NPU. Implement in current project repo (local repo), validate via direct sglang serve from local repo, and deliver one signed commit in the current repo. Invoke when user asks to adapt new models, fix model compatibility, or debug NPU inference."
---

# sglang Ascend Model Adapter

## Overview

Adapt Hugging Face or local models to run on `sglang` with minimal changes, deterministic validation, and single-commit delivery. This skill is for both already-supported models and new architectures not yet registered in sglang.

## NPU Implementation Reference

When debugging NPU-specific issues, **MUST read the reference documentation** in `./reference/` directory:

| Document | When to Read |
|----------|--------------|
| [attention_backend.md](./reference/attention_backend.md) | Attention errors, dimension mismatch, path selection issues |
| [mla_preprocess.md](./reference/mla_preprocess.md) | MLA model adaptation (DeepSeek-V2/V3), weight transformation errors |
| [rope_embedding.md](./reference/rope_embedding.md) | Position encoding errors, RoPE dimension issues |
| [kv_cache.md](./reference/kv_cache.md) | Cache errors, OOM, slot mapping issues |
| [environment_variables.md](./reference/environment_variables.md) | Performance tuning, environment configuration |
| [parallel_strategies.md](./reference/parallel_strategies.md) | Multi-GPU setup, TP/DP/EP configuration, communication errors |
| [speculative_decoding.md](./reference/speculative_decoding.md) | EAGLE/EAGLE3 setup, draft/verify errors, acceleration issues |
| [aclgraph.md](./reference/aclgraph.md) | Graph capture optimization, capture/replay errors, memory issues |
| [quantization.md](./reference/quantization.md) | ModelSlim quantization, W8A8/W4A4 errors, precision issues |

### Quick Debug Guide

| Error Type | First Reference | Key Code Files |
|------------|-----------------|----------------|
| Attention dimension mismatch | [attention_backend.md](./reference/attention_backend.md) | `ascend_backend.py:714-1087` (extend), `ascend_backend.py:1427-1677` (decode) |
| MLA weight errors | [mla_preprocess.md](./reference/mla_preprocess.md) | `mla_preprocess.py:63-200` |
| RoPE position errors | [rope_embedding.md](./reference/rope_embedding.md) | `mla_preprocess.py:248-253` |
| Cache slot errors | [kv_cache.md](./reference/kv_cache.md) | `memory_pool_npu.py` |
| Performance issues | [environment_variables.md](./reference/environment_variables.md) | - |
| TP/DP/EP communication errors | [parallel_strategies.md](./reference/parallel_strategies.md) | `parallel_state.py`, `npu_communicator.py` |
| Speculative decoding errors | [speculative_decoding.md](./reference/speculative_decoding.md) | `eagle_worker_v2.py`, `spec_info.py` |
| Graph capture/replay errors | [aclgraph.md](./reference/aclgraph.md) | `npu_graph_runner.py` |
| Quantization errors | [quantization.md](./reference/quantization.md) | `modelslim.py`, `linear_method_npu.py` |

### Key NPU Operators

| Operator | Usage | Reference |
|----------|-------|-----------|
| `npu_fused_infer_attention_score` | FIA attention (prefill/decode) | attention_backend.md |
| `_npu_paged_attention` | Paged attention (decode) | attention_backend.md |
| `npu_interleave_rope` | RoPE for MLA models | rope_embedding.md |
| `npu_kv_rmsnorm_rope_cache` | Fused RMSNorm + RoPE + Cache | mla_preprocess.md, rope_embedding.md |
| `npu_ring_mla` | MLA ring attention | attention_backend.md |
| `dist.all_reduce` (HCCL) | TP/DP communication | parallel_strategies.md |
| `torch.npu.graph` / `NPUGraph` | Graph capture and replay | aclgraph.md |
| `npu_quantize` / `npu_dynamic_quant` | Quantization | quantization.md |
| `npu_quant_matmul` | Quantized matmul | quantization.md |
| `npu_grouped_matmul` | Grouped matmul (MoE) | quantization.md |

## Hard constraints

- Never upgrade `transformers`.
- Primary implementation roots are current project repo.
- Start `sglang serve` from current project repo with adding environment variable `export PYTHONPATH=${PWD}/python:$PYTHONPATH`.
- Default API port is `8000` unless user explicitly asks otherwise.
- Feature-first default: try best to validate ACLGraph / DeepEP / DP-Attention / MTP / multimodal out-of-box.
- `--ep-size` `--moe-a2a-backend` `--deepep-mode` checks are MoE-only; for non-MoE models mark as not-applicable with evidence.
- If any feature cannot be enabled, keep evidence and explain reason in final report.
- Keep code changes minimal and focused on the target model.
- Final deliverable commit must be one single signed commit in the current working repo (`git commit -sm ...`).
- Keep final docs in Chinese and compact.
- **Dummy-first is encouraged for speed, but dummy is NOT fully equivalent to real weights.**
- **Never sign off adaptation using dummy-only evidence; real-weight gate is mandatory.**
- **When encountering NPU-specific errors, read the corresponding reference documentation first.**

## Non-invasive Development Guidelines

### Core Principle: Reuse First, Isolate Branches

When adapting new models, prioritize **code reuse** and **branch isolation**:

1. **Reuse existing code** when possible - minimizes maintenance burden
2. **Add branches, not replacements** - keep existing paths intact
3. **Isolate NPU-specific logic** - use conditional branching

### 1. Model Implementation Strategy

**Preferred: Reuse existing model class**

```python
# In models/existing_model.py - add NPU support via branching
class ExistingModelForCausalLM(nn.Module):
    def forward(self, ...):
        if is_npu():
            # NPU-specific path (new)
            return self._forward_npu(...)
        else:
            # Existing CUDA path (unchanged)
            return self._forward_cuda(...)
```

**When to create new model file:**
- Architecture is genuinely new (not in registry)
- Weight mapping differs significantly
- NPU implementation requires different module structure

### 2. Code Reuse Patterns

#### Pattern A: Branch in Existing Method

```python
# Good: Minimal change, reuse existing structure
def attention_forward(self, q, k, v):
    if is_npu():
        return self._attention_npu(q, k, v)  # Add NPU method
    return self._attention_cuda(q, k, v)     # Keep existing
```

#### Pattern B: Override Only What's Needed

```python
# Good: Inherit and override only NPU-specific parts
class NPUMyModel(MyModel):
    def forward(self, x):
        if is_npu():
            return self._forward_npu(x)
        return super().forward(x)  # Reuse parent implementation
```

#### Pattern C: Mixin for NPU Extensions

```python
# Good: Add NPU capabilities via mixin
class NPUAttentionMixin:
    def _attention_npu(self, q, k, v):
        # NPU-specific implementation
        pass

class MyModel(NPUAttentionMixin, BaseModel):
    def forward(self, ...):
        if is_npu():
            return self._attention_npu(...)
        return super().forward(...)
```

### 3. Branch Isolation Best Practices

```python
# At module level (cached)
from sglang.srt.utils import is_npu
_is_npu = is_npu()

# Clear separation
if _is_npu:
    # All NPU imports and logic here
    from sglang.srt.hardware_backend.npu.xxx import npu_func
    result = npu_func(...)
else:
    # Existing code path (unchanged)
    result = original_func(...)
```

### 4. When to Add New Files

| Scenario | Action |
|----------|--------|
| New architecture not in registry | Create `models/new_arch.py` |
| NPU-specific module (e.g., MLA attention) | Create `hardware_backend/npu/modules/xxx.py` |
| NPU-specific optimization | Create in `hardware_backend/npu/` |
| Minor NPU adjustment to existing model | Add branch in existing file |

### 5. Modification Guidelines

When modifying shared code:

1. **Add branches, don't replace**:
   ```python
   # Good
   def forward(self, x):
       if is_npu():
           return self._forward_npu(x)  # New branch
       return original_forward(x)        # Unchanged
   ```

2. **Keep existing tests passing**: NPU branch should not affect CUDA path

3. **Document the branch**: Brief comment explaining NPU-specific behavior

### 6. Code Review Checklist

Before committing:

- [ ] Reused existing model class when architecture is similar
- [ ] Added branches instead of replacing existing code
- [ ] NPU-specific code guarded by `is_npu()` check
- [ ] Existing CUDA path remains unchanged
- [ ] Existing tests still pass
- [ ] New NPU code is in appropriate location (`hardware_backend/npu/` if substantial)

## Execution playbook

### 1) Collect context

- Confirm model path (default `/models/<model-name>`; if environment differs, confirm with user explicitly).
- Confirm implementation roots (current project repo).
- Confirm delivery root (the current git repo where the final commit is expected).
- Use default expected feature set: ACLGraph + DeepEP + MTP + multimodal (if model has VL capability).
- User requirements extend this baseline, not replace it.

### 2) Analyze model first

- Inspect `config.json`, processor files, modeling files, tokenizer files.
- Identify architecture class, attention variant, quantization type, and multimodal requirements.
- Check state-dict key prefixes (and safetensors index) to infer mapping needs.
- Decide whether support already exists in `python/sglang/srt/models/registry.py`.

### 3) Choose adaptation strategy (new-model capable)

- Reuse existing sglang architecture if compatible.
- If architecture is missing or incompatible, implement native support:
    - add model adapter under `python/sglang/srt/models/`;
    - add processor under `python/sglang/srt/utils/hf_transformers_utils.py` when needed;
    - register architecture in `python/sglang/srt/models/registry.py`;
    - implement explicit weight loading/remap rules (including KV/QK norm sharding, rope variants).
- If remote code needs newer transformers symbols, do not upgrade dependency.
- If unavoidable, copy required modeling files from sibling transformers source and keep scope explicit.
- If failure is backend-specific (kernel/op/platform), patch minimal required code in current project repo.

### 4) Implement minimal code changes (in implementation roots)

- Touch only files required for this model adaptation.
- Keep weight mapping explicit and auditable.
- Avoid unrelated refactors.

### 5) Two-stage validation on Ascend (direct run)

#### Stage A: dummy fast gate (recommended first)

- Run with `--load-format dummy`.
- Goal: fast validate architecture path / operator path / API path.
- Do not treat `Application startup complete` as pass by itself; request smoke is mandatory.
- Require at least:
    - startup readiness (`/v1/models` 200),
    - one text request 200,
    - if VL model, one text+image request 200,
    - ACLGraph evidence where expected.

#### Stage B: real-weight mandatory gate (must pass before sign-off)

- Remove `--load-format dummy` and validate with real checkpoint.
- Goal: validate real-only risks:
    - weight key mapping,
    - KV/QK norm sharding with real tensor shapes,
    - load-time/runtime stability.
- Require HTTP 200 and non-empty output before declaring success.
- Do not pass Stage B on startup-only evidence.

### 6) Validate inference and features

- Send `GET /v1/models` first.
- Send at least one OpenAI-compatible text request.
- For multimodal models, require at least one text+image request.
- Validate architecture registration and loader path with logs (no unresolved architecture, no fatal missing-key errors).
- Try feature-first validation: EP + ACLGraph path first; eager path as fallback/isolation.
- If startup succeeds but first request crashes (false-ready), treat as runtime failure and continue root-cause isolation.
- For multimodal processor API mismatch (for example `skip_tensor_conversion` signature mismatch), use text-only isolation (`--limit-mm-per-prompt` set image/video/audio to 0) to separate processor issues from core weight loading issues.
- Capacity baseline by default (single machine): `context-length=128k` + `max-running-request=16`.
- Then expand concurrency (e.g., 32/64) if requested or feasible.

### 7) Backport, generate artifacts, and commit in delivery repo

- Generate tutorial markdown at implementation roots `./<ModelName>.md` following the standard template (Introduction, Supported Features, Environment Preparation with docker tabs, Deployment with serve script, Functional Verification with curl example, Accuracy Evaluation, Performance). Fill in model-specific details: HF path, hardware requirements, TP size, max-model-len, served-model-name, sample curl, and accuracy table.
- Confirm test script and tutorial doc are included in the staged files.
- Commit code changes once (single signed commit).

### 8) Prepare handoff artifacts

- Write comprehensive Chinese analysis report.
- Write compact Chinese runbook for server startup and validation commands.
- Include feature status matrix (supported / unsupported / checkpoint-missing / not-applicable).
- Include dummy-vs-real validation matrix and explicit non-equivalence notes.
- Include changed-file list, key logs, and final commit hash.

## Quality gate before final answer

- Service starts successfully from implementation roots with direct command.
- OpenAI-compatible inference request succeeds (not startup-only).
- Key feature set is attempted and reported: ACLGraph / DeepEP / MTP / multimodal.
- Capacity baseline (`128k + bs16`) result is reported, or explicit reason why not feasible.
- **Dummy stage evidence is present (if used), and real-weight stage evidence is present (mandatory).**
- Tutorial doc exists at `./<ModelName>.md` and follows the standard template (Introduction, Supported Features, Environment Preparation, Deployment, Functional Verification, Accuracy Evaluation, Performance).
- Exactly one signed commit contains all code changes in current working repo.
- Final response includes commit hash, file paths, key commands, known limits, and failure reasons where applicable.

## Completion Criteria (Mandatory Testing)

Model adaptation is **NOT complete** until the following tests pass:

### Basic Functional Tests (Mandatory)

```bash
# 1. Service health check
curl http://localhost:8000/v1/models

# 2. Simple inference test
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<model-name>", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'
```

### Performance Benchmark (Recommended)

```bash
# Run bench_serving for performance metrics
python -m sglang.bench_serving \
    --dataset-name random \
    --backend sglang \
    --host 127.0.0.1 \
    --port 8000 \
    --num-prompts 100 \
    --random-input-len 128 \
    --random-output-len 128
```

### Accuracy Test (Optional but Recommended)

```bash
# GSM8K accuracy test
python -m sglang.test.few_shot_gsm8k \
    --num-questions 100 \
    --host http://127.0.0.1 \
    --port 8000
```

### Completion Checklist

- [ ] Service starts without errors
- [ ] `/v1/models` returns 200
- [ ] At least one text inference succeeds
- [ ] (For VL models) At least one image+text inference succeeds
- [ ] Performance benchmark completed (TTFT/TPOT reported)
- [ ] (Optional) Accuracy benchmark completed
- [ ] All results documented in final report

### Transition to Comprehensive Testing

Once basic tests pass, the model adaptation is complete. For comprehensive testing:

1. **Invoke `npu-testing-workflow` skill** for:
   - Full performance benchmarking
   - Accuracy evaluation
   - Structured test report generation

2. **Provide to testing workflow**:
   - Model path and configuration
   - Working server startup command
   - Known limitations or issues

## Quick Start Command Templates

### Standard LLM Model

```bash
export PYTHONPATH=${PWD}/python:$PYTHONPATH
python -m sglang.launch_server \
    --model-path /models/<model-name> \
    --tp-size <tp> \
    --attention-backend ascend \
    --device npu \
    --host 0.0.0.0 --port 8000 \
    --cuda-graph-bs 1 2 4 8 16 32
```

### MLA Model (DeepSeek-V2/V3)

```bash
export PYTHONPATH=${PWD}/python:$PYTHONPATH
export ASCEND_USE_FIA=1
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1

python -m sglang.launch_server \
    --model-path /models/deepseek-v3 \
    --tp-size 16 \
    --enable-dp-attention \
    --attention-backend ascend \
    --device npu \
    --quantization modelslim \
    --cuda-graph-bs 8 16 24 32
```

### MoE Model with DeepEP

```bash
export PYTHONPATH=${PWD}/python:$PYTHONPATH
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

python -m sglang.launch_server \
    --model-path /models/<moe-model> \
    --tp-size 8 \
    --ep-size 8 \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --attention-backend ascend \
    --device npu
```

### Speculative Decoding (EAGLE3)

```bash
export PYTHONPATH=${PWD}/python:$PYTHONPATH
export SGLANG_ENABLE_SPEC_V2=1

python -m sglang.launch_server \
    --model-path /models/<model> \
    --speculative-algorithm EAGLE3 \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 4 \
    --attention-backend ascend \
    --device npu
```

## Debug Checklist

### Before Starting

- [ ] Confirm model path exists and contains `config.json`, tokenizer files
- [ ] Check model architecture in `python/sglang/srt/models/registry.py`
- [ ] Verify NPU environment: `torch.npu.is_available()`
- [ ] Check CANN version compatibility

### Startup Issues

- [ ] Check `PYTHONPATH` includes `${PWD}/python`
- [ ] Verify `--attention-backend ascend` is set
- [ ] Check `--device npu` is set
- [ ] Review logs for architecture registration
- [ ] Check for missing weight keys

### Inference Issues

- [ ] Test `/v1/models` endpoint first
- [ ] Send simple text request to verify inference
- [ ] Check attention backend logs for path selection
- [ ] Verify KV Cache allocation in logs
- [ ] Check for dimension mismatch errors

### Performance Issues

- [ ] Verify ACLGraph is enabled (check logs for graph capture)
- [ ] Check `--cuda-graph-bs` matches actual batch sizes
- [ ] Review environment variables for optimization
- [ ] Check HCCL communication logs
- [ ] Monitor NPU memory utilization

### Feature-Specific Checks

**For MoE Models:**
- [ ] `--ep-size` matches expert parallelism needs
- [ ] `--moe-a2a-backend deepep` is set
- [ ] DeepEP environment variables configured

**For MLA Models:**
- [ ] `ASCEND_USE_FIA=1` is set
- [ ] `SGLANG_NPU_USE_MLAPO=1` for optimization
- [ ] Check `kv_lora_rank` dimension in config

**For Speculative Decoding:**
- [ ] Draft model path is correct (if standalone)
- [ ] `--speculative-num-steps` is reasonable (3-8)
- [ ] Check accept rate in logs
