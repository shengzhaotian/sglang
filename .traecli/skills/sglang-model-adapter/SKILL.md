---
name: sglang-model-adapter
description: "Use when adapting new models for sglang on Ascend NPU, fixing model compatibility issues, or debugging NPU inference errors. Invoke when user asks to adapt models, fix model compatibility, or debug NPU inference."
---

# sglang Ascend Model Adapter

## Overview

Adapt Hugging Face or local models to run on `sglang` with minimal changes, deterministic validation, and single-commit delivery. Works for both already-supported models and new architectures.

## Input/Output Specification

### Required Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | string | `/models/<model-name>` | Model checkpoint path |
| `implementation_root` | path | `${PWD}` | Current project repo |
| `delivery_root` | path | `${PWD}` | Git repo for final commit |

### Optional Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tp_size` | int | auto | Tensor parallelism size |
| `quantization` | string | none | Quantization method (modelslim) |
| `features` | list | [ACLGraph, DeepEP, MTP, multimodal] | Features to validate |

### Outputs

| Artifact | Location | Description |
|----------|----------|-------------|
| Code changes | `implementation_root` | Minimal code modifications |
| Tutorial doc | `./<ModelName>.md` | Deployment guide |
| Analysis report | Console | Chinese analysis report |
| Commit | `delivery_root` | Single signed commit |

## Prerequisites

**MUST verify before starting** (see [shared/npu_common_reference.md](../shared/npu_common_reference.md)):

```bash
# NPU available
python -c "import torch; assert torch.npu.is_available()"

# CANN version
npu-smi info

# SGLang installed
python -c "import sglang"
```

## Hard Constraints

- Never upgrade `transformers`
- Primary implementation in current project repo
- Start `sglang serve` with `export PYTHONPATH=${PWD}/python:$PYTHONPATH`
- Default API port: `8000`
- Feature-first: validate ACLGraph/DeepEP/DP-Attention/MTP/multimodal out-of-box
- `--ep-size`/`--moe-a2a-backend`/`--deepep-mode` are MoE-only; mark N/A for non-MoE
- One single signed commit (`git commit -sm ...`)
- Final docs in Chinese, compact
- **Dummy-first encouraged, but real-weight validation is MANDATORY**
- **Accuracy validation is MANDATORY** - model loading ≠ correct outputs

## NPU Hardware Limitations

### Critical: Attention Head Count Requirement

**NPU fused attention operator requires attention head count to be a power of 2** (1, 2, 4, 8, 16, 32, 64, 128).

**Pre-flight Check**:
```python
def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0

# Calculate heads per device
heads_per_device = total_heads // tp_size
if not _is_power_of_two(heads_per_device):
    # Must use --disable-cuda-graph
    print(f"WARNING: {heads_per_device} heads/device is not power of 2")
    print(f"         CUDA Graph will be disabled")
```

**Workaround**: Use `--disable-cuda-graph` to fall back to native SDPA implementation.

### Common Head Count Scenarios

| Total Heads | TP Size | Heads/Device | Power of 2? | Action |
|-------------|---------|--------------|-------------|--------|
| 32 | 1 | 32 | ✅ Yes | CUDA Graph OK |
| 32 | 2 | 16 | ✅ Yes | CUDA Graph OK |
| 32 | 4 | 8 | ✅ Yes | CUDA Graph OK |
| 20 | 1 | 20 | ❌ No | Disable CUDA Graph |
| 20 | 2 | 10 | ❌ No | Disable CUDA Graph |
| 64 | 8 | 8 | ✅ Yes | CUDA Graph OK |

### RoPE Implementation Differences

**Problem**: `npu_mrope` is designed for multi-modal RoPE and expects different input format than standard RoPE.

**Solution**: Use `npu_interleave_rope` for standard RoPE:
```python
# WRONG: npu_mrope expects 4D tensors for multimodal
query_out, key_out = torch_npu.npu_mrope(positions, query, key, ...)

# CORRECT: npu_interleave_rope for standard RoPE
# Reshape to [batch, heads, 1, dim]
q_pe = q_pe.reshape(num_tokens, num_heads, 1, rotary_dim)
cos = cos.reshape(num_tokens, 1, 1, rotary_dim).contiguous()
sin = sin.reshape(num_tokens, 1, 1, rotary_dim).contiguous()
q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin)
```

### MLA Attention Format Handling

**Problem**: When backend is in `FORWARD_ABSORB_CORE_ATTENTION_BACKENDS`, attention receives `q_nope_out` (latent format) and `q_rope` separately. Native SDPA doesn't support this format directly.

**Solution**: Concatenate query/key parts before passing to native SDPA:
```python
if q_rope is not None:
    # q is in latent format (kv_lora_rank dimension)
    # q_rope is the rope part separately
    # Concatenate to form full query for standard attention
    q_full = torch.cat([q, q_rope], dim=-1)
    k_full = torch.cat([k, k_rope], dim=-1) if k_rope is not None else k
    # Use q_full and k_full for attention computation
```

## NPU Reference Documentation

**MUST read corresponding reference when encountering NPU-specific errors:**

| Error Type | Reference | Key Code Location |
|------------|-----------|-------------------|
| Attention dimension mismatch | [attention_backend.md](./reference/attention_backend.md) | `ascend_backend.py:714-1087` |
| MLA weight errors | [mla_preprocess.md](./reference/mla_preprocess.md) | `mla_preprocess.py:63-200` |
| RoPE position errors | [rope_embedding.md](./reference/rope_embedding.md) | `mla_preprocess.py:248-253` |
| Cache slot errors | [kv_cache.md](./reference/kv_cache.md) | `memory_pool_npu.py` |
| Performance issues | [environment_variables.md](./reference/environment_variables.md) | - |
| TP/DP/EP communication errors | [parallel_strategies.md](./reference/parallel_strategies.md) | `parallel_state.py` |
| Speculative decoding errors | [speculative_decoding.md](./reference/speculative_decoding.md) | `eagle_worker_v2.py` |
| Graph capture errors | [aclgraph.md](./reference/aclgraph.md) | `npu_graph_runner.py` |
| Quantization errors | [quantization.md](./reference/quantization.md) | `modelslim.py` |

**Full index**: [reference/README.md](./reference/README.md)

## Non-invasive Development Guidelines

### Core Principle: Reuse First, Isolate Branches

```python
# Preferred: Branch in existing method
from sglang.srt.utils import is_npu
_is_npu = is_npu()

def forward(self, x):
    if _is_npu:
        return self._forward_npu(x)  # New branch
    return original_forward(x)        # Unchanged
```

### When to Create New Files

| Scenario | Action |
|----------|--------|
| New architecture not in registry | Create `models/new_arch.py` |
| NPU-specific module | Create in `hardware_backend/npu/` |
| Minor NPU adjustment | Add branch in existing file |

### Code Review Checklist

- [ ] Reused existing model class when architecture is similar
- [ ] Added branches instead of replacing existing code
- [ ] NPU-specific code guarded by `is_npu()` check
- [ ] Existing CUDA path remains unchanged
- [ ] Existing tests still pass

## Execution Playbook

### 1) Collect Context

- Confirm model path (default `/models/<model-name>`)
- Confirm implementation roots (current project repo)
- Confirm delivery root (current git repo)
- Use default feature set: ACLGraph + DeepEP + MTP + multimodal (if VL)

### 2) Analyze Model

- Inspect `config.json`, processor files, modeling files, tokenizer files
- Identify architecture class, attention variant, quantization type
- Check state-dict key prefixes for mapping needs
- Check `python/sglang/srt/models/registry.py` for existing support

### 3) Choose Adaptation Strategy

- **Reuse**: If architecture exists in registry
- **Implement**: If missing or incompatible:
  - Add model adapter under `python/sglang/srt/models/`
  - Add processor in `hf_transformers_utils.py` if needed
  - Register architecture in `registry.py`
  - Implement weight loading/remap rules

### 4) Implement Minimal Changes

- Touch only required files
- Keep weight mapping explicit and auditable
- Avoid unrelated refactors

### 5) Two-Stage Validation

#### Stage A: Dummy Fast Gate (Recommended First)

```bash
export PYTHONPATH=${PWD}/python:$PYTHONPATH
python -m sglang.launch_server \
    --model-path /models/<model> \
    --load-format dummy \
    --attention-backend ascend \
    --device npu --port 8000
```

**Required checks**:
- `/v1/models` returns 200
- One text request returns 200
- (VL models) One text+image request returns 200
- ACLGraph evidence where expected

#### Stage B: Real-Weight Mandatory Gate

```bash
# Remove --load-format dummy
python -m sglang.launch_server \
    --model-path /models/<model> \
    --attention-backend ascend \
    --device npu --port 8000
```

**Required checks**:
- HTTP 200 and non-empty output
- Weight key mapping correct
- KV/QK norm sharding validated

### 5.5) Accuracy Validation (MANDATORY)

**Model loading successfully does NOT guarantee correct outputs.** Always run accuracy tests.

#### Quick Accuracy Test Suite

```bash
# Test 1: Simple math
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<model>", "messages": [{"role": "user", "content": "What is 15 times 7?"}], "max_tokens": 100, "temperature": 0}'

# Test 2: Knowledge question
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<model>", "messages": [{"role": "user", "content": "What is the largest planet in our solar system?"}], "max_tokens": 100, "temperature": 0}'

# Test 3: Logic reasoning
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<model>", "messages": [{"role": "user", "content": "If A > B and B > C, is A > C?"}], "max_tokens": 100, "temperature": 0}'
```

#### Expected Results Template

| Test Type | Question | Expected Answer | Model Output | Status |
|-----------|----------|-----------------|--------------|--------|
| Math | 15 × 7 = ? | 105 | ___ | ✅/❌ |
| Knowledge | Largest planet? | Jupiter | ___ | ✅/❌ |
| Logic | A>B, B>C → A>C? | Yes | ___ | ✅/❌ |

#### Special Considerations

- **Reasoning models** (e.g., GLM-4.7-flash): Use `max_tokens >= 500` to get complete responses with reasoning
- **Check `finish_reason`**: Should be `stop`, not `length`
- **Verify response completeness**: Ensure the model finishes its response

### 6) Validate Features

- Feature-first: EP + ACLGraph first; eager as fallback
- For multimodal processor issues: use `--limit-mm-per-prompt` to isolate
- Capacity baseline: `context-length=128k` + `max-running-requests=16`

### 7) Generate Artifacts and Commit

- Create `./<ModelName>.md` tutorial (Introduction, Features, Environment, Deployment, Verification, Accuracy, Performance)
- Single signed commit with all changes

## Quick Start Commands

See [shared/npu_common_reference.md](../shared/npu_common_reference.md) for environment variables.

### Standard LLM

```bash
export PYTHONPATH=${PWD}/python:$PYTHONPATH
python -m sglang.launch_server \
    --model-path /models/<model> --tp-size <tp> \
    --attention-backend ascend --device npu \
    --port 8000 --cuda-graph-bs 8 16 24 32
```

### MLA Model (DeepSeek-V2/V3)

```bash
export PYTHONPATH=${PWD}/python:$PYTHONPATH
export ASCEND_USE_FIA=1 SGLANG_NPU_USE_MLAPO=1 SGLANG_USE_FIA_NZ=1

python -m sglang.launch_server \
    --model-path /models/deepseek-v3 --tp-size 16 \
    --enable-dp-attention --attention-backend ascend --device npu \
    --quantization modelslim --cuda-graph-bs 8 16 24 32
```

### MoE with DeepEP

```bash
export PYTHONPATH=${PWD}/python:$PYTHONPATH
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

python -m sglang.launch_server \
    --model-path /models/<moe-model> --tp-size 8 --ep-size 8 \
    --moe-a2a-backend deepep --deepep-mode normal \
    --attention-backend ascend --device npu
```

### Speculative Decoding (EAGLE3)

```bash
export PYTHONPATH=${PWD}/python:$PYTHONPATH
export SGLANG_ENABLE_SPEC_V2=1

python -m sglang.launch_server \
    --model-path /models/<model> \
    --speculative-algorithm EAGLE3 \
    --speculative-num-steps 5 --speculative-eagle-topk 4 \
    --attention-backend ascend --device npu
```

## Error Handling

See [shared/npu_common_reference.md](../shared/npu_common_reference.md) for error code reference.

### Debug Checklist

| Phase | Check |
|-------|-------|
| Before Start | Model path exists, NPU available, CANN version |
| Startup | PYTHONPATH set, `--attention-backend ascend`, `--device npu` |
| Inference | `/v1/models` first, check attention path logs, verify KV Cache |
| Performance | ACLGraph enabled, `--cuda-graph-bs` matches, HCCL logs |

### Feature-Specific Checks

**MoE Models**: `--ep-size`, `--moe-a2a-backend deepep`, DeepEP env vars
**MLA Models**: `ASCEND_USE_FIA=1`, `SGLANG_NPU_USE_MLAPO=1`, check `kv_lora_rank`
**Speculative**: Draft model path correct, `--speculative-num-steps` reasonable (3-8)

## Quality Gate

- [ ] Service starts from implementation roots
- [ ] OpenAI-compatible inference succeeds (not startup-only)
- [ ] **Accuracy validation passed** (minimum 3 tests: math, knowledge, logic)
- [ ] Feature set attempted: ACLGraph/DeepEP/MTP/multimodal
- [ ] Capacity baseline (`128k + bs16`) reported
- [ ] Dummy + real-weight evidence present
- [ ] Tutorial doc exists at `./<ModelName>.md`
- [ ] One signed commit in current repo
- [ ] Final response includes: commit hash, file paths, commands, limits

## Common Pitfalls

### 1. Assuming CUDA Code Works on NPU
**Problem**: CUDA-specific implementations may fail silently or produce incorrect results on NPU.
**Solution**: Always test NPU-specific code paths with real inputs.

### 2. Skipping Accuracy Validation
**Problem**: Model loading successfully does NOT mean outputs are correct.
**Solution**: Run accuracy tests after real-weight validation.

### 3. Modifying Shared Code Without Guards
**Problem**: Changes to shared code may break CUDA path.
**Solution**: Use `is_npu()` check or create NPU-specific files in `hardware_backend/npu/`.

### 4. Ignoring Token Limits for Reasoning Models
**Problem**: Reasoning models output thinking process before answer, may get truncated.
**Solution**: Use `max_tokens >= 500` for reasoning models, check `finish_reason`.

### 5. Not Checking Head Count Before CUDA Graph
**Problem**: CUDA Graph fails with non-power-of-2 head counts.
**Solution**: Pre-flight check head count, use `--disable-cuda-graph` if needed.

## Completion Criteria

### Basic Functional Tests (Mandatory)

```bash
# 1. Health check
curl http://localhost:8000/v1/models

# 2. Inference test
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<model>", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'
```

### Performance Benchmark (Recommended)

```bash
python -m sglang.bench_serving \
    --dataset-name random --backend sglang \
    --host 127.0.0.1 --port 8000 \
    --num-prompts 100 --random-input-len 128 --random-output-len 128
```

### Transition to Testing Workflow

Once basic tests pass, invoke **npu-testing-workflow** skill for:
- Full performance benchmarking
- Accuracy evaluation
- Structured test report generation

**Handoff data**: model_path, working_command, known_limitations, basic_test_results
