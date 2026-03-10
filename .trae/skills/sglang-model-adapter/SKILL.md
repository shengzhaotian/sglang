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

**Assume environment is ready.** Only verify on error:

| Check | Command | When to Run |
|-------|---------|-------------|
| NPU available | `python -c "import torch; assert torch.npu.is_available()"` | Startup failure |
| CANN version | `npu-smi info` | Communication errors |
| SGLang installed | `python -c "import sglang"` | Import errors |

**Full reference**: [shared/npu_common_reference.md](../shared/npu_common_reference.md)

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

### Step 1: Collect Context

- Confirm model path (default `/models/<model-name>`)
- Confirm implementation roots (current project repo)
- Confirm delivery root (current git repo)
- Use default feature set: ACLGraph + DeepEP + MTP + multimodal (if VL)

### Step 2: Resource Assessment (Conditional)

**Skip if model fits single card.** Only assess when:
- Model > 60GB parameters
- TP size > 1 required

```bash
# Quick check (run only if needed)
python -c "import torch, json; \
  mem = torch.npu.get_device_properties(0).total_memory/1e9; \
  c = json.load(open('/models/<model>/config.json')); \
  print(f'NPU: {mem:.0f}GB | Model: {c.get(\"hidden_size\")}h × {c.get(\"num_hidden_layers\")}L')"
```

**Decision**: Single card if model < NPU memory × 0.9, else use TP.

### Step 3: Analyze Model

- Inspect `config.json`, processor files, modeling files, tokenizer files
- Identify architecture class, attention variant, quantization type
- Check state-dict key prefixes for mapping needs
- Check `python/sglang/srt/models/registry.py` for existing support

### Step 4: Choose Adaptation Strategy

- **Reuse**: If architecture exists in registry
- **Implement**: If missing or incompatible:
  - Add model adapter under `python/sglang/srt/models/`
  - Add processor in `hf_transformers_utils.py` if needed
  - Register architecture in `registry.py`
  - Implement weight loading/remap rules

### Step 5: Implement Minimal Changes

- Touch only required files
- Keep weight mapping explicit and auditable
- Avoid unrelated refactors

### Step 6: Two-Stage Validation

**Pre-flight: Clean residual processes**
```bash
pkill -f "sglang.launch_server" 2>/dev/null || true
```

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

#### Stage C: Accuracy Validation (MANDATORY)

**Model loading successfully does NOT guarantee correct outputs.** Always run accuracy tests.

| Test Type | Question | Expected |
|-----------|----------|----------|
| Math | 15 × 7 = ? | 105 |
| Knowledge | Largest planet? | Jupiter |
| Logic | A>B, B>C → A>C? | Yes |

**Special**: Reasoning models need `max_tokens >= 500`, check `finish_reason=stop`.

**Commands**: See [quick_start_commands.md](./reference/quick_start_commands.md)

### Step 7: Validate Features

- Feature-first: EP + ACLGraph first; eager as fallback
- For multimodal processor issues: use `--limit-mm-per-prompt` to isolate
- Capacity baseline: `context-length=128k` + `max-running-requests=16`

### Step 8: Generate Artifacts and Commit

- Create `./<ModelName>.md` tutorial (Introduction, Features, Environment, Deployment, Verification, Accuracy, Performance)
- Single signed commit with all changes

## Quick Start Commands

See [shared/npu_common_reference.md](../shared/npu_common_reference.md) for environment variables.

| Model Type | Key Flags |
|------------|-----------|
| Standard LLM | `--attention-backend ascend --device npu` |
| MLA (DeepSeek) | `--enable-dp-attention` + `ASCEND_USE_FIA=1 SGLANG_NPU_USE_MLAPO=1` |
| MoE | `--ep-size <n> --moe-a2a-backend deepep` |
| Speculative | `--speculative-algorithm EAGLE3` + `SGLANG_ENABLE_SPEC_V2=1` |

**Detailed commands**: See [quick_start_commands.md](./reference/quick_start_commands.md)

## Error Handling & Fallback

### Obstacle Detection Triggers

| Obstacle Type | Detection Pattern | Threshold |
|---------------|-------------------|-----------|
| Dimension mismatch | `shape mismatch`, `size mismatch`, dimension errors | >3 attempts |
| Fused operator inapplicable | `FIA not supported`, `operator not implemented` | >2 attempts |
| Graph capture failure | `CUDA graph`, `ACLGraph`, graph capture errors | >2 attempts |
| Custom kernel failure | `kernel not found`, `unsupported operation on NPU` | >2 attempts |
| MoE routing errors | `expert routing error`, `DeepEP initialization failed` | >2 attempts |

### Decision Flow

```
Error → Attempts ≥ threshold? → Match obstacle pattern? → Implement native fallback → Record optimization task → Validate accuracy
```

### Quick Fallback Commands

```bash
# Attention/Graph issues
python -m sglang.launch_server --model-path /models/<model> --disable-cuda-graph --device npu --port 8000

# MoE issues
export SGLANG_NPU_USE_NATIVE_MOE=1
python -m sglang.launch_server --model-path /models/<model> --attention-backend ascend --device npu --port 8000
```

### Native Implementation Templates

**Attention Fallback**:
```python
def forward(self, q, k, v, ...):
    if is_npu() and self._use_native_fallback:
        return F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(), ...)
    return self._forward_fused(q, k, v, ...)
```

**Graph Fallback**: Add `--disable-cuda-graph` flag or set `enable_graph=False` in config.

**MoE Fallback**: Use sequential expert computation instead of DeepEP when `SGLANG_NPU_USE_NATIVE_MOE=1`.

### Debug Checklist

| Phase | Check (on error) |
|-------|------------------|
| Startup | PYTHONPATH set, `--attention-backend ascend`, `--device npu` |
| Inference | `/v1/models` first, check attention path logs, verify KV Cache |
| Performance | ACLGraph enabled, `--cuda-graph-bs` matches, HCCL logs |

### Feature-Specific Checks

**MoE Models**: `--ep-size`, `--moe-a2a-backend deepep`, DeepEP env vars
**MLA Models**: `ASCEND_USE_FIA=1`, `SGLANG_NPU_USE_MLAPO=1`, check `kv_lora_rank`
**Speculative**: Draft model path correct, `--speculative-num-steps` reasonable (3-8)

### Optimization Task Recording

After implementing native fallback, create entry in `./<ModelName>_optimization_backlog.md`:
- **Component**: attention/graph/moe/custom
- **Issue**: brief description
- **Fallback Used**: native_sdpa/disable_graph/native_moe
- **Root Cause**: why optimized path failed
- **Priority**: High/Medium/Low based on performance impact

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

See [common_pitfalls.md](./reference/common_pitfalls.md) for detailed explanations.

| Pitfall | Quick Fix |
|---------|-----------|
| CUDA code on NPU | Test NPU paths with real inputs |
| Skipping accuracy test | Run 3-test suite (math/knowledge/logic) |
| Modifying shared code | Use `is_npu()` guard |
| Reasoning model truncation | `max_tokens >= 500` |
| Non-power-of-2 heads | `--disable-cuda-graph` |

## Completion Criteria

**Mandatory**: Quality Gate checklist passed + Accuracy validation (3 tests)
**Recommended**: Performance benchmark + Transition to npu-testing-workflow

See [quick_start_commands.md](./reference/quick_start_commands.md) for test commands.
