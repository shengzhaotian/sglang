---
name: sglang-model-adapter
description: "Use when adapting new models for sglang on Ascend NPU, fixing model compatibility issues, or debugging NPU inference errors. Invoke when user asks to adapt models, fix model compatibility, or debug NPU inference."
---

# sglang Ascend Model Adapter

## Overview

Adapt Hugging Face or local models to run on `sglang` with minimal changes, deterministic validation, and single-commit delivery.

## Hard Constraints

- Never upgrade `transformers`
- Start with `export PYTHONPATH=${PWD}/python:$PYTHONPATH`
- Default API port: `8000`
- One single signed commit (`git commit -sm ...`)
- Final docs in Chinese, compact
- **Resource assessment MANDATORY before launch**
- **Accuracy validation MANDATORY** - model loading ≠ correct outputs
- **Fast-Path First** - Use native fallback for quick validation, optimize later

## Pre-flight Resource Assessment

**CRITICAL: Run this BEFORE any server launch.** Never guess tp_size.

```bash
python -c "
import json, os, sys
model_path = sys.argv[1] if len(sys.argv) > 1 else '/models/default'
with open(os.path.join(model_path, 'config.json')) as f:
    c = json.load(f)

hs = c.get('hidden_size', c.get('d_model', 4096))
is_ = c.get('intermediate_size', c.get('ffn_hidden_size', hs * 4))
nl = c.get('num_hidden_layers', c.get('n_layer', 32))
vs = c.get('vocab_size', 32000)
nh = c.get('num_attention_heads', c.get('n_head', 32))
ne = c.get('num_experts', 1)
db = {'float32': 4, 'float16': 2, 'bfloat16': 2}.get(c.get('torch_dtype', 'float16'), 2)

if ne > 1:
    tp = ne * (hs * is_ * 2 + hs * hs * 2) + hs * vs + hs * hs * 2 * nl
else:
    tp = vs * hs + nl * (hs * hs * 4 + hs * is_ * 2 + hs * 3)

msg = (tp * db) / (1024**3)
try:
    import torch_npu
    npu_mem = torch.npu.mem_get_info()[1] // 1024 // 1024 // 1024
except: npu_mem = 64

req = msg * 1.3
tp_size = 1
while tp_size < 64 and req / tp_size > npu_mem * 0.85: tp_size *= 2
hpd = nh // tp_size
cg_ok = hpd > 0 and (hpd & (hpd - 1)) == 0

print(f'Model: {msg:.1f}GB | NPU: {npu_mem}GB | TP: {tp_size} | Heads/Dev: {hpd} | CUDA Graph: {cg_ok} | MoE: {ne>1}')
" /models/<model>
```

**Decision Rules:**
1. **tp_size**: Use calculated value (power of 2)
2. **CUDA Graph**: Disable if `cg_ok=False` → add `--disable-cuda-graph`
3. **MoE**: If `ne>1`, add `--ep-size <tp_size> --moe-a2a-backend deepep`
4. **Verify**: `torch.npu.device_count() >= tp_size`

## NPU Hardware Limitations

### Attention Head Count

NPU requires `heads_per_device = total_heads / tp_size` to be power of 2 (1,2,4,8,16,32,64).

| Heads | TP=1 | TP=2 | TP=4 | Action |
|-------|------|------|------|--------|
| 32 | 32✅ | 16✅ | 8✅ | CUDA Graph OK |
| 20 | 20❌ | 10❌ | 5❌ | Disable CUDA Graph |
| 64 | 64✅ | 32✅ | 16✅ | CUDA Graph OK |

### RoPE / MLA

- **Standard RoPE**: Use `torch_npu.npu_interleave_rope()`, not `npu_mrope`
- **MLA Attention**: Concatenate `q_full = torch.cat([q, q_rope], dim=-1)` before SDPA

## Execution Playbook

### 0) Fast-Path First Strategy (CRITICAL)

**Problem**: Debugging dimension mismatches or fused operator issues consumes excessive tokens and time.

**Solution**: Prioritize native/fallback implementations to achieve functional + accurate model first, then optimize.

```
┌─────────────────────────────────────────────────────────────────┐
│                    FAST-PATH DECISION TREE                       │
├─────────────────────────────────────────────────────────────────┤
│  Error encountered?                                              │
│       │                                                          │
│       ├── Dimension mismatch in fused op?                        │
│       │        └── YES → Switch to native PyTorch implementation │
│       │                  Add TODO comment for future optimization│
│       │                                                          │
│       ├── NPU kernel not available?                              │
│       │        └── YES → Use CPU fallback or standard PyTorch    │
│       │                  Mark as performance optimization point   │
│       │                                                          │
│       └── Complex fusion failing?                                │
│                └── YES → Decompose into simpler ops              │
│                          Validate accuracy first, optimize later │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation Patterns**:

1. **Branch Isolation Pattern** - Add NPU-specific fallback:
```python
if is_npu():
    # Native implementation for quick validation
    result = self._forward_native(x)
    # TODO(perf): Optimize with fused kernel after accuracy validated
else:
    # Original optimized CUDA path
    result = self._forward_fused(x)
```

2. **Decomposition Pattern** - Break complex ops:
```python
if is_npu():
    # Step-by-step native ops
    x = self.norm(x)
    x = self.proj(x)
    # TODO(perf): Fuse norm+proj for NPU
else:
    x = self.fused_norm_proj(x)
```

3. **Fallback Chain Pattern**:
```python
try:
    # Try optimized NPU kernel
    return npu_fused_attention(q, k, v)
except (RuntimeError, NotImplementedError):
    # Fallback to standard SDPA
    return F.scaled_dot_product_attention(q, k, v)
```

**Optimization Tracking**:

Create `TODO-NPU-OPT.md` in model directory:
```markdown
## Performance Optimization Points

| Location | Issue | Native Fallback | Priority |
|----------|-------|-----------------|----------|
| attention.py:123 | Fused kernel dim mismatch | Standard SDPA | High |
| mlp.py:45 | GELU fusion not supported | torch.nn.GELU | Medium |
```

**Token Budget Rule**: If debugging a single issue exceeds 3 iterations, switch to native fallback immediately.

### 1) Collect Context
- Model path (default `/models/<model-name>`)
- Implementation root (current repo)

### 2) Analyze Model
- Inspect `config.json`, modeling files
- Check `python/sglang/srt/models/registry.py` for existing support

### 3) Calculate Configuration (MANDATORY)
Run resource assessment script above. **NEVER skip.**

### 4) Choose Strategy
- **Reuse**: If architecture exists in registry
- **Implement**: Add model adapter under `models/`, register in `registry.py`

### 5) Two-Stage Validation

**Stage A: Dummy weights**
```bash
python -m sglang.launch_server --model-path /models/<model> --load-format dummy \
    --tp-size <calculated_tp> --attention-backend ascend --device npu --port 8000
```

**Stage B: Real weights**
```bash
python -m sglang.launch_server --model-path /models/<model> \
    --tp-size <calculated_tp> --attention-backend ascend --device npu --port 8000
```

### 6) Accuracy Validation (MANDATORY)

```bash
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "<model>", "messages": [{"role": "user", "content": "What is 15 times 7?"}], "max_tokens": 100}'
```

## Quick Start Commands

**Standard LLM**: `--tp-size <tp> --attention-backend ascend --device npu`

**MLA (DeepSeek-V2/V3)**: Add `--enable-dp-attention`, env: `ASCEND_USE_FIA=1 SGLANG_NPU_USE_MLAPO=1`

**MoE with DeepEP**: Add `--ep-size <tp> --moe-a2a-backend deepep`, env: `DEEP_NORMAL_MODE_USE_INT8_QUANT=1`

**Speculative (EAGLE3)**: Add `--speculative-algorithm EAGLE3`, env: `SGLANG_ENABLE_SPEC_V2=1`

See [shared/npu_common_reference.md](../shared/npu_common_reference.md) for full env vars.

## Error Reference

| Error | Reference | Location |
|-------|-----------|----------|
| Attention mismatch | [attention_backend.md](./reference/attention_backend.md) | `ascend_backend.py` |
| MLA weight errors | [mla_preprocess.md](./reference/mla_preprocess.md) | `mla_preprocess.py` |
| RoPE errors | [rope_embedding.md](./reference/rope_embedding.md) | `mla_preprocess.py` |
| Graph capture fails | [aclgraph.md](./reference/aclgraph.md) | `npu_graph_runner.py` |

Full index: [reference/README.md](./reference/README.md)

## Quality Gate

- [ ] Resource assessment completed (model size, tp_size, CUDA Graph decision)
- [ ] Available NPUs verified (device_count >= tp_size)
- [ ] Fast-path fallbacks implemented for blocking issues
- [ ] Optimization points documented in `TODO-NPU-OPT.md`
- [ ] Service starts successfully
- [ ] Accuracy validation passed (min 3 tests)
- [ ] Tutorial doc at `./<ModelName>.md`
- [ ] One signed commit

## Common Pitfalls

1. **Skipping resource assessment** → OOM/crash. ALWAYS calculate tp_size first.
2. **Ignoring head count** → CUDA Graph fails. Check power-of-2 requirement.
3. **Skipping accuracy tests** → Model loads but outputs wrong. Run validation.
4. **Modifying shared code without guards** → Breaks CUDA path. Use `is_npu()` check.
5. **Over-debugging fused ops** → Token waste. Use native fallback after 3 failed attempts.
6. **Premature optimization** → Blocks progress. Validate accuracy first, optimize later.

## Transition to Testing Workflow

Once basic tests pass, invoke **npu-testing-workflow** skill for full benchmarking.

**Handoff**: model_path, working_command, known_limitations, basic_test_results
