# Architecture Detection Patterns

This document provides detailed patterns for detecting model architecture from source code.

## Standard Transformer (Dense)

### File Patterns
- Model file name: `llama.py`, `qwen2.py`, `qwen3.py`, `mistral.py`
- No "moe" in filename or class names

### Code Patterns

```python
# Standard attention layers
self.q_proj = nn.Linear(hidden_size, num_heads * head_dim)
self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)
self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)
self.o_proj = nn.Linear(num_heads * head_dim, hidden_size)

# Standard MLP (SwiGLU)
self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

# No MoE imports
# from sglang.srt.layers.moe import ...  # NOT present
```

### Config Patterns

```json
{
  "hidden_size": 4096,
  "intermediate_size": 22016,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  // No MoE parameters
}
```

### DP-Attention Recommendation

| Use DP-Attention? | Reason |
|-------------------|--------|
| **No** | Standard TP is sufficient for dense models |

---

## MoE (Mixture of Experts)

### File Patterns
- Model file name: `*_moe.py`, `*moe*.py`
- Class names contain "MoE" or "SparseMoe"

### Code Patterns

```python
# MoE imports
from sglang.srt.layers.moe import FusedMoE, TopK
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class

# MoE block
class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, ..., num_experts, moe_intermediate_size, ...):
        self.experts = get_moe_impl_class()(
            num_experts=num_experts,
            top_k=config.num_experts_per_tok,
            intermediate_size=moe_intermediate_size,
        )
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)

# Expert parallel imports
from sglang.srt.distributed import get_moe_expert_parallel_world_size
```

### Config Patterns

```json
{
  "hidden_size": 4096,
  "num_hidden_layers": 94,
  "num_experts": 128,
  "num_experts_per_tok": 8,
  "moe_intermediate_size": 768,
  // May have decoder_sparse_step = 1 (all layers are MoE)
}
```

### Key Indicators

| Indicator | Description |
|-----------|-------------|
| `num_experts` | Number of experts in MoE layer |
| `num_experts_per_tok` | Top-k experts per token |
| `moe_intermediate_size` | Expert intermediate dimension |
| `decoder_sparse_step` | Step for sparse layers (1 = all MoE) |

### DP-Attention Recommendation

| Use DP-Attention? | Configuration | Reason |
|-------------------|---------------|--------|
| **Yes (Moderate)** | `--tp-size N --dp-size N --enable-dp-attention` | Better throughput for MoE |

---

## MLA (Multi-Head Latent Attention)

### File Patterns
- Model file for DeepSeek: `deepseek_v2.py`, `deepseek_v3.py`
- Class names contain "DeepSeekV2", "DeepSeekV3"

### Code Patterns

```python
# MLA specific layers
self.kv_a_proj_with_mqa = nn.Linear(hidden_size, kv_lora_rank + qk_rope_head_dim)
self.kv_b_proj = nn.Linear(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim))
self.q_a_proj = nn.Linear(hidden_size, q_lora_rank)
self.q_b_proj = nn.Linear(q_lora_rank, num_heads * qk_head_dim)

# Compressed KV
# kv_lora_rank is much smaller than num_heads * head_dim

# Key parameters
kv_lora_rank = 512  # Compressed KV dimension
q_lora_rank = 1536  # Compressed Q dimension
qk_rope_head_dim = 64  # RoPE dimension
```

### Config Patterns

```json
{
  "hidden_size": 7168,
  "kv_lora_rank": 512,
  "q_lora_rank": 1536,
  "qk_head_dim": 128,
  "qk_rope_head_dim": 64,
  "v_head_dim": 128,
  // MLA specific parameters
}
```

### Key Indicators

| Indicator | Description |
|-----------|-------------|
| `kv_lora_rank` | Compressed KV latent dimension |
| `q_lora_rank` | Compressed Q latent dimension |
| `qk_rope_head_dim` | RoPE dimension per head |
| `v_head_dim` | Value head dimension |

### DP-Attention Recommendation

| Use DP-Attention? | Configuration | Reason |
|-------------------|---------------|--------|
| **Yes (Strongly)** | `--tp-size N --dp-size D --enable-dp-attention` | Eliminates KV cache duplication |

---

## MLA + MoE (DeepSeek-V3/R1)

### File Patterns
- Model file: `deepseek_v3.py`, `deepseek_v2.py`
- Contains both MLA and MoE patterns

### Code Patterns

```python
# MLA attention (as above)
self.kv_a_proj_with_mqa = ...
self.kv_b_proj = ...

# MoE layers
class DeepseekV3MoE(nn.Module):
    def __init__(self, ..., n_routed_experts, n_shared_experts, ...):
        self.experts = ...  # Routed experts
        self.shared_experts = ...  # Shared experts (always active)

# Both patterns present
```

### Config Patterns

```json
{
  // MLA parameters
  "kv_lora_rank": 512,
  "q_lora_rank": 1536,

  // MoE parameters
  "n_routed_experts": 256,
  "n_shared_experts": 1,
  "moe_intermediate_size": 2048,
  "num_experts_per_tok": 8,

  // First N dense layers
  "num_dense_layers": 3,
  "num_hidden_layers": 61
}
```

### Key Indicators

| Indicator | Description |
|-----------|-------------|
| `n_routed_experts` | Number of routed experts |
| `n_shared_experts` | Number of always-active experts |
| `num_dense_layers` | First N layers are dense (non-MoE) |
| Both MLA and MoE params | Indicates MLA + MoE architecture |

### DP-Attention Recommendation

| Use DP-Attention? | Configuration | Reason |
|-------------------|---------------|--------|
| **Yes (Very Strongly)** | `--tp-size N --dp-size D --ep-size N --enable-dp-attention --moe-dense-tp-size 1` | Best throughput for MLA+MoE |

---

## Quick Detection Flowchart

```
Start
  |
  v
Check for MoE indicators (num_experts, SparseMoeBlock)
  |
  +-- Yes --> MoE or MLA+MoE
  |             |
  |             v
  |           Check for MLA indicators (kv_lora_rank)
  |             |
  |             +-- Yes --> MLA + MoE (DeepSeek-V3/R1)
  |             |             |
  |             |             v
  |             |           DP-Attention: STRONGLY RECOMMENDED
  |             |
  |             +-- No --> Pure MoE (Qwen3-235B, Mixtral)
  |                           |
  |                           v
  |                         DP-Attention: MODERATELY RECOMMENDED
  |
  +-- No --> Standard or MLA
              |
              v
            Check for MLA indicators (kv_lora_rank)
              |
              +-- Yes --> MLA (rare, usually with MoE)
              |             |
              |             v
              |           DP-Attention: STRONGLY RECOMMENDED
              |
              +-- No --> Standard (Llama, Qwen3-32B)
                            |
                            v
                          DP-Attention: NOT RECOMMENDED
```

---

## Memory Calculation by Architecture

| Architecture | Weight Calculation | KV Cache Calculation | DP-Attention Impact |
|--------------|-------------------|---------------------|---------------------|
| Standard | `embed + attn + mlp + lm_head` | `2 * num_kv_heads * head_dim * num_layers` | Minimal benefit |
| MoE | `embed + attn + gate + experts + lm_head` | Same as standard | Moderate throughput gain |
| MLA | `embed + mla_attn + mlp + lm_head` | `(kv_lora_rank + qk_rope_head_dim) * num_layers` | Significant capacity gain |
| MLA + MoE | `embed + mla_attn + gate + experts + shared + lm_head` | Same as MLA | Very significant throughput gain |

---

## Parallelism Impact by Architecture

### Without DP-Attention

| Architecture | TP Shards | EP Shards | Replicated |
|--------------|-----------|-----------|------------|
| Standard | All weights | N/A | LayerNorm |
| MoE | Attention, Embedding | Experts | Gate, LayerNorm |
| MLA | q_b, kv_b, o_proj | N/A | fused_qkv_a, LayerNorm |
| MLA + MoE | q_b, kv_b, o_proj, shared_experts | routed_experts | Gate, fused_qkv_a, LayerNorm |

### With DP-Attention

| Architecture | TP Shards | attn_tp Shards | EP Shards | Replicated |
|--------------|-----------|----------------|-----------|------------|
| Standard | MLP | Attention | N/A | LayerNorm |
| MoE | MLP | Attention | Experts | Gate, LayerNorm |
| MLA | MLP | q_b, kv_b, o_proj | N/A | fused_qkv_a, LayerNorm |
| MLA + MoE | shared_experts | q_b, kv_b, o_proj | routed_experts | Gate, fused_qkv_a, LayerNorm |

**Key**: `attn_tp_size = tp_size / dp_size`

---

## DP-Attention Memory Implications

### KV Cache Behavior

| Architecture | Without DP-Attention | With DP-Attention |
|--------------|---------------------|-------------------|
| **Standard** | Sharded by TP | Sharded by attn_tp |
| **MoE** | Sharded by TP | Sharded by attn_tp |
| **MLA** | Replicated | Replicated (but unique per DP rank) |
| **MLA + MoE** | Replicated | Replicated (but unique per DP rank) |

### Effective Capacity

```
Without DP-Attention:
  - All TP ranks process SAME batch
  - KV cache shared/duplicated
  - Capacity = 1× single batch

With DP-Attention:
  - Each DP rank processes DIFFERENT batch
  - KV cache unique per DP rank
  - Capacity = dp_size × single batch
```

### Example: DeepSeek-R1

```
Configuration: TP=32, DP=16, EP=32
attn_tp_size = 32 / 16 = 2

Without DP-Attention:
  - 32 ranks share same batch
  - Max requests: ~32

With DP-Attention:
  - 16 DP groups, each with 2 ranks
  - Each handles independent batch
  - Max requests: ~16 × 32 = 512 (16× improvement!)
```

---

## Recommended Configurations by Architecture

### Standard Models (e.g., Qwen3-32B)

```bash
# No DP-Attention needed
python -m sglang.launch_server \
    --model-path <model> \
    --tp-size 8 \
    --attention-backend ascend \
    --device npu
```

### MoE Models (e.g., Qwen3-235B-A22B)

```bash
# DP-Attention recommended
python -m sglang.launch_server \
    --model-path <model> \
    --tp-size 16 \
    --dp-size 16 \
    --enable-dp-attention \
    --enable-dp-lm-head \
    --moe-a2a-backend deepep \
    --attention-backend ascend \
    --device npu
```

### MLA + MoE Models (e.g., DeepSeek-R1)

```bash
# DP-Attention strongly recommended
python -m sglang.launch_server \
    --model-path <model> \
    --tp-size 32 \
    --dp-size 16 \
    --ep-size 32 \
    --enable-dp-attention \
    --enable-dp-lm-head \
    --moe-dense-tp-size 1 \
    --moe-a2a-backend deepep \
    --attention-backend ascend \
    --device npu
```
