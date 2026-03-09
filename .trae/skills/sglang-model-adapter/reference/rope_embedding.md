# NPU RoPE (Rotary Position Embedding) Reference

## Overview

Rotary Position Embedding (RoPE) is a core component of modern LLMs. On NPU, RoPE implementation has specific characteristics that require special attention to operator selection and data layout.

## Core Implementation Locations

```
python/sglang/srt/layers/rotary_embedding.py       # General RoPE implementation
python/sglang/srt/hardware_backend/npu/attention/
├── ascend_backend.py                              # RoPE usage in attention
└── mla_preprocess.py                              # RoPE preprocessing for MLA models
```

## NPU RoPE Operators

### npu_interleave_rope

**Purpose**: NPU-specific interleaved RoPE operator

**Function Signature**:
```python
torch.ops.npu.npu_interleave_rope(
    x,           # [Batch, Num_heads, Seq_len, Head_dim] or [B*S, N, 1, D]
    cos,         # Cosine values
    sin,         # Sine values
)
```

**Features**:
- Suitable for RoPE part of MLA models
- Input needs to be reshaped to 4D tensor
- `cos` and `sin` need to be pre-calculated and cached

**Usage Example** (from `mla_preprocess.py`):
```python
q_pe = q_pe.view(-1, self.num_local_heads, 1, self.qk_rope_head_dim)
cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)
q_pe = torch.ops.npu.npu_interleave_rope(q_pe, cos, sin)
```

### npu_kv_rmsnorm_rope_cache

**Purpose**: Fused RMSNorm + RoPE + KV Cache write

**Function Signature**:
```python
torch.ops.npu.npu_kv_rmsnorm_rope_cache(
    latent_cache,      # [B*S, 1, 1, kv_lora_rank + qk_rope_head_dim]
    rmsnorm_weight,    # LayerNorm weight
    cos,               # Cosine values
    sin,               # Sine values
    slot_mapping,      # Cache write location
    k_rope_cache,      # K RoPE cache output
    c_kv_cache,        # C KV cache output
    epsilon,           # RMSNorm epsilon
    cache_mode,        # "PA_BNSD" or "PA_NZ"
    is_output_kv,      # Whether to output KV
)
```

**Returns**: `(k_pe, k_nope, ...)` or multiple values depending on parameters

**Use Case**: KV cache preprocessing for MLA models

## RoPE Dimension Specifications

### Standard RoPE Models

| Model | RoPE Dimension | Total Head Dim |
|-------|----------------|----------------|
| LLaMA | 128 | 128 |
| Qwen2 | 128 | 128 |
| Mistral | 128 | 128 |

### MLA Models (DeepSeek-V2/V3)

| Component | Dimension | Description |
|-----------|-----------|-------------|
| `qk_rope_head_dim` | 64 | RoPE part dimension |
| `qk_nope_head_dim` | 128 | Non-RoPE part dimension |
| `qk_head_dim` | 192 | Total dimension |

**Key Difference**: MLA models split Q and K into RoPE and non-RoPE parts, with position encoding applied only to the RoPE part.

## RoPE Computation Flow

### Non-MLA Models

```
positions → rotary_emb.get_cos_sin_cache() → cos, sin
                                              ↓
q, k → rotary_emb(positions, q, k) → q_rotated, k_rotated
```

### MLA Models

```
positions → get_sin_cos() → cos, sin
                              ↓
q_pe → npu_interleave_rope(q_pe, cos, sin) → q_pe_rotated
                              ↓
latent_cache → npu_kv_rmsnorm_rope_cache(...) → k_pe, k_nope
```

## DeepSeek Yarn RoPE

DeepSeek models use special Yarn RoPE extension:

```python
# In deepseek_v2_attention_mla_npu.py
if m.use_deepseek_yarn_rope:
    cos, sin = m.rotary_emb.get_cos_sin_cache(positions, dtype, offsets=None)
    q_pe = torch_npu.npu_interleave_rope(
        q_pe.reshape(B, -1, S, m.qk_rope_head_dim),
        cos, sin,
    )
```

## Cos/Sin Cache Management

### Standard Implementation

```python
class RotaryEmbedding:
    def __init__(self, ...):
        self.cos_cached = None
        self.sin_cached = None
    
    def get_cos_sin_cache(self, positions, dtype, offsets=None):
        # Calculate or return cached cos/sin
```

### NPU Optimized Implementation

```python
# In mla_preprocess.py
def get_sin_cos(self, positions):
    cos_sin = self.rotary_emb.cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.repeat(1, 2)  # Expand dimension
    sin = sin.repeat(1, 2)
    return cos, sin
```

## Common Issue Troubleshooting

### 1. Dimension Mismatch

**Symptom**: `RuntimeError: shape mismatch in npu_interleave_rope`

**Checkpoints**:
- Is input a 4D tensor
- Is `qk_rope_head_dim` correct
- Do `cos` and `sin` dimensions match `q_pe`

**Solution**:
```python
# Ensure correct reshape
q_pe = q_pe.view(-1, num_heads, 1, qk_rope_head_dim)
cos = cos.view(-1, 1, 1, qk_rope_head_dim)
sin = sin.view(-1, 1, 1, qk_rope_head_dim)
```

### 2. Position Encoding Errors

**Symptom**: Model outputs garbled text or accuracy degradation

**Checkpoints**:
- Is `positions` tensor correct (starting from 0)
- Is `cos_sin_cache` correctly initialized
- Is the correct RoPE type used (standard vs Yarn)

### 3. Cache Not Correctly Updated

**Symptom**: Long sequence inference errors

**Checkpoints**:
- Is `slot_mapping` correctly mapping to cache locations
- Does `cache_mode` match actual cache layout
- Check if `npu_kv_rmsnorm_rope_cache` output is correctly written

### 4. MLA Model RoPE Separation Issues

**Symptom**: `q_nope` and `q_pe` separation errors

**Checkpoints**:
```python
# Correct separation
q_nope, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)

# Wrong way (dimension order error)
q_pe, q_nope = q.split([qk_rope_head_dim, qk_nope_head_dim], dim=-1)
```

## Debugging Suggestions

### 1. Print Intermediate Results

```python
def forward(self, positions, hidden_states, ...):
    cos, sin = self.get_sin_cos(positions)
    print(f"positions: {positions.shape}, cos: {cos.shape}, sin: {sin.shape}")
    print(f"q_pe before rope: {q_pe.shape}")
    q_pe = torch.ops.npu.npu_interleave_rope(q_pe, cos, sin)
    print(f"q_pe after rope: {q_pe.shape}")
```

### 2. Verify cos/sin Values

```python
# Check if cos/sin are in reasonable range
assert cos.min() >= -1.0 and cos.max() <= 1.0
assert sin.min() >= -1.0 and sin.max() <= 1.0
```

### 3. Compare with CPU Implementation

```python
# Use PyTorch standard RoPE implementation for comparison
def reference_rope(x, cos, sin):
    # Standard RoPE implementation
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
```

## Relationship with Other Modules

```
RoPE
├── Input: positions (from tokenizer)
├── Dependency: rotary_emb (created during model initialization)
├── Output: q_pe, k_pe (passed to attention backend)
└── Interaction with KV Cache: npu_kv_rmsnorm_rope_cache
```
