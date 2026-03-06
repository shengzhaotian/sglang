# NPU MLA Preprocess Reference

## Overview

The MLA (Multi-head Latent Attention) preprocessing module is a key component for MLA architecture models like DeepSeek-V2/V3 running on NPU. It is responsible for transforming compressed latent representations into computable forms.

## Core Files

```
python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py
```

## Environment Variables

| Variable Name | Description | Default |
|---------------|-------------|---------|
| `SGLANG_NPU_USE_MLAPO` | Enable MLAPO optimization | False |
| `SGLANG_USE_FIA_NZ` | Enable NZ format (requires MLAPO) | False |

## NPUFusedMLAPreprocess Class

### Initialization Parameters

```python
NPUFusedMLAPreprocess(
    fused_qkv_a_proj_with_mqa,  # Fused QKV projection layer
    q_a_layernorm,              # Q LayerNorm
    kv_a_layernorm,             # KV LayerNorm
    q_b_proj,                   # Q second projection layer
    w_kc,                       # Key compression weight
    rotary_emb,                 # Rotary position encoding
    layer_id,                   # Layer ID
    num_local_heads,            # Local head count (after TP)
    qk_nope_head_dim,           # Non-RoPE dimension (e.g., 128)
    qk_rope_head_dim,           # RoPE dimension (e.g., 64)
    v_head_dim,                 # Value dimension
    quant_config,               # Quantization configuration
)
```

### Key Dimension Specifications

| Dimension | DeepSeek-V2/V3 Typical Value | Description |
|-----------|------------------------------|-------------|
| `q_lora_rank` | 1536 | Q LoRA compression rank |
| `kv_lora_rank` | 512 | KV LoRA compression rank |
| `qk_nope_head_dim` | 128 | Query/Key non-RoPE part |
| `qk_rope_head_dim` | 64 | Query/Key RoPE part |
| `qk_head_dim` | 192 | qk_nope + qk_rope |
| `v_head_dim` | 128 | Value head dimension |

### Three Preprocessing Modes

#### 1. forward_mlapo (W8A8 Quantization Mode)

**Trigger Condition**: `qkv_a_proj` uses modelslim quantization

**Flow**:
```
hidden_states
    ↓
qkv_a_proj (quantized matmul)
    ↓
split → q_lowrank, latent_cache
    ↓
q_a_layernorm → q_b_proj → q
    ↓
split → q_nope, q_pe
    ↓
matmul(q_nope, w_kc) → q_nope_out
    ↓
npu_interleave_rope(q_pe) → q_rope_out
    ↓
npu_kv_rmsnorm_rope_cache(latent_cache) → k_rope, k_nope
```

**Key Operator**: `torch.ops.npu.mla_preprocess`

**Weight Preprocessing** (`preprocess_weights` method):
- Convert weights to NZ format
- Rearrange RoPE dimensions (`trans_rope_weight`)
- Prepare dequant_scale and quant_bias

#### 2. forward_mlaprolog (MLA Prolog Mode)

**Trigger Condition**: `quant_config.ignore` contains `kv_b_proj`

**Flow**:
```
hidden_states
    ↓
npu_mla_prolog_v3(
    token_x, weight_dq, weight_uq_qr,
    weight_uk, weight_dkv_kr,
    rmsnorm_gamma_cq, rmsnorm_gamma_ckv,
    rope_sin, rope_cos,
    kv_cache, kr_cache, cache_index
)
    ↓
q_nope, q_pe, dequant_scale_q_nope, qr, dequant_q_norm
```

**Feature**: Uses CANN's MLA Prolog operator, more efficient

#### 3. forward_absorb_prepare_npu_rms_norm_cache (Non-quantized Mode)

**Trigger Condition**: Non-quantized model

**Flow**:
```
hidden_states
    ↓
qkv_a_proj → q_lowrank, latent_cache
    ↓
q_a_layernorm → q_b_proj → q
    ↓
split → q_nope, q_pe
    ↓
matmul(q_nope, w_kc) → q_nope_out
    ↓
rotary_emb(q_pe, k_pe) → q_pe, k_pe
    ↓
npu_kv_rmsnorm_rope_cache → k_rope, k_nope
```

## Weight Transformation Details

### trans_rope_weight Function

**Purpose**: Rearrange RoPE dimensions to adapt to NPU operators

```python
def trans_rope_weight(weight, rope_dim):
    # Rearrange odd and even positions of RoPE dimension
    weight_1 = weight[..., -rope_dim::2, :]  # Even positions
    weight_2 = weight[..., -rope_dim + 1 :: 2, :]  # Odd positions
    weight[..., -rope_dim:, :] = torch.cat([weight_1, weight_2], dim=-2)
    return weight
```

### transdata Function

**Purpose**: Convert matrix to NZ (Block Major) format

```python
def transdata(nd_mat, block_size=(16, 16)):
    # Block and rearrange matrix to NPU-optimized format
    # Input: [M, N]
    # Output: [N//16, M*16, 16] (NZ format)
```

## KV Cache Operations

### get_kv_cache_and_cache_idx

```python
def get_kv_cache_and_cache_idx(self, forward_batch):
    k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(self.layer_id)
    slot_mapping = forward_batch.out_cache_loc.to(dtype=torch.int32)
    return k_cache, v_cache, slot_mapping
```

**Note**: In MLA models:
- `k_cache` stores `kv_c` (compressed KV latent)
- `v_cache` stores `k_pe` (Key RoPE part)

### Cache Modes

| Mode | Description | Trigger Condition |
|------|-------------|-------------------|
| `PA_BNSD` | Standard Paged Attention layout | Default |
| `PA_NZ` | NZ format | `SGLANG_USE_FIA_NZ=1` |
| `krope_ctkv` | K RoPE + CTKV | MLAPO mode |
| `nzcache` | NZ Cache | FIA NZ mode |

## Interaction with Attention Backend

### Data Flow

```
NPUFusedMLAPreprocess.forward()
    ↓
(q_rope, k_rope, q_nope, k_nope, ...)
    ↓
AscendAttnBackend.forward_mla() or forward_sparse()
    ↓
npu_fused_infer_attention_score or npu_ring_mla
```

### Key Interfaces

```python
# In deepseek_v2_attention_mla_npu.py
def forward_mla_prepare_npu(m, positions, hidden_states, forward_batch, ...):
    if is_mla_preprocess_enabled():
        if not hasattr(m, "mla_preprocess"):
            m.mla_preprocess = NPUFusedMLAPreprocess(...)
        (q_pe, k_pe, q_nope_out, k_nope, ...) = m.mla_preprocess.forward(...)
```

## Common Issue Troubleshooting

### 1. Dimension Errors

**Symptom**: `RuntimeError: shape mismatch`

**Checkpoints**:
- Is `kv_lora_rank` consistent with model configuration
- Is `qk_rope_head_dim` correct
- Does `num_local_heads` account for TP sharding

### 2. Quantization-related Issues

**Symptom**: Accuracy degradation or NaN

**Checkpoints**:
- Are `deq_scale` and `quant_bias` correctly loaded
- Are weights correctly converted to NZ format
- Check `input_scale` and `input_offset`

### 3. Cache Format Issues

**Symptom**: FIA operator errors

**Checkpoints**:
- Does `SGLANG_USE_FIA_NZ` match actual cache format
- Is `cache_mode` parameter correct
- Is `slot_mapping` type `int32` or `int64`

### 4. RoPE-related Issues

**Symptom**: Position encoding errors

**Checkpoints**:
- Are `cos` and `sin` correctly calculated
- Choice of `npu_interleave_rope` vs `rotary_emb`
- Check if `positions` tensor is correct

## Debugging Suggestions

1. **Add Logging**: Print tensor shapes at the beginning of `forward` method
2. **Check Weights**: Ensure `preprocess_weights` executes only once
3. **Verify Cache**: Check if `slot_mapping` values are within valid range
4. **Compare Paths**: For the same model, compare outputs of three modes
