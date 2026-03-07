# NPU Attention Backend Reference

## Overview

The NPU Attention backend is a core component of SGLang running on Huawei Ascend NPUs, responsible for handling all attention computations.

## Core Files

```
python/sglang/srt/hardware_backend/npu/attention/
├── ascend_backend.py           # Main backend implementation
├── ascend_torch_native_backend.py  # Torch Native SDPA backend
└── mla_preprocess.py           # MLA preprocessing module
```

## AscendAttnBackend Class Structure

### Initialization Parameters

| Parameter | Source | Description |
|-----------|--------|-------------|
| `device` | `model_runner.device` | NPU device |
| `page_size` | `model_runner.page_size` | KV cache page size |
| `use_mla` | `model_config.attention_arch == AttentionArch.MLA` | Whether to use MLA architecture |
| `kv_lora_rank` | `model_config.kv_lora_rank` | KV LoRA rank for MLA |
| `qk_rope_head_dim` | `model_config.qk_rope_head_dim` | RoPE dimension |
| `qk_nope_head_dim` | `model_config.qk_nope_head_dim` | Non-RoPE dimension |
| `use_fia` | `ASCEND_USE_FIA` environment variable | Whether to use FIA backend |

### Core Methods

#### 1. forward_extend (Extension Phase)

**Purpose**: Handle prefill and extend requests

**Key Path Selection**:

```
forward_extend()
├── topk_indices != None? → forward_sparse() (NSA/sparse attention)
├── is_target_verify() or is_draft_extend()? → forward_mtp() (speculative decoding)
├── use_mla == False?
│   ├── use_fia == True? → npu_fused_infer_attention_score (FIA path)
│   ├── qk_head_dim <= 128 and causal? → _npu_flash_attention_qlens
│   └── else → native_attn.run_sdpa_forward_extend
└── use_mla == True?
    ├── prefix_lens > 0? → npu_ring_mla (two-stage computation)
    └── else → FIA or native SDPA
```

**Key Code Location**: `ascend_backend.py:714-1087`

#### 2. forward_decode (Decode Phase)

**Purpose**: Handle decode requests

**Key Path Selection**:

```
forward_decode()
├── graph_mode and not enable_torch_compile? → forward_decode_graph()
├── use_mla == False?
│   ├── use_fia == True? → npu_fused_infer_attention_score
│   ├── encoder_lens is None and logit_cap == 0? → _npu_paged_attention
│   └── else → native_attn.run_sdpa_forward_decode
└── use_mla == True?
    ├── use_fia and GQA >= 8? → npu_fused_infer_attention_score
    └── else → _npu_paged_attention_mla
```

**Key Code Location**: `ascend_backend.py:1427-1677`

#### 3. forward_mtp (Speculative Decoding)

**Purpose**: Handle MTP (Multi-Token Prediction) speculative decoding

**Key Code Location**: `ascend_backend.py:1088-1267`

## NPU Operator Reference

### npu_fused_infer_attention_score (FIA)

**Purpose**: Fused attention operator provided by Huawei CANN

**Key Parameters**:

| Parameter | Description | Common Values |
|-----------|-------------|---------------|
| `input_layout` | Input layout | "TND", "BSND", "BSH" |
| `sparse_mode` | Sparse mode | 0 (no mask), 3 (causal mask) |
| `num_heads` | Number of query heads | - |
| `num_key_value_heads` | Number of KV heads | - |
| `scale` | Scale factor | `1/sqrt(head_dim)` |
| `block_table` | KV cache page table | - |
| `block_size` | Page size | - |
| `actual_seq_lengths` | Query actual sequence length | - |
| `actual_seq_lengths_kv` | KV actual sequence length | - |

**Layout Description**:
- `TND`: [Token, Num_heads, Head_dim] - Suitable for variable-length sequences
- `BSND`: [Batch, Seq, Num_heads, Head_dim] - Suitable for fixed batches
- `BSH`: [Batch, Seq, Hidden] - Suitable for single-head or fused layouts

### _npu_flash_attention_qlens

**Purpose**: Flash Attention supporting variable-length sequences

**Key Parameters**:
- `query`: [Total_tokens, Hidden]
- `key_cache`: KV cache keys
- `value_cache`: KV cache values
- `mask`: Attention mask
- `block_table`: Page table
- `seq_len`: Extension length
- `context_lens`: Context length

### _npu_paged_attention

**Purpose**: Paged Attention for decode phase

**Key Parameters**:
- `query`: [Num_tokens, Num_heads, Head_dim]
- `key_cache`: KV cache
- `value_cache`: KV cache
- `block_table`: Page table
- `context_lens`: Context length

### npu_ring_mla

**Purpose**: Ring attention computation for MLA architecture

**Key Parameters**:
- `q_nope`, `q_rope`: Non-RoPE and RoPE parts of query
- `k_nope`, `k_rope`: Non-RoPE and RoPE parts of key
- `value`: Value
- `mask`: Attention mask
- `seqlen`: Sequence length
- `kernel_type`: "kernel_type_high_precision"
- `mask_type`: "mask_type_triu" or "no_mask"
- `calc_type`: "calc_type_first_ring" or "calc_type_default"

## MLA vs Non-MLA Differences

### Non-MLA Models
- K and V stored separately
- Standard QKV projection
- Uses `_npu_paged_attention` or FIA

### MLA Models (DeepSeek-V2/V3)
- K compressed as latent representation
- Separated `q_nope` and `q_pe` (RoPE part)
- Uses `npu_ring_mla` or `_npu_paged_attention_mla`
- Requires additional `kv_b_proj` for decompression

## Common Issue Troubleshooting

### 1. Dimension Mismatch
- Check `qk_head_dim` vs `v_head_dim`
- In MLA models, `qk_head_dim != v_head_dim` is normal
- Confirm GQA ratio of `num_heads` and `num_kv_heads`

### 2. Layout Errors
- FIA path: Check `input_layout` parameter
- Non-FIA path: Check if tensor reshape is correct
- NZ format: Check `SGLANG_USE_FIA_NZ` environment variable

### 3. Mask Issues
- `sparse_mode=3` for causal attention
- `sparse_mode=0` for no mask (decode phase)
- Check if `atten_mask` is correctly initialized

### 4. Performance Issues
- Check if FIA is enabled (`ASCEND_USE_FIA=1`)
- For MLA models, check `SGLANG_NPU_USE_MLAPO`
- Check if graph mode is correctly enabled
