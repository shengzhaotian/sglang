# NPU KV Cache Reference

## Overview

KV Cache is a core component of LLM inference, used to store and reuse historical Key-Value states. NPU KV Cache implementation has specific characteristics including page-based management and NZ format support.

## Core Files

```
python/sglang/srt/hardware_backend/npu/
├── memory_pool_npu.py      # NPU memory pool implementation
├── allocator_npu.py        # NPU memory allocator
└── attention/
    ├── ascend_backend.py   # KV Cache usage in attention
    └── mla_preprocess.py   # KV Cache preprocessing for MLA models
```

## KV Cache Architecture

### Standard (Non-MLA) Models

```
KV Cache Structure:
┌─────────────────────────────────────────┐
│  Block 0  │  Block 1  │  Block 2  │ ... │
├───────────┼───────────┼───────────┼─────┤
│ K: [page_size, num_kv_heads, head_dim]  │
│ V: [page_size, num_kv_heads, head_dim]  │
└─────────────────────────────────────────┘
```

### MLA Models

```
KV Cache Structure (DeepSeek-V2/V3):
┌─────────────────────────────────────────┐
│  Block 0  │  Block 1  │  Block 2  │ ... │
├───────────┼───────────┼───────────┼─────┤
│ kv_c: [page_size, num_kv_heads, kv_lora_rank]  │  ← Compressed KV latent
│ k_pe: [page_size, num_kv_heads, qk_rope_head_dim] │  ← Key RoPE part
└─────────────────────────────────────────┘
```

**Key Difference**: MLA models don't directly store K and V, but store compressed representations `kv_c` and `k_pe`.

## Memory Pool Management

### MemoryPool Class

**Key Attributes**:

```python
class MemoryPool:
    def __init__(self, ...):
        self.page_size = page_size          # Page size (usually 1)
        self.num_layers = num_layers        # Number of layers
        self.dtype = dtype                  # Data type
        
        # KV Cache storage
        self.kv_data = {}                   # Layer ID → tensor
```

### KV Buffer Operations

```python
# Set KV buffer
def set_kv_buffer(self, layer, cache_loc, k, v):
    """
    layer: Current layer
    cache_loc: Write location (slot_mapping)
    k: Key tensor
    v: Value tensor
    """
    
# Get KV buffer
def get_kv_buffer(self, layer_id):
    """
    Returns: (k_cache, v_cache)
    """
```

### Special Handling for MLA Models

```python
# In mla_preprocess.py
def get_kv_cache_and_cache_idx(self, forward_batch):
    k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(self.layer_id)
    # Note: In MLA models
    # k_cache actually stores kv_c (compressed KV)
    # v_cache actually stores k_pe (Key RoPE)
    slot_mapping = forward_batch.out_cache_loc.to(dtype=torch.int32)
    return k_cache, v_cache, slot_mapping
```

## Page Table Management

### Block Table Structure

```python
# block_tables: [batch_size, max_seq_pages]
# Each element is a block ID pointing to the corresponding block in KV Cache

# Example: sequence length 10, page_size = 1
# block_tables[0] = [5, 8, 12, 25, 33, 41, 55, 60, 71, 82, 0, 0, ...]
#                                                          ↑ Unused
```

### Block Table Construction

```python
# In ascend_backend.py
def init_forward_metadata(self, forward_batch: ForwardBatch):
    self.forward_metadata.block_tables = (
        forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :seq_lens_max
        ][:, :: self.page_size]
        // self.page_size
    )
```

## Cache Formats

### PA_BNSD Format (Standard)

```
Layout: [Num_blocks, Block_size, Num_heads, Head_dim]

Example: num_blocks=1000, block_size=1, num_heads=32, head_dim=128
Shape: [1000, 1, 32, 128]
```

### PA_NZ Format (NPU Optimized)

```
Layout: [Num_blocks, Num_heads * Head_dim // 16, Block_size, 16]

Example: num_blocks=1000, block_size=1, num_heads=32, head_dim=128
Shape: [1000, 256, 1, 16]

Feature: 16-element blocks, suitable for NPU vectorization
```

### Format Conversion

```python
# In mla_preprocess.py
def _reshape_kv_for_fia_nz(tensor, num_heads, head_dim, page_size):
    """Convert tensor to FIA NZ format"""
    return tensor.view(-1, 1, num_heads * head_dim // 16, page_size, 16)
```

## Slot Mapping

### Concept

Slot mapping defines which position in KV Cache each token should be written to.

```python
# slot_mapping: [num_tokens]
# Each element is a linear index pointing to a position in cache

# Example:
# tokens: ["Hello", "World", "!"]
# slot_mapping: [0, 1, 2]  # Write to first three positions in cache
```

### How to Get

```python
# In forward_batch
slot_mapping = forward_batch.out_cache_loc

# Convert to required type
slot_mapping_int32 = slot_mapping.to(dtype=torch.int32)
slot_mapping_int64 = slot_mapping.to(dtype=torch.int64)
```

## Cache Write Operations

### Standard Cache Write

```python
# In ascend_backend.py forward_extend
if save_kv_cache and k is not None and v is not None:
    cache_loc = (
        forward_batch.out_cache_loc
        if not layer.is_cross_attention
        else forward_batch.encoder_out_cache_loc
    )
    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
```

### MLA Cache Write

```python
# Method 1: Direct write
forward_batch.token_to_kv_pool.set_kv_buffer(
    layer, forward_batch.out_cache_loc, kv_a.unsqueeze(1), k_pe
)

# Method 2: Write via npu_kv_rmsnorm_rope_cache
torch_npu.npu_kv_rmsnorm_rope_cache(
    latent_cache,
    layernorm_weight,
    cos, sin,
    slot_mapping,
    k_rope_cache,    # Output: k_pe cache
    c_kv_cache,      # Output: kv_c cache
    ...
)
```

## Cache Read Operations

### Standard Cache Read

```python
k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

# Reshape for attention computation
k_cache = k_cache.view(-1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim)
v_cache = v_cache.view(-1, self.page_size, layer.tp_v_head_num * layer.v_head_dim)
```

### MLA Cache Read

```python
kv_c = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
k_pe = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

# For FIA NZ format
if is_fia_nz():
    k_rope_cache = _reshape_kv_for_fia_nz(k_pe, ...)
    c_kv_cache = _reshape_kv_for_fia_nz(kv_c, ...)
else:
    k_rope_cache = k_pe.view(-1, self.page_size, ...)
    c_kv_cache = kv_c.view(-1, self.page_size, ...)
```

## Common Issue Troubleshooting

### 1. Cache Location Errors

**Symptom**: Incorrect inference results or crash

**Checkpoints**:
- Is `slot_mapping` correct
- Is `block_tables` correctly constructed
- Are `req_pool_indices` valid

**Debug Code**:
```python
print(f"slot_mapping: {forward_batch.out_cache_loc}")
print(f"block_tables shape: {self.forward_metadata.block_tables.shape}")
print(f"seq_lens: {forward_batch.seq_lens}")
```

### 2. Dimension Mismatch

**Symptom**: `RuntimeError: shape mismatch`

**Checkpoints**:
- Does `num_heads` account for TP sharding
- Is `head_dim` correct (MLA: `kv_lora_rank` vs `qk_head_dim`)
- Is `page_size` consistent

### 3. Format Errors

**Symptom**: FIA operator errors

**Checkpoints**:
- Is `SGLANG_USE_FIA_NZ` set correctly
- Does cache format match operator expectation
- Check `cache_mode` parameter

### 4. Out of Memory

**Symptom**: OOM errors

**Checkpoints**:
- Is `mem_fraction_static` set reasonably
- Is `max_model_len` too large
- Check actual cache size

**Calculation Formula**:
```python
# KV Cache size per token
bytes_per_token = (
    2  # K + V
    * num_layers
    * num_kv_heads
    * head_dim
    * dtype_bytes  # 2 for fp16/bf16
)

# Total Cache size
total_cache_size = max_model_len * bytes_per_token * max_running_requests
```

## Performance Optimization Suggestions

### 1. Page Size Selection

- Small `page_size` (1): High memory utilization, but high management overhead
- Large `page_size` (16, 32): Low management overhead, but potential fragmentation

### 2. Format Selection

- `PA_BNSD`: Good compatibility, easy debugging
- `PA_NZ`: Better NPU performance, but requires specific environment variables

### 3. Pre-allocation

```python
# Pre-allocate enough cache space
--mem-fraction-static 0.85  # Reserve 85% memory for static allocation
```

## Relationship with Other Modules

```
KV Cache
├── Input: token positions (from scheduler)
├── Management: MemoryPool, req_to_token_pool
├── Consumer: Attention Backend
└── Special Handling: MLA Preprocess (compressed representation)
```
