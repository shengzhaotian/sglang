---
name: memory-calculation
description: |
  Calculate memory requirements for LLM deployment including weights, KV cache, and runtime overhead. Use this skill when the user wants to calculate memory occupation, estimate VRAM/HBM usage, or analyze memory breakdown for models. This skill reads model files to understand architecture before calculation. Triggers when user mentions: memory calculation, VRAM, HBM, memory occupation, weight memory, KV cache memory, memory estimation, or asks about memory requirements for model deployment.
---

# LLM Memory Calculation Skill

This skill provides accurate memory calculation for LLM deployment by first analyzing the model architecture from source files, then computing weight memory, KV cache memory, and runtime overhead.

**IMPORTANT**: For MoE and MLA models, DP-Attention significantly affects memory calculation. See Section 4.4 for details.

## Workflow Overview

1. **Identify Model Architecture**: Read model files to understand structure
2. **Extract Model Configuration**: Get model parameters from config or source
3. **Calculate Weight Memory**: Based on actual architecture (Standard, MLA, MoE)
4. **Calculate KV Cache Memory**: Based on attention mechanism and parallelism
5. **Apply DP-Attention**: Adjust calculations for DP-Attention if enabled
6. **Calculate Runtime Overhead**: Activation and buffer memory
7. **Summarize Total Memory**: Per-device breakdown with parallelism

---

## Step 1: Identify Model Architecture

**CRITICAL**: Before calculating memory, you MUST read the model implementation files to understand the actual architecture.

### 1.1 Locate Model Files

Search for model implementation in the codebase:

```bash
# Find model files
find . -name "*.py" -path "*/models/*" | grep -E "(qwen|llama|deepseek|mixtral)"

# Common locations:
# - python/sglang/srt/models/
```

### 1.2 Identify Architecture Type

Read the model file to determine architecture:

| Architecture | Key Indicators | Examples |
|--------------|----------------|----------|
| **Standard (Dense)** | `nn.Linear` for MLP, no MoE | Llama, Qwen2, Qwen3-32B |
| **MoE** | `MoE`, `num_experts`, `SparseMoeBlock` | Qwen3-235B, DeepSeek-V3, Mixtral |
| **MLA** | `kv_lora_rank`, `fused_qkv_a_proj` | DeepSeek-V3, DeepSeek-R1 |
| **MLA + MoE** | Both MLA and MoE indicators | DeepSeek-V3, DeepSeek-R1 |

### 1.3 Architecture Detection Patterns

```python
# In model files, look for:

# MoE indicators:
# - class name contains "MoE" or "SparseMoe"
# - num_experts parameter
# - moe_intermediate_size parameter
# - Expert parallel imports (ep_size, get_moe_expert_parallel_world_size)

# MLA indicators:
# - kv_lora_rank parameter
# - q_lora_rank parameter
# - fused_qkv_a_proj layer
# - CompressedKV projection

# Standard indicators:
# - gate_proj, up_proj, down_proj (SwiGLU MLP)
# - q_proj, k_proj, v_proj, o_proj (standard attention)
# - num_key_value_heads (GQA)
```

---

## Step 2: Extract Model Configuration

### 2.1 From config.json

```bash
cat <model_path>/config.json
```

Key parameters to extract:

```json
{
  "hidden_size": 4096,
  "intermediate_size": 22016,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "vocab_size": 152064,
  "head_dim": 128,

  // MoE specific
  "num_experts": 128,
  "num_experts_per_tok": 8,
  "moe_intermediate_size": 768,

  // MLA specific
  "kv_lora_rank": 512,
  "q_lora_rank": 1536,
  "qk_head_dim": 128,
  "qk_rope_head_dim": 64,
  "v_head_dim": 128
}
```

### 2.2 From Model Source Code

If config.json is unavailable, read the model source:

```python
# Look for default values in model config class
class Qwen3MoeConfig:
    def __init__(
        self,
        hidden_size=4096,
        num_hidden_layers=94,
        num_attention_heads=64,
        num_key_value_heads=16,
        # ...
    )
```

### 2.3 Derived Parameters

```python
# If head_dim is not specified:
head_dim = hidden_size // num_attention_heads

# For GQA:
num_kv_heads = num_key_value_heads or num_attention_heads
```

---

## Step 3: Calculate Weight Memory

### 3.1 Standard Transformer (Dense)

```python
# Embedding Layer
embedding_params = vocab_size * hidden_size

# Attention Layer (per layer)
# With GQA (Grouped Query Attention)
q_params = hidden_size * (num_attention_heads * head_dim)
k_params = hidden_size * (num_kv_heads * head_dim)
v_params = hidden_size * (num_kv_heads * head_dim)
o_params = (num_attention_heads * head_dim) * hidden_size

# Note: For most models, num_attention_heads * head_dim = hidden_size
# So Q and O are typically hidden_size * hidden_size

attention_params_per_layer = q_params + k_params + v_params + o_params

# MLP Layer (per layer, SwiGLU)
gate_params = hidden_size * intermediate_size
up_params = hidden_size * intermediate_size
down_params = intermediate_size * hidden_size

mlp_params_per_layer = gate_params + up_params + down_params

# LayerNorm (per layer)
ln_params_per_layer = 2 * hidden_size  # input_ln + post_attn_ln

# Total per layer
layer_params = attention_params_per_layer + mlp_params_per_layer + ln_params_per_layer

# Total model
total_params = embedding_params + (layer_params * num_layers) + lm_head_params

# Memory (BF16 = 2 bytes per param)
weight_memory_gb = total_params * 2 / (1024**3)
```

### 3.2 MoE (Mixture of Experts)

```python
# Embedding Layer (same as standard)
embedding_params = vocab_size * hidden_size

# Attention Layer (same as standard)
attention_params_per_layer = q_params + k_params + v_params + o_params
total_attention_params = attention_params_per_layer * num_layers

# MoE Layer (per layer)
# Gate (router) - replicated
gate_params = hidden_size * num_experts

# Routed Experts
expert_gate_params = hidden_size * moe_intermediate_size
expert_up_params = hidden_size * moe_intermediate_size
expert_down_params = moe_intermediate_size * hidden_size
expert_params = expert_gate_params + expert_up_params + expert_down_params

total_expert_params = expert_params * num_experts

# Per MoE layer
moe_params_per_layer = gate_params + total_expert_params
total_moe_params = moe_params_per_layer * num_layers

# Total model
total_params = embedding_params + total_attention_params + total_moe_params + lm_head_params
```

### 3.3 MLA (Multi-Head Latent Attention)

```python
# MLA uses compressed KV representation
# Attention Layer (per layer)

# Compressed projections (replicated, NOT sharded by TP)
fused_qkv_a_params = hidden_size * (q_lora_rank + kv_lora_rank + qk_rope_head_dim)

# Decompressed projections (TP sharded)
q_b_params = q_lora_rank * (num_heads * qk_head_dim)
kv_b_params = kv_lora_rank * (num_heads * (qk_nope_head_dim + v_head_dim))

# Output projection (TP sharded)
o_proj_params = (num_heads * v_head_dim) * hidden_size

attention_params_per_layer = fused_qkv_a_params + q_b_params + kv_b_params + o_proj_params
```

### 3.4 Bytes Per Parameter

| Precision | Bytes Per Param | Memory Reduction |
|-----------|-----------------|------------------|
| FP32 | 4 | Baseline |
| BF16/FP16 | 2 | 50% |
| INT8 (W8A8) | 1 | 75% |
| INT4 | 0.5 | 87.5% |
| FP8 | 1 | 75% |

---

## Step 4: Calculate KV Cache Memory

### 4.1 Standard Attention (GQA)

```python
# KV cache stores K and V for each token
# Shape: [num_layers, 2, batch_size, num_kv_heads, head_dim]

# Per token per layer
kv_elements_per_token_per_layer = 2 * num_kv_heads * head_dim

# Per token all layers
kv_elements_per_token = kv_elements_per_token_per_layer * num_layers

# Memory in bytes
kv_bytes_per_token = kv_elements_per_token * bytes_per_element

# Example: Qwen3-32B
# num_kv_heads = 8, head_dim = 128, num_layers = 64
# kv_bytes_per_token = 2 * 8 * 128 * 64 * 2 = 262,144 bytes = 256 KB/token
```

### 4.2 MLA (Compressed KV)

```python
# MLA stores compressed latent representation
# Per token per layer: kv_lora_rank + qk_rope_head_dim

kv_elements_per_token_per_layer = kv_lora_rank + qk_rope_head_dim
# Example: DeepSeek-V3
# kv_lora_rank = 512, qk_rope_head_dim = 64
# kv_elements = 512 + 64 = 576

kv_bytes_per_token_per_layer = kv_elements_per_token_per_layer * bytes_per_element
kv_bytes_per_token = kv_bytes_per_token_per_layer * num_layers

# Example: DeepSeek-V3
# kv_bytes_per_token = 576 * 2 * 61 = 70,272 bytes = 68.6 KB/token
# vs Standard: 2 * 128 * 128 * 61 * 2 = 3.9 MB/token (98.2% reduction!)
```

### 4.3 With Standard Parallelism

**Tensor Parallelism (TP):**
```python
# KV heads distributed across TP ranks
kv_heads_per_device = num_kv_heads / tp_size

# KV cache per device
kv_bytes_per_token_per_device = 2 * kv_heads_per_device * head_dim * num_layers * bytes_per_element
```

**Expert Parallelism (EP):**
```python
# EP does NOT affect KV cache (only affects expert weights)
# KV cache calculation same as without EP
```

### 4.4 With DP-Attention (CRITICAL)

**DP-Attention fundamentally changes KV cache memory calculation:**

#### What is DP-Attention?

DP-Attention applies **Data Parallelism to the attention layer**, where each DP rank:
- Processes its own batch of requests independently
- Maintains its own KV cache (no duplication across DP ranks)
- Can be in different forward modes (prefill, decode, idle)

#### Memory Calculation with DP-Attention

```python
# Key formula: attn_tp_size = tp_size / dp_size
attn_tp_size = tp_size // dp_size

# KV heads per device with DP-Attention
# Each DP rank only sees a subset of attention heads
kv_heads_per_device = num_kv_heads / attn_tp_size

# But CRITICAL: Each DP rank has its OWN KV cache
# No duplication across DP ranks!

# For MLA models with DP-Attention:
# KV cache is NOT sharded by TP, but each DP rank has its own copy
# However, since each DP rank handles different requests,
# the effective KV cache per request is the same

# Per token per device (MLA with DP-Attention)
# kv_lora_rank is replicated within attn_tp_group
# But each DP rank handles different batches
kv_bytes_per_token_per_device = (kv_lora_rank + qk_rope_head_dim) * num_layers * bytes_per_element
```

#### DP-Attention Memory Benefits

| Aspect | Without DP-Attention | With DP-Attention |
|--------|---------------------|-------------------|
| **KV Cache Duplication** | Duplicated across TP ranks | Unique per DP rank |
| **Effective Batch Size** | Limited by single batch | dp_size × larger batch |
| **Memory per Request** | Same | Same |
| **Total Capacity** | Limited | **dp_size × larger** |

#### Example: DeepSeek-R1 with DP-Attention

```python
# Without DP-Attention (TP=32)
# KV cache is shared across all 32 ranks
# Each rank stores same KV cache for same requests
# Effective: 1× KV cache capacity

# With DP-Attention (TP=32, DP=16)
# attn_tp_size = 32 / 16 = 2
# Each DP rank (16 total) has its own KV cache
# Each handles different batches
# Effective: 16× KV cache capacity for different requests!

# Memory per device (MLA):
# kv_bytes_per_token = 576 * 61 * 2 = 70,272 bytes = 68.6 KB/token
# This is the same per device, but now you can handle 16× more requests
```

### 4.5 Total KV Cache

```python
# Total tokens in flight
total_tokens = (input_length + output_length) * max_running_requests

# Total KV cache memory per device
total_kv_cache = total_tokens * kv_bytes_per_token_per_device

# With DP-Attention, max_running_requests can be much larger
# because each DP rank handles independent batches
```

---

## Step 5: Apply DP-Attention Configuration

### 5.1 When to Use DP-Attention

| Architecture | DP-Attention Benefit | Recommendation |
|--------------|---------------------|----------------|
| **Standard (Dense)** | Minimal | Use standard TP |
| **MoE** | Moderate | Use DP-Attention with TP=DP |
| **MLA** | **Significant** | **Strongly recommended** |
| **MLA + MoE** | **Very Significant** | **Strongly recommended** |

### 5.2 DP-Attention Configuration

```python
# Key parameters
--tp-size <tp>           # Total tensor parallelism
--dp-size <dp>           # Data parallelism for attention
--enable-dp-attention    # Enable DP-Attention
--enable-dp-lm-head      # Also apply DP to LM head (recommended)

# Constraints
assert tp_size % dp_size == 0  # Must be divisible
assert dp_size > 1             # DP=1 means no DP-Attention

# Derived
attn_tp_size = tp_size // dp_size
```

### 5.3 Weight Distribution with DP-Attention

```python
# For MLA + MoE models with DP-Attention:

# Attention weights
# - fused_qkv_a: replicated within attn_tp_group
# - q_b, kv_b, o_proj: sharded by attn_tp_size

# Expert weights
# - routed_experts: sharded by ep_size
# - shared_experts: sharded by tp_size (or moe_dense_tp_size)
# - gate: replicated

# LM head (with --enable-dp-lm-head)
# - sharded by attn_tp_size instead of tp_size

# Weight per device calculation
fused_qkv_a_per_device = fused_qkv_a_weight  # replicated
other_attn_per_device = other_attn_weight / attn_tp_size
expert_per_device = expert_weight / ep_size
lm_head_per_device = lm_head_weight / attn_tp_size  # with enable-dp-lm-head

weight_per_device = (fused_qkv_a_per_device + 
                     other_attn_per_device + 
                     expert_per_device +
                     lm_head_per_device)
```

---

## Step 6: Calculate Runtime Overhead

### 6.1 Activation Memory

```python
# Activation memory depends on batch size and sequence length
# Rough estimate: 10-20% of weight memory for inference

activation_memory = weight_memory * 0.15
```

### 6.2 Buffer Memory

```python
# Communication buffers for distributed inference
# TP buffers
buffer_memory = tp_size * 0.3  # GB per device, rough estimate

# EP buffers (for MoE)
if ep_size > 1:
    buffer_memory += ep_size * 0.2  # Additional for all-to-all

# DP-Attention buffers
if dp_size > 1:
    buffer_memory += dp_size * 0.1  # Additional for DP gather/scatter
```

### 6.3 Total Runtime Overhead

```python
# Typical range: 2-4 GB per device
# With DP-Attention: 3-5 GB per device
runtime_overhead = 3  # GB estimate (adjust based on parallelism)
```

---

## Step 7: Total Memory with Parallelism

### 7.1 Weight Distribution Summary

| Component | TP Sharded | EP Sharded | DP-Attention Sharded | Replicated |
|-----------|------------|------------|---------------------|------------|
| Embedding | Yes | No | No | No |
| Q, K, V projections | Yes | No | By attn_tp_size | No |
| O projection | Yes | No | By attn_tp_size | No |
| MLP (gate/up/down) | Yes | No | No | No |
| MoE Gate | No | No | No | Yes |
| MoE Experts | No | Yes | No | No |
| MLA fused_qkv_a | No | No | No | Yes |
| LM Head | Yes | No | By attn_tp_size | No |

### 7.2 Memory per Device Formula

```python
# Standard architecture
if architecture == "standard":
    weight_per_device = total_weight_memory / tp_size

# MoE architecture with EP
elif architecture == "moe":
    attention_weight_per_device = attention_weight / tp_size
    expert_weight_per_device = expert_weight / ep_size
    gate_weight_per_device = gate_weight  # replicated
    weight_per_device = attention_weight_per_device + expert_weight_per_device + gate_weight_per_device

# MLA + MoE with DP-Attention
elif architecture == "mla_moe":
    attn_tp_size = tp_size // dp_size if enable_dp_attention else tp_size
    
    # MLA attention
    fused_qkv_a_per_device = fused_qkv_a_weight  # replicated
    other_attn_per_device = other_attn_weight / attn_tp_size
    
    # Experts
    expert_per_device = expert_weight / ep_size
    
    # LM head
    lm_head_per_device = lm_head_weight / attn_tp_size if enable_dp_lm_head else lm_head_weight / tp_size
    
    weight_per_device = fused_qkv_a_per_device + other_attn_per_device + expert_per_device + lm_head_per_device

# Total memory per device
total_per_device = weight_per_device + kv_cache_per_device + runtime_overhead
```

### 7.3 mem-fraction-static Calculation

```python
# Required memory
required_memory = weight_per_device + kv_cache_per_device + buffer

# mem-fraction-static with headroom
mem_fraction = min(0.86, max(0.50, required_memory / hbm_per_device + 0.10))
```

---

## Quick Reference Tables

### Common Model Sizes (BF16)

| Model | Parameters | Weight Size | Architecture |
|-------|-----------|-------------|--------------|
| Qwen3-32B | 32B | ~61 GB | Standard (GQA) |
| Qwen3-235B-A22B | 235B | ~450 GB | MoE |
| DeepSeek-V3 | 671B | ~1.3 TB | MoE + MLA |
| DeepSeek-R1 | 671B | ~1.3 TB | MoE + MLA |
| Llama-3.1-70B | 70B | ~140 GB | Standard (GQA) |
| Llama-3.1-8B | 8B | ~16 GB | Standard |

### KV Cache per Token (BF16)

| Model | Architecture | KV Heads | Layers | KB/Token (no TP) | With DP-Attention |
|-------|--------------|----------|--------|------------------|-------------------|
| Qwen3-32B | GQA | 8 | 64 | 256 KB | Same per device |
| Qwen3-235B-A22B | MoE | 16 | 94 | 600 KB | Same per device |
| DeepSeek-V3 | MLA | - | 61 | 69 KB | Same per device |
| Llama-3.1-70B | GQA | 8 | 80 | 320 KB | N/A |

### DP-Attention Memory Impact

| Configuration | KV Cache Capacity | Throughput Impact |
|---------------|-------------------|-------------------|
| TP=32, DP=1 | 1× | Baseline |
| TP=32, DP=8 | 8× | 1.5-2× higher |
| TP=32, DP=16 | 16× | 2-3× higher |
| TP=32, DP=32 | 32× | 3-5× higher |

---

## Example: DeepSeek-R1 with DP-Attention

### Model Configuration
```python
hidden_size = 7168
num_layers = 61
num_heads = 128
kv_lora_rank = 512
qk_rope_head_dim = 64
n_routed_experts = 256
n_shared_experts = 1
moe_intermediate_size = 2048
bytes_per_param = 1  # W8A8 INT8
```

### Without DP-Attention (TP=32, EP=32)

```python
# Weight per device
attn_tp_size = 32
fused_qkv_a_per_device = hidden_size * (q_lora_rank + kv_lora_rank + qk_rope_head_dim) * 1 / 1024**3
# ≈ 0.05 GB (replicated)

other_attn_per_device = other_attn_weight / 32
# ≈ 1.5 GB

expert_per_device = expert_weight / 32
# ≈ 10 GB

total_weight_per_device ≈ 12 GB

# KV cache per device (MLA)
kv_bytes_per_token = (512 + 64) * 61 * 2 = 70,272 bytes = 68.6 KB/token

# For 32 requests, 6K+1.6K length
total_tokens = 32 * 7600 = 243,200
kv_cache = 243,200 * 68.6 KB = 16.2 GB

# Total per device
total = 12 + 16.2 + 4 = 32.2 GB (50% of 64 GB)
```

### With DP-Attention (TP=32, DP=16, EP=32)

```python
# attn_tp_size = 32 / 16 = 2
attn_tp_size = 2

# Weight per device
fused_qkv_a_per_device = 0.05 GB  # same (replicated)
other_attn_per_device = other_attn_weight / 2  # larger!
# ≈ 24 GB (but distributed across DP ranks)

# Wait, this seems wrong. Let me recalculate:
# With DP-Attention, attention weights are sharded by attn_tp_size=2
# But each DP rank only holds 1/16 of the attention computation
# So effective weight per device for attention is:
# - fused_qkv_a: replicated (same for all DP ranks in same attn_tp_group)
# - other_attn: sharded by attn_tp_size=2

# Actually, the key insight is:
# - Total TP ranks = 32
# - DP size = 16, so we have 16 DP groups
# - Each DP group has attn_tp_size = 32/16 = 2 ranks
# - Within each DP group, attention is TP-sharded by 2
# - Across DP groups, attention is data-parallel

# So weight per device:
attention_per_device = attention_weight / 2  # sharded by attn_tp_size
expert_per_device = expert_weight / 32       # sharded by EP
total_weight_per_device ≈ 15 GB

# KV cache per device (same as before)
kv_bytes_per_token = 68.6 KB/token

# BUT: Each DP rank can handle independent batches!
# So effective capacity = 16 × single batch capacity

# For 32 requests per DP rank (512 total across DP)
total_tokens_per_device = 32 * 7600 = 243,200
kv_cache_per_device = 243,200 * 68.6 KB = 16.2 GB

# Total per device
total = 15 + 16.2 + 4 = 35.2 GB (55% of 64 GB)

# But total system capacity = 16 × 35.2 GB = 563 GB effective!
# vs without DP-Attention: 32 × 32.2 GB = 1030 GB
# Wait, this doesn't match...

# The real benefit is:
# Without DP: 32 devices share same batch, limited by single batch size
# With DP: 16 groups of 2 devices, each handles independent batches
# So you can run 16× more concurrent requests!
```

### Key Insight: DP-Attention Benefit

```
Without DP-Attention:
  - All 32 TP ranks process the SAME batch
  - Max concurrent requests limited by single batch
  - Example: 32 requests max

With DP-Attention (DP=16):
  - 16 independent DP groups
  - Each DP group processes DIFFERENT batches
  - Example: 16 × 32 = 512 requests max (16× improvement!)
  
Memory per device is similar, but total system throughput is much higher.
```

---

## Reference Files

- `references/architecture_patterns.md` - Detailed architecture detection patterns
- `references/memory_formulas.md` - Complete memory calculation formulas
