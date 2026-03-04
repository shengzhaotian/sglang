# Memory Calculation Formulas

This document provides complete memory calculation formulas for different model architectures.

## Weight Memory Formulas

### Standard Transformer

```
Total Params = Embedding + Attention + MLP + LayerNorm + LM_Head

Embedding = vocab_size × hidden_size

Attention (per layer) = Q + K + V + O
  Q = hidden_size × num_heads × head_dim
  K = hidden_size × num_kv_heads × head_dim
  V = hidden_size × num_kv_heads × head_dim
  O = num_heads × head_dim × hidden_size

MLP (per layer, SwiGLU) = Gate + Up + Down
  Gate = hidden_size × intermediate_size
  Up = hidden_size × intermediate_size
  Down = intermediate_size × hidden_size

LayerNorm (per layer) = 2 × hidden_size

LM_Head = vocab_size × hidden_size (if not tied)

Weight Memory = Total Params × bytes_per_param
```

### MoE Transformer

```
Total Params = Embedding + Attention + MoE + LayerNorm + LM_Head

Embedding = vocab_size × hidden_size

Attention (per layer) = Same as standard

MoE (per layer) = Gate + Experts
  Gate = hidden_size × num_experts (replicated)
  Expert = (Gate + Up + Down) × num_experts
    Expert_Gate = hidden_size × moe_intermediate_size
    Expert_Up = hidden_size × moe_intermediate_size
    Expert_Down = moe_intermediate_size × hidden_size

LayerNorm (per layer) = 2 × hidden_size

LM_Head = vocab_size × hidden_size
```

### MLA Transformer

```
Total Params = Embedding + MLA_Attention + MLP + LayerNorm + LM_Head

Embedding = vocab_size × hidden_size

MLA_Attention (per layer) = fused_qkv_a + q_b + kv_b + o_proj
  fused_qkv_a = hidden_size × (q_lora_rank + kv_lora_rank + qk_rope_head_dim) [replicated]
  q_b = q_lora_rank × num_heads × qk_head_dim
  kv_b = kv_lora_rank × num_heads × (qk_nope_head_dim + v_head_dim)
  o_proj = num_heads × v_head_dim × hidden_size

MLP = Same as standard

LayerNorm = Same as standard

LM_Head = vocab_size × hidden_size
```

### MLA + MoE Transformer (DeepSeek-V3)

```
Total Params = Embedding + MLA_Attention + MoE + Shared_Experts + LayerNorm + LM_Head

MLA_Attention = Same as MLA

MoE (per layer) = Gate + Routed_Experts
  Gate = hidden_size × n_routed_experts
  Routed_Expert = (Gate + Up + Down) × n_routed_experts

Shared_Experts (per layer) = Gate + Up + Down
  shared_intermediate_size = moe_intermediate_size × n_shared_experts
  Shared_Gate = hidden_size × shared_intermediate_size
  Shared_Up = hidden_size × shared_intermediate_size
  Shared_Down = shared_intermediate_size × hidden_size
```

---

## KV Cache Memory Formulas

### Standard Attention (GQA)

```
KV elements per token per layer = 2 × num_kv_heads × head_dim

KV elements per token = KV elements per layer × num_layers

KV memory per token = KV elements per token × bytes_per_element

With TP:
  kv_heads_per_device = num_kv_heads / tp_size
  KV memory per token per device = 2 × kv_heads_per_device × head_dim × num_layers × bytes_per_element
```

### MLA (Compressed)

```
KV elements per token per layer = kv_lora_rank + qk_rope_head_dim

KV elements per token = KV elements per layer × num_layers

KV memory per token = KV elements per token × bytes_per_element

Note: MLA KV cache is typically replicated (not sharded by TP)
```

### Total KV Cache

```
Total tokens = (input_length + output_length) × max_running_requests

Total KV cache = Total tokens × KV memory per token per device
```

---

## DP-Attention Memory Formulas

### What is DP-Attention?

DP-Attention applies Data Parallelism to attention layers, where:
- Each DP rank processes its own batch independently
- Each DP rank maintains its own KV cache (no duplication)
- attn_tp_size = tp_size / dp_size

### KV Cache with DP-Attention

```
# Key formula
attn_tp_size = tp_size / dp_size

# For Standard Attention with DP-Attention
kv_heads_per_device = num_kv_heads / attn_tp_size
KV memory per token per device = 2 × kv_heads_per_device × head_dim × num_layers × bytes_per_element

# For MLA with DP-Attention
# KV cache is replicated within attn_tp_group
# But each DP rank handles different batches
KV memory per token per device = (kv_lora_rank + qk_rope_head_dim) × num_layers × bytes_per_element

# Key insight: Memory per device is similar, but total system capacity is dp_size × larger
```

### Weight Distribution with DP-Attention

```
# Attention weights
fused_qkv_a_per_device = fused_qkv_a_weight  # replicated
other_attn_per_device = other_attn_weight / attn_tp_size  # sharded by attn_tp_size

# Expert weights (unchanged)
expert_per_device = expert_weight / ep_size  # sharded by EP
gate_per_device = gate_weight  # replicated

# LM head (with --enable-dp-lm-head)
lm_head_per_device = lm_head_weight / attn_tp_size  # sharded by attn_tp_size

# Total weight per device
total_weight_per_device = fused_qkv_a_per_device + other_attn_per_device 
                        + expert_per_device + gate_per_device + lm_head_per_device
```

### DP-Attention Memory Benefits

```
# Without DP-Attention
# - All TP ranks process the SAME batch
# - KV cache duplicated across TP ranks
# - Limited by single batch size

# With DP-Attention
# - Each DP rank processes DIFFERENT batches
# - KV cache unique per DP rank
# - Effective capacity = dp_size × single batch capacity

# Example: TP=32, DP=16
# - attn_tp_size = 32 / 16 = 2
# - 16 independent DP groups
# - Each handles different requests
# - 16× more concurrent requests possible
```

---

## Parallelism Distribution Formulas

### Tensor Parallelism (TP)

```
Weight per device = Total weight / tp_size

For components that are TP-sharded:
  - Embedding: vocab_size / tp_size
  - Q, K, V projections: heads / tp_size
  - O projection: heads / tp_size
  - MLP gate/up: intermediate_size / tp_size
  - MLP down: intermediate_size / tp_size
  - LM Head: vocab_size / tp_size

For components that are replicated:
  - LayerNorm: full size
  - MoE Gate: full size
  - MLA fused_qkv_a: full size
```

### Expert Parallelism (EP)

```
Expert weight per device = Total expert weight / ep_size

Experts per device = num_experts / ep_size

For MoE with EP:
  - Routed experts: sharded by EP
  - Gate: replicated
  - Attention: sharded by TP
  - Shared experts (if any): sharded by TP
```

### DP-Attention Parallelism

```
attn_tp_size = tp_size / dp_size

For components with DP-Attention:
  - fused_qkv_a: replicated (same as without DP)
  - q_b, kv_b, o_proj: sharded by attn_tp_size
  - LM head (with enable-dp-lm-head): sharded by attn_tp_size

Constraints:
  - tp_size % dp_size == 0
  - dp_size > 1
```

### Combined TP + EP + DP-Attention

```
Weight per device = TP_sharded_weight / attn_tp_size 
                  + EP_sharded_weight / ep_size 
                  + Replicated_weight

Example for MLA + MoE with DP-Attention:
  attn_tp_size = tp_size / dp_size
  
  # Attention
  fused_qkv_a_per_device = fused_qkv_a_weight  # replicated
  other_attn_per_device = other_attn_weight / attn_tp_size
  
  # Experts
  expert_per_device = expert_weight / ep_size
  gate_per_device = gate_weight  # replicated
  
  # LM head
  lm_head_per_device = lm_head_weight / attn_tp_size  # with enable-dp-lm-head
  
  total_per_device = fused_qkv_a_per_device + other_attn_per_device
                   + expert_per_device + gate_per_device + lm_head_per_device
```

---

## Runtime Memory Formulas

### Activation Memory

```
Activation memory ≈ weight_memory × 0.10 to 0.20

For inference, activation memory is relatively small.
```

### Buffer Memory

```
TP buffer ≈ tp_size × 0.3 GB

EP buffer ≈ ep_size × 0.2 GB (for all-to-all communication)

DP-Attention buffer ≈ dp_size × 0.1 GB (for gather/scatter)

Total buffer ≈ TP buffer + EP buffer + DP buffer
```

### Total Runtime Overhead

```
Runtime overhead = Activation + Buffer + Framework overhead

Typical range: 2-4 GB per device
With DP-Attention: 3-5 GB per device
```

---

## mem-fraction-static Formula

```
Required memory = Weight per device + KV cache per device + Buffer

mem_fraction = Required memory / HBM per device

mem_fraction_static = min(0.86, max(0.50, mem_fraction + 0.10))

The +0.10 provides headroom for dynamic allocations.
```

---

## Quick Calculation Examples

### Qwen3-32B (BF16, TP=8)

```
# Config
hidden_size = 4096
intermediate_size = 27648
num_layers = 64
num_heads = 40
num_kv_heads = 8
vocab_size = 152064

# Weight calculation
embedding = 152064 × 4096 × 2 = 1.16 GB
attention = (4096 × 40 × 128 + 4096 × 8 × 128 × 2 + 40 × 128 × 4096) × 64 × 2
          = (20.9M + 8.4M + 8.4M + 20.9M) × 64 × 2 = 7.5 GB
mlp = (4096 × 27648 × 3) × 64 × 2 = 41.5 GB
lm_head = 1.16 GB
total = 51.3 GB

# Per device (TP=8)
weight_per_device = 51.3 / 8 = 6.4 GB

# KV cache
kv_heads_per_device = 8 / 8 = 1
kv_per_token = 2 × 1 × 128 × 64 × 2 = 32 KB/token

# For 32 requests, 6K total length
total_tokens = 6144 × 32 = 196,608
kv_cache = 196,608 × 32 KB = 6.1 GB

# Total per device
total = 6.4 + 6.1 + 3 = 15.5 GB (24% of 64 GB)
```

### Qwen3-235B-A22B with DP-Attention (W8A8, TP=16, DP=16)

```
# Config
hidden_size = 4096
num_layers = 94
num_heads = 64
num_kv_heads = 16
num_experts = 128
moe_intermediate_size = 768
vocab_size = 152064

# With DP-Attention
attn_tp_size = 16 / 16 = 1

# Weight calculation (INT8 = 1 byte)
embedding = 152064 × 4096 × 1 = 0.58 GB
attention = (4096 × 64 × 128 + 4096 × 16 × 128 × 2 + 64 × 128 × 4096) × 94 × 1
          = 7.9 GB
experts = (4096 × 768 × 3) × 128 × 94 × 1 = 113.2 GB
gate = 4096 × 128 × 94 × 1 = 0.05 GB
lm_head = 0.58 GB
total = 122.3 GB

# Per device (attn_tp_size=1 for attention, EP not used here)
attention_per_device = 7.9 / 1 = 7.9 GB  # Each DP rank has full attention
experts_per_device = 113.2 / 16 = 7.1 GB  # Experts distributed by DP
embedding_per_device = 0.58 / 1 = 0.58 GB
gate_per_device = 0.05 GB  # replicated
total_per_device = 15.6 GB

# KV cache (attn_tp_size=1, so no sharding)
kv_heads_per_device = 16 / 1 = 16
kv_per_token = 2 × 16 × 128 × 94 × 2 = 758 KB/token

# For 32 requests per DP rank (512 total across DP)
total_tokens_per_device = 4096 × 32 = 131,072
kv_cache = 131,072 × 758 KB = 97.6 GB  # Exceeds capacity!

# Need to reduce requests per DP rank
max_kv = 64 × 0.75 - 15.6 - 3 = 29.4 GB
max_tokens = 29.4 GB / 758 KB = 38,786 tokens
max_requests_per_dp = 38,786 / 4096 = 9 requests

# Total across all DP ranks = 9 × 16 = 144 requests
```

### DeepSeek-V3 with DP-Attention (W8A8, TP=32, DP=8, EP=32)

```
# Config (simplified)
hidden_size = 7168
num_layers = 61 (3 dense + 58 MoE)
num_heads = 128
kv_lora_rank = 512
qk_rope_head_dim = 64
n_routed_experts = 256
n_shared_experts = 1
moe_intermediate_size = 2048

# With DP-Attention
attn_tp_size = 32 / 8 = 4

# MLA KV cache (compressed)
kv_per_token = (512 + 64) × 61 × 2 = 70,272 bytes = 69 KB/token
# Same per device regardless of DP

# Weight per device
# - fused_qkv_a: replicated
# - other_attn: sharded by attn_tp_size=4
# - experts: sharded by EP=32
# - lm_head: sharded by attn_tp_size=4

# Key benefit: 8 DP groups can handle independent batches
# 8× more concurrent requests than without DP-Attention
```

---

## DP-Attention Decision Guide

### When to Use DP-Attention

| Architecture | DP-Attention Benefit | Recommendation |
|--------------|---------------------|----------------|
| Standard (Dense) | Minimal | Use standard TP |
| MoE | Moderate | Use DP-Attention with TP=DP |
| MLA | **Significant** | **Strongly recommended** |
| MLA + MoE | **Very Significant** | **Strongly recommended** |

### DP-Attention Configuration Examples

```
# Qwen3-235B-A22B (MoE)
--tp-size 16 --dp-size 16 --enable-dp-attention --enable-dp-lm-head

# DeepSeek-R1 (MLA + MoE) - Low Latency
--tp-size 32 --dp-size 16 --ep-size 32 --enable-dp-attention --enable-dp-lm-head --moe-dense-tp-size 1

# DeepSeek-R1 (MLA + MoE) - High Throughput
--tp-size 32 --dp-size 32 --ep-size 32 --enable-dp-attention --enable-dp-lm-head --moe-dense-tp-size 1
```
