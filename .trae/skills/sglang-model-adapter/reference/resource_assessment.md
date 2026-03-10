# Resource & Parallel Strategy Assessment

## Critical Memory Components

SGLang memory usage consists of:

| Component | Description | Key Factors |
|-----------|-------------|-------------|
| **Model Weights** | Embedding + Attention + MLP/Experts | vocab, hidden, layers, intermediate, experts |
| **KV Cache Pool** | Token-level cache for inference | layers, kv_heads, head_dim, dtype |
| **Runtime Reserved** | Activation + Graph buffers + Metadata | chunked_prefill, cuda_graph_bs, tp_size |

## Memory Estimation Formula

### Standard LLM

```
Weights = embed + attn + mlp
  embed = vocab × hidden × dtype_size
  attn  = layers × (4 × hidden² + 2 × hidden × head_dim × kv_heads) × dtype_size
  mlp   = layers × 3 × hidden × intermediate × dtype_size

KV Cache per token = 2 × layers × kv_heads × head_dim × dtype_size
Runtime Reserved ≈ 1-3 GB (depends on cuda_graph_bs, chunked_prefill_size, tp_size)
```

### MoE Model

```
Weights = embed + attn + router + experts
  router   = layers × hidden × num_experts × dtype_size
  experts  = layers × num_experts × (3 × hidden × expert_intermediate) × dtype_size
  + shared_expert if present
```

### MLA Model (DeepSeek-V2/V3)

```
Weights = embed + mla_proj + mlp
  MLA uses compressed KV: kv_lora_rank instead of full kv_heads × head_dim

KV Cache per token = layers × (kv_lora_rank + qk_rope_head_dim) × dtype_size
```

## Complete Assessment Script

```python
#!/usr/bin/env python3
"""
NPU Memory Assessment for SGLang
Run: python assess_memory.py --model-path /models/<model>
"""
import argparse
import json
import math

def estimate_model_memory(config_path: str, tp_size: int = 1, dtype: str = "fp16") -> dict:
    """Estimate model memory requirements based on SGLang actual logic."""
    with open(config_path) as f:
        cfg = json.load(f)
    
    dtype_size = 2 if dtype in ("fp16", "bf16") else 4 if dtype in ("fp32", "bf32") else 1
    
    hidden = cfg.get("hidden_size", 0)
    layers = cfg.get("num_hidden_layers", 0)
    vocab = cfg.get("vocab_size", 0)
    intermediate = cfg.get("intermediate_size", hidden * 4)
    num_heads = cfg.get("num_attention_heads", hidden)
    num_kv_heads = cfg.get("num_key_value_heads", num_heads)
    head_dim = cfg.get("head_dim", hidden // num_heads)
    
    kv_heads_per_device = max(1, num_kv_heads // tp_size)
    
    embed_weight = vocab * hidden * dtype_size
    
    q_proj = hidden * hidden * dtype_size
    k_proj = hidden * (num_kv_heads * head_dim) * dtype_size
    v_proj = hidden * (num_kv_heads * head_dim) * dtype_size
    o_proj = (num_kv_heads * head_dim) * hidden * dtype_size
    attn_per_layer = q_proj + k_proj + v_proj + o_proj
    
    mlp_per_layer = 3 * hidden * intermediate * dtype_size
    
    is_moe = cfg.get("num_experts", 0) > 0
    moe_weight = 0
    if is_moe:
        num_experts = cfg.get("num_experts", 0)
        expert_intermediate = cfg.get("moe_intermediate_size", intermediate)
        router_weight = hidden * num_experts * dtype_size
        expert_weight = num_experts * 3 * hidden * expert_intermediate * dtype_size
        shared_expert_intermediate = cfg.get("shared_expert_intermediate_size", 0)
        shared_expert_weight = 3 * hidden * shared_expert_intermediate * dtype_size if shared_expert_intermediate > 0 else 0
        moe_weight = layers * (router_weight + expert_weight + shared_expert_weight)
    
    is_mla = cfg.get("kv_lora_rank", 0) > 0
    if is_mla:
        kv_lora_rank = cfg.get("kv_lora_rank", 0)
        qk_rope_head_dim = cfg.get("qk_rope_head_dim", 0)
        kv_cache_per_token = layers * (kv_lora_rank + qk_rope_head_dim) * dtype_size
    else:
        kv_cache_per_token = 2 * layers * kv_heads_per_device * head_dim * dtype_size
    
    total_weights = embed_weight + layers * (attn_per_layer + mlp_per_layer) + moe_weight
    
    weights_per_device = total_weights / tp_size
    
    reserved_mem_gb = 1.5
    reserved_mem_gb += 0.5 * (tp_size > 1)
    reserved_mem_gb += 0.5 * is_moe
    reserved_mem_gb += 0.3 * is_mla
    
    min_kv_tokens = 8192
    min_kv_cache_gb = (kv_cache_per_token * min_kv_tokens) / (1024**3)
    
    total_min_gb = weights_per_device / (1024**3) + reserved_mem_gb + min_kv_cache_gb
    
    return {
        "model_type": "MoE" if is_moe else "MLA" if is_mla else "Standard",
        "hidden_size": hidden,
        "num_layers": layers,
        "vocab_size": vocab,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate,
        "weights_total_gb": total_weights / (1024**3),
        "weights_per_device_gb": weights_per_device / (1024**3),
        "kv_cache_per_token_bytes": kv_cache_per_token,
        "reserved_mem_gb": reserved_mem_gb,
        "min_kv_cache_gb": min_kv_cache_gb,
        "total_min_per_device_gb": total_min_gb,
        "tp_size": tp_size,
    }


def recommend_config(model_req: dict, npu_mem_gb: float, num_npus: int) -> dict:
    """Recommend optimal configuration for given hardware."""
    
    usable_mem = npu_mem_gb * 0.88
    
    if model_req["total_min_per_device_gb"] <= usable_mem:
        return {
            "tp_size": 1,
            "feasible": True,
            "mem_fraction_static": round((usable_mem - model_req["reserved_mem_gb"]) / npu_mem_gb, 3),
            "max_total_tokens_estimate": int((usable_mem - model_req["weights_per_device_gb"] - model_req["reserved_mem_gb"]) 
                                              * (1024**3) / model_req["kv_cache_per_token_bytes"]),
            "recommendation": "Single NPU sufficient"
        }
    
    min_tp = math.ceil(model_req["total_min_per_device_gb"] / usable_mem)
    tp_size = 1
    while tp_size < min_tp:
        tp_size *= 2
    
    if tp_size > num_npus:
        return {
            "tp_size": tp_size,
            "feasible": False,
            "reason": f"Need {tp_size} NPUs, only {num_npus} available"
        }
    
    recalc = estimate_model_memory(
        config_path=args.model_path if 'args' in dir() else "",
        tp_size=tp_size
    )
    
    return {
        "tp_size": tp_size,
        "feasible": True,
        "mem_fraction_static": round((usable_mem - recalc["reserved_mem_gb"]) / npu_mem_gb, 3),
        "max_total_tokens_estimate": int((usable_mem - recalc["weights_per_device_gb"] - recalc["reserved_mem_gb"]) 
                                          * (1024**3) / recalc["kv_cache_per_token_bytes"]),
        "recommendation": f"Use --tp-size {tp_size}"
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to model config.json")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--npu-mem", type=float, default=64, help="NPU memory in GB")
    parser.add_argument("--num-npus", type=int, default=8, help="Number of available NPUs")
    args = parser.parse_args()
    
    import os
    config_path = args.model_path
    if os.path.isdir(config_path):
        config_path = os.path.join(config_path, "config.json")
    
    model_req = estimate_model_memory(config_path, args.tp_size)
    
    print("=" * 60)
    print("MODEL ANALYSIS")
    print("=" * 60)
    print(f"Type: {model_req['model_type']}")
    print(f"Hidden: {model_req['hidden_size']}, Layers: {model_req['num_layers']}")
    print(f"Heads: {model_req['num_heads']} (KV: {model_req['num_kv_heads']})")
    print(f"Head Dim: {model_req['head_dim']}, Intermediate: {model_req['intermediate_size']}")
    print(f"Vocab: {model_req['vocab_size']}")
    
    print("\n" + "=" * 60)
    print("MEMORY ESTIMATION")
    print("=" * 60)
    print(f"Total Weights: {model_req['weights_total_gb']:.2f} GB")
    print(f"Weights/Device (TP={args.tp_size}): {model_req['weights_per_device_gb']:.2f} GB")
    print(f"KV Cache/Token: {model_req['kv_cache_per_token_bytes']} bytes")
    print(f"Runtime Reserved: {model_req['reserved_mem_gb']:.2f} GB")
    print(f"Min Total/Device: {model_req['total_min_per_device_gb']:.2f} GB")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    rec = recommend_config(model_req, args.npu_mem, args.num_npus)
    print(f"Feasible: {rec['feasible']}")
    print(f"TP Size: {rec['tp_size']}")
    if rec['feasible']:
        print(f"Mem Fraction Static: {rec['mem_fraction_static']}")
        print(f"Est. Max Tokens: {rec['max_total_tokens_estimate']:,}")
        print(f"Recommendation: {rec['recommendation']}")
    else:
        print(f"Reason: {rec['reason']}")
    
    print("\n" + "=" * 60)
    print("SUGGESTED COMMAND")
    print("=" * 60)
    if rec['feasible']:
        tp_flag = f"--tp-size {rec['tp_size']}" if rec['tp_size'] > 1 else ""
        mem_flag = f"--mem-fraction-static {rec['mem_fraction_static']}"
        print(f"python -m sglang.launch_server \\")
        print(f"    --model-path {args.model_path} \\")
        print(f"    --attention-backend ascend \\")
        print(f"    --device npu {tp_flag} {mem_flag} \\")
        print(f"    --port 8000")
```

## Quick Assessment Commands

```bash
# One-liner memory check
python -c "
import torch, json, sys
cfg = json.load(open(sys.argv[1])) if sys.argv[1].endswith('.json') else json.load(open(f'{sys.argv[1]}/config.json'))
h, l, v = cfg.get('hidden_size',0), cfg.get('num_hidden_layers',0), cfg.get('vocab_size',0)
w = (v*h + l*(4*h*h + 6*h*h)) * 2 / 1e9
print(f'~{w:.1f}GB weights | NPU: {torch.npu.get_device_properties(0).total_memory/1e9:.0f}GB')
" /models/<model>/config.json

# Full assessment
python assess_memory.py --model-path /models/<model> --npu-mem 64 --num-npus 8
```

## Decision Matrix

| Model Size | NPU Memory | TP Size | Notes |
|------------|------------|---------|-------|
| < 20B | 64GB | 1 | Single card, mem_fraction ~0.85 |
| 20B-70B | 64GB | 2-4 | TP required, check kv_heads divisibility |
| 70B-200B | 64GB×8 | 8 | Full node, MoE may need EP |
| > 200B | 64GB×16+ | 16+ | Multi-node, consider PP |

## Key Validation Points

Before starting server, verify:

1. **TP divisibility**: `num_kv_heads % tp_size == 0`
2. **Head count power-of-2** (for ACLGraph): `(num_heads // tp_size) & ((num_heads // tp_size) - 1) == 0`
3. **Memory headroom**: `weights_per_device < npu_mem × 0.7`

```bash
# Validate before launch
python -c "
import json, sys
cfg = json.load(open('$MODEL_PATH/config.json'))
h = cfg.get('num_attention_heads', 0)
kv = cfg.get('num_key_value_heads', h)
tp = $TP_SIZE
print(f'Heads/Device: {h//tp}, KV Heads/Device: {kv//tp}')
assert h % tp == 0, f'Heads not divisible by TP'
assert kv % tp == 0, f'KV heads not divisible by TP'
heads_per_dev = h // tp
is_pow2 = heads_per_dev & (heads_per_dev - 1) == 0
print(f'ACLGraph compatible: {is_pow2} (heads_per_device={heads_per_dev})')
"
```
