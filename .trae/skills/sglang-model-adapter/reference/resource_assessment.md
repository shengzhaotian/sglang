# Resource & Parallel Strategy Assessment

## Memory Estimation

### Formula

```
Weights (FP16) = vocab × hidden × 2 + layers × (4 × hidden² + 2 × hidden × intermediate) × 2
Activation ≈ Weights × 0.2
KV Cache = 2 × layers × heads × head_dim × context_len × batch_size × 2
Total Required = (Weights + Activation + KV Cache) × 1.2 (safety margin)
```

### Python Implementation

```python
def estimate_model_memory(config_path: str) -> dict:
    """Estimate model memory requirements."""
    import json
    with open(config_path) as f:
        cfg = json.load(f)
    
    hidden_size = cfg.get("hidden_size", 0)
    num_layers = cfg.get("num_hidden_layers", 0)
    vocab_size = cfg.get("vocab_size", 0)
    intermediate_size = cfg.get("intermediate_size", hidden_size * 4)
    num_heads = cfg.get("num_attention_heads", 0)
    
    embed_weight = vocab_size * hidden_size * 2
    attn_weight = num_layers * (4 * hidden_size * hidden_size) * 2
    mlp_weight = num_layers * (2 * hidden_size * intermediate_size) * 2
    total_weights = embed_weight + attn_weight + mlp_weight
    
    kv_per_token = 2 * num_layers * num_heads * (hidden_size // num_heads) * 2
    activation_mem = total_weights * 0.2
    
    return {
        "weights_gb": total_weights / 1e9,
        "activation_gb": activation_mem / 1e9,
        "kv_per_token_kb": kv_per_token / 1024,
        "total_min_gb": (total_weights + activation_mem) / 1e9
    }
```

## Hardware Detection

```python
def detect_npu_resources() -> dict:
    """Detect available NPU resources."""
    import torch
    
    if not torch.npu.is_available():
        return {"error": "NPU not available"}
    
    device_count = torch.npu.device_count()
    devices = []
    for i in range(device_count):
        props = torch.npu.get_device_properties(i)
        devices.append({
            "id": i,
            "name": props.name,
            "total_memory_gb": props.total_memory / 1e9
        })
    
    return {
        "device_count": device_count,
        "devices": devices,
        "total_memory_gb": sum(d["total_memory_gb"] for d in devices)
    }
```

## Parallel Strategy Recommendation

```python
def recommend_parallel_strategy(model_req: dict, hardware: dict, context_len: int = 4096) -> dict:
    """Recommend minimal parallel strategy."""
    
    single_card_mem = hardware["devices"][0]["total_memory_gb"]
    model_min_mem = model_req["total_min_gb"]
    kv_cache_gb = (model_req["kv_per_token_kb"] * context_len * 64) / 1e6
    total_required = (model_min_mem + kv_cache_gb) * 1.2
    
    if total_required <= single_card_mem * 0.9:
        return {
            "strategy": "single_card",
            "tp_size": 1,
            "feasible": True,
            "memory_usage": f"{total_required:.1f}GB / {single_card_mem:.1f}GB"
        }
    
    min_tp = int(total_required / (single_card_mem * 0.8)) + 1
    tp_size = 1
    while tp_size < min_tp:
        tp_size *= 2
    
    if tp_size <= hardware["device_count"]:
        return {
            "strategy": "tensor_parallel",
            "tp_size": tp_size,
            "feasible": True,
            "memory_usage": f"{total_required/tp_size:.1f}GB per card"
        }
    
    return {
        "strategy": "insufficient_resources",
        "feasible": False,
        "reason": f"Need {tp_size} cards, only {hardware['device_count']} available"
    }
```

## Decision Table

| Condition | Strategy | TP Size |
|-----------|----------|---------|
| Total ≤ Single Card × 0.9 | Single Card | 1 |
| Total > Single Card × 0.9 | Tensor Parallel | min power-of-2 ≥ Total/(Card×0.8) |
| TP > Available NPUs | Infeasible | - |

## Quick Commands

```bash
# Hardware check
python -c "import torch; [print(f'NPU {i}: {torch.npu.get_device_properties(i).total_memory/1e9:.1f}GB') for i in range(torch.npu.device_count())]"

# Model config check
python -c "import json; c=json.load(open('/models/<model>/config.json')); print(f'Hidden:{c.get(\"hidden_size\")} Layers:{c.get(\"num_hidden_layers\")} Vocab:{c.get(\"vocab_size\")}')"
```
