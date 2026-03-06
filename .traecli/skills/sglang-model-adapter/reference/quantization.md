# NPU Quantization Reference

## Overview

Quantization is a key technique for reducing model memory footprint and improving inference performance. SGLang supports multiple quantization schemes on NPU. This document details NPU-related quantization implementation.

## Core Files

```
python/sglang/srt/layers/quantization/
├── base_config.py                  # Quantization configuration base class
├── base_scheme.py                  # Quantization scheme base class
├── modelslim/
│   ├── modelslim.py                # ModelSlim quantization configuration
│   └── schemes/
│       ├── modelslim_w8a8_int8.py  # W8A8 INT8 scheme
│       ├── modelslim_w4a4_int4.py  # W4A4 INT4 scheme
│       ├── modelslim_w4a8_int8_moe.py  # W4A8 MoE scheme
│       └── modelslim_w8a8_int8_moe.py  # W8A8 MoE scheme
├── compressed_tensors/             # Compressed Tensors format
├── fp8.py                          # FP8 quantization
├── awq.py                          # AWQ quantization
└── gptq.py                         # GPTQ quantization

python/sglang/srt/hardware_backend/npu/quantization/
├── linear_method_npu.py            # NPU linear layer quantization method
└── fused_moe_method_npu.py         # NPU MoE quantization method
```

## Supported Quantization Types

### NPU-Specific Quantization

| Type | Weight Bit-width | Activation Bit-width | Use Case |
|------|------------------|----------------------|----------|
| W8A8 (Static) | INT8 | INT8 | Static quantization, requires calibration |
| W8A8 (Dynamic) | INT8 | INT8 (dynamic) | Dynamic quantization, no calibration needed |
| W4A4 (Dynamic) | INT4 | INT4 (dynamic) | Extreme compression |
| W4A8 MoE | INT4 | INT8 (dynamic) | MoE models |
| W8A8 MoE | INT8 | INT8 (dynamic) | MoE models |

### General Quantization (Partial NPU Support)

| Type | Description | NPU Support |
|------|-------------|-------------|
| FP8 | 8-bit floating point | ⚠️ Limited support |
| AWQ | Activation-aware | ⚠️ Limited support |
| GPTQ | Post-training | ⚠️ Limited support |
| GGUF | llama.cpp format | ❌ Not supported |

## ModelSlim Quantization

### Overview

ModelSlim is a model quantization tool provided by Huawei. SGLang supports loading ModelSlim-quantized models.

### Configuration File

ModelSlim quantized models require a `quant_model_description.json` file:

```json
{
    "model.layers.0.self_attn.q_proj.weight": "W8A8_DYNAMIC",
    "model.layers.0.self_attn.k_proj.weight": "W8A8_DYNAMIC",
    "model.layers.0.self_attn.v_proj.weight": "W8A8_DYNAMIC",
    "model.layers.0.self_attn.o_proj.weight": "W8A8_DYNAMIC",
    "model.layers.0.mlp.gate_proj.weight": "W8A8_DYNAMIC",
    "model.layers.0.mlp.up_proj.weight": "W8A8_DYNAMIC",
    "model.layers.0.mlp.down_proj.weight": "W8A8_DYNAMIC",
    "model.layers.0.input_layernorm.weight": "FLOAT",
    "model.layers.0.post_attention_layernorm.weight": "FLOAT",
    "ignore": ["lm_head"],
    "packed_modules_mapping": {}
}
```

### Quantization Type Descriptions

| Identifier | Type | Description |
|------------|------|-------------|
| `W8A8` | Static W8A8 | Requires input scale/offset |
| `W8A8_DYNAMIC` | Dynamic W8A8 | Computes activation scale at runtime |
| `W4A4_DYNAMIC` | Dynamic W4A4 | Computes activation scale at runtime |
| `FLOAT` | Not quantized | Maintains floating-point precision |

### Launch Command

```bash
python -m sglang.launch_server \
    --model-path /models/quantized-model \
    --quantization modelslim \
    --attention-backend ascend \
    --device npu
```

## W8A8 INT8 Quantization Details

### Static Quantization (W8A8)

**Weight Processing**:

```python
# In linear_method_npu.py
def process_weights_after_loading(self, layer: torch.nn.Module):
    # Transpose and convert to NPU format
    layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
    layer.weight.data = npu_format_cast(layer.weight.data)
    
    # Flatten scale and offset
    layer.weight_scale.data = layer.weight_scale.data.flatten()
    layer.weight_offset.data = layer.weight_offset.data.flatten()
    
    # Expand input scale/offset
    expanding_factor = layer.weight.data.shape[0]
    layer.aclnn_input_scale = layer.input_scale.data.repeat(expanding_factor)
    layer.aclnn_input_scale_reciprocal = 1 / layer.aclnn_input_scale
    layer.aclnn_input_offset = layer.input_offset.data.repeat(expanding_factor)
```

**Forward Computation**:

```python
def apply(self, layer, x, bias=None):
    # 1. Quantize input
    x = torch.ops.npu.npu_quantize(
        x,
        layer.aclnn_input_scale_reciprocal,
        layer.aclnn_input_offset,
        torch.qint8,
        -1,
        False,
    )
    
    # 2. Quantized matmul
    return torch.ops.npu.npu_quant_matmul(
        x,
        layer.weight,
        layer.deq_scale,
        bias=layer.quant_bias,
        output_dtype=original_dtype,
    )
```

### Dynamic Quantization (W8A8_DYNAMIC)

**Feature**: Dynamically computes activation scale at runtime, no pre-calibration required.

**Forward Computation**:

```python
def apply(self, layer, x, bias=None):
    # 1. Dynamic quantization (compute scale at runtime)
    quant_out, dynamic_scale = torch.ops.npu.npu_dynamic_quant(x)
    
    # 2. Quantized matmul
    return torch.ops.npu.npu_quant_matmul(
        quant_out,
        layer.weight,
        layer.weight_scale,
        pertoken_scale=dynamic_scale,
        bias=bias,
        output_dtype=original_dtype,
    )
```

## W4A4 INT4 Quantization Details

**Weight Processing**:

```python
def process_weights_after_loading(self, layer):
    layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
    layer.weight_scale.data = layer.weight_scale.data.flatten()
    layer.weight_offset.data = layer.weight_offset.data.flatten()
    
    # Convert to INT4 packed format
    layer.weight.data = torch.ops.npu.npu_convert_weight_to_int4pack(
        layer.weight.data.to(torch.int32)
    )
```

**Forward Computation**:

```python
def apply(self, layer, x, bias=None):
    # Dynamic quantization to INT4
    quant_out, dynamic_scale = torch.ops.npu.npu_dynamic_quant(
        x, dst_type=torch.quint4x2
    )
    
    return torch.ops.npu.npu_quant_matmul(
        quant_out,
        layer.weight,
        layer.weight_scale,
        pertoken_scale=dynamic_scale,
        bias=bias,
        output_dtype=original_dtype,
    )
```

## MoE Quantization

### W8A8 MoE

**Forward Computation**:

```python
def npu_fused_experts(hidden_states, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids, top_k):
    # 1. MoE routing initialization
    hidden_states, expanded_row_idx, expanded_expert_idx = \
        torch.ops.npu.npu_moe_init_routing(
            hidden_states, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens
        )
    
    # 2. Compute expert tokens
    expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, num_experts
    )
    
    # 3. gate_up_proj (dynamic quantization)
    hidden_states, pertoken_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w13],
        scale=[w13_scale],
        per_token_scale=[pertoken_scale],
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]
    
    # 4. swiglu activation
    hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
    
    # 5. down_proj
    hidden_states, pertoken_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        scale=[w2_scale],
        per_token_scale=[pertoken_scale],
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]
    
    return hidden_states
```

### W4A8 MoE

Similar to W8A8 MoE, but weights are in INT4 format.

## RMSNorm Quantization Adaptation

W8A8 quantization requires modifying RMSNorm to add bias:

```python
# In modelslim.py
def npu_wrapper_rmsnorm_init(func):
    def init(self, hidden_size, **extra_args):
        func(self, hidden_size, **extra_args)
        self.ignore_anti = True
        # Add bias for quantization
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
    return init

def npu_wrapper_rmsnorm_forward(func):
    def _rmsnorm_forward_oot(self, x, residual=None):
        from sgl_kernel_npu.norm.add_rmsnorm_bias import add_rmsnorm_bias
        
        if residual is not None:
            out, residual_out = add_rmsnorm_bias(
                x, residual, self.weight.data, self.bias, self.variance_epsilon
            )
            return out.to(x.dtype), residual_out
        
        out = torch.ops.npu.npu_rms_norm(x, self.weight.data, self.variance_epsilon)[0]
        out = out + self.bias
        return out.to(x.dtype)
    return _rmsnorm_forward_oot
```

## Key NPU Operators

### npu_quantize

**Purpose**: Quantize floating-point tensor to integer

```python
torch.ops.npu.npu_quantize(
    x,                    # Input tensor
    scale,                # Quantization scale
    offset,               # Quantization offset
    dst_type,             # Target type (torch.qint8, torch.quint4x2)
    axis,                 # Quantization axis
    squeeze,              # Whether to squeeze
)
```

### npu_dynamic_quant

**Purpose**: Dynamic quantization, returns quantized result and scale

```python
quant_out, dynamic_scale = torch.ops.npu.npu_dynamic_quant(
    x,                    # Input tensor
    dst_type=torch.qint8  # Target type
)
```

### npu_quant_matmul

**Purpose**: Quantized matrix multiplication

```python
torch.ops.npu.npu_quant_matmul(
    x,                    # Quantized input
    weight,               # Quantized weight
    scale,                # Dequantization scale
    pertoken_scale=None,  # Dynamic scale (optional)
    bias=None,            # Bias
    output_dtype=torch.float16,
)
```

### npu_grouped_matmul

**Purpose**: Grouped matrix multiplication (MoE)

```python
torch.ops.npu.npu_grouped_matmul(
    x=[hidden_states],
    weight=[w],
    scale=[scale],
    per_token_scale=[pertoken_scale],
    split_item=2,
    group_list_type=0,
    group_type=0,
    group_list=expert_tokens,
    output_dtype=torch.float16,
)
```

### npu_format_cast

**Purpose**: Convert to NPU-optimized format

```python
from sglang.srt.hardware_backend.npu.utils import npu_format_cast
weight = npu_format_cast(weight)  # Convert to NZ format
```

## Configuration Examples

### DeepSeek-V3 Quantized Model

```bash
export ASCEND_USE_FIA=1
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1

python -m sglang.launch_server \
    --model-path /models/deepseek-v3-w8a8 \
    --quantization modelslim \
    --tp-size 16 \
    --enable-dp-attention \
    --attention-backend ascend \
    --device npu
```

### Qwen2.5 MoE Quantized Model

```bash
python -m sglang.launch_server \
    --model-path /models/qwen2.5-moe-w4a8 \
    --quantization modelslim \
    --tp-size 8 \
    --ep-size 8 \
    --moe-a2a-backend deepep \
    --attention-backend ascend \
    --device npu
```

## Common Issue Troubleshooting

### 1. Quantization Configuration File Not Found

**Symptom**: `FileNotFoundError: quant_model_description.json`

**Solution**:
- Confirm model directory contains `quant_model_description.json`
- Check `--quantization modelslim` parameter

### 2. Weight Dimension Mismatch

**Symptom**: `RuntimeError: shape mismatch in npu_quant_matmul`

**Checkpoints**:
- Is weight correctly transposed
- Is `npu_format_cast` correctly applied
- Check `weight_scale` and `weight_offset` dimensions

### 3. Significant Accuracy Degradation

**Symptom**: Model output quality significantly degraded

**Checkpoints**:
- Confirm quantization configuration matches model
- Check if `ignore` list is correct
- Verify `FLOAT` layers are correctly skipped

### 4. Dynamic Quantization Performance Issues

**Symptom**: Dynamic quantization slower than static quantization

**Reason**: Dynamic quantization requires computing scale at runtime

**Recommendations**:
- For stable input distributions, use static quantization
- For variable input distributions, use dynamic quantization

### 5. MoE Quantization Errors

**Symptom**: `RuntimeError in npu_grouped_matmul`

**Checkpoints**:
- Is `expert_tokens` correctly computed
- Is weight format correct
- Check `group_list` parameter

## Performance Comparison

| Quantization Type | Memory Usage | Inference Speed | Accuracy Loss |
|-------------------|--------------|-----------------|---------------|
| FP16/BF16 | 100% | Baseline | None |
| W8A8 Static | ~50% | 1.5-2x | Small |
| W8A8 Dynamic | ~50% | 1.3-1.8x | Small |
| W4A4 Dynamic | ~25% | 1.2-1.5x | Moderate |

## Relationship with Other Modules

```
Quantization
├── Configuration Loading: ModelSlimConfig.from_config()
├── Weight Creation: ModelSlimLinearMethod.create_weights()
├── Weight Processing: process_weights_after_loading()
├── Forward Computation: apply() → npu_quant_matmul()
├── MoE: npu_fused_experts()
└── RMSNorm: npu_rms_norm + bias
```

## Debugging Suggestions

### 1. Check Quantization Configuration

```python
import json
with open("/models/quantized-model/quant_model_description.json") as f:
    config = json.load(f)
print(json.dumps(config, indent=2))
```

### 2. Verify Weight Format

```python
# Add in process_weights_after_loading
print(f"Weight shape: {layer.weight.shape}")
print(f"Weight dtype: {layer.weight.dtype}")
print(f"Weight scale shape: {layer.weight_scale.shape}")
```

### 3. Check Quantization Accuracy

```python
# Compare output before and after quantization
with torch.no_grad():
    output_fp16 = model_fp16(input_ids)
    output_quant = model_quant(input_ids)
    diff = (output_fp16 - output_quant).abs().mean()
    print(f"Mean absolute difference: {diff}")
```
