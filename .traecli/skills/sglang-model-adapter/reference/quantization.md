# NPU Quantization Reference

## 概述

量化是降低模型内存占用和提升推理性能的关键技术。SGLang 在 NPU 上支持多种量化方案，本文档详细介绍 NPU 相关的量化实现。

## 核心文件

```
python/sglang/srt/layers/quantization/
├── base_config.py                  # 量化配置基类
├── base_scheme.py                  # 量化方案基类
├── modelslim/
│   ├── modelslim.py                # ModelSlim 量化配置
│   └── schemes/
│       ├── modelslim_w8a8_int8.py  # W8A8 INT8 方案
│       ├── modelslim_w4a4_int4.py  # W4A4 INT4 方案
│       ├── modelslim_w4a8_int8_moe.py  # W4A8 MoE 方案
│       └── modelslim_w8a8_int8_moe.py  # W8A8 MoE 方案
├── compressed_tensors/             # Compressed Tensors 格式
├── fp8.py                          # FP8 量化
├── awq.py                          # AWQ 量化
└── gptq.py                         # GPTQ 量化

python/sglang/srt/hardware_backend/npu/quantization/
├── linear_method_npu.py            # NPU 线性层量化方法
└── fused_moe_method_npu.py         # NPU MoE 量化方法
```

## 支持的量化类型

### NPU 专用量化

| 类型 | 权重位宽 | 激活位宽 | 适用场景 |
|------|----------|----------|----------|
| W8A8 (Static) | INT8 | INT8 | 静态量化，需要校准 |
| W8A8 (Dynamic) | INT8 | INT8 (动态) | 动态量化，无需校准 |
| W4A4 (Dynamic) | INT4 | INT4 (动态) | 极致压缩 |
| W4A8 MoE | INT4 | INT8 (动态) | MoE 模型 |
| W8A8 MoE | INT8 | INT8 (动态) | MoE 模型 |

### 通用量化 (NPU 部分支持)

| 类型 | 说明 | NPU 支持 |
|------|------|----------|
| FP8 | 8-bit 浮点 | ⚠️ 有限支持 |
| AWQ | Activation-aware | ⚠️ 有限支持 |
| GPTQ | Post-training | ⚠️ 有限支持 |
| GGUF | llama.cpp 格式 | ❌ 不支持 |

## ModelSlim 量化

### 概述

ModelSlim 是华为提供的模型量化工具，SGLang 支持加载 ModelSlim 量化后的模型。

### 配置文件

ModelSlim 量化模型需要 `quant_model_description.json` 文件：

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

### 量化类型说明

| 标识 | 类型 | 说明 |
|------|------|------|
| `W8A8` | 静态 W8A8 | 需要输入 scale/offset |
| `W8A8_DYNAMIC` | 动态 W8A8 | 运行时计算激活 scale |
| `W4A4_DYNAMIC` | 动态 W4A4 | 运行时计算激活 scale |
| `FLOAT` | 不量化 | 保持浮点精度 |

### 启动命令

```bash
python -m sglang.launch_server \
    --model-path /models/quantized-model \
    --quantization modelslim \
    --attention-backend ascend \
    --device npu
```

## W8A8 INT8 量化详解

### 静态量化 (W8A8)

**权重处理**:

```python
# 在 linear_method_npu.py 中
def process_weights_after_loading(self, layer: torch.nn.Module):
    # 转置并转换为 NPU 格式
    layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
    layer.weight.data = npu_format_cast(layer.weight.data)
    
    # 展开 scale 和 offset
    layer.weight_scale.data = layer.weight_scale.data.flatten()
    layer.weight_offset.data = layer.weight_offset.data.flatten()
    
    # 扩展输入 scale/offset
    expanding_factor = layer.weight.data.shape[0]
    layer.aclnn_input_scale = layer.input_scale.data.repeat(expanding_factor)
    layer.aclnn_input_scale_reciprocal = 1 / layer.aclnn_input_scale
    layer.aclnn_input_offset = layer.input_offset.data.repeat(expanding_factor)
```

**前向计算**:

```python
def apply(self, layer, x, bias=None):
    # 1. 量化输入
    x = torch.ops.npu.npu_quantize(
        x,
        layer.aclnn_input_scale_reciprocal,
        layer.aclnn_input_offset,
        torch.qint8,
        -1,
        False,
    )
    
    # 2. 量化矩阵乘
    return torch.ops.npu.npu_quant_matmul(
        x,
        layer.weight,
        layer.deq_scale,
        bias=layer.quant_bias,
        output_dtype=original_dtype,
    )
```

### 动态量化 (W8A8_DYNAMIC)

**特点**: 运行时动态计算激活的 scale，无需预先校准。

**前向计算**:

```python
def apply(self, layer, x, bias=None):
    # 1. 动态量化 (运行时计算 scale)
    quant_out, dynamic_scale = torch.ops.npu.npu_dynamic_quant(x)
    
    # 2. 量化矩阵乘
    return torch.ops.npu.npu_quant_matmul(
        quant_out,
        layer.weight,
        layer.weight_scale,
        pertoken_scale=dynamic_scale,
        bias=bias,
        output_dtype=original_dtype,
    )
```

## W4A4 INT4 量化详解

**权重处理**:

```python
def process_weights_after_loading(self, layer):
    layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
    layer.weight_scale.data = layer.weight_scale.data.flatten()
    layer.weight_offset.data = layer.weight_offset.data.flatten()
    
    # 转换为 INT4 packed 格式
    layer.weight.data = torch.ops.npu.npu_convert_weight_to_int4pack(
        layer.weight.data.to(torch.int32)
    )
```

**前向计算**:

```python
def apply(self, layer, x, bias=None):
    # 动态量化为 INT4
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

## MoE 量化

### W8A8 MoE

**前向计算**:

```python
def npu_fused_experts(hidden_states, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids, top_k):
    # 1. MoE 路由初始化
    hidden_states, expanded_row_idx, expanded_expert_idx = \
        torch.ops.npu.npu_moe_init_routing(
            hidden_states, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens
        )
    
    # 2. 计算 expert tokens
    expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, num_experts
    )
    
    # 3. gate_up_proj (动态量化)
    hidden_states, pertoken_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w13],
        scale=[w13_scale],
        per_token_scale=[pertoken_scale],
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]
    
    # 4. swiglu 激活
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

与 W8A8 MoE 类似，但权重为 INT4 格式。

## RMSNorm 量化适配

W8A8 量化需要修改 RMSNorm 以添加 bias：

```python
# 在 modelslim.py 中
def npu_wrapper_rmsnorm_init(func):
    def init(self, hidden_size, **extra_args):
        func(self, hidden_size, **extra_args)
        self.ignore_anti = True
        # 添加 bias 用于量化
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

## 关键 NPU 算子

### npu_quantize

**用途**: 将浮点 tensor 量化为整数

```python
torch.ops.npu.npu_quantize(
    x,                    # 输入 tensor
    scale,                # 量化 scale
    offset,               # 量化 offset
    dst_type,             # 目标类型 (torch.qint8, torch.quint4x2)
    axis,                 # 量化轴
    squeeze,              # 是否压缩
)
```

### npu_dynamic_quant

**用途**: 动态量化，返回量化结果和 scale

```python
quant_out, dynamic_scale = torch.ops.npu.npu_dynamic_quant(
    x,                    # 输入 tensor
    dst_type=torch.qint8  # 目标类型
)
```

### npu_quant_matmul

**用途**: 量化矩阵乘法

```python
torch.ops.npu.npu_quant_matmul(
    x,                    # 量化输入
    weight,               # 量化权重
    scale,                # 反量化 scale
    pertoken_scale=None,  # 动态 scale (可选)
    bias=None,            # 偏置
    output_dtype=torch.float16,
)
```

### npu_grouped_matmul

**用途**: 分组矩阵乘法 (MoE)

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

**用途**: 转换为 NPU 优化格式

```python
from sglang.srt.hardware_backend.npu.utils import npu_format_cast
weight = npu_format_cast(weight)  # 转换为 NZ 格式
```

## 配置示例

### DeepSeek-V3 量化模型

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

### Qwen2.5 MoE 量化模型

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

## 常见问题排查

### 1. 量化配置文件未找到

**症状**: `FileNotFoundError: quant_model_description.json`

**解决方案**:
- 确认模型目录包含 `quant_model_description.json`
- 检查 `--quantization modelslim` 参数

### 2. 权重维度不匹配

**症状**: `RuntimeError: shape mismatch in npu_quant_matmul`

**检查点**:
- 权重是否正确转置
- `npu_format_cast` 是否正确应用
- 检查 `weight_scale` 和 `weight_offset` 维度

### 3. 精度下降严重

**症状**: 模型输出质量明显下降

**检查点**:
- 确认量化配置与模型匹配
- 检查 `ignore` 列表是否正确
- 验证 `FLOAT` 层是否被正确跳过

### 4. 动态量化性能问题

**症状**: 动态量化比静态量化慢

**原因**: 动态量化需要运行时计算 scale

**建议**:
- 对于稳定输入分布，使用静态量化
- 对于变化输入分布，使用动态量化

### 5. MoE 量化错误

**症状**: `RuntimeError in npu_grouped_matmul`

**检查点**:
- `expert_tokens` 是否正确计算
- 权重格式是否正确
- 检查 `group_list` 参数

## 性能对比

| 量化类型 | 内存占用 | 推理速度 | 精度损失 |
|----------|----------|----------|----------|
| FP16/BF16 | 100% | 基准 | 无 |
| W8A8 Static | ~50% | 1.5-2x | 小 |
| W8A8 Dynamic | ~50% | 1.3-1.8x | 小 |
| W4A4 Dynamic | ~25% | 1.2-1.5x | 中等 |

## 与其他模块的关系

```
Quantization
├── 配置加载: ModelSlimConfig.from_config()
├── 权重创建: ModelSlimLinearMethod.create_weights()
├── 权重处理: process_weights_after_loading()
├── 前向计算: apply() → npu_quant_matmul()
├── MoE: npu_fused_experts()
└── RMSNorm: npu_rms_norm + bias
```

## 调试建议

### 1. 检查量化配置

```python
import json
with open("/models/quantized-model/quant_model_description.json") as f:
    config = json.load(f)
print(json.dumps(config, indent=2))
```

### 2. 验证权重格式

```python
# 在 process_weights_after_loading 中添加
print(f"Weight shape: {layer.weight.shape}")
print(f"Weight dtype: {layer.weight.dtype}")
print(f"Weight scale shape: {layer.weight_scale.shape}")
```

### 3. 检查量化精度

```python
# 对比量化前后输出
with torch.no_grad():
    output_fp16 = model_fp16(input_ids)
    output_quant = model_quant(input_ids)
    diff = (output_fp16 - output_quant).abs().mean()
    print(f"Mean absolute difference: {diff}")
```
