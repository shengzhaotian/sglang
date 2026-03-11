# SGLang NPU API参考

## 模型注册

```python
from sglang.srt.models import register_model

@register_model("MyModel")
class MyModelForCausalLM(nn.Module):
    pass
```

## Attention Backend

```python
from sglang.srt.layers.attention import AttentionBackend

class NPUAttention(AttentionBackend):
    def forward(self, q, k, v, mask): pass
```

## 量化接口

```python
from sglang.srt.layers.quantization import QuantizationMethod

class W8A8Quant(QuantizationMethod):
    def quantize(self, weight): pass
```

## 关键参数

| 参数 | 说明 | 默认 |
|------|------|------|
| tp_size | 张量并行 | 1 |
| dp_size | 数据并行 | 1 |
| mem_fraction_static | 静态内存 | 0.9 |
| chunked_prefill_size | 分块大小 | 8192 |
| cuda_graph_batch_sizes | Graph批大小 | [1,2,4,8] |
