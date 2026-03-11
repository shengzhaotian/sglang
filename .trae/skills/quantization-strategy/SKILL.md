---
name: "quantization-strategy"
description: "生成量化策略。当确定量化方案、评估精度影响时调用。"
---

# Quantization Strategy

## 参数
`model_info`(必需): 模型信息
`performance_target`(可选): low_latency|high_throughput|balanced

## 返回
```json
{"recommended_strategy":"W8A8|W4A8|BF16|FP16","available_strategies":[{"name":"","precision_impact":"","memory_reduction":"","speedup_estimate":""}],"calibration_required":true,"calibration_dataset":"random|pileval|custom","warnings":[]}
```

## 错误
| 错误 | 返回 |
|------|------|
| 量化不支持 | `{"available_strategies":[],"reason":""}` |

## 依赖: npu-operator-mapper
