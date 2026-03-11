---
name: "config-tuner"
description: "自动调优配置。当迭代优化部署参数时调用。"
---

# Config Tuner

## 参数
`performance_target`(必需): low_latency|high_throughput|balanced
`constraints`(可选): 约束条件

## 返回
```json
{"tuning_status":"success|failed|timeout","best_config":{"tp_size":0,"dp_size":0,"chunked_prefill_size":0,"cuda_graph_batch_sizes":[]},"tuning_history":[],"final_score":0}
```

## 错误
| 错误 | 返回 |
|------|------|
| 无法优化 | `{"tuning_status":"failed","best_config":{}}` |

## 依赖: resource-optimizer
