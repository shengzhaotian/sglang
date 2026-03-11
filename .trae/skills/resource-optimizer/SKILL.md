---
name: "resource-optimizer"
description: "优化资源配置。当调整TP/DP、内存分配时调用。"
---

# Resource Optimizer

## 参数
`benchmark_results`(必需): 基准测试结果
`hardware_info`(必需): 硬件资源信息

## 返回
```json
{"optimization_status":"improved|unchanged|degraded","recommendations":{"tp_size":0,"dp_size":0,"cuda_graph_batch_sizes":[],"max_num_seqs":0,"mem_fraction_static":0},"expected_improvement":{"throughput_increase":"","latency_decrease":""},"warnings":[]}
```

## 错误
| 错误 | 返回 |
|------|------|
| 资源不足 | `{"optimization_status":"degraded"}` |

## 依赖: perf-benchmark
