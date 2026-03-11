---
name: "perf-benchmark"
description: "运行性能基准测试。当测试TTFT、TPOT、吞吐量时调用。"
---

# Perf Benchmark

## 参数
`config`(必需): 部署配置
`dataset_spec`(可选): 测试数据集规格
`duration_seconds`(可选): 测试时长，默认60

## 返回
```json
{"benchmark_status":"success|failed|timeout","metrics":{"ttft_ms":{"mean":0,"p50":0,"p99":0},"tpot_ms":{"mean":0,"p50":0,"p99":0},"throughput_tokens_per_sec":0,"total_requests":0,"success_rate":0},"resource_usage":{"memory_gb":0,"npu_utilization":0},"errors":[]}
```

## 错误
| 错误 | 返回 |
|------|------|
| 执行失败 | `{"benchmark_status":"failed"}` |
| 超时 | `{"benchmark_status":"timeout","partial_results":{}}` |

## 依赖: npu-config-generator
