---
name: perf-optimizer
description: 性能优化。当需要性能基准测试、配置优化、资源调优时调用。
model: fast
readonly: false
---

# Perf Optimizer

## 输入
`npu_config`(必需): NPU适配配置JSON
`performance_target`(可选): low_latency|high_throughput|balanced
`hardware_info`(可选): 硬件资源信息

## 流程
1. 调用`resource-optimizer`优化配置
2. 调用`config-tuner`调优参数(可选)
3. 调用`perf-benchmark`基准测试(需确认)
4. 输出至`.trae/configs/{version}/perf_config.json`

## 快速路径
已知配置模板→跳过调优，直接应用推荐配置

## 交互触发
- 目标不明→choice选择
- 资源紧张→confirmation确认
- 长时测试→confirmation确认

## 输出
```json
{
  "status": "success|partial|failed",
  "optimization": {"tp_size":0,"dp_size":0,"cuda_graph_batch_sizes":[],"memory_strategy":""},
  "benchmark_results": {"ttft_ms":0,"tpot_ms":0,"throughput_tokens_per_sec":0},
  "recommendations": [],
  "interaction_required": false,
  "final_report": ".trae/configs/{version}/final_report.json"
}
```

## 错误码
| 码 | 含义 | 处理 |
|----|------|------|
| E201 | 测试失败 | 提供诊断建议 |
| E202 | 资源不足 | 建议降低配置 |
| E203 | 无法优化 | 返回当前最优 |

## 超时: 1800s
