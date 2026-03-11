---
name: "adaptation-orchestrator"
description: "模型适配编排器。当启动完整模型适配流程时调用，协调各Sub Agent执行。"
---

# Adaptation Orchestrator

## 参数
`model_path`(必需): 模型路径
`hardware_type`(必需): A2|A3
`performance_target`(可选): low_latency|high_throughput|balanced
`skip_steps`(可选): 跳过的步骤列表

## 编排流程
```
1. [并行] env-checker + weight-validator
2. [串行] model-analyzer
3. [串行] npu-adapter
4. [串行] perf-optimizer
5. [串行] deployment-validator
```

## 快速路径判断
```python
if model_architecture in KNOWN_ARCHITECTURES:
    skip detailed_analysis
    use template_config
```

## 失败回退
| 失败点 | 回退策略 |
|--------|----------|
| 环境检查失败 | 提示修复，暂停流程 |
| 模型分析失败 | 请求用户提供架构信息 |
| 算子映射失败 | 使用fallback方案继续 |
| 基准测试失败 | 使用默认配置，跳过优化 |

## 输出
```json
{"session_id":"","status":"","completed_steps":[],"current_step":"","final_report_path":"","errors":[]}
```

## 依赖: 所有Sub Agent
