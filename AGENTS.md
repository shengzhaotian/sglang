# NPU Model Adaptation Agents

## Sub Agents Index

| Agent | 触发条件 | 输出 |
|-------|----------|------|
| [model-analyzer](.trae/subagents/model-analyzer.md) | 分析新模型架构、评估NPU兼容性 | 模型分析报告 |
| [npu-adapter](.trae/subagents/npu-adapter.md) | NPU算子映射、量化配置 | NPU适配配置 |
| [perf-optimizer](.trae/subagents/perf-optimizer.md) | 性能基准测试、配置优化 | 优化配置与测试报告 |

## Orchestration

调用 [adaptation-orchestrator](.trae/skills/adaptation-orchestrator/SKILL.md) 启动完整流程：

```
[并行] env-checker + weight-validator
    ↓
[串行] model-analyzer
    ↓
[串行] npu-adapter
    ↓
[串行] perf-optimizer
    ↓
[串行] deployment-validator
```

## Quick Reference

### Known Architectures (Fast Path)
- Qwen, Llama, DeepSeek, Mistral → 直接匹配模板

### Error Codes
| Range | Agent |
|-------|-------|
| E001-E099 | model-analyzer |
| E101-E199 | npu-adapter |
| E201-E299 | perf-optimizer |

### Output Paths
```
.trae/configs/{version}/
├── model_analysis.json
├── npu_adaptation.json
├── perf_config.json
└── final_report.json
```

## Skills Dependency

```
model-config-parser
    ↓
architecture-detector
    ↓
compatibility-checker → env-checker
    ↓                       ↓
npu-operator-mapper    weight-validator
    ↓
quantization-strategy
    ↓
npu-config-generator
    ↓
perf-benchmark
    ↓
resource-optimizer
    ↓
config-tuner
    ↓
deployment-validator
```
