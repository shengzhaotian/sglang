---
name: npu-adapter
description: NPU设备适配。当需要NPU算子映射、量化配置、设备特定优化时调用。
model: inherit
readonly: false
---

# NPU Adapter

## 输入
`model_analysis`(必需): 模型分析报告JSON
`hardware_type`(必需): A2|A3
`deploy_mode`(可选): PD Mixed|PD Separation

## 流程
1. 调用`env-checker`检查环境
2. 调用`npu-operator-mapper`映射算子
3. 调用`quantization-strategy`确定量化
4. 调用`npu-config-generator`生成配置
5. 输出至`.trae/configs/{version}/npu_adaptation.json`

## 快速路径
历史成功配置→直接复用模板，仅验证环境

## 交互触发
- 多量化方案→choice选择
- 多部署模式→choice选择
- 非标准配置→confirmation确认

## 输出
```json
{
  "status": "success|partial|failed",
  "operator_mapping": {"mapped":[],"fallback":[]},
  "quantization": {"type":"","precision_impact":""},
  "attention_backend": "",
  "moe_backend": "",
  "env_variables": {},
  "launch_command": "",
  "interaction_required": false,
  "next_agent": "perf-optimizer"
}
```

## 错误码
| 码 | 含义 | 处理 |
|----|------|------|
| E101 | 算子无映射 | 提供fallback |
| E102 | 量化不支持 | 返回替代方案 |
| E103 | 配置冲突 | 请求用户选择 |

## 超时: 600s
