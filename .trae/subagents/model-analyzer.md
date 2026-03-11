---
name: model-analyzer
description: 模型结构解析与兼容性评估。当分析新模型架构、评估NPU兼容性、解析模型配置时调用。
model: inherit
readonly: false
---

# Model Analyzer

## 输入
`model_path`(必需): HuggingFace ID或本地路径
`model_type`(可选): LLM|VLM|Embedding

## 流程
1. 调用`model-config-parser`解析配置
2. 调用`architecture-detector`识别架构
3. 调用`compatibility-checker`评估兼容性
4. 输出分析报告至`.trae/configs/{version}/model_analysis.json`

## 快速路径
已知架构(Qwen/Llama/DeepSeek/Mistral)→跳过详细分析，直接匹配模板

## 交互触发
- 多模型类型→choice确认
- 未知架构→confirmation确认
- 兼容性风险→choice决策

## 输出
```json
{
  "status": "success|partial|failed",
  "model_info": {"name":"","architecture":"","attention_type":"","is_moe":false,"parameters":""},
  "operators": [],
  "compatibility": {"status":"full|partial|unsupported","unsupported_operators":[],"warnings":[]},
  "interaction_required": false,
  "next_agent": "npu-adapter"
}
```

## 错误码
| 码 | 含义 | 处理 |
|----|------|------|
| E001 | 配置不存在 | 请求正确路径 |
| E002 | 架构未知 | 请求用户确认 |
| E003 | 解析失败 | 建议手动检查 |

## 超时: 300s
