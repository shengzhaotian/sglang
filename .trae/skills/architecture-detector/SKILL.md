---
name: "architecture-detector"
description: "检测模型架构类型。当识别架构、attention类型、MoE结构时调用。"
---

# Architecture Detector

## 参数
`config_dict`(必需): 模型配置字典

## 返回
```json
{"architecture_family":"Qwen|Llama|DeepSeek|Mistral|Other","attention_type":"MHA|GQA|MLA","is_moe":false,"num_experts":0,"num_experts_per_tok":0,"rope_type":"default|linear|none","key_features":[]}
```

## 错误
| 错误 | 返回 |
|------|------|
| 未知架构 | `{"architecture_family":"unknown","suggestion":""}` |

## 依赖: model-config-parser
