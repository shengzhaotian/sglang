---
name: "model-config-parser"
description: "解析HuggingFace模型配置。当读取config.json、提取架构参数时调用。"
---

# Model Config Parser

## 参数
`model_path`(必需): HuggingFace ID或本地路径
`model_type`(可选): LLM|VLM|Embedding

## 返回
```json
{"config_path":"","architectures":[],"hidden_size":0,"num_attention_heads":0,"num_hidden_layers":0,"intermediate_size":0,"vocab_size":0,"max_position_embeddings":0,"torch_dtype":"","custom_config":{}}
```

## 错误
| 错误 | 返回 |
|------|------|
| 文件不存在 | `{"error":"config_not_found"}` |
| 解析失败 | `{"error":"parse_failed"}` |

## 依赖: 无
