---
name: "weight-validator"
description: "验证模型权重完整性。当检查权重文件、验证SHA256时调用。"
---

# Weight Validator

## 参数
`model_path`(必需): 模型权重路径
`check_sha256`(可选): 是否验证SHA256，默认false

## 返回
```json
{"validation_status":"valid|invalid|incomplete","weight_files":[{"name":"","size_gb":0,"sha256":""}],"total_size_gb":0,"format":"safetensors|bin","missing_files":[],"corrupted_files":[]}
```

## 错误
| 错误 | 返回 |
|------|------|
| 文件不存在 | `{"validation_status":"invalid"}` |
| SHA256失败 | `{"validation_status":"invalid","corrupted_files":[]}` |

## 依赖: 无
