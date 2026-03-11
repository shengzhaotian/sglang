---
name: "npu-operator-mapper"
description: "生成NPU算子映射。当将模型算子映射到NPU实现时调用。"
---

# NPU Operator Mapper

## 参数
`required_operators`(必需): 所需算子列表
`hardware_type`(必需): A2|A3

## 返回
```json
{"mapping_status":"complete|partial|failed","operator_mapping":{},"fallback_operators":[],"unsupported_operators":[],"performance_notes":[]}
```

## 错误
| 错误 | 返回 |
|------|------|
| 算子无映射 | `{"mapping_status":"partial","unsupported_operators":[]}` |

## 依赖: compatibility-checker
