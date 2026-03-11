---
name: "compatibility-checker"
description: "检查NPU兼容性。当评估模型与NPU硬件兼容性时调用。"
---

# Compatibility Checker

## 参数
`architecture_info`(必需): 架构检测结果
`target_hardware`(必需): A2|A3

## 返回
```json
{"compatibility_status":"full|partial|unsupported","supported_features":[],"unsupported_operators":[],"fallback_available":true,"warnings":[],"recommendations":[]}
```

## 错误
| 错误 | 返回 |
|------|------|
| 无法确定 | `{"compatibility_status":"unknown"}` |

## 依赖: architecture-detector
