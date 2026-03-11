---
name: "env-checker"
description: "检查NPU环境依赖。当验证驱动、CANN、PyTorch NPU环境时调用。"
---

# Env Checker

## 参数
`hardware_type`(必需): A2|A3

## 返回
```json
{"environment_status":"ready|incomplete|error","checks":{"npu_driver":{"status":"","version":""},"cann":{"status":"","version":""},"torch_npu":{"status":"","version":""},"triton_ascend":{"status":"","version":""},"sgl_kernel_npu":{"status":"","version":""},"memory":{"status":"","available_gb":0}},"missing_dependencies":[],"recommendations":[]}
```

## 错误
| 错误 | 返回 |
|------|------|
| 检查失败 | `{"environment_status":"error"}` |

## 依赖: 无
