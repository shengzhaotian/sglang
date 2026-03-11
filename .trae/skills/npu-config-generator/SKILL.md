---
name: "npu-config-generator"
description: "生成NPU部署配置。当生成启动命令、环境变量时调用。"
---

# NPU Config Generator

## 参数
`adaptation_info`(必需): NPU适配信息
`deploy_mode`(可选): PD Mixed|PD Separation

## 返回
```json
{"launch_command":"","env_variables":{},"config_file":{"path":"","content":{}},"port":30000,"additional_args":{}}
```

## 错误
| 错误 | 返回 |
|------|------|
| 配置冲突 | `{"error":"config_conflict","conflicts":[]}` |

## 依赖: quantization-strategy
