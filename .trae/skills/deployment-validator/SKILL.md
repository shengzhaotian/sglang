---
name: "deployment-validator"
description: "验证部署结果。当验证服务启动、精度、性能时调用。"
---

# Deployment Validator

## 参数
`server_url`(必需): 服务URL
`test_config`(可选): 测试配置

## 返回
```json
{"validation_status":"passed|failed|partial","tests":{"health_check":{"status":"","response_time_ms":0},"basic_inference":{"status":"","output_valid":true},"accuracy_test":{"status":"","accuracy_loss":""},"performance_test":{"status":"","meets_target":true},"stability_test":{"status":"","duration_minutes":0}},"issues":[],"recommendations":[]}
```

## 错误
| 错误 | 返回 |
|------|------|
| 服务不可达 | `{"validation_status":"failed","error":"connection_failed"}` |

## 依赖: config-tuner
