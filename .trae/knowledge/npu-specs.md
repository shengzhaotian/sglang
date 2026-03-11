# NPU设备技术规格

## A2 vs A3 对比

| 项目 | A2 (910B1) | A3 (910B2/B3) |
|------|------------|---------------|
| 算力(FP16) | 313 TFLOPS | 376 TFLOPS |
| 内存 | 64GB HBM | 64GB HBM |
| 带宽 | 1.2 TB/s | 1.6 TB/s |
| 互联 | 392 GB/s | 400 GB/s |

## 量化支持

| 类型 | A2 | A3 |
|------|----|----|
| BF16/FP16 | ✅ | ✅ |
| W8A8 INT8 | ✅ | ✅ |
| W4A8 INT8 | ⚠️ | ✅ |
| W8A8 FP8 | ❌ | ✅ |

## 推荐配置

| 模型规模 | 硬件 | 卡数 |
|----------|------|------|
| 7B-14B | A2 | 1-2 |
| 32B-70B | A2/A3 | 4-8 |
| 100B+ | A3 | 8+ |

## 关键环境变量

```bash
export ENABLE_SGLANG_NPU=1
export ASCEND_GLOBAL_LOG_LEVEL=3
export HCCL_CONNECT_TIMEOUT=7200
```
