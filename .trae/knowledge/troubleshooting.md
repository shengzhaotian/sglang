# 常见问题排查

## 算子不支持

**症状**: `Operator XXX not supported`

**解决**: 1.使用fallback 2.联系开发团队 3.等效算子替换

## 内存溢出

**症状**: `OOM`

**解决**: 1.降低mem_fraction_static 2.减小max_num_seqs 3.增加TP size

## 精度问题

**症状**: 输出质量下降

**解决**: 1.检查量化配置 2.验证权重加载 3.对比参考输出

## 性能不达标

**排查**: 1.NPU利用率 2.通信瓶颈 3.batch sizes 4.CUDA Graph

## 启动失败

**排查**: 1.端口占用 2.模型路径 3.环境变量 4.日志详情

```bash
ASCEND_GLOBAL_LOG_LEVEL=0 python -m sglang.launch_server ...
```
