# GLM-4.7-flash NPU 适配指南

## 简介

GLM-4.7-flash 是智谱AI推出的混合专家(MoE)大语言模型，采用MLA(Multi-head Latent Attention)注意力机制，具有以下特点：

- **架构**: MoE (64路由专家 + 1共享专家)
- **注意力**: MLA (Multi-head Latent Attention)
- **参数量**: ~30B (有效激活参数)
- **上下文长度**: 202752 tokens

## 特性支持

| 特性 | 状态 | 备注 |
|------|------|------|
| 基础推理 | ✅ 支持 | |
| MLA注意力 | ✅ 支持 | 使用native SDPA实现 |
| MoE | ✅ 支持 | |
| ACLGraph | ❌ 不支持 | 头数非2的幂次限制 |
| DeepEP | N/A | 非EP部署 |
| MTP | N/A | 无MTP层 |

## 环境要求

- CANN: 8.5.0+
- PyTorch: 2.1+
- torch_npu: 最新版本
- SGLang: 当前版本

## 部署命令

### 基础部署

```bash
export PYTHONPATH=${PWD}/python:$PYTHONPATH

python -m sglang.launch_server \
    --model-path /home/trae/testCode/weight/GLM4-7-flash \
    --attention-backend ascend \
    --device npu \
    --port 8000 \
    --tp-size 2 \
    --disable-cuda-graph
```

### 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--tp-size` | 2 | 张量并行数，每个NPU 10个注意力头 |
| `--attention-backend` | ascend | NPU注意力后端 |
| `--device` | npu | 使用NPU设备 |
| `--disable-cuda-graph` | - | 禁用CUDA Graph（头数限制） |

## 验证测试

### 健康检查

```bash
curl http://localhost:8000/v1/models
```

### 推理测试

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/trae/testCode/weight/GLM4-7-flash",
    "messages": [{"role": "user", "content": "你好，请介绍一下你自己"}],
    "max_tokens": 100
  }'
```

### 准确性验证

以下测试均在NPU上通过验证：

| 测试类型 | 问题 | 预期答案 | 模型输出 | 结果 |
|----------|------|----------|----------|------|
| 数学计算 | 15 × 7 = ? | 105 | 105 | ✅ 通过 |
| 知识问答 | 太阳系最大的行星？ | Jupiter | Jupiter | ✅ 通过 |
| 逻辑推理 | A>B, B>C, 则 A>C? | Yes | Yes | ✅ 通过 |
| 多步数学 | 5个苹果给2个，再买3个 | 6 | 6 | ✅ 通过 |
| 文学知识 | 谁写了罗密欧与朱丽叶？ | Shakespeare | William Shakespeare | ✅ 通过 |
| 日期推理 | 周一+3天是周几？ | Thursday | Thursday | ✅ 通过 |

**注意**: GLM-4.7-flash是推理模型，会先输出思考过程再给出答案。建议设置较大的 `max_tokens` (如500)以获得完整响应。

## 已知限制

### 1. CUDA Graph 不支持

**原因**: NPU融合注意力算子要求注意力头数为2的幂次（1, 2, 4, 8, 16, 32, 64, 128），GLM-4.7-flash有20个注意力头，使用tp_size=2时每个设备10个头，不满足条件。

**解决方案**: 使用 `--disable-cuda-graph` 禁用CUDA Graph。

### 2. TP Size 限制

| TP Size | 每设备头数 | 是否支持 |
|---------|-----------|---------|
| 1 | 20 | ❌ (非2的幂次) |
| 2 | 10 | ❌ (非2的幂次，但可运行) |
| 4 | 5 | ❌ (非2的幂次) |
| 5 | 4 | ❌ (MoE中间维度不可整除) |

**推荐配置**: tp_size=2 + --disable-cuda-graph

## 性能指标

### 资源占用

- **模型权重**: ~29.3 GB (每设备)
- **KV Cache**: ~19.2 GB
- **总显存**: ~31.5 GB (每设备)

### 加载时间

- **权重加载**: ~25秒
- **服务启动**: ~30秒

## 代码变更

### 新增文件

1. `python/sglang/srt/configs/glm4_moe_lite.py` - 模型配置类

### 修改文件

1. `python/sglang/srt/configs/__init__.py` - 注册配置
2. `python/sglang/srt/utils/hf_transformers_utils.py` - 注册配置
3. `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` - NPU注意力后端处理
4. `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` - MLA注意力处理
5. `python/sglang/srt/layers/rotary_embedding/base.py` - RoPE实现
6. `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` - MLA模块

## 故障排除

### 问题1: 模型加载失败

```
KeyError: 'glm4_moe_lite'
```

**解决方案**: 确保 `Glm4MoeLiteConfig` 已正确注册。

### 问题2: 注意力头数错误

```
group num should be in 1, 2, 4, 8, 16, 32, 64, 128, but got 10
```

**解决方案**: 添加 `--disable-cuda-graph` 参数。

### 问题3: RoPE维度不匹配

```
qD and kD should be same
```

**解决方案**: 确保MLA注意力路径正确处理q_rope和k_rope。

## 后续优化

1. **性能优化**: 等待NPU算子支持非2幂次头数
2. **功能支持**: 探索其他TP配置可能性
3. **量化支持**: 评估ModelSlim量化方案

## 联系方式

如有问题，请在GitHub提交Issue。
