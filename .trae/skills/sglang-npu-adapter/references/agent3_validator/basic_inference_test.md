# 基础推理测试方法

## 1. 测试流程

### 1.1 完整测试流程

```
1. 模型加载测试
   └─ 验证模型能正确加载到内存

2. 基础推理测试
   └─ 验证模型能正常生成输出

3. 输出验证
   └─ 验证输出格式和内容正确

4. 性能测试（可选）
   └─ 测量吞吐量和延迟
```

### 1.2 测试优先级

| 优先级 | 测试项 | 说明 |
|--------|--------|------|
| P0 | 模型加载 | 必须通过 |
| P0 | 基础推理 | 必须通过 |
| P1 | 输出验证 | 建议通过 |
| P2 | 性能测试 | 可选 |

---

## 2. SGLang启动命令

### 2.1 基础启动

```bash
python -m sglang.launch_server \
    --model-path /path/to/model \
    --port 30000
```

### 2.2 NPU启动

```bash
python -m sglang.launch_server \
    --model-path /path/to/model \
    --port 30000 \
    --device npu \
    --attention-backend ascend
```

### 2.3 常用参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model-path` | 模型路径 | `/path/to/model` |
| `--port` | 服务端口 | `30000` |
| `--tp` | Tensor Parallel大小 | `2` |
| `--device` | 设备类型 | `npu` |
| `--attention-backend` | Attention后端 | `ascend` |
| `--context-length` | 上下文长度 | `4096` |
| `--max-running-requests` | 最大并发请求数 | `16` |

---

## 3. 测试请求

### 3.1 curl测试

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 50
  }'
```

### 3.2 Python测试

```python
import requests

response = requests.post(
    "http://localhost:30000/v1/chat/completions",
    json={
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 50,
    }
)
print(response.json())
```

### 3.3 OpenAI客户端测试

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="none")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=50,
)
print(response.choices[0].message.content)
```

---

## 4. 测试用例

### 4.1 短文本推理

```json
{
    "model": "default",
    "messages": [{"role": "user", "content": "1+1=?"}],
    "max_tokens": 10
}
```

**预期**：模型输出类似 "2" 或 "2." 的简短回答

### 4.2 长文本推理

```json
{
    "model": "default",
    "messages": [{"role": "user", "content": "请写一篇关于人工智能的500字文章"}],
    "max_tokens": 1000
}
```

**预期**：模型输出一篇完整的文章

### 4.3 多轮对话

```json
{
    "model": "default",
    "messages": [
        {"role": "user", "content": "我叫张三"},
        {"role": "assistant", "content": "你好，张三！"},
        {"role": "user", "content": "我叫什么名字？"}
    ],
    "max_tokens": 50
}
```

**预期**：模型回答 "张三"

---

## 5. 错误诊断

### 5.1 启动失败

**现象**：服务无法启动

**排查步骤**：
1. 检查模型路径是否正确
2. 检查设备是否可用
3. 检查内存是否充足
4. 查看错误日志

**常见错误**：
| 错误 | 原因 | 解决方案 |
|------|------|----------|
| Model not found | 模型路径错误 | 检查路径 |
| Out of memory | 内存不足 | 减小TP/BS |
| Device not available | 设备不可用 | 检查设备状态 |

### 5.2 推理失败

**现象**：请求返回错误

**排查步骤**：
1. 检查请求格式是否正确
2. 检查服务日志
3. 检查模型配置

**常见错误**：
| 错误 | 原因 | 解决方案 |
|------|------|----------|
| Invalid request | 请求格式错误 | 检查JSON格式 |
| Context too long | 上下文超长 | 减小输入长度 |
| Internal error | 模型内部错误 | 查看详细日志 |

### 5.3 输出异常

**现象**：输出内容异常（乱码、重复等）

**排查步骤**：
1. 检查模型权重是否正确加载
2. 检查配置是否正确
3. 与HuggingFace输出对比

---

## 6. 日志查看

### 6.1 启动日志

启动时关注以下信息：
```
[INFO] Loading model from /path/to/model
[INFO] Model architecture: Qwen2ForCausalLM
[INFO] Using attention backend: ascend
[INFO] Server started at http://0.0.0.0:30000
```

### 6.2 运行日志

运行时关注：
- 请求处理时间
- 内存使用情况
- 错误和警告信息

---

## 7. 参考代码

| 文件 | 说明 |
|------|------|
| `python/sglang/launch_server.py` | 服务启动入口 |
| `python/sglang/srt/server_args.py` | 服务参数定义 |
| `python/sglang/srt/hardware_backend/npu/` | NPU后端实现 |
