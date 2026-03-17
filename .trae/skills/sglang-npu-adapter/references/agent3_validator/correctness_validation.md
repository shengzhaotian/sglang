# 正确性验证方法

## 1. 与HuggingFace对比

### 1.1 对比方法

将SGLang输出与HuggingFace模型输出对比，验证正确性。

### 1.2 输出logits对比

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载HF模型
hf_model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 获取HF输出
inputs = tokenizer("Hello", return_tensors="pt")
with torch.no_grad():
    hf_outputs = hf_model(**inputs)
    hf_logits = hf_outputs.logits

# 与SGLang输出对比
# sglang_logits = ...
diff = torch.abs(hf_logits - sglang_logits).max()
print(f"Max logits diff: {diff}")
```

### 1.3 允许误差范围

| 精度 | 允许误差 |
|------|----------|
| FP32 | < 1e-5 |
| FP16/BF16 | < 1e-3 |
| FP8 | < 1e-2 |

---

## 2. 验证流程

### 2.1 基础验证

```
1. 加载HF模型和SGLang模型
2. 使用相同输入
3. 对比输出logits
4. 检查误差是否在允许范围内
```

### 2.2 采样输出对比

```python
# 使用相同seed采样
torch.manual_seed(42)
hf_output = hf_model.generate(**inputs, max_new_tokens=50)

torch.manual_seed(42)
sglang_output = sglang_generate(inputs, max_new_tokens=50)

# 对比输出文本
print(f"HF: {tokenizer.decode(hf_output[0])}")
print(f"SGLang: {tokenizer.decode(sglang_output[0])}")
```

### 2.3 多样本验证

使用多个测试样本验证：
- 短文本
- 长文本
- 特殊字符
- 多语言

---

## 3. 常见问题

### 3.1 精度差异来源

| 来源 | 说明 |
|------|------|
| 算子实现差异 | 不同后端的数值精度 |
| 量化误差 | FP8/INT8量化 |
| RoPE实现 | 位置编码计算方式 |

### 3.2 数值稳定性问题

**现象**：输出NaN或Inf

**排查**：
1. 检查输入数据
2. 检查权重加载
3. 检查计算过程

### 3.3 验证失败分析

如果验证失败：
1. 检查模型配置是否正确
2. 检查权重加载是否完整
3. 检查Attention后端
4. 检查RoPE配置

---

## 4. 自动化验证脚本

```python
def validate_model(model_path, test_prompts):
    """验证模型正确性"""
    # 加载HF模型
    hf_model = load_hf_model(model_path)
    
    # 加载SGLang模型
    sglang_model = load_sglang_model(model_path)
    
    results = []
    for prompt in test_prompts:
        hf_output = hf_model.generate(prompt)
        sglang_output = sglang_model.generate(prompt)
        
        diff = compute_diff(hf_output, sglang_output)
        results.append({
            "prompt": prompt,
            "diff": diff,
            "passed": diff < threshold
        })
    
    return results
```

---

## 5. 参考代码

| 文件 | 说明 |
|------|------|
| `python/sglang/srt/models/` | 模型实现 |
| `python/sglang/test/` | 测试用例 |
