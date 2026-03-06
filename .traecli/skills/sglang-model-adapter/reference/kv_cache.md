# NPU KV Cache Reference

## 概述

KV Cache 是 LLM 推理的核心组件，用于存储和复用历史 Key-Value 状态。NPU 上的 KV Cache 实现有其特殊性，包括页式管理、NZ 格式支持等。

## 核心文件

```
python/sglang/srt/hardware_backend/npu/
├── memory_pool_npu.py      # NPU 内存池实现
├── allocator_npu.py        # NPU 内存分配器
└── attention/
    ├── ascend_backend.py   # KV Cache 在 attention 中的使用
    └── mla_preprocess.py   # MLA 模型的 KV Cache 预处理
```

## KV Cache 架构

### 标准 (非 MLA) 模型

```
KV Cache 结构:
┌─────────────────────────────────────────┐
│  Block 0  │  Block 1  │  Block 2  │ ... │
├───────────┼───────────┼───────────┼─────┤
│ K: [page_size, num_kv_heads, head_dim]  │
│ V: [page_size, num_kv_heads, head_dim]  │
└─────────────────────────────────────────┘
```

### MLA 模型

```
KV Cache 结构 (DeepSeek-V2/V3):
┌─────────────────────────────────────────┐
│  Block 0  │  Block 1  │  Block 2  │ ... │
├───────────┼───────────┼───────────┼─────┤
│ kv_c: [page_size, num_kv_heads, kv_lora_rank]  │  ← 压缩的 KV latent
│ k_pe: [page_size, num_kv_heads, qk_rope_head_dim] │  ← Key 的 RoPE 部分
└─────────────────────────────────────────┘
```

**关键区别**: MLA 模型不直接存储 K 和 V，而是存储压缩表示 `kv_c` 和 `k_pe`。

## 内存池管理

### MemoryPool 类

**关键属性**:

```python
class MemoryPool:
    def __init__(self, ...):
        self.page_size = page_size          # 页大小 (通常为 1)
        self.num_layers = num_layers        # 层数
        self.dtype = dtype                  # 数据类型
        
        # KV Cache 存储
        self.kv_data = {}                   # 层 ID → tensor
```

### KV Buffer 操作

```python
# 设置 KV buffer
def set_kv_buffer(self, layer, cache_loc, k, v):
    """
    layer: 当前层
    cache_loc: 写入位置 (slot_mapping)
    k: Key tensor
    v: Value tensor
    """
    
# 获取 KV buffer
def get_kv_buffer(self, layer_id):
    """
    返回: (k_cache, v_cache)
    """
```

### MLA 模型的特殊处理

```python
# 在 mla_preprocess.py 中
def get_kv_cache_and_cache_idx(self, forward_batch):
    k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(self.layer_id)
    # 注意: MLA 模型中
    # k_cache 实际存储 kv_c (压缩的 KV)
    # v_cache 实际存储 k_pe (Key RoPE)
    slot_mapping = forward_batch.out_cache_loc.to(dtype=torch.int32)
    return k_cache, v_cache, slot_mapping
```

## Page Table 管理

### Block Table 结构

```python
# block_tables: [batch_size, max_seq_pages]
# 每个元素是一个 block ID，指向 KV Cache 中的对应块

# 示例: 序列长度 10, page_size = 1
# block_tables[0] = [5, 8, 12, 25, 33, 41, 55, 60, 71, 82, 0, 0, ...]
#                                                          ↑ 未使用
```

### Block Table 构建

```python
# 在 ascend_backend.py 中
def init_forward_metadata(self, forward_batch: ForwardBatch):
    self.forward_metadata.block_tables = (
        forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :seq_lens_max
        ][:, :: self.page_size]
        // self.page_size
    )
```

## Cache 格式

### PA_BNSD 格式 (标准)

```
Layout: [Num_blocks, Block_size, Num_heads, Head_dim]

示例: num_blocks=1000, block_size=1, num_heads=32, head_dim=128
Shape: [1000, 1, 32, 128]
```

### PA_NZ 格式 (NPU 优化)

```
Layout: [Num_blocks, Num_heads * Head_dim // 16, Block_size, 16]

示例: num_blocks=1000, block_size=1, num_heads=32, head_dim=128
Shape: [1000, 256, 1, 16]

特点: 16 元素分块，适合 NPU 向量化
```

### 格式转换

```python
# 在 mla_preprocess.py 中
def _reshape_kv_for_fia_nz(tensor, num_heads, head_dim, page_size):
    """将 tensor 转换为 FIA NZ 格式"""
    return tensor.view(-1, 1, num_heads * head_dim // 16, page_size, 16)
```

## Slot Mapping

### 概念

Slot mapping 定义了每个 token 应该写入 KV Cache 的哪个位置。

```python
# slot_mapping: [num_tokens]
# 每个元素是一个线性索引，指向 cache 中的位置

# 示例:
# tokens: ["Hello", "World", "!"]
# slot_mapping: [0, 1, 2]  # 写入 cache 的前三个位置
```

### 获取方式

```python
# 在 forward_batch 中
slot_mapping = forward_batch.out_cache_loc

# 转换为所需类型
slot_mapping_int32 = slot_mapping.to(dtype=torch.int32)
slot_mapping_int64 = slot_mapping.to(dtype=torch.int64)
```

## Cache 写入操作

### 标准 Cache 写入

```python
# 在 ascend_backend.py 的 forward_extend 中
if save_kv_cache and k is not None and v is not None:
    cache_loc = (
        forward_batch.out_cache_loc
        if not layer.is_cross_attention
        else forward_batch.encoder_out_cache_loc
    )
    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
```

### MLA Cache 写入

```python
# 方式 1: 直接写入
forward_batch.token_to_kv_pool.set_kv_buffer(
    layer, forward_batch.out_cache_loc, kv_a.unsqueeze(1), k_pe
)

# 方式 2: 通过 npu_kv_rmsnorm_rope_cache 写入
torch_npu.npu_kv_rmsnorm_rope_cache(
    latent_cache,
    layernorm_weight,
    cos, sin,
    slot_mapping,
    k_rope_cache,    # 输出: k_pe cache
    c_kv_cache,      # 输出: kv_c cache
    ...
)
```

## Cache 读取操作

### 标准 Cache 读取

```python
k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

# reshape 用于 attention 计算
k_cache = k_cache.view(-1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim)
v_cache = v_cache.view(-1, self.page_size, layer.tp_v_head_num * layer.v_head_dim)
```

### MLA Cache 读取

```python
kv_c = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
k_pe = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

# 对于 FIA NZ 格式
if is_fia_nz():
    k_rope_cache = _reshape_kv_for_fia_nz(k_pe, ...)
    c_kv_cache = _reshape_kv_for_fia_nz(kv_c, ...)
else:
    k_rope_cache = k_pe.view(-1, self.page_size, ...)
    c_kv_cache = kv_c.view(-1, self.page_size, ...)
```

## 常见问题排查

### 1. Cache 位置错误

**症状**: 推理结果错误或 crash

**检查点**:
- `slot_mapping` 是否正确
- `block_tables` 是否正确构建
- `req_pool_indices` 是否有效

**调试代码**:
```python
print(f"slot_mapping: {forward_batch.out_cache_loc}")
print(f"block_tables shape: {self.forward_metadata.block_tables.shape}")
print(f"seq_lens: {forward_batch.seq_lens}")
```

### 2. 维度不匹配

**症状**: `RuntimeError: shape mismatch`

**检查点**:
- `num_heads` 是否考虑了 TP 分片
- `head_dim` 是否正确 (MLA: `kv_lora_rank` vs `qk_head_dim`)
- `page_size` 是否一致

### 3. 格式错误

**症状**: FIA 算子报错

**检查点**:
- `SGLANG_USE_FIA_NZ` 是否设置正确
- cache 格式是否与算子期望一致
- 检查 `cache_mode` 参数

### 4. 内存不足

**症状**: OOM 错误

**检查点**:
- `mem_fraction_static` 设置是否合理
- `max_model_len` 是否过大
- 检查实际 cache 大小

**计算公式**:
```python
# 每 token 的 KV Cache 大小
bytes_per_token = (
    2  # K + V
    * num_layers
    * num_kv_heads
    * head_dim
    * dtype_bytes  # 2 for fp16/bf16
)

# 总 Cache 大小
total_cache_size = max_model_len * bytes_per_token * max_running_requests
```

## 性能优化建议

### 1. 页大小选择

- 小 `page_size` (1): 内存利用率高，但管理开销大
- 大 `page_size` (16, 32): 管理开销小，但可能有碎片

### 2. 格式选择

- `PA_BNSD`: 通用性好，调试方便
- `PA_NZ`: NPU 性能更优，但需要特定环境变量

### 3. 预分配

```python
# 提前分配足够的 cache 空间
--mem-fraction-static 0.85  # 预留 85% 内存给静态分配
```

## 与其他模块的关系

```
KV Cache
├── 输入: token positions (来自 scheduler)
├── 管理: MemoryPool, req_to_token_pool
├── 消费: Attention Backend
└── 特殊处理: MLA Preprocess (压缩表示)
```
