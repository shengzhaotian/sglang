# NPU ACLGraph Reference

## 概述

ACLGraph (Ascend Computing Language Graph) 是华为昇腾 NPU 上的图捕获和执行优化技术。通过捕获计算图并重放，可以显著减少 kernel launch 开销，提升推理性能。本文档介绍 SGLang 中 ACLGraph 的实现和配置。

## 核心文件

```
python/sglang/srt/hardware_backend/npu/graph_runner/
├── npu_graph_runner.py             # NPU Graph Runner 主实现
├── eagle_draft_npu_graph_runner.py # EAGLE Draft Graph Runner
└── eagle_draft_extend_npu_graph_runner.py  # EAGLE Extend Graph Runner

python/sglang/srt/compilation/
├── npu_piecewise_backend.py        # NPU 分段图编译后端
└── compilation_config.py           # 编译配置

python/sglang/srt/model_executor/
├── cuda_graph_runner.py            # 基础 Graph Runner (CUDA/NPU 共用)
└── model_runner.py                 # 模型执行器
```

## ACLGraph 原理

### 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│                    ACLGraph 工作流程                         │
├─────────────────────────────────────────────────────────────┤
│  1. Capture 阶段:                                           │
│     - 执行一次前向传播                                       │
│     - 捕获所有 NPU 操作到图中                                │
│     - 存储图和内存池                                         │
│                                                             │
│  2. Replay 阶段:                                            │
│     - 更新输入数据 (input_ids, positions, seq_lens 等)      │
│     - 重放捕获的图                                           │
│     - 获取输出                                               │
└─────────────────────────────────────────────────────────────┘
```

### 性能优势

| 场景 | 无 Graph | 有 Graph | 提升 |
|------|----------|----------|------|
| 小批次 decode | 高 kernel launch 开销 | 低开销 | 2-3x |
| 固定形状计算 | 动态调度 | 静态执行 | 1.5-2x |
| 连续推理 | 每次重新调度 | 一次捕获多次重放 | 显著 |

## NPUGraphRunner 类

### 类结构

```python
class NPUGraphRunner(CudaGraphRunner):
    """NPU Graph Runner，继承自 CudaGraphRunner"""
    
    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        self._init_arch_map()
    
    def _init_arch_map(self):
        # 根据注意力架构类型设置更新属性
        self.attr_name = {
            AttentionArch.MLA: "actual_seq_lengths_kv",
            AttentionArch.MHA: "context_lens",
        }
```

### 关键方法

#### 1. 创建 NPU Graph

```python
def _create_device_graph(self):
    """创建 NPU Graph 对象"""
    return torch.npu.NPUGraph()
```

#### 2. 捕获 Graph

```python
def _capture_graph(self, graph, pool, stream, run_once_fn):
    """捕获 NPU Graph"""
    if self.enable_torch_compile:
        skip_guard_context = torch.compiler.set_stance(skip_guard_eval_unsafe=True)
    else:
        skip_guard_context = empty_context()

    with skip_guard_context, torch.npu.graph(
        graph,
        pool=pool,
        stream=stream,
        auto_dispatch_capture=True,
    ):
        out = run_once_fn()
    return out
```

#### 3. 重放 Graph

```python
def replay(self, forward_batch: ForwardBatch, ...):
    """重放 NPU Graph"""
    # 准备输入
    self.replay_prepare(forward_batch, pp_proxy_tensors)
    
    # 更新序列长度 (使用线程避免阻塞)
    seq_lens = forward_batch.seq_lens.cpu().tolist() + [0] * (self.bs - self.raw_bs)
    thread = threading.Thread(target=self._update_inputs, args=(seq_lens,))
    thread.start()
    
    # 重放图
    self.graphs[self.bs].replay()
    thread.join()
    
    return output
```

#### 4. 更新输入

```python
def _update_inputs(self, seq_lens):
    """更新 Graph 输入 (通过 CPU)"""
    if isinstance(self.update_attr_type, torch.Tensor):
        seq_lens = torch.from_numpy(np.array(seq_lens).astype(np.int32))
    
    self.graphs[self.bs].update(
        cpu_update_input=[{self.update_attr_name: seq_lens}]
    )
```

## 配置参数

### 服务端参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cuda-graph-bs` | None | Graph 捕获的批次大小列表 |
| `--cuda-graph-max-bs` | 自动 | Graph 最大批次大小 |
| `--disable-cuda-graph` | False | 禁用 Graph |
| `--disable-cuda-graph-padding` | False | 禁用 Graph Padding |
| `--enable-torch-compile` | False | 启用 torch.compile |
| `--enable-profile-cuda-graph` | False | 启用 Graph 捕获性能分析 |

### Piecewise Graph 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--disable-piecewise-cuda-graph` | False | 禁用分段 Graph |
| `--enforce-piecewise-cuda-graph` | False | 强制启用分段 Graph |
| `--piecewise-cuda-graph-max-tokens` | None | 分段 Graph 最大 token 数 |
| `--piecewise-cuda-graph-tokens` | None | 分段 Graph token 列表 |
| `--piecewise-cuda-graph-compiler` | "eager" | 分段 Graph 编译器 |

## 配置示例

### 基本配置

```bash
python -m sglang.launch_server \
    --model-path /models/llama-7b \
    --cuda-graph-bs 1 2 4 8 16 32 \
    --device npu
```

### 大批次配置

```bash
python -m sglang.launch_server \
    --model-path /models/llama-70b \
    --tp-size 8 \
    --cuda-graph-max-bs 160 \
    --cuda-graph-bs 1 2 4 8 16 32 64 128 160 \
    --device npu
```

### 禁用 Graph (调试用)

```bash
python -m sglang.launch_server \
    --model-path /models/llama-7b \
    --disable-cuda-graph \
    --device npu
```

### 启用 torch.compile

```bash
python -m sglang.launch_server \
    --model-path /models/llama-7b \
    --enable-torch-compile \
    --device npu
```

### 性能分析

```bash
python -m sglang.launch_server \
    --model-path /models/llama-7b \
    --enable-profile-cuda-graph \
    --device npu

# 分析结果保存在 /tmp/graph_capture_profile/
```

## Piecewise CUDA Graph

### 概念

Piecewise CUDA Graph 将模型分成多个段，每段独立捕获和执行，适用于：
- 动态形状场景
- 条件分支较多的模型
- 内存受限场景

### NPUPiecewiseBackend

```python
class NPUPiecewiseBackend(CUDAPiecewiseBackend):
    def __call__(self, *args):
        runtime_shape = args[self.sym_shape_indices[0]]
        
        if entry.cudagraph is None:
            # 首次执行，捕获图
            npugraph = torch.npu.NPUGraph()
            with torch.npu.graph(npugraph, pool=self.graph_pool):
                output = entry.runnable(*args)
            entry.cudagraph = npugraph
        else:
            # 后续执行，重放图
            entry.cudagraph.replay()
```

### 配置示例

```bash
python -m sglang.launch_server \
    --model-path /models/llama-7b \
    --enforce-piecewise-cuda-graph \
    --piecewise-cuda-graph-max-tokens 1024 \
    --device npu
```

## 禁用 Graph 的场景

以下场景会自动禁用 Graph：

| 场景 | 原因 |
|------|------|
| 启用 LoRA | 动态权重修改 |
| 启用 EP (Expert Parallelism) | 动态专家路由 |
| 启用 PP (Pipeline Parallelism) | 跨设备通信 |
| 启用 Mamba 模型 | 动态状态管理 |
| 启用 Encoder-Decoder | 双向注意力 |
| 启用 Custom AllReduce | 动态通信 |
| GGUF 量化模型 | 自定义反量化操作 |

```python
# 在 server_args.py 中
def _handle_piecewise_cuda_graph(self):
    if self.enable_lora:
        self.disable_piecewise_cuda_graph = True
    if self.ep_size > 1:
        self.disable_piecewise_cuda_graph = True
    # ... 其他条件
```

## 内存管理

### Graph 内存池

```python
# Graph 使用共享内存池
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph, pool=self.graph_pool):
    output = model(input)
```

### 内存估算

```python
# Graph 内存 ≈ cuda_graph_max_bs * 激活值大小
reserved_mem = cuda_graph_max_bs * 2  # GB (估算)
```

### 内存优化建议

```bash
# 减少内存占用
--cuda-graph-max-bs 80  # 降低最大批次
--mem-fraction-static 0.85  # 调整静态分配比例
```

## 与 Attention Backend 的交互

### Graph 模式下的 Attention

```python
# 在 ascend_backend.py 中
def init_forward_metadata_replay_cuda_graph(self, ...):
    """Graph 重放时初始化 metadata"""
    metadata = self.graph_metadata[bs]
    
    # 更新 block_tables
    metadata.block_tables[:bs, :max_seq_pages].copy_(
        self.req_to_token[req_pool_indices[:bs], :max_len][:, :: self.page_size]
        // self.page_size
    )
    
    # 更新 seq_lens
    metadata.seq_lens[:bs].copy_(seq_lens[:bs])
    
    self.graph_mode = True
```

### MLA 模型的特殊处理

```python
# MLA 模型需要更新 actual_seq_lengths_kv
def _get_update_attr_name(self):
    return self.attr_name[AttentionArch.MLA]  # "actual_seq_lengths_kv"

def _update_inputs(self, seq_lens):
    self.graphs[self.bs].update(
        cpu_update_input=[{"actual_seq_lengths_kv": seq_lens}]
    )
```

## 常见问题排查

### 1. Graph 捕获失败

**症状**: `RuntimeError: NPU graph capture failed`

**检查点**:
- 是否有动态控制流 (if/while)
- Tensor 形状是否固定
- 是否有不支持的算子

**解决方案**:
```bash
# 禁用 Graph 排查问题
--disable-cuda-graph

# 或使用分段 Graph
--enforce-piecewise-cuda-graph
```

### 2. Graph 重放结果错误

**症状**: 输出与 eager 模式不一致

**检查点**:
- 输入是否正确更新
- `seq_lens` 是否正确传递
- 检查 `_update_inputs` 是否被调用

**调试代码**:
```python
# 在 replay 方法中添加日志
logger.info(f"seq_lens: {seq_lens}")
logger.info(f"update_attr_name: {self.update_attr_name}")
```

### 3. 内存不足

**症状**: OOM 错误

**检查点**:
- `--cuda-graph-max-bs` 是否过大
- Graph 数量是否过多
- 内存池是否正确共享

**解决方案**:
```bash
# 降低批次大小
--cuda-graph-max-bs 80

# 减少批次列表
--cuda-graph-bs 1 2 4 8 16 32
```

### 4. 性能未提升

**症状**: Graph 模式性能与 eager 相近

**检查点**:
- 是否正确捕获 (检查日志)
- 批次大小是否匹配
- 是否有频繁的 Graph 切换

**调优建议**:
```bash
# 匹配实际批次大小
--cuda-graph-bs 8 16 24 32  # 根据实际负载调整

# 启用 torch.compile
--enable-torch-compile
```

## 性能调优建议

### 1. 批次大小选择

```bash
# 根据实际负载选择
# 低延迟场景: 小批次
--cuda-graph-bs 1 2 4 8

# 高吞吐场景: 大批次
--cuda-graph-bs 16 32 64 128
```

### 2. 与其他特性组合

| 组合 | 兼容性 | 建议 |
|------|--------|------|
| Graph + TP | ✅ | 推荐 |
| Graph + DP | ✅ | 推荐 |
| Graph + EP | ❌ | 自动禁用 |
| Graph + LoRA | ❌ | 自动禁用 |
| Graph + Speculative | ✅ | 需要特殊处理 |
| Graph + torch.compile | ✅ | 实验性 |

### 3. 预热策略

```python
# Graph 捕获需要预热
# 首次请求可能较慢，后续请求会快
```

## 与其他模块的关系

```
ACLGraph
├── 初始化: NPUGraphRunner.__init__()
├── 捕获: _capture_graph() → torch.npu.graph()
├── 重放: replay() → graph.replay()
├── 输入更新: _update_inputs() → graph.update()
├── Attention: AscendAttnBackend.graph_mode
├── 内存: 共享 graph_pool
└── 编译: torch.compile, NPUPiecewiseBackend
```

## 调试建议

### 1. 启用详细日志

```python
import logging
logging.getLogger("sglang.srt.model_executor").setLevel(logging.DEBUG)
```

### 2. 检查 Graph 状态

```python
# 在 NPUGraphRunner 中
print(f"Captured graphs: {list(self.graphs.keys())}")
print(f"Graph pool: {self.graph_pool}")
```

### 3. 性能分析

```bash
# 启用性能分析
--enable-profile-cuda-graph

# 查看分析结果
ls /tmp/graph_capture_profile/
```

### 4. 对比 Eager 模式

```bash
# 先用 eager 模式验证正确性
--disable-cuda-graph

# 再启用 Graph 优化
--cuda-graph-bs 1 2 4 8
```
