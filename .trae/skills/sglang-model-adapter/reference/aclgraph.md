# NPU ACLGraph Reference

## Overview

ACLGraph (Ascend Computing Language Graph) is a graph capture and execution optimization technology on Huawei Ascend NPUs. By capturing and replaying computation graphs, it significantly reduces kernel launch overhead and improves inference performance. This document introduces ACLGraph implementation and configuration in SGLang.

## Core Files

```
python/sglang/srt/hardware_backend/npu/graph_runner/
├── npu_graph_runner.py             # NPU Graph Runner main implementation
├── eagle_draft_npu_graph_runner.py # EAGLE Draft Graph Runner
└── eagle_draft_extend_npu_graph_runner.py  # EAGLE Extend Graph Runner

python/sglang/srt/compilation/
├── npu_piecewise_backend.py        # NPU piecewise graph compilation backend
└── compilation_config.py           # Compilation configuration

python/sglang/srt/model_executor/
├── cuda_graph_runner.py            # Base Graph Runner (CUDA/NPU shared)
└── model_runner.py                 # Model executor
```

## ACLGraph Principles

### Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    ACLGraph Workflow                            │
├─────────────────────────────────────────────────────────────────┤
│  1. Capture Phase:                                              │
│     - Execute one forward pass                                  │
│     - Capture all NPU operations into graph                     │
│     - Store graph and memory pool                               │
│                                                                 │
│  2. Replay Phase:                                               │
│     - Update input data (input_ids, positions, seq_lens, etc.)  │
│     - Replay captured graph                                     │
│     - Get output                                                │
└─────────────────────────────────────────────────────────────────┘
```

### Performance Advantages

| Scenario | Without Graph | With Graph | Improvement |
|----------|---------------|------------|-------------|
| Small batch decode | High kernel launch overhead | Low overhead | 2-3x |
| Fixed shape computation | Dynamic scheduling | Static execution | 1.5-2x |
| Continuous inference | Re-schedule each time | Capture once, replay many | Significant |

## NPUGraphRunner Class

### Class Structure

```python
class NPUGraphRunner(CudaGraphRunner):
    """NPU Graph Runner, inherits from CudaGraphRunner"""
    
    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        self._init_arch_map()
    
    def _init_arch_map(self):
        # Set update attributes based on attention architecture type
        self.attr_name = {
            AttentionArch.MLA: "actual_seq_lengths_kv",
            AttentionArch.MHA: "context_lens",
        }
```

### Key Methods

#### 1. Create NPU Graph

```python
def _create_device_graph(self):
    """Create NPU Graph object"""
    return torch.npu.NPUGraph()
```

#### 2. Capture Graph

```python
def _capture_graph(self, graph, pool, stream, run_once_fn):
    """Capture NPU Graph"""
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

#### 3. Replay Graph

```python
def replay(self, forward_batch: ForwardBatch, ...):
    """Replay NPU Graph"""
    # Prepare input
    self.replay_prepare(forward_batch, pp_proxy_tensors)
    
    # Update sequence lengths (use thread to avoid blocking)
    seq_lens = forward_batch.seq_lens.cpu().tolist() + [0] * (self.bs - self.raw_bs)
    thread = threading.Thread(target=self._update_inputs, args=(seq_lens,))
    thread.start()
    
    # Replay graph
    self.graphs[self.bs].replay()
    thread.join()
    
    return output
```

#### 4. Update Inputs

```python
def _update_inputs(self, seq_lens):
    """Update Graph inputs (via CPU)"""
    if isinstance(self.update_attr_type, torch.Tensor):
        seq_lens = torch.from_numpy(np.array(seq_lens).astype(np.int32))
    
    self.graphs[self.bs].update(
        cpu_update_input=[{self.update_attr_name: seq_lens}]
    )
```

## Configuration Parameters

### Server Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cuda-graph-bs` | None | Batch size list for graph capture |
| `--cuda-graph-max-bs` | Auto | Graph maximum batch size |
| `--disable-cuda-graph` | False | Disable Graph |
| `--disable-cuda-graph-padding` | False | Disable Graph Padding |
| `--enable-torch-compile` | False | Enable torch.compile |
| `--enable-profile-cuda-graph` | False | Enable Graph capture profiling |

### Piecewise Graph Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--disable-piecewise-cuda-graph` | False | Disable piecewise Graph |
| `--enforce-piecewise-cuda-graph` | False | Force enable piecewise Graph |
| `--piecewise-cuda-graph-max-tokens` | None | Piecewise Graph max tokens |
| `--piecewise-cuda-graph-tokens` | None | Piecewise Graph token list |
| `--piecewise-cuda-graph-compiler` | "eager" | Piecewise Graph compiler |

## Configuration Examples

### Basic Configuration

```bash
python -m sglang.launch_server \
    --model-path /models/llama-7b \
    --cuda-graph-bs 1 2 4 8 16 32 \
    --device npu
```

### Large Batch Configuration

```bash
python -m sglang.launch_server \
    --model-path /models/llama-70b \
    --tp-size 8 \
    --cuda-graph-max-bs 160 \
    --cuda-graph-bs 1 2 4 8 16 32 64 128 160 \
    --device npu
```

### Disable Graph (for Debugging)

```bash
python -m sglang.launch_server \
    --model-path /models/llama-7b \
    --disable-cuda-graph \
    --device npu
```

### Enable torch.compile

```bash
python -m sglang.launch_server \
    --model-path /models/llama-7b \
    --enable-torch-compile \
    --device npu
```

### Profiling

```bash
python -m sglang.launch_server \
    --model-path /models/llama-7b \
    --enable-profile-cuda-graph \
    --device npu

# Profile results saved in /tmp/graph_capture_profile/
```

## Piecewise CUDA Graph

### Concept

Piecewise CUDA Graph splits the model into multiple segments, each captured and executed independently, suitable for:
- Dynamic shape scenarios
- Models with many conditional branches
- Memory-constrained scenarios

### NPUPiecewiseBackend

```python
class NPUPiecewiseBackend(CUDAPiecewiseBackend):
    def __call__(self, *args):
        runtime_shape = args[self.sym_shape_indices[0]]
        
        if entry.cudagraph is None:
            # First execution, capture graph
            npugraph = torch.npu.NPUGraph()
            with torch.npu.graph(npugraph, pool=self.graph_pool):
                output = entry.runnable(*args)
            entry.cudagraph = npugraph
        else:
            # Subsequent execution, replay graph
            entry.cudagraph.replay()
```

### Configuration Example

```bash
python -m sglang.launch_server \
    --model-path /models/llama-7b \
    --enforce-piecewise-cuda-graph \
    --piecewise-cuda-graph-max-tokens 1024 \
    --device npu
```

## Scenarios Where Graph is Disabled

Graph is automatically disabled in the following scenarios:

| Scenario | Reason |
|----------|--------|
| LoRA enabled | Dynamic weight modification |
| EP (Expert Parallelism) enabled | Dynamic expert routing |
| PP (Pipeline Parallelism) enabled | Cross-device communication |
| Mamba models enabled | Dynamic state management |
| Encoder-Decoder enabled | Bidirectional attention |
| Custom AllReduce enabled | Dynamic communication |
| GGUF quantized models | Custom dequantization operations |

```python
# In server_args.py
def _handle_piecewise_cuda_graph(self):
    if self.enable_lora:
        self.disable_piecewise_cuda_graph = True
    if self.ep_size > 1:
        self.disable_piecewise_cuda_graph = True
    # ... other conditions
```

## Memory Management

### Graph Memory Pool

```python
# Graph uses shared memory pool
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph, pool=self.graph_pool):
    output = model(input)
```

### Memory Estimation

```python
# Graph memory ≈ cuda_graph_max_bs * activation size
reserved_mem = cuda_graph_max_bs * 2  # GB (estimate)
```

### Memory Optimization Suggestions

```bash
# Reduce memory usage
--cuda-graph-max-bs 80  # Lower max batch
--mem-fraction-static 0.85  # Adjust static allocation ratio
```

## Interaction with Attention Backend

### Attention in Graph Mode

```python
# In ascend_backend.py
def init_forward_metadata_replay_cuda_graph(self, ...):
    """Initialize metadata during Graph replay"""
    metadata = self.graph_metadata[bs]
    
    # Update block_tables
    metadata.block_tables[:bs, :max_seq_pages].copy_(
        self.req_to_token[req_pool_indices[:bs], :max_len][:, :: self.page_size]
        // self.page_size
    )
    
    # Update seq_lens
    metadata.seq_lens[:bs].copy_(seq_lens[:bs])
    
    self.graph_mode = True
```

### Special Handling for MLA Models

```python
# MLA models need to update actual_seq_lengths_kv
def _get_update_attr_name(self):
    return self.attr_name[AttentionArch.MLA]  # "actual_seq_lengths_kv"

def _update_inputs(self, seq_lens):
    self.graphs[self.bs].update(
        cpu_update_input=[{"actual_seq_lengths_kv": seq_lens}]
    )
```

## Common Issue Troubleshooting

### 1. Graph Capture Failure

**Symptom**: `RuntimeError: NPU graph capture failed`

**Checkpoints**:
- Is there dynamic control flow (if/while)
- Are tensor shapes fixed
- Are there unsupported operators

**Solution**:
```bash
# Disable Graph to troubleshoot
--disable-cuda-graph

# Or use piecewise Graph
--enforce-piecewise-cuda-graph
```

### 2. Graph Replay Result Errors

**Symptom**: Output inconsistent with eager mode

**Checkpoints**:
- Are inputs correctly updated
- Is `seq_lens` correctly passed
- Check if `_update_inputs` is called

**Debug Code**:
```python
# Add logging in replay method
logger.info(f"seq_lens: {seq_lens}")
logger.info(f"update_attr_name: {self.update_attr_name}")
```

### 3. Out of Memory

**Symptom**: OOM errors

**Checkpoints**:
- Is `--cuda-graph-max-bs` too large
- Are there too many graphs
- Is memory pool correctly shared

**Solution**:
```bash
# Lower batch size
--cuda-graph-max-bs 80

# Reduce batch list
--cuda-graph-bs 1 2 4 8 16 32
```

### 4. No Performance Improvement

**Symptom**: Graph mode performance similar to eager

**Checkpoints**:
- Is graph correctly captured (check logs)
- Do batch sizes match
- Is there frequent Graph switching

**Tuning Suggestions**:
```bash
# Match actual batch sizes
--cuda-graph-bs 8 16 24 32  # Adjust based on actual workload

# Enable torch.compile
--enable-torch-compile
```

## Performance Tuning Suggestions

### 1. Batch Size Selection

```bash
# Choose based on actual workload
# Low latency scenario: small batches
--cuda-graph-bs 1 2 4 8

# High throughput scenario: large batches
--cuda-graph-bs 16 32 64 128
```

### 2. Combination with Other Features

| Combination | Compatibility | Suggestion |
|-------------|---------------|------------|
| Graph + TP | ✅ | Recommended |
| Graph + DP | ✅ | Recommended |
| Graph + EP | ❌ | Auto-disabled |
| Graph + LoRA | ❌ | Auto-disabled |
| Graph + Speculative | ✅ | Requires special handling |
| Graph + torch.compile | ✅ | Experimental |

### 3. Warmup Strategy

```python
# Graph capture requires warmup
# First request may be slow, subsequent requests will be fast
```

## Relationship with Other Modules

```
ACLGraph
├── Initialization: NPUGraphRunner.__init__()
├── Capture: _capture_graph() → torch.npu.graph()
├── Replay: replay() → graph.replay()
├── Input Update: _update_inputs() → graph.update()
├── Attention: AscendAttnBackend.graph_mode
├── Memory: Shared graph_pool
└── Compilation: torch.compile, NPUPiecewiseBackend
```

## Debugging Suggestions

### 1. Enable Verbose Logging

```python
import logging
logging.getLogger("sglang.srt.model_executor").setLevel(logging.DEBUG)
```

### 2. Check Graph Status

```python
# In NPUGraphRunner
print(f"Captured graphs: {list(self.graphs.keys())}")
print(f"Graph pool: {self.graph_pool}")
```

### 3. Profiling

```bash
# Enable profiling
--enable-profile-cuda-graph

# View profile results
ls /tmp/graph_capture_profile/
```

### 4. Compare with Eager Mode

```bash
# First verify correctness with eager mode
--disable-cuda-graph

# Then enable Graph optimization
--cuda-graph-bs 1 2 4 8
```
