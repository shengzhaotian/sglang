# NPU Speculative Decoding Reference

## Overview

Speculative Decoding is a technique to accelerate LLM inference by using a small draft model to predict multiple tokens, which are then verified by the target model, reducing decoding latency. This document introduces speculative decoding implementation on NPU.

## Core Files

```
python/sglang/srt/speculative/
├── spec_info.py                   # Speculative algorithm definitions
├── spec_utils.py                  # Utility functions
├── eagle_worker.py                # EAGLE Worker (non-overlapping)
├── eagle_worker_v2.py             # EAGLE Worker V2 (overlapping)
├── eagle_info.py                  # EAGLE input/output definitions
├── eagle_info_v2.py               # EAGLE V2 input/output definitions
├── eagle_utils.py                 # EAGLE utility functions
├── standalone_worker.py           # Standalone Worker
├── ngram_worker.py                # NGRAM Worker
├── multi_layer_eagle_worker.py    # Multi-layer EAGLE Worker
└── draft_utils.py                 # Draft utilities

python/sglang/srt/hardware_backend/npu/graph_runner/
├── eagle_draft_npu_graph_runner.py      # NPU EAGLE Draft Graph
└── eagle_draft_extend_npu_graph_runner.py  # NPU EAGLE Extend Graph
```

## Speculative Algorithm Types

### SpeculativeAlgorithm Enum

```python
class SpeculativeAlgorithm(Enum):
    EAGLE = auto()       # EAGLE algorithm
    EAGLE3 = auto()      # EAGLE3 algorithm (improved)
    STANDALONE = auto()  # Independent draft model
    NGRAM = auto()       # NGRAM autoregressive
    NONE = auto()        # No speculative decoding
```

### Algorithm Comparison

| Algorithm | Draft Model | Features | Use Case |
|-----------|-------------|----------|----------|
| EAGLE | Lightweight extension | Shares weights with Target | EAGLE-trained models |
| EAGLE3 | Improved EAGLE | Higher acceptance rate | EAGLE3-trained models |
| STANDALONE | Independent small model | Flexible selection | General scenarios |
| NGRAM | No model needed | Zero extra overhead | Simple scenarios |

## EAGLE Speculative Decoding

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    EAGLE Speculative Decoding Flow              │
├─────────────────────────────────────────────────────────────────┤
│  1. Draft Phase:                                                │
│     Input → Draft Model → Generate N candidate tokens           │
│                                                                 │
│  2. Verify Phase:                                               │
│     Candidate tokens → Target Model → Parallel verification     │
│                                                                 │
│  3. Accept/Reject:                                              │
│     Decide which tokens to accept based on probability          │
│     Accepted tokens added to output sequence                    │
└─────────────────────────────────────────────────────────────────┘
```

### EAGLE Worker Structure

```python
class EagleDraftWorker(BaseDraftWorker):
    def __init__(self, server_args, ...):
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        
        # Draft worker uses shared memory pool
        self.req_to_token_pool = target_worker.get_memory_pool()
        
        # Initialize draft model
        self.draft_worker = TpModelWorker(
            server_args=server_args,
            is_draft_worker=True,
            ...
        )
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--speculative-algorithm` | Speculative algorithm | None |
| `--speculative-draft-model-path` | Draft model path | None |
| `--speculative-num-steps` | Draft steps | 5 |
| `--speculative-eagle-topk` | EAGLE top-k | 8 |
| `--speculative-num-draft-tokens` | Draft token count | 64 |
| `--speculative-accept-threshold-single` | Single token accept threshold | 1.0 |
| `--speculative-accept-threshold-acc` | Cumulative accept threshold | 1.0 |

### NPU-Specific Implementation

#### Graph Runner

```python
# In eagle_draft_npu_graph_runner.py
class EAGLEDraftNpuGraphRunner:
    """NPU EAGLE Draft Graph Runner"""
    
    def __init__(self, ...):
        # Initialize NPU graph
        self.graph = torch.npu.CUDAGraph()
        
    def capture(self, ...):
        # Capture NPU graph
        with torch.npu.graph(self.graph):
            output = self.model(...)
```

#### Attention Backend

```python
# In ascend_backend.py
def forward_mtp(self, q, k, v, layer, forward_batch, ...):
    """Handle MTP (Multi-Token Prediction) speculative decoding"""
    
    if self.use_mla:
        # MTP handling for MLA models
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        
        # Use FIA for batched attention
        attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            query, k_cache, v_cache,
            block_table=self.forward_metadata.block_tables,
            ...
        )
```

## Configuration Examples

### EAGLE Configuration

```bash
python -m sglang.launch_server \
    --model-path /models/llama-70b \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 8 \
    --speculative-num-draft-tokens 64 \
    --device npu
```

### EAGLE3 Configuration

```bash
python -m sglang.launch_server \
    --model-path /models/llama-70b \
    --speculative-algorithm EAGLE3 \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 4 \
    --device npu
```

### STANDALONE Configuration

```bash
python -m sglang.launch_server \
    --model-path /models/llama-70b \
    --speculative-algorithm STANDALONE \
    --speculative-draft-model-path /models/llama-7b \
    --speculative-num-steps 5 \
    --device npu
```

### NGRAM Configuration

```bash
python -m sglang.launch_server \
    --model-path /models/llama-70b \
    --speculative-algorithm NGRAM \
    --speculative-ngram-min-match-window-size 1 \
    --speculative-ngram-max-match-window-size 12 \
    --device npu
```

## Environment Variables

| Variable Name | Description | Default |
|---------------|-------------|---------|
| `SGLANG_ENABLE_SPEC_V2` | Enable Spec V2 | 0 |
| `SGLANG_ENABLE_OVERLAP_PLAN_STREAM` | Enable stream overlap | 0 |

```bash
# Enable speculative decoding V2 and stream overlap
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
```

## Inference Flow Details

### Draft Phase

```python
def forward_draft(self, forward_batch: ForwardBatch):
    """Draft model forward pass"""
    
    # 1. Prepare input
    hidden_states = forward_batch.hidden_states
    
    # 2. Draft model inference (multiple steps)
    for step in range(self.speculative_num_steps):
        # Get draft tokens
        draft_logits = self.draft_worker.forward(hidden_states)
        draft_tokens = sample(draft_logits, topk=self.topk)
        
        # Update hidden states
        hidden_states = embed(draft_tokens)
    
    # 3. Return draft tokens
    return draft_tokens
```

### Verify Phase

```python
def forward_verify(self, forward_batch: ForwardBatch):
    """Target model verification"""
    
    # 1. Prepare tree attention mask
    tree_mask = build_tree_mask(draft_tokens)
    
    # 2. Target model parallel verification of all draft tokens
    target_logits = self.target_worker.forward(
        forward_batch,
        attention_mask=tree_mask
    )
    
    # 3. Accept/reject decision
    accepted_tokens = verify_tokens(
        draft_tokens, target_logits,
        threshold=self.accept_threshold
    )
    
    return accepted_tokens
```

### Tree Attention

```python
# In eagle_utils.py
def build_tree_kernel_efficient(
    draft_tokens: torch.Tensor,
    topk: int,
    ...
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build mask and position info for tree attention"""
    
    # Tree structure:
    #       [root]
    #      /  |  \
    #    [t1] [t2] [t3]  ← top-k candidates
    #    /|    |    |\
    #  ...    ...  ... ...
```

## Interaction with DP Attention

```python
# In eagle_worker_v2.py
if server_args.enable_dp_attention and self.speculative_algorithm.is_eagle3():
    # EAGLE3 + DP Attention requires special TP context
    ctx = draft_tp_context(get_attention_tp_group())
else:
    ctx = empty_context()

with ctx:
    self.draft_worker = TpModelWorker(...)
```

## Common Issue Troubleshooting

### 1. Draft Model Loading Failure

**Symptom**: `RuntimeError: Failed to load draft model`

**Checkpoints**:
- Is `--speculative-draft-model-path` correct
- Is draft model compatible with target model
- Is memory sufficient

### 2. Low Acceptance Rate

**Symptom**: Speculative decoding acceleration not obvious

**Checkpoints**:
- `--speculative-accept-threshold-single` setting
- Draft model quality
- Is `--speculative-eagle-topk` appropriate

**Tuning Suggestions**:
```bash
# Lower acceptance threshold
--speculative-accept-threshold-single 0.9
--speculative-accept-threshold-acc 0.95

# Adjust top-k
--speculative-eagle-topk 4
```

### 3. Graph Capture Failure

**Symptom**: `RuntimeError: NPU graph capture failed`

**Checkpoints**:
- Is there dynamic control flow
- Check if tensor shapes are fixed
- Check `--cuda-graph-bs` setting

### 4. Out of Memory

**Symptom**: OOM errors

**Checkpoints**:
- Draft model extra memory
- KV Cache double requirement
- `--mem-fraction-static` setting

**Solution**:
```bash
# Reduce memory usage
--mem-fraction-static 0.75
--speculative-num-draft-tokens 32
```

## Performance Tuning Suggestions

### 1. Choose Appropriate Algorithm

| Scenario | Recommended Algorithm |
|----------|----------------------|
| EAGLE-trained models | EAGLE3 |
| General acceleration | STANDALONE |
| No extra model | NGRAM |
| Highest speedup | EAGLE3 + DP Attention |

### 2. Parameter Tuning

```bash
# High acceptance rate configuration
--speculative-num-steps 5
--speculative-eagle-topk 4
--speculative-num-draft-tokens 32

# High throughput configuration
--speculative-num-steps 8
--speculative-eagle-topk 8
--speculative-num-draft-tokens 64
```

### 3. Combination with Other Features

```bash
# EAGLE3 + DP Attention + DeepEP (Recommended)
python -m sglang.launch_server \
    --model-path /models/deepseek-v3 \
    --speculative-algorithm EAGLE3 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --device npu
```

## Relationship with Other Modules

```
Speculative Decoding
├── Draft Model: TpModelWorker (is_draft_worker=True)
├── Target Model: TpModelWorker
├── Attention: AscendAttnBackend.forward_mtp()
├── Graph: EAGLEDraftNpuGraphRunner
├── Scheduling: Scheduler (overlap schedule)
└── Memory: Shared req_to_token_pool
```

## Debugging Suggestions

### 1. Print Acceptance Rate

```python
# Add logging in verify phase
accepted_count = sum(accepted_tokens)
total_count = len(draft_tokens)
accept_rate = accepted_count / total_count
logger.info(f"Accept rate: {accept_rate:.2%}")
```

### 2. Check Draft Output

```python
# Add logging in draft phase
logger.info(f"Draft tokens: {draft_tokens}")
logger.info(f"Draft logits shape: {draft_logits.shape}")
```

### 3. Verify Tree Mask

```python
# Check if tree mask is correct
assert tree_mask.shape == (num_draft_tokens, num_draft_tokens)
assert tree_mask.dtype == torch.bool
```
