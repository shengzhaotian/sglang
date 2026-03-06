# SGLang Architecture Overview

This document provides a high-level architecture overview for agents and developers working with the SGLang codebase.

## 1. Project Overview

SGLang is a high-performance serving framework for large language models (LLMs) and multimodal models. It delivers low-latency and high-throughput inference from single GPU to large distributed clusters.

### Core Components

- **python/sglang/**: Main Python package
  - `srt/`: SGLang Runtime (SRT) - the backend engine for running models
  - `lang/`: Frontend language for structured generation
  - `multimodal_gen/`: Inference framework for image/video generation
  - `cli/`: Command-line interface (`sglang` command)
  - `api.py`: Public APIs

- **sgl-kernel/**: Optimized CUDA/ROCm compute kernels for LLM inference (separate PyPI package)

- **sgl-model-gateway/**: Rust-based model gateway for routing, load balancing, and multi-model serving

- **test/**: Test suites organized by category (registered/, srt/, unit/, manual/)

### Key Runtime Components (python/sglang/srt/)

- `models/`: Model implementations (Llama, Qwen, DeepSeek, etc.)
- `layers/`: Neural network layers (attention, MoE, quantization)
- `managers/`: Scheduler, tokenizer, and batch management
- `mem_cache/`: Memory pools and radix cache for KV cache
- `distributed/`: Tensor/pipeline/expert parallelism
- `entrypoints/`: HTTP server, gRPC server, OpenAI-compatible APIs

## 2. Build & Commands

### Installation

```bash
# From PyPI (recommended)
pip install uv
uv pip install sglang

# From source
git clone https://github.com/sgl-project/sglang.git
cd sglang
pip install -e "python"
```

### Development Setup

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### Running Tests

```bash
# Run a single test file
python3 test/srt/test_srt_endpoint.py

# Run a single test case
python3 test/srt/test_srt_endpoint.py TestSRTEndpoint.test_simple_decode

# Run test suite
python3 test/run_suite.py --hw cuda --suite stage-b-test-small-1-gpu
```

### Launching Server

```bash
# Basic server
python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --port 30000

# With Docker
docker run --gpus all --shm-size 32g -p 30000:30000 \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0
```

### sgl-kernel Build

```bash
cd sgl-kernel
make build
# With resource limits
make build MAX_JOBS=2
```

## 3. Code Style

### Formatting Tools

- **black**: Code formatting (26.1.0)
- **isort**: Import sorting (7.0.0)
- **ruff**: Linting (F401, F821 checks)
- **clang-format**: C++/CUDA formatting

### Key Conventions

- Avoid code duplication - extract shared functions for code appearing 5+ times
- Minimize device synchronization (`tensor.item()`, `tensor.cpu()`) - use vectorized code
- Cache runtime checks in model forward pass when results are consistent across layers
- Keep files under 2,000 lines - split into smaller files if needed
- Keep test files under 500 seconds execution time
- For new hardware/features: prefer new files over modifying existing code
- Place common path (NVIDIA/existing code) as first branch in if/else blocks

### Pre-commit Hooks

Configured in `.pre-commit-config.yaml`:
- trailing-whitespace, end-of-file-fixer
- check-yaml, check-toml, check-ast
- black-jupyter, isort, ruff
- clang-format for C++/CUDA
- codespell for spelling

## 4. Testing

### Framework

- Primary: Python `unittest` framework
- Secondary: `pytest` (asyncio_mode = auto)

### Test Organization

- `test/registered/`: CI-registered tests using registry system
- `test/srt/`: Runtime backend tests
- `test/unit/`: Unit tests
- `test/manual/`: Manual/nightly tests

### CI Registry System

Tests register with decorators:

```python
from sglang.test.ci.ci_register import register_cuda_ci, register_amd_ci

# Per-commit test (5090)
register_cuda_ci(est_time=80, suite="stage-b-test-small-1-gpu")

# Per-commit test (H100 for large models)
register_cuda_ci(est_time=120, suite="stage-b-test-large-1-gpu")

# Nightly test
register_cuda_ci(est_time=200, suite="nightly-1-gpu", nightly=True)
```

### Test Suites

- **Stage A**: `stage-a-test-1`, `stage-a-test-2`, `stage-a-test-cpu`
- **Stage B**: `stage-b-test-small-1-gpu` (5090), `stage-b-test-large-1-gpu` (H100), `stage-b-test-large-2-gpu`
- **Stage C**: 4-GPU and 8-GPU tests
- **Nightly**: `nightly-1-gpu`, `nightly-2-gpu`, etc.

### Writing Tests

- Use smaller models for faster tests
- Reuse server launches across test cases
- Use minimal GPUs needed
- Add `unittest.main()` for unittest, `sys.exit(pytest.main([__file__]))` for pytest

### Accuracy Testing

```bash
python3 -m sglang.launch_server --model Qwen/Qwen2-7B-Instruct
python3 -m sglang.test.few_shot_gsm8k --num-questions 200
```

## 5. Security

### API Authentication

- Router API key via `--api-key` flag
- Per-worker API keys for dynamic registration
- Control plane authentication: API keys or JWT/OIDC

### TLS/mTLS Support

- Gateway TLS: `--tls-cert-path`, `--tls-key-path`
- Worker mTLS: `--client-cert-path`, `--client-key-path`, `--ca-cert-path`

### RBAC Roles

- `admin`: Full control plane access
- `user`: Inference/data plane APIs only

### Data Privacy

- History backends: memory, none, oracle, postgres, redis
- Conversation state kept at router tier for enterprise privacy

## 6. Configuration

### Environment Variables

Prefix: `SGLANG_` or `SGL_`

**Key Variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `SGLANG_USE_MODELSCOPE` | Use ModelScope models | `false` |
| `SGLANG_ENABLE_TORCH_COMPILE` | Enable torch.compile | `true` |
| `SGLANG_SET_CPU_AFFINITY` | CPU affinity | `0` |
| `SGLANG_IS_FLASHINFER_AVAILABLE` | FlashInfer check | `true` |
| `SGLANG_SKIP_P2P_CHECK` | Skip P2P check | `false` |

**Performance Tuning:**

| Variable | Description |
|----------|-------------|
| `SGLANG_ENABLE_JIT_DEEPGEMM` | JIT DeepGEMM kernels |
| `SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS` | Parallel compilation workers |
| `SGLANG_CUSTOM_ALLREDUCE_ALGO` | All-reduce algorithm |

### Server Arguments

Key arguments for `launch_server`:
- `--model-path`: Model path or HuggingFace ID
- `--tp-size`: Tensor parallelism size
- `--dp-size`: Data parallelism size
- `--attention-backend`: FlashInfer, Triton, etc.
- `--quantization`: FP8, INT4, AWQ, GPTQ, etc.

### Hardware Support

- NVIDIA GPUs: GB200, B300, H100, A100, etc.
- AMD GPUs: MI300X, MI325X (via ROCm)
- Intel Xeon CPUs
- Google TPUs
- Ascend NPUs
- Intel XPUs

## 7. Ascend NPU Support

### Architecture

NPU-specific code is isolated in `hardware_backend/npu/`:

```
python/sglang/srt/hardware_backend/npu/
├── attention/           # NPU attention backends (FIA, Paged Attention)
├── graph_runner/        # ACLGraph implementation
├── modules/             # NPU-specific model modules (e.g., MLA)
├── moe/                 # MoE optimizations
├── quantization/        # ModelSlim quantization support
└── utils.py             # NPU utilities
```

### Installation

```bash
# Prerequisites: CANN toolkit and torch_npu
pip install torch-npu

# Install SGLang from source
pip install -e "python"
```

### Launching Server

```bash
# Basic NPU server
python3 -m sglang.launch_server \
    --model-path /models/llama-7b \
    --attention-backend ascend \
    --device npu \
    --port 30000

# With tensor parallelism
python3 -m sglang.launch_server \
    --model-path /models/deepseek-v3 \
    --tp-size 16 \
    --attention-backend ascend \
    --device npu
```

### Key Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ASCEND_USE_FIA` | Enable FIA attention backend | `false` |
| `SGLANG_NPU_USE_MLAPO` | Enable MLA preprocessing optimization | `false` |
| `SGLANG_USE_FIA_NZ` | Enable NZ format for FIA | `false` |
| `HCCL_BUFFSIZE` | HCCL buffer size | `1600` |

### Code Conventions for NPU

1. **Branch Isolation**: Use `is_npu()` for NPU-specific paths
   ```python
   from sglang.srt.utils import is_npu
   _is_npu = is_npu()
   
   if _is_npu:
       # NPU-specific implementation
       pass
   else:
       # CUDA/general implementation
       pass
   ```

2. **Code Location**: NPU-specific implementations go in `hardware_backend/npu/`

3. **Reuse First**: Reuse existing model classes when possible, add NPU branches

4. **Testing**: Always verify existing CUDA path still works after NPU changes

### NPU-Specific Features

| Feature | Status | Notes |
|---------|--------|-------|
| FIA (Fused Infer Attention) | ✅ | Enable with `ASCEND_USE_FIA=1` |
| ACLGraph | ✅ | Graph capture for decode optimization |
| DeepEP | ✅ | MoE communication optimization |
| DP Attention | ✅ | Data parallel attention |
| Speculative Decoding | ✅ | EAGLE/EAGLE3 support |
| ModelSlim Quantization | ✅ | W8A8, W4A4 quantization |

### Testing on NPU

```bash
# Run NPU-specific tests
python3 test/srt/test_srt_endpoint.py --device npu

# Run with NPU hardware flag
python3 test/run_suite.py --hw npu --suite stage-b-test-small-1-gpu
```

### NPU-Specific Skills

For NPU-related tasks, use the specialized skills:

| Skill | Purpose | When to Invoke |
|-------|---------|----------------|
| `sglang-model-adapter` | Model adaptation and debugging | Adapting new models, fixing NPU compatibility issues |
| `npu-testing-workflow` | Testing and benchmarking | Performance evaluation, test report generation |

### NPU Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    NPU Model Development Workflow                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Model Adaptation (sglang-model-adapter)                     │
│     ├── Analyze model architecture                              │
│     ├── Implement NPU support (branch isolation)                │
│     ├── Validate with dummy weights                             │
│     └── Validate with real weights                              │
│                         ↓                                        │
│  2. Completion Criteria                                         │
│     ├── Basic functional tests (mandatory)                      │
│     ├── Performance benchmark (recommended)                     │
│     └── Accuracy test (optional)                                │
│                         ↓                                        │
│  3. Comprehensive Testing (npu-testing-workflow)                │
│     ├── Service deployment scripts                              │
│     ├── Client benchmarking                                     │
│     └── Test report generation                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Workflow**: Use `sglang-model-adapter` first for model adaptation, then `npu-testing-workflow` for validation and benchmarking.

## CI Commands

Authorized users can comment on PRs:
- `/tag-run-ci-label`: Add "run-ci" label
- `/rerun-failed-ci`: Rerun failed tests
- `/rerun-stage <stage-name>`: Rerun specific stage

## Updating sgl-kernel

1. Submit PR to update sgl-kernel source
2. Bump sgl-kernel version (triggers PyPI release)
3. Update version in `python/pyproject.toml` and caller code
