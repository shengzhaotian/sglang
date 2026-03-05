# Test and Fix Model Accuracy Skill

This skill tests model accuracy on NPU devices using datasets like GSM8K and fixes any accuracy issues found.

## Purpose

Automate the workflow of:
1. Testing model accuracy on NPU with standard benchmarks
2. Comparing results with official accuracy benchmarks
3. Identifying and fixing accuracy issues in the codebase
4. Committing fixes to the repository

## Prerequisites

- Ascend NPU device (A2 or A3 series)
- CANN toolkit installed
- Model weights available locally or downloadable from ModelScope/HuggingFace
- GSM8K or other benchmark datasets

## Usage

### Basic Usage

```bash
# Test Qwen3-8B on GSM8K
python test/registered/ascend/llm_models/test_ascend_qwen3_8b.py
```

### Test with Custom Parameters

The test uses the GSM8KAscendMixin which provides:
- 5-shot GSM8K evaluation
- 200 questions by default
- Automatic server launch and teardown

## Test Files

Test files are located in:
- `test/registered/ascend/llm_models/` - NPU model tests
- `python/sglang/test/ascend/gsm8k_ascend_mixin.py` - GSM8K test mixin
- `python/sglang/test/ascend/test_ascend_utils.py` - Model path definitions

## Creating a New Model Test

```python
import unittest
from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

class TestQwen38B(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify model accuracy on GSM8K dataset."""

    model = QWEN3_8B_WEIGHTS_PATH
    accuracy = 0.75  # Expected accuracy threshold
    other_args = [
        "--chunked-prefill-size", "256",
        "--attention-backend", "ascend",
        "--disable-cuda-graph",
    ]

if __name__ == "__main__":
    unittest.main()
```

## Expected Accuracy Benchmarks

### Qwen3 Models on GSM8K (5-shot)
- Qwen3-0.6B: ~0.38
- Qwen3-8B: ~0.75 (actual: 0.955 on NPU)
- Qwen3-32B: ~0.85

### Other Models
- Llama-3.1-8B-Instruct: ~0.50
- DeepSeek-V3: ~0.80

## Workflow

1. **Model Setup**
   - Check if model exists locally
   - Download model if not available using ModelScope
   - Configure NPU environment variables

2. **Server Launch**
   - Launch SGLang server with NPU backend
   - Configure attention backend and memory settings
   - Wait for server readiness

3. **Accuracy Testing**
   - Run benchmark dataset evaluation
   - Collect accuracy metrics
   - Compare with expected accuracy

4. **Issue Detection**
   - If accuracy < expected:
     - Analyze model configuration
     - Check for known issues
     - Identify potential fixes

5. **Fix Implementation**
   - Apply fixes to model implementation
   - Re-run tests to verify fixes
   - Document changes made

6. **Commit Changes**
   - Stage modified files
   - Create commit with descriptive message

## Common Accuracy Issues

### 1. Quantization Issues
- **Symptom**: Accuracy drop with quantized models
- **Fix**: Adjust quantization parameters
- **Files**: `python/sglang/srt/layers/quantization/`

### 2. Attention Backend Issues
- **Symptom**: Incorrect attention calculations
- **Fix**: Update attention kernel or backend configuration
- **Files**: `python/sglang/srt/layers/attention/`

### 3. Numerical Precision Issues
- **Symptom**: Small accuracy degradation
- **Fix**: Adjust precision settings
- **Files**: `python/sglang/srt/model_executor/`

## Troubleshooting

### Model Download Fails
```python
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen3-8B', cache_dir='/root/.cache/modelscope')
```

### Server Fails to Start
- Check NPU availability: `npu-smi info`
- Verify CANN installation
- Check memory: reduce `--mem-fraction-static`

### Accuracy Lower Than Expected
1. Check model configuration
2. Verify attention backend
3. Test with different parameters
4. Compare with reference implementation

## Integration with CI

Tests can be registered with CI system:

```python
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)
```

## Example: Successful Test Run

```
Accuracy: 0.955
Invalid: 0.000
Latency: 36.874 s
Output throughput: 636.741 token/s

----------------------------------------------------------------------
Ran 1 test in 81.008s

OK
```

## References

- [SGLang Documentation](docs/platforms/ascend_npu.md)
- [NPU Testing Guide](docs/platforms/ascend_npu_best_practice.md)
- [Model Support List](docs/platforms/ascend_npu_support_models.md)
