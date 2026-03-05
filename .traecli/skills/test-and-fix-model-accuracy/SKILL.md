---
name: test-and-fix-model-accuracy
description: "Test model accuracy on NPU devices using benchmarks and fix accuracy issues. Invoke when user wants to run accuracy benchmarks, compare model performance against expected results, or debug accuracy degradation on Ascend NPU."
---

# Test and Fix Model Accuracy Skill

This skill tests model accuracy on NPU devices using standard benchmarks and helps identify and fix accuracy issues.

## When to Use

Invoke this skill when:
- User wants to run accuracy benchmarks on NPU
- User needs to compare model performance against expected results
- User reports accuracy degradation or unexpected results
- User asks to debug or fix accuracy issues

## Purpose

Automate the workflow of:
1. Testing model accuracy with standard benchmarks
2. Comparing results with expected accuracy thresholds
3. Identifying root causes of accuracy issues
4. Implementing and verifying fixes
5. Committing validated changes

## Prerequisites

- Ascend NPU device available
- CANN toolkit installed
- Model weights accessible
- Benchmark dataset available (GSM8K, MMLU, etc.)

## Execution Steps

### 1. Gather Context

Ask user for:
- Model path or HuggingFace/ModelScope ID
- Expected accuracy threshold (if known)
- Benchmark dataset preference (default: GSM8K)
- Any specific accuracy issues observed

### 2. Locate Test Infrastructure

Search for existing test files:
```bash
# Find NPU test files
find . -path "*/ascend/*" -name "*.py" | grep -i test

# Find accuracy test mixins
find . -name "*gsm8k*" -o -name "*accuracy*" -o -name "*benchmark*"
```

Key locations to check:
- `test/registered/ascend/` - NPU-specific tests
- `python/sglang/test/` - Test utilities and mixins

### 3. Prepare Test Environment

```bash
export PYTHONPATH=<project-root>/python:$PYTHONPATH
```

Ensure model is accessible. If download needed:
```python
# For ModelScope
from modelscope import snapshot_download
snapshot_download('<model-id>', cache_dir='<cache-path>')

# For HuggingFace
from huggingface_hub import snapshot_download
snapshot_download('<model-id>', cache_dir='<cache-path>')
```

### 4. Run Accuracy Test

Execute test with appropriate configuration:
```bash
python <test-file> --model-path <model-path> [other-args]
```

Or use Python test framework:
```bash
python -m pytest <test-file> -v
```

### 5. Analyze Results

Compare actual accuracy with expected:
- If accuracy >= expected: Report success
- If accuracy < expected: Proceed to diagnosis

### 6. Diagnose Issues

Common accuracy issue categories:

| Category | Symptoms | Investigation Areas |
|----------|----------|---------------------|
| Quantization | Accuracy drop with quantized models | `layers/quantization/` |
| Attention | Incorrect attention calculations | `layers/attention/` |
| Precision | Small accuracy degradation | `model_executor/` |
| Weight Loading | Missing or incorrect weights | Model adapter, registry |
| Backend | NPU-specific issues | Backend configuration |

### 7. Implement Fix

Based on diagnosis:
1. Identify affected code files
2. Make minimal, targeted changes
3. Document the fix rationale
4. Re-run test to verify

### 8. Commit Changes

```bash
git add <modified-files>
git commit -m "fix(accuracy): <description of fix>"
```

## Creating a New Accuracy Test

If no suitable test exists, create one:

```python
import unittest
from sglang.test.test_utils import CustomTestCase

class TestModelAccuracy(CustomTestCase):
    """Test model accuracy on benchmark dataset."""
    
    model = "<model-path>"
    expected_accuracy = 0.75  # Adjust based on model
    benchmark = "gsm8k"  # or "mmlu", "hellaswag", etc.
    
    other_args = [
        "--attention-backend", "ascend",
        # Add other necessary args
    ]
    
    def test_accuracy(self):
        # Run benchmark and check accuracy >= expected
        pass

if __name__ == "__main__":
    unittest.main()
```

## Common Accuracy Benchmarks

| Benchmark | Description | Typical Use |
|-----------|-------------|-------------|
| GSM8K | Grade school math | Reasoning ability |
| MMLU | Multi-subject QA | Knowledge |
| HellaSwag | Sentence completion | Common sense |
| HumanEval | Code generation | Coding ability |

## Troubleshooting

| Issue | Diagnostic | Solution |
|-------|------------|----------|
| Model download fails | Check network/auth | Use mirror or local path |
| Server fails to start | `npu-smi info` | Check NPU availability |
| Accuracy lower than expected | Compare with reference | Check config, backend, weights |
| Test timeout | Check server logs | Increase timeout or reduce load |

## Quality Gate

Before reporting completion:
- [ ] Accuracy test executed successfully
- [ ] Results compared with expected threshold
- [ ] If accuracy issue found: root cause identified and fix implemented
- [ ] Fix verified with re-run
- [ ] Changes committed with descriptive message
- [ ] Summary report provided to user

## Output Format

Provide user with:
1. Accuracy results (actual vs expected)
2. If issues found: diagnosis and fix summary
3. Commit hash (if changes made)
4. Recommendations for further testing (if applicable)
