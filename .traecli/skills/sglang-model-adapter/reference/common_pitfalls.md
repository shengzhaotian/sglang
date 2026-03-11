# Common Pitfalls

## 1. Assuming CUDA Code Works on NPU

**Problem**: CUDA-specific implementations may fail silently or produce incorrect results on NPU.
**Solution**: Always test NPU-specific code paths with real inputs.

## 2. Skipping Accuracy Validation

**Problem**: Model loading successfully does NOT mean outputs are correct.
**Solution**: Run accuracy tests after real-weight validation.

## 3. Modifying Shared Code Without Guards

**Problem**: Changes to shared code may break CUDA path.
**Solution**: Use `is_npu()` check or create NPU-specific files in `hardware_backend/npu/`.

## 4. Ignoring Token Limits for Reasoning Models

**Problem**: Reasoning models output thinking process before answer, may get truncated.
**Solution**: Use `max_tokens >= 500` for reasoning models, check `finish_reason`.

## 5. Not Checking Head Count Before CUDA Graph

**Problem**: CUDA Graph fails with non-power-of-2 head counts.
**Solution**: Pre-flight check head count, use `--disable-cuda-graph` if needed.

## 6. Wrong Parallel Strategy for Model Size

**Problem**: OOM errors due to insufficient memory planning.
**Solution**: Always run Resource Assessment (Step 1.5) before loading.
