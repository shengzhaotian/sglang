# Quick Start Commands

## Standard LLM

```bash
export PYTHONPATH=${PWD}/python:$PYTHONPATH
python -m sglang.launch_server \
    --model-path /models/<model> --tp-size <tp> \
    --attention-backend ascend --device npu \
    --port 8000 --cuda-graph-bs 8 16 24 32
```

## MLA Model (DeepSeek-V2/V3)

```bash
export PYTHONPATH=${PWD}/python:$PYTHONPATH
export ASCEND_USE_FIA=1 SGLANG_NPU_USE_MLAPO=1 SGLANG_USE_FIA_NZ=1

python -m sglang.launch_server \
    --model-path /models/deepseek-v3 --tp-size 16 \
    --enable-dp-attention --attention-backend ascend --device npu \
    --quantization modelslim --cuda-graph-bs 8 16 24 32
```

## MoE with DeepEP

```bash
export PYTHONPATH=${PWD}/python:$PYTHONPATH
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

python -m sglang.launch_server \
    --model-path /models/<moe-model> --tp-size 8 --ep-size 8 \
    --moe-a2a-backend deepep --deepep-mode normal \
    --attention-backend ascend --device npu
```

## Speculative Decoding (EAGLE3)

```bash
export PYTHONPATH=${PWD}/python:$PYTHONPATH
export SGLANG_ENABLE_SPEC_V2=1

python -m sglang.launch_server \
    --model-path /models/<model> \
    --speculative-algorithm EAGLE3 \
    --speculative-num-steps 5 --speculative-eagle-topk 4 \
    --attention-backend ascend --device npu
```

## Accuracy Test Commands

```bash
# Math test
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<model>", "messages": [{"role": "user", "content": "What is 15 times 7?"}], "max_tokens": 100, "temperature": 0}'

# Knowledge test
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<model>", "messages": [{"role": "user", "content": "What is the largest planet in our solar system?"}], "max_tokens": 100, "temperature": 0}'

# Logic test
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<model>", "messages": [{"role": "user", "content": "If A > B and B > C, is A > C?"}], "max_tokens": 100, "temperature": 0}'
```

## Performance Benchmark

```bash
python -m sglang.bench_serving \
    --dataset-name random --backend sglang \
    --host 127.0.0.1 --port 8000 \
    --num-prompts 100 --random-input-len 128 --random-output-len 128
```
