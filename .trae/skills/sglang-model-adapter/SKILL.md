---
name: sglang-model-adapter
description: "Adapt and debug existing or new models for sglang on Ascend NPU. Implement in current project repo (local repo), validate via direct sglang serve from local repo, and deliver one signed commit in the current repo. Invoke when user asks to adapt new models, fix model compatibility, or debug NPU inference."
---

# sglang Ascend Model Adapter

## Overview

Adapt Hugging Face or local models to run on `sglang` with minimal changes, deterministic validation, and single-commit delivery. This skill is for both already-supported models and new architectures not yet registered in sglang.

## Hard constraints

- Never upgrade `transformers`.
- Primary implementation roots are current project repo.
- Start `sglang serve` from current project repo with adding environment variable `export PYTHONPATH=${PWD}/python:$PYTHONPATH`.
- Default API port is `8000` unless user explicitly asks otherwise.
- Feature-first default: try best to validate ACLGraph / DeepEP / DP-Attention / MTP / multimodal out-of-box.
- `--ep-size` `--moe-a2a-backend` `--deepep-mode` checks are MoE-only; for non-MoE models mark as not-applicable with evidence.
- If any feature cannot be enabled, keep evidence and explain reason in final report.
- Keep code changes minimal and focused on the target model.
- Final deliverable commit must be one single signed commit in the current working repo (`git commit -sm ...`).
- Keep final docs in Chinese and compact.
- **Dummy-first is encouraged for speed, but dummy is NOT fully equivalent to real weights.**
- **Never sign off adaptation using dummy-only evidence; real-weight gate is mandatory.**
- **CUDA graph batch size optimization**: Use minimal `--cuda-graph-bs` values (e.g., `1 2 4 8`) to reduce capture time and memory overhead. Default large batch sizes can cause excessive capture time and OOM during graph capture phase.

## Execution playbook

### 1) Collect context

- Confirm model path (default `/models/<model-name>`; if environment differs, confirm with user explicitly).
- Confirm implementation roots (current project repo).
- Confirm delivery root (the current git repo where the final commit is expected).
- Use default expected feature set: ACLGraph + DeepEP + MTP + multimodal (if model has VL capability).
- User requirements extend this baseline, not replace it.

### 2) Analyze model first

- Inspect `config.json`, processor files, modeling files, tokenizer files.
- Identify architecture class, attention variant, quantization type, and multimodal requirements.
- Check state-dict key prefixes (and safetensors index) to infer mapping needs.
- Decide whether support already exists in `python/sglang/srt/models/registry.py`.

### 3) Memory and configuration pre-check (CRITICAL)

**Goal: Avoid invalid startup attempts by validating memory requirements before model launch.**

- Calculate model weight memory footprint:
    - Parse `config.json` to get: `hidden_size` (H), `num_hidden_layers` (L), `intermediate_size` (H'), `vocab_size` (V), `num_attention_heads`, `num_key_value_heads`.
    - **Standard Transformer (MHA attention)** parameter count formula:
      ```
      params = 2 * V * H + H + L * (4 * H² + 3 * H * H' + 2 * H)
      ```
    - **GQA/MQA attention** (e.g., Llama3, Qwen2): adjust attention params:
      ```
      attention_params_per_layer = H² + 2 * H * (H / num_heads) * num_kv_heads + H²
      ```
      where `num_kv_heads` is typically smaller than `num_attention_heads`.
    - **MoE models** (e.g., DeepSeek, Mixtral): add expert parameters:
      ```
      expert_params = num_experts * (3 * H * expert_intermediate_size)
      shared_expert_params = 3 * H * shared_intermediate_size (if applicable)
      router_params = num_experts * H
      ```
    - Memory size: `weight_memory_GB = params * dtype_bytes / (1024³)`
    - For quantized models (FP8/INT8/INT4), multiply by compression ratio (0.5/0.5/0.25).
- Estimate KV cache memory requirement:
    - **Standard MHA/GQA KV cache** per token:
      ```
      kv_cache_per_token = 2 * L * num_kv_heads * head_dim * dtype_bytes
      ```
      where `head_dim = hidden_size / num_attention_heads`.
    - **MLA architecture** (e.g., DeepSeek-V2/V3): uses compressed KV:
      ```
      kv_cache_per_token = L * (kv_lora_rank + qk_rope_head_dim) * dtype_bytes
      ```
    - Total KV cache for context:
      ```
      kv_cache_total = kv_cache_per_token * context_length * max_running_requests
      kv_cache_GB = kv_cache_total / (1024³)
      ```
    - Default context length: 128k for baseline, adjust based on model config.
    - Default max_running_requests: 16.
- Validate against single-card HBM capacity:
    - Get NPU HBM size (e.g., Ascend 910B: 64GB, Ascend 910A: 32GB).
    - Account for activation overhead (~5-8GB for typical inference).
    - Total memory required: `model_weights + kv_cache + activation_overhead`.
    - If `total_memory > single_card_hbm`:
        - Calculate minimum required TP size: `ceil(total_memory / single_card_hbm)`.
        - **MUST NOT attempt tp=1 launch**; skip to recommended TP configuration.
        - Log memory analysis and TP recommendation in the report.
- Generate configuration report:
    - Model parameter count (billions).
    - Model weight size (GB).
    - KV cache size per token (bytes) and total for target context (GB).
    - Total memory requirement vs available HBM.
    - Recommended TP size with justification.
- **Gate condition**: If memory validation fails (insufficient HBM for tp=1), explicitly warn user and provide correct TP configuration before proceeding to Stage A.
- **Validation example** (LLaMA-7B, FP16, context=128k, bs=16):
    - Params: 2×32000×4096 + 4096 + 32×(4×4096² + 3×4096×11008 + 2×4096) ≈ 6.74B
    - Weights: 6.74B × 2 bytes ≈ 12.5 GB
    - KV cache: 2×32×32×128×131072×16×2 ≈ 32 GB (GQA with 8 kv_heads reduces to ~8 GB)
    - Total: ~12.5 + 8 + 6 ≈ 26.5 GB (fits in 32GB HBM with tp=1)

### 4) Choose adaptation strategy (new-model capable)

- Reuse existing sglang architecture if compatible.
- If architecture is missing or incompatible, implement native support:
    - add model adapter under `python/sglang/srt/models/`;
    - add processor under `python/sglang/srt/utils/hf_transformers_utils.py` when needed;
    - register architecture in `python/sglang/srt/models/registry.py`;
    - implement explicit weight loading/remap rules (including KV/QK norm sharding, rope variants).
- If remote code needs newer transformers symbols, do not upgrade dependency.
- If unavoidable, copy required modeling files from sibling transformers source and keep scope explicit.
- If failure is backend-specific (kernel/op/platform), patch minimal required code in current project repo.

### 5) Implement minimal code changes (in implementation roots)

- Touch only files required for this model adaptation.
- Keep weight mapping explicit and auditable.
- Avoid unrelated refactors.

### 6) Two-stage validation on Ascend (direct run)

#### Stage A: dummy fast gate (recommended first)

- Run with `--load-format dummy`.
- Goal: fast validate architecture path / operator path / API path.
- Do not treat `Application startup complete` as pass by itself; request smoke is mandatory.
- Require at least:
    - startup readiness (`/v1/models` 200),
    - one text request 200,
    - if VL model, one text+image request 200,
    - ACLGraph evidence where expected.

#### Stage B: real-weight mandatory gate (must pass before sign-off)

- Remove `--load-format dummy` and validate with real checkpoint.
- Goal: validate real-only risks:
    - weight key mapping,
    - KV/QK norm sharding with real tensor shapes,
    - load-time/runtime stability.
- Require HTTP 200 and non-empty output before declaring success.
- Do not pass Stage B on startup-only evidence.

### 7) Validate inference and features

- Send `GET /v1/models` first.
- Send at least one OpenAI-compatible text request.
- For multimodal models, require at least one text+image request.
- Validate architecture registration and loader path with logs (no unresolved architecture, no fatal missing-key errors).
- Try feature-first validation: EP + ACLGraph path first; eager path as fallback/isolation.
- If startup succeeds but first request crashes (false-ready), treat as runtime failure and continue root-cause isolation.
- For multimodal processor API mismatch (for example `skip_tensor_conversion` signature mismatch), use text-only isolation (`--limit-mm-per-prompt` set image/video/audio to 0) to separate processor issues from core weight loading issues.
- Capacity baseline by default (single machine): `context-length=128k` + `max-running-request=16`.
- Then expand concurrency (e.g., 32/64) if requested or feasible.

### 8) Accuracy validation via API (MANDATORY for real-weight gate)

**Goal: Validate model inference accuracy through API requests with expected output comparison.**

- Prerequisite: Model service must be running with real weights (Stage B passed).
- Execute accuracy validation sequence:
    1. **Service health check**:
       ```bash
       curl -s http://localhost:8000/v1/models | jq .
       ```
    2. **Text generation accuracy test**:
       ```bash
       curl -s http://localhost:8000/v1/chat/completions \
         -H "Content-Type: application/json" \
         -d '{
           "model": "<served-model-name>",
           "messages": [{"role": "user", "content": "Hello, how are you?"}],
           "max_tokens": 50,
           "temperature": 0
         }' | jq .
       ```
       - Verify: HTTP 200, non-empty `choices[0].message.content`.
    3. **Deterministic output validation** (for models with known outputs):
       - Use fixed prompt with `temperature: 0` for reproducible outputs.
       - Compare output against expected response pattern or reference output.
       - For instruction-tuned models, verify response follows instruction format.
    4. **Multimodal accuracy test** (if applicable):
       ```bash
       curl -s http://localhost:8000/v1/chat/completions \
         -H "Content-Type: application/json" \
         -d '{
           "model": "<served-model-name>",
           "messages": [{
             "role": "user",
             "content": [
               {"type": "text", "text": "Describe this image."},
               {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<base64_image>"}}
             ]
           }],
           "max_tokens": 100,
           "temperature": 0
         }' | jq .
       ```
    5. **Reasoning capability test** (optional, for reasoning models):
       - Submit math/logic problems with known answers.
       - Verify model produces correct reasoning steps and conclusions.
- Accuracy validation report must include:
    - All curl commands executed.
    - Raw API responses (HTTP status, response body).
    - Pass/fail assessment for each test case.
    - Any unexpected outputs or errors with analysis.
- **Gate condition**: Accuracy validation MUST pass before sign-off. If any test fails, root-cause analysis is required.

### 9) Backport, generate artifacts, and commit in delivery repo

- Generate tutorial markdown at implementation roots `./<ModelName>.md` following the standard template (Introduction, Supported Features, Environment Preparation with docker tabs, Deployment with serve script, Functional Verification with curl example, Accuracy Evaluation, Performance). Fill in model-specific details: HF path, hardware requirements, TP size, max-model-len, served-model-name, sample curl, and accuracy table.
- Confirm test script and tutorial doc are included in the staged files.
- Commit code changes once (single signed commit).

### 10) Prepare handoff artifacts

- Write comprehensive Chinese analysis report.
- Write compact Chinese runbook for server startup and validation commands.
- Include feature status matrix (supported / unsupported / checkpoint-missing / not-applicable).
- Include dummy-vs-real validation matrix and explicit non-equivalence notes.
- Include changed-file list, key logs, and final commit hash.

## Quality gate before final answer

- **Memory pre-check report exists** with model weight size, KV cache estimate, and TP recommendation.
- **Configuration validated**: TP size matches memory requirement (no tp=1 attempt when insufficient HBM).
- Service starts successfully from implementation roots with direct command.
- OpenAI-compatible inference request succeeds (not startup-only).
- **Accuracy validation completed**: curl commands executed, responses recorded, pass/fail assessed.
- Key feature set is attempted and reported: ACLGraph / DeepEP / MTP / multimodal.
- Capacity baseline (`32k + bs2`) result is reported, or explicit reason why not feasible.
- **Dummy stage evidence is present (if used), and real-weight stage evidence is present (mandatory).**
- Tutorial doc exists at `./<ModelName>.md` and follows the standard template (Introduction, Supported Features, Environment Preparation, Deployment, Functional Verification, Accuracy Evaluation, Performance).
- Exactly one signed commit contains all code changes in current working repo.
- Final response includes commit hash, file paths, key commands, known limits, and failure reasons where applicable.
