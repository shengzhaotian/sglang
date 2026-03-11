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

### 3) Choose adaptation strategy (new-model capable)

- Reuse existing sglang architecture if compatible.
- If architecture is missing or incompatible, implement native support:
    - add model adapter under `python/sglang/srt/models/`;
    - add processor under `python/sglang/srt/utils/hf_transformers_utils.py` when needed;
    - register architecture in `python/sglang/srt/models/registry.py`;
    - implement explicit weight loading/remap rules (including KV/QK norm sharding, rope variants).
- If remote code needs newer transformers symbols, do not upgrade dependency.
- If unavoidable, copy required modeling files from sibling transformers source and keep scope explicit.
- If failure is backend-specific (kernel/op/platform), patch minimal required code in current project repo.

### 4) Implement minimal code changes (in implementation roots)

- Touch only files required for this model adaptation.
- Keep weight mapping explicit and auditable.
- Avoid unrelated refactors.

### 5) Two-stage validation on Ascend (direct run)

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

### 6) Validate inference and features

- Send `GET /v1/models` first.
- Send at least one OpenAI-compatible text request.
- For multimodal models, require at least one text+image request.
- Validate architecture registration and loader path with logs (no unresolved architecture, no fatal missing-key errors).
- Try feature-first validation: EP + ACLGraph path first; eager path as fallback/isolation.
- If startup succeeds but first request crashes (false-ready), treat as runtime failure and continue root-cause isolation.
- For multimodal processor API mismatch (for example `skip_tensor_conversion` signature mismatch), use text-only isolation (`--limit-mm-per-prompt` set image/video/audio to 0) to separate processor issues from core weight loading issues.
- Capacity baseline by default (single machine): `context-length=128k` + `max-running-request=16`.
- Then expand concurrency (e.g., 32/64) if requested or feasible.

### 7) Backport, generate artifacts, and commit in delivery repo

- Generate tutorial markdown at implementation roots `./<ModelName>.md` following the standard template (Introduction, Supported Features, Environment Preparation with docker tabs, Deployment with serve script, Functional Verification with curl example, Accuracy Evaluation, Performance). Fill in model-specific details: HF path, hardware requirements, TP size, max-model-len, served-model-name, sample curl, and accuracy table.
- Confirm test script and tutorial doc are included in the staged files.
- Commit code changes once (single signed commit).

### 8) Prepare handoff artifacts

- Write comprehensive Chinese analysis report.
- Write compact Chinese runbook for server startup and validation commands.
- Include feature status matrix (supported / unsupported / checkpoint-missing / not-applicable).
- Include dummy-vs-real validation matrix and explicit non-equivalence notes.
- Include changed-file list, key logs, and final commit hash.

## Quality gate before final answer

- Service starts successfully from implementation roots with direct command.
- OpenAI-compatible inference request succeeds (not startup-only).
- Key feature set is attempted and reported: ACLGraph / DeepEP / MTP / multimodal.
- Capacity baseline (`128k + bs16`) result is reported, or explicit reason why not feasible.
- **Dummy stage evidence is present (if used), and real-weight stage evidence is present (mandatory).**
- Tutorial doc exists at `./<ModelName>.md` and follows the standard template (Introduction, Supported Features, Environment Preparation, Deployment, Functional Verification, Accuracy Evaluation, Performance).
- Exactly one signed commit contains all code changes in current working repo.
- Final response includes commit hash, file paths, key commands, known limits, and failure reasons where applicable.
