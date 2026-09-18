# Kimi-K3 MLA C8: A5 decode and static DSpark integration

Base: `shengzhaotian/sglang:a5-k3-0828`, commit
`2287a82a9655dd9a98f5555ffd68c06acc47614d`.
The quantization/FIA contract follows
[PR #29641](https://github.com/sgl-project/sglang/pull/29641), head
`a0f61a3f23c15280beb170471747535d7da8b74d`.

## Supported path

- Target MLA decode and DSpark **static** target verification on Ascend.
- Absorbed Q: FP8 E4M3, dynamic per-token-head FP32 descale.
- Shared latent K/V: FP8 E4M3, checkpoint per-tensor FP32 descale per layer.
  The latent uses `fa_k.scale`, whose offset must be zero. `fa_v` parameters
  remain registered and loaded but do not supply a second latent/FIA scale;
  K and V checkpoint scales need not match.
- The independent 64-dimensional side stays BF16. Kimi NoPE skips rotation,
  not this side contribution. Output is BF16 before the existing V projection.
- Prefill attention stays BF16, including its existing prefix loop. It writes
  the same quantized latent cache and dequantizes historical latent values when
  reading prefix pages. No prefill FIA optimization is included.
- This uses CANN FIA v2, not a new Triton/TileLang attention kernel.
- Eager and NPUGraph share the C8 attention path. Graph capture uses the full
  fixed batch bucket; replay refreshes the existing page table and FIA v2's
  host `actual_seq_kvlen`. Static DSpark uses the final CPU verify boundary
  without adding its block width twice, and IDLE ranks update lengths to zero.

The Kimi-only `FAKQuant` compatibility override ignores the known-bad per-layer
`quant_type` label when both FA K/V scale entries declare `FAQuant`. It does not
rewrite the descriptor or change other models' scheme selection. Missing,
nonfinite or nonpositive K scales fail instead of defaulting to 1.
This follows the ModelSlim Kimi adapter and PR #29641: cache quantization uses
`1 / fa_k.scale`, and FIA latent K/V descaling uses `fa_k.scale`. For example,
the checkpoint values K=`0.00885009765625`, V=`0.01129150390625` are valid;
the runtime uses K without changing either checkpoint parameter.

For the FIA contract, prepare supplies `Q_side / (s_q * s_k)` in BF16, so a
uniform `s_q * s_k` descale on the combined QK term preserves the side term.
Only the latent is quantized in the persistent cache; the stored side is raw BF16.

## Runtime settings

Add to an otherwise working A5 target launch:

```text
--attention-backend ascend
--kv-cache-dtype fp8_e4m3
--cuda-graph-bs 8 16
```

For DSpark, additionally use the existing draft model/steps/token arguments and:

```text
--speculative-draft-kv-cache-dtype bf16
```

Set `SGLANG_RAGGED_VERIFY_MODE=static`. The existing DSpark implementation
requires PP=1; this patch does not change that restriction. `auto` does not
automatically select target C8. Explicit draft BF16 is required because otherwise
the draft inherits the target dtype; the draft MHA C8 path is outside this patch.

Target C8 graph support is implemented but still needs A5 capture/replay
validation. CP/A2A, ragged verify, disaggregation and CPU cache offload are not
supported by this integration. Existing BF16 paths are unchanged.

**Pending memory-budget correction:** physical target MLA cache uses
`512 * 1 + 64 * 2 = 640` bytes/token/layer, excluding the reserved page. The
existing shared configurator currently estimates 576 for dense FP8 MLA. That
common-code change is awaiting approval and is not included here. Do not rely
on automatic capacity sizing for a near-full-memory launch; cap cache capacity
and check actual allocation until this is corrected.

## Validation

Run the five `test_kimi_mla_fp8_*.py` files with pytest in an installed SGLang
environment. These CPU tests extract the real functions/classes and mock device
operators. They check loading, dtype/shape/scale plumbing, cache relocation,
prefill cache compatibility, decode/verify dispatch, fixed-bucket metadata and
replay length updates. They do not establish that the installed CANN supports
these operator shapes or graph capture, or that end-to-end A5 accuracy/performance
meets the target.

A5 acceptance still needs: checkpoint load; prefill then multi-step decode;
chunked/prefix prefill; static DSpark accepted/rejected proposals; and output
accuracy against BF16 using identical prompts. Measure performance separately.
Compare C8 eager with graph over multiple steps, changing batch sizes and crossing
page boundaries; include DP active-to-IDLE-to-active transitions. Check zero-Q
dynamic scales and FP8 cache scatter during actual graph capture/replay.
