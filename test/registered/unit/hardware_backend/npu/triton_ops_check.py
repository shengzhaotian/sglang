"""Numeric checks for the in-tree Triton-Ascend kernels, against torch references.

Run as a script with ``TRITON_INTERPRET=1`` in the *environment* (Triton decides
between the compiler and the interpreter when ``@triton.jit`` is applied, so
setting it from inside the process is too late):

    TRITON_INTERPRET=1 python triton_ops_check.py

``test_npu_triton_ops.py`` drives exactly that in a subprocess, so the
interpreter switch cannot leak into the rest of a pytest run. On an Ascend box
the same file runs without ``TRITON_INTERPRET`` to check the compiled kernels.

Two notes on the tolerances:

* The Triton *interpreter* truncates float32 -> bfloat16 instead of rounding to
  nearest even, so a bf16 result can sit one ULP below the torch reference.
  bf16 comparisons therefore allow one bf16 ULP; float32 comparisons are the
  strict ones.
* Summation order differs from torch's pairwise reduction, so a float32
  reduction result is compared with a few float32 ULPs of slack rather than
  bit-exactly.
"""

import math
import sys

import torch

from sglang.srt.hardware_backend.npu.dcp.ops import _lse_pack_cols, lse_combine
from sglang.srt.hardware_backend.npu.triton_ops.dcp_merge import (
    dcp_pack_send,
    lse_combine_shards,
)
from sglang.srt.hardware_backend.npu.triton_ops.kv_store import dcp_store_mla_kv
from sglang.srt.hardware_backend.npu.triton_ops.split_qk_norm import split_qk_rmsnorm

MAX_ULP = 1.0  # one bfloat16 ULP: the interpreter's truncating store
FP32_SLACK = 2**-19  # a few float32 ULPs, for a differently ordered reduction

_CHECKS = []


def check(fn):
    _CHECKS.append(fn)
    return fn


def _rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
    got, want = got.double(), want.double()
    scale = want.abs().amax().clamp_min(1e-12)
    return ((got - want).abs().amax() / scale).item()


def _max_ulp(got: torch.Tensor, want: torch.Tensor) -> float:
    """Largest distance between ``got`` and ``want`` in bfloat16 ULPs.

    The bfloat16 spacing at ``v`` is ``2 ** (floor(log2|v|) - 7)``, so a value
    rounded (or, in the Triton interpreter, truncated) from a slightly
    different float32 intermediate sits at most 1 ULP away.
    """
    got, want = got.double(), want.double()
    mag = want.abs()
    spacing = torch.where(
        mag > 0, torch.exp2(torch.floor(torch.log2(mag.clamp_min(1e-300))) - 7), 1e-300
    )
    return ((got - want).abs() / spacing).amax().item()


def _rms_norm_ref(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    f = x.float()
    return (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(
        x.dtype
    )


@check
def split_qk_rmsnorm_fp32():
    """float32 in / out: only the reduction order may differ."""
    torch.manual_seed(0)
    t, q_dim, k_dim, r_dim = 5, 1536, 512, 64
    x = torch.randn(t, q_dim + k_dim + r_dim)
    qw, kw = torch.randn(q_dim), torch.randn(k_dim)
    q, k_nope, k_pe = split_qk_rmsnorm(x, qw, kw, q_dim, k_dim, r_dim, 1e-6, 1e-5)
    q_err = _rel_err(q, _rms_norm_ref(x[:, :q_dim], qw, 1e-6))
    k_err = _rel_err(
        k_nope.squeeze(1), _rms_norm_ref(x[:, q_dim : q_dim + k_dim], kw, 1e-5)
    )
    assert torch.equal(k_pe.squeeze(1), x[:, q_dim + k_dim :]), "rope slice altered"
    assert q_err < FP32_SLACK and k_err < FP32_SLACK, (q_err, k_err)
    return f"q {q_err:.2e} k {k_err:.2e} rel, rope slice bit-exact"


@check
def split_qk_rmsnorm_bf16():
    """bfloat16 in / out, and a non-contiguous row stride."""
    torch.manual_seed(1)
    t, q_dim, k_dim, r_dim = 9, 96, 64, 16
    wide = torch.randn(t, q_dim + k_dim + r_dim + 7, dtype=torch.bfloat16)
    x = wide[:, : q_dim + k_dim + r_dim]  # row stride != row width
    qw = torch.randn(q_dim, dtype=torch.bfloat16)
    kw = torch.randn(k_dim, dtype=torch.bfloat16)
    q, k_nope, k_pe = split_qk_rmsnorm(x, qw, kw, q_dim, k_dim, r_dim, 1e-6, 1e-6)
    q_ulp = _max_ulp(q, _rms_norm_ref(x[:, :q_dim], qw, 1e-6))
    k_ulp = _max_ulp(
        k_nope.squeeze(1), _rms_norm_ref(x[:, q_dim : q_dim + k_dim], kw, 1e-6)
    )
    assert torch.equal(k_pe.squeeze(1), x[:, q_dim + k_dim :]), "rope slice altered"
    assert q_ulp <= MAX_ULP and k_ulp <= MAX_ULP, (q_ulp, k_ulp)
    return f"q {q_ulp:.2f} k {k_ulp:.2f} bf16 ULP, rope slice bit-exact"


@check
def dcp_kv_store():
    """Owner filter + both writes, against the torch reference."""
    torch.manual_seed(2)
    dcp_size, slots, d_c, d_r = 4, 24, 12, 6
    loc = torch.tensor([-1, 2, 6, 9, 10, 5, 0, 3, 7, 11])
    cache_k = torch.randn(loc.numel(), d_c, dtype=torch.bfloat16)
    cache_v = torch.randn(loc.numel(), d_r, dtype=torch.bfloat16)
    for rank in range(dcp_size):
        k_buf = torch.full((slots, d_c), -7.0, dtype=torch.bfloat16)
        v_buf = torch.full((slots, d_r), -7.0, dtype=torch.bfloat16)
        k_ref, v_ref = k_buf.clone(), v_buf.clone()
        dcp_store_mla_kv(k_buf, v_buf, cache_k, cache_v, loc, dcp_size, rank)
        for i, l in enumerate(loc.tolist()):
            if l >= 0 and l % dcp_size == rank:
                k_ref[l // dcp_size] = cache_k[i]
                v_ref[l // dcp_size] = cache_v[i]
        assert torch.equal(k_buf, k_ref), f"latent cache differs at rank {rank}"
        assert torch.equal(v_buf, v_ref), f"rope cache differs at rank {rank}"
    # dcp_size == 1 degenerates to "write every non-negative loc".
    k_buf = torch.zeros(slots, d_c, dtype=torch.bfloat16)
    v_buf = torch.zeros(slots, d_r, dtype=torch.bfloat16)
    dcp_store_mla_kv(k_buf, v_buf, cache_k, cache_v, loc, 1, 0)
    for i, l in enumerate(loc.tolist()):
        if l >= 0:
            assert torch.equal(k_buf[l], cache_k[i]), f"dcp=1 latent row {l}"
            assert torch.equal(v_buf[l], cache_v[i]), f"dcp=1 rope row {l}"
    return "bit-exact for every rank, negative locs skipped, dcp=1 degenerates"


@check
def dcp_pack_send_matches_torch():
    """The pack kernel reproduces cat + view + transpose + contiguous, bit for bit."""
    torch.manual_seed(3)
    n, b, h, d = 8, 5, 3, 32
    full = torch.randn(b, n * h, d + 4, dtype=torch.bfloat16)
    out = full[:, :, :d]  # a non-contiguous head slice, as FIA hands it over
    lse = torch.randn(b, n * h)
    got = dcp_pack_send(out, lse, n)
    cols = _lse_pack_cols(out.dtype)
    want = (
        torch.cat([out, lse.reshape(b, n * h, 1).view(out.dtype)], dim=-1)
        .view(b, n, h, d + cols)
        .transpose(0, 1)
        .contiguous()
    )
    assert torch.equal(got, want), "packed send buffer differs"
    assert torch.equal(
        got.view(torch.float32)[..., d // cols], lse.view(b, n, h).transpose(0, 1)
    )
    return "bit-exact payload and unpacked float32 LSE"


def _combine_reference(outs, lses, extra_out=None, extra_lse=None):
    if extra_out is not None:
        outs = torch.cat([outs, extra_out.unsqueeze(0)], dim=0)
        lses = torch.cat([lses, extra_lse.unsqueeze(0)], dim=0)
    n, d = outs.shape[0], outs.shape[-1]
    merged = lse_combine(outs.reshape(n, -1, d), lses.reshape(n, -1))
    finite = torch.where(torch.isfinite(lses), lses, torch.tensor(-math.inf))
    return merged, torch.logsumexp(finite.reshape(n, -1), dim=0)


def _random_shards(n, b, h, d, seed):
    torch.manual_seed(seed)
    outs = torch.randn(n, b, h, d, dtype=torch.bfloat16)
    lses = torch.randn(n, b, h) * 3
    lses[0, 0, 0] = math.inf  # FIA's empty-shard sentinel
    lses[1, 0, 0] = math.nan
    lses[:, 1, 1] = math.inf  # a row no rank holds KV for
    outs[:, 1, 1] = math.nan  # ... whose outputs are garbage
    return outs, lses


@check
def lse_combine_shards_matches_torch():
    n, b, h, d = 8, 4, 3, 64
    outs, lses = _random_shards(n, b, h, d, seed=4)
    got, got_lse = lse_combine_shards(outs, lses, return_lse=True)
    want, want_lse = _combine_reference(outs, lses)
    empty = ~torch.isfinite(want_lse)
    assert bool((got.reshape(-1, d)[empty] == 0).all()), "empty rows must merge to 0"
    assert bool((got_lse[empty] == -math.inf).all()), "empty rows need an -inf LSE"
    ulp = _max_ulp(got[~empty], want[~empty])
    lse_err = (got_lse[~empty] - want_lse[~empty]).abs().amax().item()
    assert ulp <= MAX_ULP, ulp
    assert lse_err < 1e-5, lse_err
    return f"out {ulp:.2f} bf16 ULP, lse {lse_err:.2e} abs, empty rows 0 / -inf"


@check
def lse_combine_shards_extra_shard():
    """The verify window merged as the (N+1)-th shard == merging over N+1."""
    n, b, h, d = 8, 4, 3, 64
    outs, lses = _random_shards(n, b, h, d, seed=5)
    extra_out = torch.randn(b, h, d, dtype=torch.bfloat16)
    extra_lse = torch.randn(b, h) * 3
    got = lse_combine_shards(outs, lses, extra_out, extra_lse)
    want, _ = _combine_reference(outs, lses, extra_out, extra_lse)
    ulp = _max_ulp(got, want)
    assert ulp <= MAX_ULP, ulp
    return f"{ulp:.2f} bf16 ULP, incl. rows whose only valid shard is the extra one"


@check
def lse_combine_shards_strided():
    """Strided shard views (as returned by the packed a2a) give the same result."""
    n, b, h, d = 4, 3, 2, 16
    outs, lses = _random_shards(n, b, h, d, seed=6)
    packed = torch.empty(n, b, h, d + 2, dtype=torch.bfloat16)
    packed[..., :d] = outs
    packed.view(torch.float32)[..., d // 2] = lses
    got = lse_combine_shards(packed[..., :d], packed.view(torch.float32)[..., d // 2])
    want = lse_combine_shards(outs.contiguous(), lses.contiguous())
    assert torch.equal(got, want), "strided views must merge identically"
    return "strided payload / LSE views match the contiguous merge"


@check
def lse_combine_shards_single_shard():
    """One valid shard must come back unchanged (up to the output cast)."""
    torch.manual_seed(7)
    n, b, h, d = 3, 2, 2, 8
    outs = torch.randn(n, b, h, d, dtype=torch.bfloat16)
    lses = torch.full((n, b, h), math.inf)
    lses[1] = torch.randn(b, h)
    got, got_lse = lse_combine_shards(outs, lses, return_lse=True)
    assert torch.equal(got, outs[1].reshape(b * h, d)), "single shard altered"
    assert torch.allclose(got_lse, lses[1].reshape(-1), atol=1e-6), "single shard LSE"
    return "bit-exact passthrough"


SKIP_EXIT_CODE = 77


def _triton_probe() -> str:
    """Empty string when Triton can actually run a kernel here.

    sglang installs a stub ``triton`` module on platforms without one
    (``sglang/_platform_stubs.py``); it imports and decorates fine and only
    fails at launch, so the probe has to launch something.
    """
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _copy_kernel(src_ptr, dst_ptr, N: tl.constexpr):
            offs = tl.arange(0, N)
            tl.store(dst_ptr + offs, tl.load(src_ptr + offs))

        src = torch.arange(4, dtype=torch.float32)
        dst = torch.zeros(4)
        _copy_kernel[(1,)](src, dst, N=4)
        if not torch.equal(src, dst):
            return "the probe kernel did not write its output"
    except Exception as exc:  # noqa: BLE001 - any failure means "unusable here"
        return f"{type(exc).__name__}: {exc}"
    return ""


def main() -> int:
    reason = _triton_probe()
    if reason:
        print(f"SKIP: triton cannot run kernels here ({reason})")
        return SKIP_EXIT_CODE
    failures = 0
    for fn in _CHECKS:
        try:
            detail = fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {fn.__name__}: {exc}")
        else:
            print(f"ok   {fn.__name__}: {detail}")
    print(f"{len(_CHECKS) - failures}/{len(_CHECKS)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
