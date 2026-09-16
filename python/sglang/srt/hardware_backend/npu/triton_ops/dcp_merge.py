"""Triton-Ascend kernels for the DCP cross-rank attention merge.

Two kernels replace the op sequence the torch merge needs (see
``hardware_backend/npu/dcp/ops.py``):

``pack``     ``cat`` + ``view`` + ``transpose`` + ``contiguous``  ->  one kernel
             writing this rank's partials straight into the rank-major
             all_to_all send buffer, with the fp32 LSE reinterpreted as
             trailing payload-dtype columns.
``combine``  ``isfinite`` + ``where`` x2 + ``unbind`` +
             ``npu_attention_update``                             ->  one kernel
             that treats a non-finite shard LSE as weight 0 in-kernel (as
             vllm-ascend's fused_sfa_dcp_lse_combine does) and accumulates in
             float32, so ``SGLANG_NPU_DCP_MERGE_FP32`` has nothing left to do.

The combine kernel takes an optional extra shard, which is how the DSPARK
target-verify path merges its current window as the (N+1)-th shard instead of
running a second merge over the cross-rank result.
"""

import torch
import triton
import triton.language as tl

from sglang.srt.hardware_backend.npu.triton_ops.utils import row_grid

# Triton resolves only constexpr / jit globals inside a kernel, so these
# reach the kernel as constexpr arguments rather than module globals
# (Ascend reports "cannot access global variable ... from within @jit").
_FINITE_MAX = 3.3e38  # above any real LSE, below +inf
_NEG_INF = float("-inf")

# Cap on the combine kernel's row tile (BLOCK_R). CANNON's triton_tile_cost
# probe only prices tl.load / tl.store / tl.dot tiles, not the tl.zeros fp32
# accumulator or the fp32 cast/multiply temporaries a row of the merge holds
# alongside it (see DEVELOPING.md); by hand, one row of the merge at D = 512
# holds roughly acc (BLOCK_R*512*4 B) + the bf16 payload load
# (BLOCK_R*512*2 B) + its fp32 cast and the weighted product
# (2 * BLOCK_R*512*4 B) at once, i.e. ~14 B/element. At BLOCK_R = 16 that is
# ~115 KiB against a 192 KiB Ascend910B UB; BLOCK_R = 32 would be ~230 KiB,
# over budget once the excluded temporaries are counted by hand, so the cap
# stays at 16 even though the probe alone would call 32 WITHIN. This is a
# host-side wrapper constant, not referenced inside @triton.jit.
_MAX_BLOCK_R = 16


def _lse_pack_cols(dtype: torch.dtype) -> int:
    elem = dtype.itemsize
    assert 4 % elem == 0, f"cannot pack an fp32 LSE into {dtype} columns"
    return 4 // elem


@triton.jit
def _pack_send_kernel(
    out_ptr,  # [B, N * h, D] partials, model dtype
    lse_ptr,  # [B, N * h] float32
    send_ptr,  # [N, B, h, D + cols] model dtype, contiguous
    send_f32_ptr,  # the same buffer viewed as float32
    o_b,
    o_h,  # element strides of out
    l_b,
    l_h,  # element strides of lse
    n_rows,  # B * N * h
    heads,  # N * h
    h,  # heads per rank
    bsz,
    SEND_ROW: tl.constexpr,  # elements per send row (D + cols)
    SEND_ROW_F32: tl.constexpr,  # the same row in float32 elements
    LSE_COL: tl.constexpr,  # float32 column holding the LSE
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """send[j, b, i] = (out[b, j * h + i], lse[b, j * h + i]) for every row."""
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    rows_per_program = (n_rows + n_programs - 1) // n_programs
    start_row = pid * rows_per_program
    end_row = tl.minimum(start_row + rows_per_program, n_rows)

    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    for row in range(start_row, end_row):
        b = row // heads
        head = row % heads
        j = head // h
        i = head % h
        dst = (j * bsz + b) * h + i
        payload = tl.load(out_ptr + b * o_b + head * o_h + cols, mask=mask, other=0)
        tl.store(send_ptr + dst * SEND_ROW + cols, payload, mask=mask)
        lse = tl.load(lse_ptr + b * l_b + head * l_h)
        tl.store(send_f32_ptr + dst * SEND_ROW_F32 + LSE_COL, lse)


@triton.jit
def _lse_combine_kernel(
    out_ptr,  # [R, D] merged output, model dtype, contiguous
    lse_out_ptr,  # [R] merged float32 LSE (natural log)
    src_ptr,  # [N, B, h, D] shard outputs, model dtype, D contiguous
    src_lse_ptr,  # [N, B, h] shard LSEs, float32
    x_ptr,  # [B, h, D] extra shard, or src_ptr when unused
    x_lse_ptr,  # [B, h] extra shard LSE, or src_lse_ptr when unused
    s_n,
    s_b,
    s_h,  # element strides of src
    l_n,
    l_b,
    l_h,  # element strides of src_lse
    x_b,
    x_h,  # element strides of x
    xl_b,
    xl_h,  # element strides of x_lse
    n_rows,  # B * h
    h,
    n_shards,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    FINITE_MAX: tl.constexpr,
    NEG_INF: tl.constexpr,
):
    """Exact softmax merge of ``n_shards`` (+1) partial attentions over disjoint KV.

    Vectorised over ``BLOCK_R`` rows at a time as well as ``BLOCK_D`` columns:
    each shard's LSE load is a ``[BLOCK_R]`` vector and its payload load a
    ``[BLOCK_R, BLOCK_D]`` tile, instead of one scalar LSE load plus one
    ``[BLOCK_D]`` vector load per row per shard. Same two-pass
    max-then-weighted-sum algorithm and the same sentinel handling as before,
    just batched across the row dimension so the per-shard loop issues one
    wide load instead of ``BLOCK_R`` narrow ones.

    A shard whose LSE is not finite (FIA returns +inf for an empty local KV)
    gets weight 0 and its output is never read into the accumulator; a row with
    no valid shard merges to 0 with an LSE of -inf, matching ``lse_combine``.
    """
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    rows_per_program = (n_rows + n_programs - 1) // n_programs
    start_row = pid * rows_per_program
    end_row = tl.minimum(start_row + rows_per_program, n_rows)

    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    row_base = tl.arange(0, BLOCK_R)

    for row0 in range(start_row, end_row, BLOCK_R):
        rows = row0 + row_base
        row_mask = rows < end_row
        b = rows // h
        i = rows % h

        # Pass 1: the largest finite LSE over the shards (the softmax pivot),
        # one [BLOCK_R] vector load per shard instead of BLOCK_R scalar loads.
        max_lse = tl.full([BLOCK_R], -FINITE_MAX, dtype=tl.float32)
        for n in range(n_shards):
            lse = tl.load(
                src_lse_ptr + n * l_n + b * l_b + i * l_h,
                mask=row_mask,
                other=-FINITE_MAX,
            )
            valid = (lse == lse) & (lse < FINITE_MAX) & (lse > -FINITE_MAX)
            max_lse = tl.maximum(max_lse, tl.where(valid, lse, -FINITE_MAX))
        if HAS_EXTRA:
            lse = tl.load(
                x_lse_ptr + b * xl_b + i * xl_h, mask=row_mask, other=-FINITE_MAX
            )
            valid = (lse == lse) & (lse < FINITE_MAX) & (lse > -FINITE_MAX)
            max_lse = tl.maximum(max_lse, tl.where(valid, lse, -FINITE_MAX))
        any_valid = max_lse > -FINITE_MAX
        pivot = tl.where(any_valid, max_lse, 0.0)

        # Pass 2: weighted sum of the shard outputs in float32, one
        # [BLOCK_R, BLOCK_D] tile load per shard instead of BLOCK_R separate
        # [BLOCK_D] loads.
        acc = tl.zeros([BLOCK_R, BLOCK_D], dtype=tl.float32)
        denom = tl.zeros([BLOCK_R], dtype=tl.float32)
        tile_mask = row_mask[:, None] & col_mask[None, :]
        for n in range(n_shards):
            lse = tl.load(
                src_lse_ptr + n * l_n + b * l_b + i * l_h,
                mask=row_mask,
                other=-FINITE_MAX,
            )
            valid = (lse == lse) & (lse < FINITE_MAX) & (lse > -FINITE_MAX)
            weight = tl.where(valid, tl.exp(lse - pivot), 0.0)
            ptrs = (
                src_ptr + n * s_n + b[:, None] * s_b + i[:, None] * s_h + cols[None, :]
            )
            vals = tl.load(ptrs, mask=tile_mask, other=0).to(tl.float32)
            acc += weight[:, None] * tl.where(valid[:, None], vals, 0.0)
            denom += weight
        if HAS_EXTRA:
            lse = tl.load(
                x_lse_ptr + b * xl_b + i * xl_h, mask=row_mask, other=-FINITE_MAX
            )
            valid = (lse == lse) & (lse < FINITE_MAX) & (lse > -FINITE_MAX)
            weight = tl.where(valid, tl.exp(lse - pivot), 0.0)
            ptrs = x_ptr + b[:, None] * x_b + i[:, None] * x_h + cols[None, :]
            vals = tl.load(ptrs, mask=tile_mask, other=0).to(tl.float32)
            acc += weight[:, None] * tl.where(valid[:, None], vals, 0.0)
            denom += weight

        scale = tl.where(denom > 0.0, 1.0 / tl.where(denom > 0.0, denom, 1.0), 0.0)
        merged = acc * scale[:, None]
        out_ptrs = out_ptr + rows[:, None] * D + cols[None, :]
        tl.store(out_ptrs, merged.to(out_ptr.dtype.element_ty), mask=tile_mask)
        if RETURN_LSE:
            merged_lse = tl.where(any_valid, pivot + tl.log(denom), NEG_INF)
            tl.store(lse_out_ptr + rows, merged_lse, mask=row_mask)


def dcp_pack_send(out: torch.Tensor, lse: torch.Tensor, n: int) -> torch.Tensor:
    """[B, N*h, D] + [B, N*h] float32 -> rank-major send buffer [N, B, h, D+cols].

    ``out`` / ``lse`` may be non-contiguous head slices of the FIA outputs; the
    kernel reads them with their own strides, so no staging copy is made.
    """
    b, heads, d = out.shape
    assert heads % n == 0, f"num_heads ({heads}) must be divisible by dcp ({n})"
    h = heads // n
    cols = _lse_pack_cols(out.dtype)
    lse_col = d * out.dtype.itemsize // 4
    assert (
        lse_col * 4 == d * out.dtype.itemsize
    ), f"head dim {d} in {out.dtype} does not align an fp32 LSE column"
    lse = lse.float()
    send = out.new_empty((n, b, h, d + cols))
    n_rows = b * heads
    _pack_send_kernel[(row_grid(n_rows),)](
        out,
        lse,
        send,
        send.view(torch.float32),
        out.stride(0),
        out.stride(1),
        lse.stride(0),
        lse.stride(1),
        n_rows,
        heads,
        h,
        b,
        SEND_ROW=d + cols,
        SEND_ROW_F32=(d + cols) // cols,
        LSE_COL=lse_col,
        D=d,
        BLOCK_D=triton.next_power_of_2(d),
    )
    return send


def lse_combine_shards(
    outs: torch.Tensor,
    lses: torch.Tensor,
    extra_out: torch.Tensor = None,
    extra_lse: torch.Tensor = None,
    return_lse: bool = False,
):
    """Merge partial attentions: outs [N, B, h, D], lses [N, B, h] (natural log).

    ``extra_out`` [B, h, D] / ``extra_lse`` [B, h] add one more shard (the
    DSPARK verify window merged as the (N+1)-th shard). Returns [B*h, D] in the
    shard dtype, plus the merged LSE [B*h] float32 with ``return_lse``.
    Accumulation is float32 regardless of the shard dtype.
    """
    n, b, h, d = outs.shape
    assert outs.stride(3) == 1, "shard outputs need a contiguous head dim"
    lses = lses.float()
    n_rows = b * h
    merged = outs.new_empty((n_rows, d))
    merged_lse = torch.empty(
        n_rows if return_lse else 0, dtype=torch.float32, device=outs.device
    )
    if n_rows == 0:
        # Ascend rejects a (0,) grid at launch; row_grid() already floors the
        # grid to 1 program, but skipping the launch entirely avoids relying
        # on that and matches the n_rows == 0 guard used elsewhere in this
        # file (dcp_store_mla_kv, split_qk_rmsnorm).
        return (merged, merged_lse) if return_lse else merged
    has_extra = extra_out is not None
    if has_extra:
        assert extra_out.stride(2) == 1, "the extra shard needs a contiguous head dim"
        extra_lse = extra_lse.float()
    grid = row_grid(n_rows)
    rows_per_program = -(-n_rows // grid)  # ceil div
    block_r = min(triton.next_power_of_2(rows_per_program), _MAX_BLOCK_R)
    _lse_combine_kernel[(grid,)](
        merged,
        merged_lse if return_lse else lses,
        outs,
        lses,
        extra_out if has_extra else outs,
        extra_lse if has_extra else lses,
        outs.stride(0),
        outs.stride(1),
        outs.stride(2),
        lses.stride(0),
        lses.stride(1),
        lses.stride(2),
        extra_out.stride(0) if has_extra else 0,
        extra_out.stride(1) if has_extra else 0,
        extra_lse.stride(0) if has_extra else 0,
        extra_lse.stride(1) if has_extra else 0,
        n_rows,
        h,
        n,
        D=d,
        BLOCK_D=triton.next_power_of_2(d),
        BLOCK_R=block_r,
        HAS_EXTRA=has_extra,
        RETURN_LSE=return_lse,
        FINITE_MAX=_FINITE_MAX,
        NEG_INF=_NEG_INF,
    )
    if return_lse:
        return merged, merged_lse
    return merged


def dcp_exchange_a2a_triton(out: torch.Tensor, lse: torch.Tensor, group):
    """Packed A2A exchange with the Triton pack kernel.

    out [B, N*h, D] + lse [B, N*h] -> ([N, B, h, D], [N, B, h] float32), views
    into the received buffer (the LSE is read back through its float32 view).
    """
    n = group.world_size
    d = out.shape[-1]
    cols = _lse_pack_cols(out.dtype)
    send = dcp_pack_send(out, lse, n)
    recv = torch.empty_like(send)
    group.all_to_all_single(recv.view(-1), send.view(-1))
    return recv[..., :d], recv.view(torch.float32)[..., d // cols]


def dcp_merge_a2a_triton(
    out: torch.Tensor,
    lse: torch.Tensor,
    group,
    return_lse: bool = False,
    merge_fp32: bool = False,
):
    """A2A merge with the Triton pack + combine kernels: out [B, N*h, D],
    lse [B, N*h] natural log -> [B, h, D] (with ``return_lse`` also the merged
    LSE [B, h] float32).

    ``merge_fp32`` is accepted for signature parity and ignored: the combine
    kernel always accumulates in float32 and casts once on the way out.
    """
    b, heads, d = out.shape
    h = heads // group.world_size
    recv_out, recv_lse = dcp_exchange_a2a_triton(out, lse, group)
    merged = lse_combine_shards(recv_out, recv_lse, return_lse=return_lse)
    if return_lse:
        merged, merged_lse = merged
        return merged.view(b, h, d), merged_lse.view(b, h)
    return merged.view(b, h, d)
