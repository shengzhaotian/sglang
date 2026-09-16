"""Pure-torch helpers for decode context parallel (DCP) on the Ascend NPU path.

Layout (see ``PagedTokenToKVPoolAllocator`` built with ``page_size * dcp_size``):
a request position ``p`` lives at virtual slot ``v = page * P * c + p % (P * c)``;
rank ``v % c`` owns it and stores it at physical slot ``v // c``. Hence rank
``r`` finds its tokens of virtual page ``page`` in physical page ``page``,
position-ordered, and every rank uses the same block table
``req_to_token[:, 0:len:P*c] // (P*c)`` with block size ``P``.

Nothing here imports torch_npu, so the math is unit-testable on CPU. Functions
that communicate take a ``GroupCoordinator``-like ``group`` argument.
"""

import math
from typing import List, NamedTuple, Optional, Sequence, Union

import torch

from sglang.srt.layers.dcp.layout import get_dcp_lens


def dcp_physical_write_loc(
    loc: torch.Tensor, dcp_size: int, dcp_rank: int, pad_slot: int = 0
) -> torch.Tensor:
    """Map virtual write locs to this rank's physical slots.

    Tokens owned by another rank (and negative locs) go to ``pad_slot`` so the
    write keeps a static shape (graph-safe, no boolean indexing).
    """
    if dcp_size == 1:
        return loc
    owned = (loc >= 0) & (loc % dcp_size == dcp_rank)
    return torch.where(owned, loc // dcp_size, loc.new_full((), pad_slot))


def dcp_block_tables(
    req_to_token_rows: torch.Tensor, max_len, page_size: int, dcp_size: int
) -> torch.Tensor:
    """Per-rank FIA block table: virtual page ids (== physical page ids)."""
    stride = page_size * dcp_size
    return (req_to_token_rows[:, 0:max_len:stride] // stride).to(torch.int32)


def dcp_local_seq_lens(
    seq_lens: Union[torch.Tensor, Sequence[int]], dcp_size: int, dcp_rank: int
) -> Union[torch.Tensor, List[int]]:
    """KV length this rank holds per request (owner rule pos % c == rank)."""
    if isinstance(seq_lens, torch.Tensor):
        return get_dcp_lens(seq_lens, dcp_size, dcp_rank)
    if dcp_size == 1:
        return list(seq_lens)
    return [int(n) // dcp_size + int(dcp_rank < int(n) % dcp_size) for n in seq_lens]


def dcp_verify_history_local_lens(
    seq_lens_with_w: Union[torch.Tensor, Sequence[int]],
    w: int,
    dcp_size: int,
    dcp_rank: int,
) -> Union[torch.Tensor, List[int]]:
    """History KV length this rank holds for a target-verify window.

    ``seq_lens_with_w`` already counts the ``w`` verify tokens (written to the
    cache before attention); history is the global prefix before them,
    ``max(len - w, 0)`` (0 for idle / graph-padding rows), sharded by the
    owner rule pos % c == rank.
    """
    if isinstance(seq_lens_with_w, torch.Tensor):
        history = (seq_lens_with_w - w).clamp_min(0)
    else:
        history = [max(int(n) - int(w), 0) for n in seq_lens_with_w]
    return dcp_local_seq_lens(history, dcp_size, dcp_rank)


def dcp_interleave_pages(gathered: torch.Tensor, total_len: int) -> torch.Tensor:
    """[c, npages, P, *tail] (rank-major) -> [npages*P*c, *tail][:total_len].

    Row ``j`` of physical page ``page`` on rank ``r`` holds position
    ``page * P * c + j * c + r``, so page-major, row-major, rank-minor order is
    the global position order.
    """
    c, npages, page_size = gathered.shape[:3]
    tail = gathered.shape[3:]
    rows = gathered.movedim(0, 2).reshape(npages * page_size * c, *tail)
    return rows[:total_len]


class DcpPrefixChunk(NamedTuple):
    """Global prefix positions [start, end) and the DCP pages that cover them."""

    start: int
    end: int
    first_page: int  # index into the request's DCP block table
    num_pages: int
    row_offset: int  # start - first_page * page_size * dcp_size


def dcp_prefix_chunk_plan(
    prefix_len: int, chunk_tokens: int, page_size: int, dcp_size: int
) -> List[DcpPrefixChunk]:
    """Split a request's global prefix [0, prefix_len) into chunks.

    Depends only on global lengths, so every DCP rank derives the same plan and
    issues the same all-gathers in the same order (none for prefix_len == 0).
    """
    assert chunk_tokens > 0, f"chunk_tokens must be positive, got {chunk_tokens}"
    stride = page_size * dcp_size
    plan = []
    for start in range(0, int(prefix_len), chunk_tokens):
        end = min(start + chunk_tokens, int(prefix_len))
        first_page = start // stride
        num_pages = (end + stride - 1) // stride - first_page
        plan.append(
            DcpPrefixChunk(
                start, end, first_page, num_pages, start - first_page * stride
            )
        )
    return plan


def dcp_gather_chunk_rows(
    local_pages: torch.Tensor, row_offset: int, num_rows: int, group
) -> torch.Tensor:
    """Globally ordered rows of one prefix chunk from every rank's local pages.

    local_pages: [num_pages, P, *tail], this rank's physical pages selected by
    the (rank independent) DCP block table, so all ranks share the shape.
    Returns rows [row_offset, row_offset + num_rows) of the interleaved pages.
    """
    c = group.world_size
    gathered = group.all_gather(local_pages.contiguous(), dim=0)
    gathered = gathered.view(c, *local_pages.shape)
    return dcp_interleave_pages(gathered, row_offset + num_rows)[row_offset:]


def lse_combine(outs: torch.Tensor, lses: torch.Tensor) -> torch.Tensor:
    """Exact softmax merge of N partial attentions over disjoint KV shards.

    outs: [N, B, H, D], lses: [N, B, H] natural log. A shard with a non-finite
    LSE (no KV on that rank) contributes nothing; if every shard is empty the
    result is 0.
    """
    out_dtype = outs.dtype
    lses = lses.float()
    valid = torch.isfinite(lses)
    max_lse = torch.where(valid, lses, -math.inf).amax(dim=0)
    max_lse = torch.where(torch.isfinite(max_lse), max_lse, 0.0)
    weights = torch.where(valid, torch.exp(lses - max_lse), 0.0)
    denom = weights.sum(dim=0, keepdim=True)
    weights = torch.where(denom > 0, weights / denom.clamp_min(1e-30), weights)
    outs = torch.where(valid.unsqueeze(-1), outs.float(), 0.0)
    return (weights.unsqueeze(-1) * outs).sum(dim=0).to(out_dtype)


def lse_logsumexp_valid(lses: torch.Tensor) -> torch.Tensor:
    """Merged natural-log LSE over dim 0, ignoring non-finite shards.

    lses: [N, ...]; an element with no valid shard gets -inf.
    """
    lses = lses.float()
    return torch.logsumexp(torch.where(torch.isfinite(lses), lses, -math.inf), dim=0)


def _lse_pack_cols(dtype: torch.dtype) -> int:
    elem = dtype.itemsize
    assert 4 % elem == 0, f"cannot pack an fp32 LSE into {dtype} columns"
    return 4 // elem


def _dcp_a2a_packed_exchange(out: torch.Tensor, lse: torch.Tensor, group):
    """Packed A2A exchange: out [B, N*h, D], lse [B, N*h] -> ([N, B, h, D], [N, B, h]).

    send[j] carries this rank's partial for the heads rank j keeps, with the
    fp32 LSE reinterpreted as trailing ``out.dtype`` columns, so output + LSE
    move in one all_to_all_single. The buffer keeps the float dtype (no uint8
    byte view) for HCCL. recv[j] = rank j's partial for this rank's heads;
    the returned out keeps ``out.dtype`` and the LSE is float32.

    ``out`` / ``lse`` may be non-contiguous head slices of the FIA outputs:
    packing is one cat plus one rank-major copy.
    """
    n = group.world_size
    b, heads, d = out.shape
    assert heads % n == 0, f"num_heads ({heads}) must be divisible by dcp ({n})"
    h = heads // n
    cols = _lse_pack_cols(out.dtype)
    assert (d + cols) % cols == 0, f"head dim {d} not packable with {cols} cols"

    # [B, N*h] fp32 -> [B, N*h, cols] out.dtype byte view (no copy for a head
    # slice: reshape only appends a unit dim, whose stride is 1).
    lse_cols = lse.float().reshape(b, heads, 1).view(out.dtype)
    send = (
        torch.cat([out, lse_cols], dim=-1)
        .view(b, n, h, d + cols)
        .transpose(0, 1)
        .contiguous()
    )
    recv = torch.empty_like(send)
    group.all_to_all_single(recv.view(-1), send.view(-1))
    return recv[..., :d], recv.view(torch.float32)[..., d // cols]


# Finite stand-in for an invalid LSE (+inf FIA sentinel for an empty local KV,
# -inf, NaN): exp(-1e30 - lse) underflows to 0 against any finite shard, and
# when every shard is invalid the zeroed outputs merge to 0.
_INVALID_LSE = -1e30


def _torch_npu_attention_update(lse_list, out_list, update_type):
    import torch_npu  # lazy: keep this module importable on CPU

    return torch_npu.npu_attention_update(lse_list, out_list, update_type)


# Patched with a pure-torch reference in CPU tests.
_attention_update_op = _torch_npu_attention_update


def _attention_update_call(lses, outs, return_lse, any_valid):
    """torch_npu.npu_attention_update over sanitised shards: lse_i [T] float32
    natural log, out_i [T, D] -> out [T, D] in the outputs' dtype; with
    ``return_lse`` (update_type=1) also the merged LSE [T], -inf where no shard
    is valid."""
    if not return_lse:
        out, _ = _attention_update_op(lses, outs, 0)
        return out
    out, lse = _attention_update_op(lses, outs, 1)
    lse = lse.float().reshape(any_valid.shape)
    return out, torch.where(any_valid, lse, -math.inf)


def npu_attention_update(
    lse_list: Sequence[torch.Tensor],
    out_list: Sequence[torch.Tensor],
    return_lse: bool = False,
    merge_fp32: bool = False,
    mask_out: bool = True,
):
    """Merge partial attentions over disjoint KV.

    lse_i [*S] natural log, out_i [*S, D] -> [prod(S), D] (and the merged LSE
    [prod(S)] with ``return_lse``). Non-finite shard LSEs (FIA returns +inf for
    an empty local KV) get zero weight; an element with no valid shard merges
    to 0, like ``lse_combine``. Outputs are merged in their dtype (bf16 gives
    the same bf16 result as merging in fp32 and casting back); ``merge_fp32``
    casts them to float32 first. ``mask_out=False`` skips zeroing the outputs
    of invalid shards, for callers whose outputs are finite by construction.
    Inputs may be non-contiguous slices.
    """
    lses, outs = [], []
    any_valid = None
    for lse, out in zip(lse_list, out_list):
        d = out.shape[-1]
        if merge_fp32:
            out = out.float()
        valid = torch.isfinite(lse)
        if return_lse:
            any_valid = valid if any_valid is None else any_valid | valid
        lses.append(torch.where(valid, lse.float(), _INVALID_LSE).reshape(-1))
        if mask_out:
            out = torch.where(valid.unsqueeze(-1), out, 0.0)
        outs.append(out.reshape(-1, d))
    if any_valid is not None:
        any_valid = any_valid.reshape(-1)
    return _attention_update_call(lses, outs, return_lse, any_valid)


def npu_attention_update_stacked(
    lses: torch.Tensor,
    outs: torch.Tensor,
    return_lse: bool = False,
    merge_fp32: bool = False,
    extra_lse: Optional[torch.Tensor] = None,
    extra_out: Optional[torch.Tensor] = None,
):
    """``npu_attention_update`` over stacked shards: lses [N, *S], outs [N, *S, D]
    -> [prod(S), D] (and lse [prod(S)] with ``return_lse``).

    The sanitising is vectorised over the shard axis (isfinite + two where for
    all N shards), then the op gets the N unbound rows. ``extra_lse`` [*S] /
    ``extra_out`` [*S, D] add one more shard (the DSPARK verify window merged
    as the (N+1)-th shard); like the ``mask_out=False`` callers, its outputs go
    in unmasked, so they must be finite.
    """
    n, d = outs.shape[0], outs.shape[-1]
    if merge_fp32:
        outs = outs.float()
    valid = torch.isfinite(lses)
    lses = torch.where(valid, lses.float(), _INVALID_LSE).view(n, -1)
    outs = torch.where(valid.unsqueeze(-1), outs, 0.0).view(n, -1, d)
    any_valid = valid.any(dim=0).view(-1) if return_lse else None
    lse_list, out_list = list(lses.unbind(0)), list(outs.unbind(0))
    if extra_out is not None:
        extra_valid = torch.isfinite(extra_lse)
        if return_lse:
            any_valid = any_valid | extra_valid.reshape(-1)
        lse_list.append(
            torch.where(extra_valid, extra_lse.float(), _INVALID_LSE).reshape(-1)
        )
        out_list.append(
            (extra_out.float() if merge_fp32 else extra_out).reshape(-1, d)
        )
    return _attention_update_call(lse_list, out_list, return_lse, any_valid)


def dcp_merge_a2a(
    out: torch.Tensor,
    lse: torch.Tensor,
    group,
    return_lse: bool = False,
    merge_fp32: bool = False,
):
    """A2A merge with the pure-torch ``lse_combine`` (reference): out [B, N*h, D],
    lse [B, N*h] natural log -> [B, h, D] (with ``return_lse`` also the merged
    LSE [B, h] float32, -inf where no shard is valid)."""
    recv_out, recv_lse = _dcp_a2a_packed_exchange(out, lse, group)
    merged = lse_combine(recv_out, recv_lse)
    if not return_lse:
        return merged
    return merged, lse_logsumexp_valid(recv_lse)


def dcp_merge_a2a_vllm(
    out: torch.Tensor,
    lse: torch.Tensor,
    group,
    return_lse: bool = False,
    merge_fp32: bool = False,
):
    """vllm-ascend style A2A merge: out [B, N*h, D], lse [B, N*h] -> [B, h, D]
    (with ``return_lse`` also the merged natural-log LSE [B, h] float32).

    Mirrors ``_process_attn_out_lse`` + ``_npu_attention_update``: fp32
    ``cat(out, lse)`` permuted to [N*h, D+1, B], one all_to_all_single (chunk
    j of dim 0 = head group j goes to rank j), then ``npu_attention_update``
    over the N received shards in float32.
    """
    n = group.world_size
    out_dtype = out.dtype
    b, heads, d = out.shape
    assert heads % n == 0, f"num_heads ({heads}) must be divisible by dcp ({n})"
    h = heads // n
    send = torch.cat([out.float(), lse.float().unsqueeze(-1)], dim=-1)
    send = send.permute(1, 2, 0).contiguous()  # [N*h, D+1, B]
    recv = torch.empty_like(send)
    group.all_to_all_single(recv, send)
    # recv chunk j = rank j's partial for this rank's heads.
    x = recv.permute(2, 0, 1).reshape(b, n, h, d + 1).permute(1, 0, 2, 3)
    outs, lses = x.split([d, 1], dim=-1)  # [N, B, h, D], [N, B, h, 1]
    merged = npu_attention_update(
        list(lses.reshape(n, b * h).unbind(0)),
        list(outs.reshape(n, b * h, d).unbind(0)),
        return_lse=return_lse,
    )
    if return_lse:
        merged, merged_lse = merged
        return merged.view(b, h, d).to(out_dtype), merged_lse.view(b, h)
    return merged.view(b, h, d).to(out_dtype)


def dcp_merge_a2a_npu(
    out: torch.Tensor,
    lse: torch.Tensor,
    group,
    return_lse: bool = False,
    merge_fp32: bool = False,
):
    """A2A merge (packed exchange + ``npu_attention_update``): out [B, N*h, D],
    lse [B, N*h] natural log -> [B, h, D] (with ``return_lse`` also the merged
    LSE [B, h] float32).

    One all_to_all_single carries out (model dtype) with the fp32 LSE packed as
    trailing columns; the received [N, B, h, *] shards are sanitised in one
    vectorised pass and merged in the model dtype (float32 with
    ``merge_fp32``).
    """
    out_dtype = out.dtype
    b, heads, d = out.shape
    h = heads // group.world_size
    recv_out, recv_lse = _dcp_a2a_packed_exchange(out, lse, group)
    merged = npu_attention_update_stacked(
        recv_lse, recv_out, return_lse=return_lse, merge_fp32=merge_fp32
    )
    if return_lse:
        merged, merged_lse = merged
        return merged.view(b, h, d).to(out_dtype), merged_lse.view(b, h)
    return merged.view(b, h, d).to(out_dtype)


def dcp_merge_ag_rs(
    out: torch.Tensor,
    lse: torch.Tensor,
    group,
    return_lse: bool = False,
    merge_fp32: bool = False,
):
    """AG+RS merge: out [B, N*h, D], lse [B, N*h] natural log -> [B, h, D]
    (with ``return_lse`` also the merged LSE [B, h] float32)."""
    n = group.world_size
    b, heads, d = out.shape
    lses = group.all_gather(lse.float().contiguous(), dim=0).view(n, b, heads)
    # NaN and FIA's +inf empty-shard sentinel both mean "no local KV".
    lses = torch.where(torch.isnan(lses) | torch.isposinf(lses), -math.inf, lses)
    local_lse = lses[group.rank_in_group]
    scale = torch.exp(local_lse - torch.logsumexp(lses, dim=0))
    scale = torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)
    corrected = torch.nan_to_num(out.float(), nan=0.0, posinf=0.0, neginf=0.0)
    # Reduce in fp32 like cp_lse_ag_out_rs_mla, then cast back.
    corrected = corrected * scale.unsqueeze(-1)
    merged = group.reduce_scatter_along_dim(corrected, dim=1).to(out.dtype)
    if not return_lse:
        return merged
    h = heads // n
    rank = group.rank_in_group
    return merged, lse_logsumexp_valid(lses[:, :, rank * h : (rank + 1) * h])


def dcp_merge_a2a_triton(
    out: torch.Tensor,
    lse: torch.Tensor,
    group,
    return_lse: bool = False,
    merge_fp32: bool = False,
):
    """A2A merge with the Triton-Ascend pack + combine kernels: out [B, N*h, D],
    lse [B, N*h] natural log -> [B, h, D] (with ``return_lse`` also the merged
    LSE [B, h] float32).

    Same exchange as ``dcp_merge_a2a_npu`` (payload in the model dtype, fp32
    LSE packed as trailing columns), with the pack copies and the sanitise +
    ``npu_attention_update`` sequence each replaced by one kernel.
    ``merge_fp32`` is ignored: the kernel always accumulates in float32.
    """
    from sglang.srt.hardware_backend.npu.triton_ops.dcp_merge import (
        dcp_merge_a2a_triton as _merge,
    )

    return _merge(out, lse, group, return_lse=return_lse, merge_fp32=merge_fp32)


_A2A_MERGES = dict(
    npu=dcp_merge_a2a_npu,
    vllm=dcp_merge_a2a_vllm,
    torch=dcp_merge_a2a,
    triton=dcp_merge_a2a_triton,
)
DCP_MERGE_IMPLS = tuple(_A2A_MERGES)
# Merge implementations whose a2a exchange can be split from the merge, so a
# caller can add a local shard to the received ones (SGLANG_NPU_DCP_VERIFY_FUSED_MERGE).
DCP_SPLIT_MERGE_IMPLS = ("npu", "torch", "triton")


def dcp_a2a_exchange(out: torch.Tensor, lse: torch.Tensor, group, merge_impl: str):
    """The packed a2a of an A2A merge, without the merge itself.

    out [B, N*h, D] + lse [B, N*h] natural log -> ([N, B, h, D], [N, B, h]
    float32): every DCP rank's partial for the heads this rank keeps.
    """
    if merge_impl == "triton":
        from sglang.srt.hardware_backend.npu.triton_ops.dcp_merge import (
            dcp_exchange_a2a_triton,
        )

        return dcp_exchange_a2a_triton(out, lse, group)
    return _dcp_a2a_packed_exchange(out, lse, group)


def dcp_merge_shards(
    outs: torch.Tensor,
    lses: torch.Tensor,
    merge_impl: str,
    extra_out: Optional[torch.Tensor] = None,
    extra_lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    merge_fp32: bool = False,
):
    """Merge already-exchanged shards: outs [N, B, h, D], lses [N, B, h] natural
    log -> [B * h, D] (and the merged LSE [B * h] with ``return_lse``).

    ``extra_out`` [B, h, D] / ``extra_lse`` [B, h] are merged as one more shard
    in the same pass, which is how the DSPARK verify window is folded in
    instead of running a second merge over the cross-rank result.
    """
    if merge_impl == "triton":
        from sglang.srt.hardware_backend.npu.triton_ops.dcp_merge import (
            lse_combine_shards,
        )

        return lse_combine_shards(
            outs, lses, extra_out, extra_lse, return_lse=return_lse
        )
    if merge_impl == "torch":
        if extra_out is not None:
            outs = torch.cat([outs, extra_out.unsqueeze(0)], dim=0)
            lses = torch.cat([lses, extra_lse.unsqueeze(0)], dim=0)
        n, d = outs.shape[0], outs.shape[-1]
        merged = lse_combine(outs.reshape(n, -1, d), lses.reshape(n, -1))
        if not return_lse:
            return merged
        return merged, lse_logsumexp_valid(lses.reshape(n, -1))
    return npu_attention_update_stacked(
        lses,
        outs,
        return_lse=return_lse,
        merge_fp32=merge_fp32,
        extra_lse=extra_lse,
        extra_out=extra_out,
    )


def dcp_merge(
    out: torch.Tensor,
    lse: torch.Tensor,
    group,
    comm_backend: str,
    merge_impl: str,
    return_lse: bool = False,
    merge_fp32: bool = False,
):
    """Cross-rank LSE merge of this rank's all-head partials.

    out [B, N*h, D], lse [B, N*h] natural log -> out [B, h, D] (model dtype);
    with ``return_lse`` also the merged LSE [B, h] float32, -inf where no rank
    holds KV. ``comm_backend`` 'a2a' merges with ``merge_impl`` (one of
    DCP_MERGE_IMPLS, SGLANG_NPU_DCP_MERGE_IMPL); anything else uses AG+RS.
    ``merge_fp32`` only affects 'npu'.
    """
    if comm_backend == "a2a":
        merge = _A2A_MERGES[merge_impl]
    else:
        merge = dcp_merge_ag_rs
    return merge(out, lse, group, return_lse=return_lse, merge_fp32=merge_fp32)


def mla_decode_with_lse_torch(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    c_kv_pages: torch.Tensor,
    k_rope_pages: torch.Tensor,
    block_table: torch.Tensor,
    local_lens: Union[torch.Tensor, Sequence[int]],
    scale: float,
):
    """Reference single-query MLA decode over this rank's local KV pages.

    q_nope [B, H, Dc], q_rope [B, H, Dr]; c_kv_pages [pages, P, (1,) Dc],
    k_rope_pages [pages, P, (1,) Dr] in logical (token-major) order;
    block_table [B, max_pages]. Returns (out [B, H, Dc], lse [B, H]) with a
    natural-log LSE; a request with local_len == 0 gets out = 0, lse = -inf.
    """
    bsz, heads, d_c = q_nope.shape
    page_size = c_kv_pages.shape[1]
    c_kv_pages = c_kv_pages.reshape(c_kv_pages.shape[0], page_size, -1)
    k_rope_pages = k_rope_pages.reshape(k_rope_pages.shape[0], page_size, -1)
    if isinstance(local_lens, torch.Tensor):
        local_lens = local_lens.tolist()

    out = torch.zeros(bsz, heads, d_c, dtype=torch.float32, device=q_nope.device)
    lse = torch.full((bsz, heads), -math.inf, dtype=torch.float32, device=q_nope.device)
    for i in range(bsz):
        n = int(local_lens[i])
        if n <= 0:
            continue
        npages = (n + page_size - 1) // page_size
        pages = block_table[i, :npages].long()
        kv = c_kv_pages[pages].reshape(-1, c_kv_pages.shape[-1])[:n].float()
        kr = k_rope_pages[pages].reshape(-1, k_rope_pages.shape[-1])[:n].float()
        scores = (
            torch.einsum("hd,td->ht", q_nope[i].float(), kv)
            + torch.einsum("hd,td->ht", q_rope[i].float(), kr)
        ) * scale
        lse[i] = torch.logsumexp(scores, dim=-1)
        out[i] = torch.softmax(scores, dim=-1) @ kv
    return out.to(q_nope.dtype), lse
