"""Triton-Ascend kernel for the DCP-filtered MLA KV cache write.

The torch path spells the owner filter and the two cache writes as
``>=`` ``%`` ``==`` ``&`` ``//`` ``new_full`` ``where`` and two
``npu_scatter_nd_update_`` calls: tokens this rank does not own are rewritten
to the reserved slot 0 so the write keeps a static (graph-safe) shape, and the
latent and rope caches are scattered separately.

One kernel does all of it: ownership is evaluated per token inside the kernel
and becomes the store mask, so a token owned by another rank writes nothing at
all instead of landing on slot 0.
"""

import torch
import triton
import triton.language as tl

from sglang.srt.hardware_backend.npu.triton_ops.utils import row_grid


@triton.jit
def _dcp_store_mla_kv_kernel(
    k_buf_ptr,  # [num_slots, K_DIM] latent cache
    v_buf_ptr,  # [num_slots, V_DIM] rope cache
    k_src_ptr,  # [T, K_DIM]
    v_src_ptr,  # [T, V_DIM]
    loc_ptr,  # [T] virtual write locations
    k_src_row,
    v_src_row,
    n_tokens,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    K_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Write the tokens this rank owns (loc % dcp == rank) to loc // dcp."""
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    rows_per_program = (n_tokens + n_programs - 1) // n_programs
    start_row = pid * rows_per_program
    end_row = tl.minimum(start_row + rows_per_program, n_tokens)

    k_cols = tl.arange(0, BLOCK_K)
    v_cols = tl.arange(0, BLOCK_V)
    for row in range(start_row, end_row):
        loc = tl.load(loc_ptr + row)
        owned = (loc >= 0) & (loc % DCP_SIZE == DCP_RANK)
        slot = loc // DCP_SIZE
        k_mask = (k_cols < K_DIM) & owned
        tl.store(
            k_buf_ptr + slot * K_DIM + k_cols,
            tl.load(k_src_ptr + row * k_src_row + k_cols, mask=k_mask, other=0),
            mask=k_mask,
        )
        v_mask = (v_cols < V_DIM) & owned
        tl.store(
            v_buf_ptr + slot * V_DIM + v_cols,
            tl.load(v_src_ptr + row * v_src_row + v_cols, mask=v_mask, other=0),
            mask=v_mask,
        )


def dcp_store_mla_kv(
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    loc: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
) -> None:
    """Filtered MLA KV write: k_buffer [S, Dc], v_buffer [S, Dr] (flat slots),
    cache_k [T, Dc], cache_v [T, Dr], loc [T] virtual locations.

    ``dcp_size == 1`` still runs (the filter degenerates to ``loc >= 0``), so
    the kernel is a drop-in for the plain two-scatter write as well.
    """
    k_dim = k_buffer.shape[-1]
    v_dim = v_buffer.shape[-1]
    k_buffer = k_buffer.view(-1, k_dim)
    v_buffer = v_buffer.view(-1, v_dim)
    cache_k = cache_k.view(-1, k_dim)
    cache_v = cache_v.view(-1, v_dim)
    loc = loc.view(-1)
    n_tokens = loc.numel()
    assert cache_k.shape[0] == n_tokens and cache_v.shape[0] == n_tokens, (
        f"{n_tokens} write locations but cache_k {tuple(cache_k.shape)} / "
        f"cache_v {tuple(cache_v.shape)}"
    )
    if n_tokens == 0:
        return
    _dcp_store_mla_kv_kernel[(row_grid(n_tokens),)](
        k_buffer,
        v_buffer,
        cache_k,
        cache_v,
        loc,
        cache_k.stride(0),
        cache_v.stride(0),
        n_tokens,
        DCP_SIZE=dcp_size,
        DCP_RANK=dcp_rank,
        K_DIM=k_dim,
        V_DIM=v_dim,
        BLOCK_K=triton.next_power_of_2(k_dim),
        BLOCK_V=triton.next_power_of_2(v_dim),
    )
