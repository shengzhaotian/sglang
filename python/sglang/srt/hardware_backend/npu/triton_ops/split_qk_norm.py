"""Triton-Ascend kernel for the MLA latent split + q/k RMSNorm.

The fused ``qkv_a_proj`` writes one row-major [T, q_lora + kv_lora + rope]
tensor. Splitting it gives three *strided* views, and ``npu_rms_norm`` follows
aclnn's dense-stride contract, so the framework inserts a ``contiguous()``
before each of the two norms -- a copy that buys nothing, since the reduction
dimension of each slice is already contiguous.

This kernel reads the strided slices directly and emits ``q_norm``, ``k_norm``
and the rope slice in one pass, halving the launches (three ops -> one) and
dropping both staging copies.

It is *not* the vendor ``sgl_kernel_npu.norm.fused_split_qk_norm``: Kimi-K3
disables that one as numerically non-equivalent
(``models/kimi_k3.py``: ``_disable_npu_fused_split_qk_norm``). This kernel
keeps the reference arithmetic exactly -- accumulate the sum of squares in
float32, scale by ``rsqrt(mean + eps)``, multiply by the float32 weight, and
cast once on the way out -- so it can be checked element-wise against
``npu_rms_norm`` before it is switched on.
"""

import torch
import triton
import triton.language as tl

from sglang.srt.hardware_backend.npu.triton_ops.utils import row_grid


@triton.jit
def _split_qk_rmsnorm_kernel(
    x_ptr,  # [T, q_dim + k_dim + r_dim]
    q_weight_ptr,  # [q_dim]
    k_weight_ptr,  # [k_dim]
    q_out_ptr,  # [T, q_dim]
    k_out_ptr,  # [T, k_dim]
    r_out_ptr,  # [T, r_dim]
    x_row,  # element stride of an x row
    n_rows,
    q_eps,
    k_eps,
    Q_DIM: tl.constexpr,
    K_DIM: tl.constexpr,
    R_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """q_out = rmsnorm(x[:, :Q]), k_out = rmsnorm(x[:, Q:Q+K]), r_out = x[:, Q+K:]."""
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    rows_per_program = (n_rows + n_programs - 1) // n_programs
    start_row = pid * rows_per_program
    end_row = tl.minimum(start_row + rows_per_program, n_rows)

    q_cols = tl.arange(0, BLOCK_Q)
    q_mask = q_cols < Q_DIM
    k_cols = tl.arange(0, BLOCK_K)
    k_mask = k_cols < K_DIM
    r_cols = tl.arange(0, BLOCK_R)
    r_mask = r_cols < R_DIM

    for row in range(start_row, end_row):
        row_ptr = x_ptr + row * x_row

        q_vals = tl.load(row_ptr + q_cols, mask=q_mask, other=0.0).to(tl.float32)
        q_scale = tl.rsqrt(tl.sum(q_vals * q_vals) / Q_DIM + q_eps)
        q_weight = tl.load(q_weight_ptr + q_cols, mask=q_mask, other=0.0).to(tl.float32)
        q_out = q_vals * q_scale * q_weight
        tl.store(
            q_out_ptr + row * Q_DIM + q_cols,
            q_out.to(q_out_ptr.dtype.element_ty),
            mask=q_mask,
        )

        k_vals = tl.load(row_ptr + Q_DIM + k_cols, mask=k_mask, other=0.0).to(
            tl.float32
        )
        k_scale = tl.rsqrt(tl.sum(k_vals * k_vals) / K_DIM + k_eps)
        k_weight = tl.load(k_weight_ptr + k_cols, mask=k_mask, other=0.0).to(tl.float32)
        k_out = k_vals * k_scale * k_weight
        tl.store(
            k_out_ptr + row * K_DIM + k_cols,
            k_out.to(k_out_ptr.dtype.element_ty),
            mask=k_mask,
        )

        r_vals = tl.load(row_ptr + Q_DIM + K_DIM + r_cols, mask=r_mask, other=0.0)
        tl.store(r_out_ptr + row * R_DIM + r_cols, r_vals, mask=r_mask)


def split_qk_rmsnorm(
    qkv_latent: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    q_eps: float,
    k_eps: float,
):
    """[T, q_lora + kv_lora + rope] -> (q [T, q_lora], k_nope [T, 1, kv_lora],
    k_pe [T, 1, rope]), the two normed parts and the untouched rope slice."""
    assert qkv_latent.dim() == 2, f"expected [T, D], got {tuple(qkv_latent.shape)}"
    total = q_lora_rank + kv_lora_rank + qk_rope_head_dim
    assert qkv_latent.shape[1] == total, (
        f"latent width {qkv_latent.shape[1]} != {q_lora_rank} + {kv_lora_rank} + "
        f"{qk_rope_head_dim}"
    )
    assert qkv_latent.stride(1) == 1, "the latent rows must be contiguous"
    n_rows = qkv_latent.shape[0]
    q = qkv_latent.new_empty((n_rows, q_lora_rank))
    k_nope = qkv_latent.new_empty((n_rows, 1, kv_lora_rank))
    k_pe = qkv_latent.new_empty((n_rows, 1, qk_rope_head_dim))
    if n_rows == 0:
        return q, k_nope, k_pe
    _split_qk_rmsnorm_kernel[(row_grid(n_rows),)](
        qkv_latent,
        q_weight,
        k_weight,
        q,
        k_nope,
        k_pe,
        qkv_latent.stride(0),
        n_rows,
        q_eps,
        k_eps,
        Q_DIM=q_lora_rank,
        K_DIM=kv_lora_rank,
        R_DIM=qk_rope_head_dim,
        BLOCK_Q=triton.next_power_of_2(q_lora_rank),
        BLOCK_K=triton.next_power_of_2(kv_lora_rank),
        BLOCK_R=triton.next_power_of_2(qk_rope_head_dim),
    )
    return q, k_nope, k_pe
