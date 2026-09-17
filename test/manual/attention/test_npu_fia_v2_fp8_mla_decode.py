"""Standalone Ascend FIA v2 FP8 MLA decode correctness case.

The default shape models one TP rank of Kimi-K3 MLA decode:

* context length: 140,000 tokens per request
* batch size: 2
* global query heads: 24, TP size: 4, local query heads: 6
* one latent KV head, D_nope=512, D_rope=64
* paged FP8 E4M3 latent KV and BF16 RoPE cache, block size 128

The native reference follows the CANN ``fia_fullquant_mla_test`` golden rather
than implementing a generic BF16 attention. In particular, it models:

1. per-token/per-head FP8 query and per-tensor FP8 latent-KV dequant scales;
2. fused latent and RoPE score accumulation;
3. online softmax over paged KV blocks; and
4. block-local FP8 requantization of softmax probabilities before ``P @ V``.

Run on an Atlas A5 environment with a torch_npu build exposing FIA v2:

    python test/manual/attention/test_npu_fia_v2_fp8_mla_decode.py

Use ``--seq-len 4096`` for a quick smoke test. The default case is intentionally
large and allocates roughly 180 MiB for the two physical cache tensors alone.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import torch


FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


@dataclass(frozen=True)
class CaseConfig:
    batch_size: int = 2
    seq_len: int = 140_000
    global_query_heads: int = 24
    tensor_parallel_size: int = 4
    kv_heads: int = 1
    latent_dim: int = 512
    rope_dim: int = 64
    block_size: int = 128
    reference_chunk_tokens: int = 8192
    seed: int = 20260917

    @property
    def local_query_heads(self) -> int:
        return self.global_query_heads // self.tensor_parallel_size

    @property
    def pages_per_request(self) -> int:
        return math.ceil(self.seq_len / self.block_size)

    def validate(self) -> None:
        if self.global_query_heads % self.tensor_parallel_size != 0:
            raise ValueError("global_query_heads must be divisible by TP size")
        if self.local_query_heads % self.kv_heads != 0:
            raise ValueError("local query heads must be divisible by KV heads")
        if self.block_size != 128:
            raise ValueError("FIA v2 FP8 MLA paged decode requires block_size=128")
        if self.reference_chunk_tokens % self.block_size != 0:
            raise ValueError("reference_chunk_tokens must be block aligned")


@dataclass
class CaseInputs:
    query: torch.Tensor
    query_rope: torch.Tensor
    latent_cache: torch.Tensor
    key_rope_cache: torch.Tensor
    block_table: torch.Tensor
    dequant_scale_query: torch.Tensor
    dequant_scale_kv: torch.Tensor
    actual_seq_kvlen: list[int]


def _fill_fp8_cache(
    shape: tuple[int, ...], generator: torch.Generator, chunk_pages: int = 64
) -> torch.Tensor:
    """Create real FP8 payload without allocating a full-size FP32 temporary."""
    cache = torch.empty(shape, dtype=FP8_DTYPE, device="cpu")
    for start in range(0, shape[0], chunk_pages):
        end = min(start + chunk_pages, shape[0])
        values = torch.randint(
            -64,
            65,
            (end - start, *shape[1:]),
            dtype=torch.int16,
            generator=generator,
        ).float()
        cache[start:end].copy_(values.to(FP8_DTYPE))
    return cache


def _fill_bf16_cache(
    shape: tuple[int, ...], generator: torch.Generator, chunk_pages: int = 64
) -> torch.Tensor:
    cache = torch.empty(shape, dtype=torch.bfloat16, device="cpu")
    for start in range(0, shape[0], chunk_pages):
        end = min(start + chunk_pages, shape[0])
        values = torch.randn(
            (end - start, *shape[1:]),
            dtype=torch.float32,
            generator=generator,
        )
        cache[start:end].copy_((values * 0.25).to(torch.bfloat16))
    return cache


def build_case(config: CaseConfig) -> CaseInputs:
    """Build BSND query and BnBsH paged-cache inputs on CPU."""
    config.validate()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)

    total_pages = config.batch_size * config.pages_per_request
    physical_page_ids = torch.randperm(total_pages, generator=generator).to(
        torch.int32
    )
    block_table = physical_page_ids.view(
        config.batch_size, config.pages_per_request
    ).contiguous()

    # MLA Decode PA layout for BSND query: BnBsH. N_kv is one, so H=D.
    latent_cache = _fill_fp8_cache(
        (total_pages, config.block_size, config.latent_dim), generator
    )
    key_rope_cache = _fill_bf16_cache(
        (total_pages, config.block_size, config.rope_dim), generator
    )

    query_fp32 = torch.randn(
        (
            config.batch_size,
            1,
            config.local_query_heads,
            config.latent_dim,
        ),
        dtype=torch.float32,
        generator=generator,
    )
    query_absmax = query_fp32.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-8)
    quant_scale_query = FP8_MAX / query_absmax
    query = (query_fp32 * quant_scale_query).clamp_(-FP8_MAX, FP8_MAX).to(
        FP8_DTYPE
    )
    # BSND query uses BSN query scale layout.
    dequant_scale_query = quant_scale_query.reciprocal().squeeze(-1).float()

    query_rope = (
        torch.randn(
            (
                config.batch_size,
                1,
                config.local_query_heads,
                config.rope_dim,
            ),
            dtype=torch.float32,
            generator=generator,
        )
        * 0.25
    ).to(torch.bfloat16)

    # Decode MLA full-quant uses one per-tensor FP32 scale for both latent K/V.
    dequant_scale_kv = torch.tensor([1.0 / 16.0], dtype=torch.float32)
    return CaseInputs(
        query=query.contiguous(),
        query_rope=query_rope.contiguous(),
        latent_cache=latent_cache.contiguous(),
        key_rope_cache=key_rope_cache.contiguous(),
        block_table=block_table,
        dequant_scale_query=dequant_scale_query.contiguous(),
        dequant_scale_kv=dequant_scale_kv,
        actual_seq_kvlen=[config.seq_len] * config.batch_size,
    )


def move_case(inputs: CaseInputs, device: torch.device) -> CaseInputs:
    return CaseInputs(
        query=inputs.query.to(device),
        query_rope=inputs.query_rope.to(device),
        latent_cache=inputs.latent_cache.to(device),
        key_rope_cache=inputs.key_rope_cache.to(device),
        block_table=inputs.block_table.to(device),
        dequant_scale_query=inputs.dequant_scale_query.to(device),
        dequant_scale_kv=inputs.dequant_scale_kv.to(device),
        actual_seq_kvlen=inputs.actual_seq_kvlen,
    )


@torch.no_grad()
def run_fia_v2(inputs: CaseInputs, config: CaseConfig) -> torch.Tensor:
    """Run the production FIA v2 FP8 MLA paged-decode operator once."""
    import torch_npu

    output, _ = torch_npu.npu_fused_infer_attention_score_v2(
        inputs.query,
        inputs.latent_cache,
        inputs.latent_cache,
        query_rope=inputs.query_rope,
        key_rope=inputs.key_rope_cache,
        actual_seq_kvlen=inputs.actual_seq_kvlen,
        block_table=inputs.block_table,
        dequant_scale_query=inputs.dequant_scale_query,
        dequant_scale_key=inputs.dequant_scale_kv,
        dequant_scale_value=inputs.dequant_scale_kv,
        num_query_heads=config.local_query_heads,
        num_key_value_heads=config.kv_heads,
        softmax_scale=1.0 / math.sqrt(config.latent_dim),
        input_layout="BSND",
        sparse_mode=0,
        block_size=config.block_size,
        query_quant_mode=3,
        key_quant_mode=0,
        value_quant_mode=0,
        query_dtype=FP8_DTYPE,
        key_dtype=FP8_DTYPE,
        value_dtype=FP8_DTYPE,
        query_rope_dtype=torch.bfloat16,
        key_rope_dtype=torch.bfloat16,
        dequant_scale_query_dtype=torch.float32,
        dequant_scale_key_dtype=torch.float32,
        dequant_scale_value_dtype=torch.float32,
        out_dtype=torch.bfloat16,
    )
    return output


@torch.no_grad()
def native_fullquant_mla_reference(
    inputs: CaseInputs, config: CaseConfig
) -> torch.Tensor:
    """Memory-bounded native equivalent of CANN's FP8 MLA decode golden.

    This intentionally consumes physical FP8 PA pages and follows block-table
    order. It never materializes the full 140K BF16/FP32 K/V sequence.
    """
    device = inputs.query.device
    batch = config.batch_size
    heads = config.local_query_heads
    pages_per_chunk = config.reference_chunk_tokens // config.block_size
    softmax_scale = 1.0 / math.sqrt(config.latent_dim)

    # FIA's MLA full-quant bmm1 accumulates FP8 nope and BF16 rope products in
    # L0C, then applies the Q and K descales to their combined score.
    query_raw = inputs.query[:, 0].float()
    query_rope_raw = inputs.query_rope[:, 0].float()
    dequant_q = inputs.dequant_scale_query[:, 0].float()
    dequant_k = inputs.dequant_scale_kv.reshape(-1)[0].float()
    dequant_v = inputs.dequant_scale_kv.reshape(-1)[0].float()

    running_max = torch.full(
        (batch, heads), -torch.inf, dtype=torch.float32, device=device
    )
    running_sum = torch.zeros(
        (batch, heads), dtype=torch.float32, device=device
    )
    running_out = torch.zeros(
        (batch, heads, config.latent_dim), dtype=torch.float32, device=device
    )

    for page_start in range(0, config.pages_per_request, pages_per_chunk):
        page_end = min(page_start + pages_per_chunk, config.pages_per_request)
        page_ids = inputs.block_table[:, page_start:page_end].long()

        latent = inputs.latent_cache[page_ids].float()
        key_rope = inputs.key_rope_cache[page_ids].float()
        logical_start = page_start * config.block_size
        valid_tokens = min(
            config.seq_len - logical_start,
            latent.shape[1] * config.block_size,
        )

        # Keep the kernel's S2 tile boundary (128 tokens) visible while
        # vectorizing several pages into one native-op launch.
        score = torch.matmul(
            query_raw.unsqueeze(1), latent.transpose(-1, -2)
        ).permute(0, 2, 1, 3)
        score.add_(
            torch.matmul(
                query_rope_raw.unsqueeze(1), key_rope.transpose(-1, -2)
            ).permute(0, 2, 1, 3)
        )
        score.mul_(dequant_q[:, :, None, None] * dequant_k * softmax_scale)
        if valid_tokens % config.block_size:
            score[:, :, -1, valid_tokens % config.block_size :].fill_(-torch.inf)

        page_max = score.amax(dim=-1)
        probability = torch.exp(score - page_max.unsqueeze(-1))
        page_sum = probability.sum(dim=-1)

        # For one S2 tile, the official golden's rowMax_A algebra reduces to
        # quantizing exp(score - page_max) with scale 448.
        probability_fp8 = (probability * FP8_MAX).clamp_(
            -FP8_MAX, FP8_MAX
        ).to(FP8_DTYPE)
        page_out = torch.matmul(
            probability_fp8.permute(0, 2, 1, 3).float(), latent
        ).permute(0, 2, 1, 3)

        chunk_max = page_max.amax(dim=-1)
        page_weight = torch.exp(page_max - chunk_max.unsqueeze(-1))
        chunk_sum = (page_sum * page_weight).sum(dim=-1)
        chunk_out = (page_out * page_weight.unsqueeze(-1)).sum(dim=-2)

        new_max = torch.maximum(running_max, chunk_max)
        old_weight = torch.exp(running_max - new_max)
        old_weight = torch.where(
            torch.isfinite(running_max), old_weight, torch.zeros_like(old_weight)
        )
        chunk_weight = torch.exp(chunk_max - new_max)
        running_out.mul_(old_weight.unsqueeze(-1)).add_(
            chunk_out * chunk_weight.unsqueeze(-1)
        )
        running_sum.mul_(old_weight).add_(chunk_sum * chunk_weight)
        running_max = new_max

    output = running_out / running_sum.unsqueeze(-1).clamp_min_(1e-20)
    output.mul_(dequant_v / FP8_MAX)
    return output.unsqueeze(1).to(torch.bfloat16)


def compare_outputs(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 0.0078125,
    atol: float = 0.0001,
    required_close_fraction: float = 0.995,
) -> None:
    """Apply the BF16 accuracy gate used by the CANN full-quant MLA tests."""
    actual_fp32 = actual.float().cpu()
    expected_fp32 = expected.float().cpu()
    close = torch.isclose(actual_fp32, expected_fp32, rtol=rtol, atol=atol)
    close_fraction = close.float().mean().item()
    abs_error = (actual_fp32 - expected_fp32).abs()
    relative_error = abs_error / expected_fp32.abs().clamp_min(1e-10)

    print(
        "accuracy: "
        f"close={close_fraction:.6%}, "
        f"max_abs={abs_error.max().item():.6e}, "
        f"max_rel={relative_error.max().item():.6e}, "
        f"rtol={rtol}, atol={atol}"
    )
    if close_fraction < required_close_fraction:
        bad = (~close).nonzero(as_tuple=False)
        first_bad = bad[0].tolist() if bad.numel() else None
        raise AssertionError(
            f"FIA v2 accuracy failed: {close_fraction:.6%} < "
            f"{required_close_fraction:.6%}; first mismatch={first_bad}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seq-len", type=int, default=140_000)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--reference-chunk-tokens", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("This case requires torch_npu and an Ascend NPU") from exc

    config = CaseConfig(
        seq_len=args.seq_len,
        reference_chunk_tokens=args.reference_chunk_tokens,
    )
    config.validate()
    device = torch.device(args.device)

    print(
        "case: "
        f"B={config.batch_size}, S={config.seq_len}, "
        f"Nq(global/local)={config.global_query_heads}/"
        f"{config.local_query_heads}, TP={config.tensor_parallel_size}, "
        f"Nkv={config.kv_heads}, D={config.latent_dim}, "
        f"Drope={config.rope_dim}, block={config.block_size}"
    )
    inputs = move_case(build_case(config), device)

    expected = native_fullquant_mla_reference(inputs, config)
    for _ in range(args.warmup):
        run_fia_v2(inputs, config)
    torch.npu.synchronize()

    begin = time.perf_counter()
    actual = None
    for _ in range(args.repeat):
        actual = run_fia_v2(inputs, config)
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - begin) * 1000.0 / args.repeat

    assert actual is not None
    print(f"FIA v2 latency: {elapsed_ms:.3f} ms/run")
    print(f"output: shape={tuple(actual.shape)}, dtype={actual.dtype}")
    compare_outputs(actual, expected)
    print("PASS")


if __name__ == "__main__":
    main()
