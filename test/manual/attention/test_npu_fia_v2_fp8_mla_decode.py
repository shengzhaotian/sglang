"""Standalone Ascend FIA v2 FP8 MLA decode/target-verify correctness case.

The default shape models one TP rank of Kimi-K3 MLA target verification:

* committed prefix length: 140,000 tokens per request
* batch size: 2
* global query heads: 96, TP size: 4, local query heads: 24
* FIA query heads: 32 (the real 24 heads are zero-padded to a power of two)
* DSpark target verify: gamma=7, hence eight causal query tokens per request
* one latent KV head, D_nope=512, D_rope=64
* paged FP8 E4M3 latent KV and BF16 RoPE cache, block size 128

The native reference follows the CANN ``fia_fullquant_mla_test`` golden rather
than implementing a generic BF16 attention. In particular, it models:

1. per-token/per-head FP8 query and per-tensor FP8 latent-KV dequant scales;
2. fused latent and RoPE score accumulation;
3. online softmax over paged KV blocks; and
4. block-local FP8 requantization of softmax probabilities before ``P @ V``.

The default mode exercises DSpark target verification.  ``--mode decode`` keeps
the one-token decode regression, while ``--mode both`` runs both cases.
This case covers the uniform/static verify layout used by the default DSpark
path. Packed ragged-TND metadata and NPUGraph replay remain integration-level
coverage because they depend on SGLang's verify planner and graph updater.

Run on an Atlas A5 environment with a torch_npu build exposing FIA v2:

    python test/manual/attention/test_npu_fia_v2_fp8_mla_decode.py

Use ``--seq-len 4096`` for a quick smoke test. The default case is intentionally
large and allocates roughly 180 MiB for the two physical cache tensors alone.

Use ``--packed-repo-check report`` to additionally probe whether the installed
FIA v2 can consume MLAProlog V3's combined 656-byte KV/RoPE/scale repository.
The probe reports both the direct packed-cache ABI and zero-copy typed views.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, replace

import torch


FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0
MLA_KV_TILE_SIZE = 128
FP32_BYTES = 4
BF16_BYTES = 2


@dataclass(frozen=True)
class CaseConfig:
    batch_size: int = 2
    seq_len: int = 140_000
    global_query_heads: int = 96
    tensor_parallel_size: int = 4
    query_seq_len: int = 8
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
        return math.ceil(self.total_kv_len / self.block_size)

    @property
    def padded_query_heads(self) -> int:
        return 1 << (self.local_query_heads - 1).bit_length()

    @property
    def total_kv_len(self) -> int:
        # The verify candidates are written to the paged cache before FIA is
        # launched.  The causal mask determines which candidate rows each
        # query is allowed to observe.
        return self.seq_len + self.query_seq_len

    def validate(self) -> None:
        if self.global_query_heads % self.tensor_parallel_size != 0:
            raise ValueError("global_query_heads must be divisible by TP size")
        if self.local_query_heads % self.kv_heads != 0:
            raise ValueError("local query heads must be divisible by KV heads")
        if self.query_seq_len < 1:
            raise ValueError("query_seq_len must be positive")
        if self.query_seq_len > 2048:
            raise ValueError("query_seq_len exceeds the 2048x2048 causal mask")
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
    attention_mask: torch.Tensor | None
    actual_seq_kvlen: list[int]


@dataclass
class PackedRepoViews:
    """Typed views over MLAProlog's combined KV-cache repository."""

    packed_cache: torch.Tensor
    latent: torch.Tensor
    key_rope: torch.Tensor
    tile_scale: torch.Tensor


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
            config.query_seq_len,
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
                config.query_seq_len,
                config.local_query_heads,
                config.rope_dim,
            ),
            dtype=torch.float32,
            generator=generator,
        )
        * 0.25
    ).to(torch.bfloat16)

    # sparse_mode=3 aligns this upper-triangular mask to the bottom-right of KV:
    # verify row j sees the committed prefix and candidate rows [0, j].  CANN's
    # documented full-quant MLA tests use the fixed 2048x2048 causal mask.
    attention_mask = None
    if config.query_seq_len > 1:
        attention_mask = torch.triu(
            torch.ones((2048, 2048), dtype=torch.bool), diagonal=1
        )

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
        attention_mask=attention_mask,
        actual_seq_kvlen=[config.total_kv_len] * config.batch_size,
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
        attention_mask=(
            inputs.attention_mask.to(device)
            if inputs.attention_mask is not None
            else None
        ),
        actual_seq_kvlen=inputs.actual_seq_kvlen,
    )


def _pad_query_heads(
    inputs: CaseInputs, config: CaseConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad K3's 24 local heads to the 32 heads required by FIA v2."""
    pad_heads = config.padded_query_heads - config.local_query_heads
    if pad_heads == 0:
        return inputs.query, inputs.query_rope, inputs.dequant_scale_query

    shape_prefix = inputs.query.shape[:2]
    query_padding = torch.zeros(
        (*shape_prefix, pad_heads, config.latent_dim),
        dtype=inputs.query.dtype,
        device=inputs.query.device,
    )
    rope_padding = torch.zeros(
        (*shape_prefix, pad_heads, config.rope_dim),
        dtype=inputs.query_rope.dtype,
        device=inputs.query_rope.device,
    )
    scale_padding = torch.ones(
        (*shape_prefix, pad_heads),
        dtype=inputs.dequant_scale_query.dtype,
        device=inputs.dequant_scale_query.device,
    )
    query = torch.cat((inputs.query, query_padding), dim=2).contiguous()
    query_rope = torch.cat((inputs.query_rope, rope_padding), dim=2).contiguous()
    query_scale = torch.cat(
        (inputs.dequant_scale_query, scale_padding), dim=2
    ).contiguous()

    return query, query_rope, query_scale


def _gather_pa_pages(cache: torch.Tensor, page_ids: torch.Tensor) -> torch.Tensor:
    """Gather ``[B, P]`` physical pages without FP8 advanced indexing.

    Ascend ``aclnnIndex`` does not accept FLOAT8_E4M3FN as ``self``. FP8 and
    UINT8 have the same element size, so gathering the bit-view with
    ``index_select`` preserves every payload bit and can safely be viewed back.
    """
    batch, pages = page_ids.shape
    flat_ids = page_ids.reshape(-1).to(torch.int64)
    if cache.dtype == FP8_DTYPE:
        gathered = torch.index_select(cache.view(torch.uint8), 0, flat_ids)
        return gathered.view(FP8_DTYPE).reshape(batch, pages, *cache.shape[1:])
    gathered = torch.index_select(cache, 0, flat_ids)
    return gathered.reshape(batch, pages, *cache.shape[1:])


def build_mlaprolog_packed_repo(
    inputs: CaseInputs, config: CaseConfig
) -> PackedRepoViews:
    """Build the exact MLAProlog V3 per-token-per-group cache layout.

    ``kv_cache_quant_mode=3``, ``ckvkr_repo_mode=1`` and
    ``quant_scale_repo_mode=1`` store one byte-addressed entry as::

        [FP8 latent (512 B), BF16 RoPE (128 B), FP32 scales (16 B)]

    This helper uses the already validated standalone FIA inputs as payload.
    It intentionally keeps the resulting latent/RoPE/scale tensors as aliased
    (non-contiguous) views, because making them contiguous would no longer test
    whether FIA can consume MLAProlog's cache repository without a conversion.
    """
    if config.latent_dim % MLA_KV_TILE_SIZE != 0:
        raise ValueError("latent_dim must be divisible by MLA_KV_TILE_SIZE")

    tile_count = config.latent_dim // MLA_KV_TILE_SIZE
    packed_dim = (
        config.latent_dim
        + config.rope_dim * BF16_BYTES
        + tile_count * FP32_BYTES
    )
    expected_packed_dim = 656
    if packed_dim != expected_packed_dim:
        raise ValueError(
            f"K3 MLA packed Dtile must be {expected_packed_dim}, got {packed_dim}"
        )

    pages, block_size = inputs.latent_cache.shape[:2]
    packed_bytes = torch.empty(
        (pages, block_size, packed_dim),
        dtype=torch.uint8,
        device=inputs.latent_cache.device,
    )
    latent_end = config.latent_dim
    rope_end = latent_end + config.rope_dim * BF16_BYTES

    packed_bytes[..., :latent_end].copy_(inputs.latent_cache.view(torch.uint8))
    packed_bytes[..., latent_end:rope_end].copy_(
        inputs.key_rope_cache.view(torch.uint8)
    )
    tile_scale_source = (
        inputs.dequant_scale_kv.reshape(1, 1, 1)
        .expand(pages, block_size, tile_count)
        .contiguous()
    )
    packed_bytes[..., rope_end:].copy_(tile_scale_source.view(torch.uint8))

    packed_cache = packed_bytes.view(FP8_DTYPE)
    latent = packed_cache[..., :latent_end]
    key_rope = packed_bytes[..., latent_end:rope_end].view(torch.bfloat16)
    tile_scale = packed_bytes[..., rope_end:].view(torch.float32)

    # These checks validate both the byte offsets and dtype reinterpretation.
    # Compare FP8 through uint8 because aclnnIndex/equality support for FP8 is
    # incomplete on some torch_npu versions.
    if not torch.equal(
        latent.view(torch.uint8), inputs.latent_cache.view(torch.uint8)
    ):
        raise AssertionError("packed latent bytes differ from the source cache")
    if not torch.equal(key_rope, inputs.key_rope_cache):
        raise AssertionError("packed RoPE bytes differ from the source cache")
    if not torch.equal(tile_scale, tile_scale_source):
        raise AssertionError("packed tile-scale bytes differ from the source scales")

    return PackedRepoViews(
        packed_cache=packed_cache,
        latent=latent,
        key_rope=key_rope,
        tile_scale=tile_scale,
    )


@torch.no_grad()
def run_fia_v2_direct_packed_repo(
    inputs: CaseInputs, config: CaseConfig, repo: PackedRepoViews
) -> torch.Tensor:
    """Probe whether FIA implicitly understands MLAProlog's 656-byte ABI.

    No external K/V scale or key_rope is passed: both are embedded in
    ``repo.packed_cache``. A successful call therefore means FIA recognizes
    the same repository contract as MLAProlog. A D mismatch, missing-RoPE or
    missing-scale error is evidence that this FIA ABI does not.
    """
    import torch_npu

    query, _, dequant_scale_query = _pad_query_heads(inputs, config)
    output, _ = torch_npu.npu_fused_infer_attention_score_v2(
        query,
        repo.packed_cache,
        repo.packed_cache,
        actual_seq_kvlen=inputs.actual_seq_kvlen,
        block_table=inputs.block_table,
        dequant_scale_query=dequant_scale_query,
        num_query_heads=config.padded_query_heads,
        num_key_value_heads=config.kv_heads,
        softmax_scale=1.0 / math.sqrt(config.latent_dim),
        input_layout="BSND",
        sparse_mode=3 if config.query_seq_len > 1 else 0,
        block_size=config.block_size,
        query_quant_mode=3,
        key_quant_mode=0,
        value_quant_mode=0,
        query_dtype=FP8_DTYPE,
        key_dtype=FP8_DTYPE,
        value_dtype=FP8_DTYPE,
        dequant_scale_query_dtype=torch.float32,
        out_dtype=torch.bfloat16,
    )
    return output[:, :, : config.local_query_heads, :].contiguous()


@torch.no_grad()
def run_fia_v2_packed_alias_views(
    inputs: CaseInputs, config: CaseConfig, repo: PackedRepoViews
) -> torch.Tensor:
    """Probe FIA using zero-copy typed views of the same packed repository.

    This is the fallback integration shape if FIA cannot consume Dtile=656
    directly. The four FP32 scales per token are deliberately passed without
    collapsing them to a scalar: accepting this call would prove FIA supports
    MLAProlog's per-token/per-128-group scale contract. Rejecting its shape or
    dtype proves that a conversion or a different attention ABI is required.
    """
    import torch_npu

    query, query_rope, dequant_scale_query = _pad_query_heads(inputs, config)
    output, _ = torch_npu.npu_fused_infer_attention_score_v2(
        query,
        repo.latent,
        repo.latent,
        query_rope=query_rope,
        key_rope=repo.key_rope,
        atten_mask=inputs.attention_mask,
        actual_seq_kvlen=inputs.actual_seq_kvlen,
        block_table=inputs.block_table,
        dequant_scale_query=dequant_scale_query,
        dequant_scale_key=repo.tile_scale,
        dequant_scale_value=repo.tile_scale,
        num_query_heads=config.padded_query_heads,
        num_key_value_heads=config.kv_heads,
        softmax_scale=1.0 / math.sqrt(config.latent_dim),
        input_layout="BSND",
        sparse_mode=3 if config.query_seq_len > 1 else 0,
        block_size=config.block_size,
        query_quant_mode=3,
        key_quant_mode=6,
        value_quant_mode=6,
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
    return output[:, :, : config.local_query_heads, :].contiguous()


def _format_probe_error(exc: RuntimeError) -> str:
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    return " | ".join(lines[:4])


def probe_fia_packed_repo(
    inputs: CaseInputs, config: CaseConfig, expectation: str
) -> None:
    """Run both possible FIA consumers and enforce the requested expectation."""
    repo = build_mlaprolog_packed_repo(inputs, config)
    print(
        "packed repo: "
        f"shape={tuple(repo.packed_cache.shape)}, dtype={repo.packed_cache.dtype}, "
        f"latent_stride={repo.latent.stride()}, "
        f"rope_stride={repo.key_rope.stride()}, "
        f"scale_shape={tuple(repo.tile_scale.shape)}"
    )

    results: dict[str, tuple[bool, torch.Tensor | RuntimeError]] = {}
    probes = {
        "direct Dtile=656": run_fia_v2_direct_packed_repo,
        "zero-copy alias views": run_fia_v2_packed_alias_views,
    }
    for name, probe in probes.items():
        try:
            output = probe(inputs, config, repo)
            torch.npu.synchronize()
            results[name] = (True, output)
            print(
                f"packed FIA probe [{name}]: SUPPORTED, "
                f"output={tuple(output.shape)} {output.dtype}"
            )
        except RuntimeError as exc:
            results[name] = (False, exc)
            print(
                f"packed FIA probe [{name}]: REJECTED, "
                f"reason={_format_probe_error(exc)}"
            )

    any_supported = any(supported for supported, _ in results.values())
    if expectation == "supported" and not any_supported:
        raise AssertionError(
            "FIA rejected both MLAProlog packed-cache consumption strategies"
        )
    if expectation == "unsupported" and any_supported:
        supported_names = [
            name for name, (supported, _) in results.items() if supported
        ]
        raise AssertionError(
            "FIA unexpectedly accepted MLAProlog packed cache through: "
            + ", ".join(supported_names)
        )


@torch.no_grad()
def run_fia_v2(inputs: CaseInputs, config: CaseConfig) -> torch.Tensor:
    """Run the production FIA v2 FP8 MLA paged-decode operator once."""
    import torch_npu

    query, query_rope, dequant_scale_query = _pad_query_heads(inputs, config)

    output, _ = torch_npu.npu_fused_infer_attention_score_v2(
        query,
        inputs.latent_cache,
        inputs.latent_cache,
        query_rope=query_rope,
        key_rope=inputs.key_rope_cache,
        atten_mask=inputs.attention_mask,
        # CANN's IFA MLA path requires actualSeqLengthsQ to be empty for
        # non-TND layouts.  BSND obtains Sq directly from query.shape[1].
        actual_seq_kvlen=inputs.actual_seq_kvlen,
        block_table=inputs.block_table,
        dequant_scale_query=dequant_scale_query,
        dequant_scale_key=inputs.dequant_scale_kv,
        dequant_scale_value=inputs.dequant_scale_kv,
        num_query_heads=config.padded_query_heads,
        num_key_value_heads=config.kv_heads,
        softmax_scale=1.0 / math.sqrt(config.latent_dim),
        input_layout="BSND",
        sparse_mode=3 if config.query_seq_len > 1 else 0,
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
    return output[:, :, : config.local_query_heads, :].contiguous()


@torch.no_grad()
def run_fia_v2_stepwise(inputs: CaseInputs, config: CaseConfig) -> torch.Tensor:
    """Compare a causal block with independent one-token decode calls.

    Row ``j`` uses the same query and physical cache but exposes only
    ``prefix+j+1`` KV rows.  This catches an inverted verify mask and
    bottom-right causal-alignment off-by-one independently of the native golden.
    """
    if config.query_seq_len == 1:
        return run_fia_v2(inputs, config)

    decode_config = replace(config, query_seq_len=1)
    outputs = []
    for query_index in range(config.query_seq_len):
        step_inputs = replace(
            inputs,
            query=inputs.query[:, query_index : query_index + 1],
            query_rope=inputs.query_rope[:, query_index : query_index + 1],
            dequant_scale_query=inputs.dequant_scale_query[
                :, query_index : query_index + 1
            ],
            attention_mask=None,
            actual_seq_kvlen=[config.seq_len + query_index + 1]
            * config.batch_size,
        )
        outputs.append(run_fia_v2(step_inputs, decode_config))
    return torch.cat(outputs, dim=1)


@torch.no_grad()
def native_fullquant_mla_reference(
    inputs: CaseInputs, config: CaseConfig
) -> torch.Tensor:
    """Memory-bounded native equivalent of CANN's FP8 MLA golden.

    This intentionally consumes physical FP8 PA pages and follows block-table
    order. It never materializes the full 140K BF16/FP32 K/V sequence.  For a
    multi-token verify block, query row ``j`` sees exactly ``prefix + j + 1``
    KV rows, matching sparse_mode=3's bottom-right causal alignment.
    """
    device = inputs.query.device
    batch = config.batch_size
    heads = config.local_query_heads
    query_len = config.query_seq_len
    pages_per_chunk = config.reference_chunk_tokens // config.block_size
    softmax_scale = 1.0 / math.sqrt(config.latent_dim)

    # FIA's MLA full-quant bmm1 accumulates FP8 nope and BF16 rope products in
    # L0C, then applies the Q and K descales to their combined score.
    query_raw = inputs.query.float()
    query_rope_raw = inputs.query_rope.float()
    dequant_q = inputs.dequant_scale_query.float()
    dequant_k = inputs.dequant_scale_kv.reshape(-1)[0].float()
    dequant_v = inputs.dequant_scale_kv.reshape(-1)[0].float()

    running_max = torch.full(
        (batch, heads, query_len),
        -torch.inf,
        dtype=torch.float32,
        device=device,
    )
    running_sum = torch.zeros(
        (batch, heads, query_len), dtype=torch.float32, device=device
    )
    running_out = torch.zeros(
        (batch, heads, query_len, config.latent_dim),
        dtype=torch.float32,
        device=device,
    )

    for page_start in range(0, config.pages_per_request, pages_per_chunk):
        page_end = min(page_start + pages_per_chunk, config.pages_per_request)
        page_ids = inputs.block_table[:, page_start:page_end]

        latent = _gather_pa_pages(inputs.latent_cache, page_ids).float()
        key_rope = _gather_pa_pages(inputs.key_rope_cache, page_ids).float()
        logical_start = page_start * config.block_size
        # Keep the kernel's S2 tile boundary (128 tokens) visible while
        # vectorizing several pages into one native-op launch.
        # score layout: [B, H, Sq, pages, 128]
        score = torch.matmul(
            query_raw.permute(0, 2, 1, 3).unsqueeze(2),
            latent.transpose(-1, -2).unsqueeze(1),
        ).permute(0, 1, 3, 2, 4)
        score.add_(
            torch.matmul(
                query_rope_raw.permute(0, 2, 1, 3).unsqueeze(2),
                key_rope.transpose(-1, -2).unsqueeze(1),
            ).permute(0, 1, 3, 2, 4)
        )
        score.mul_(
            dequant_q.permute(0, 2, 1)[:, :, :, None, None]
            * dequant_k
            * softmax_scale
        )

        logical_positions = torch.arange(
            logical_start,
            logical_start + latent.shape[1] * config.block_size,
            dtype=torch.int64,
            device=device,
        ).view(1, 1, 1, latent.shape[1], config.block_size)
        visible_kv_lens = (
            config.seq_len
            + torch.arange(query_len, dtype=torch.int64, device=device)
            + 1
        ).view(1, 1, query_len, 1, 1)
        score.masked_fill_(logical_positions >= visible_kv_lens, -torch.inf)

        page_max = score.amax(dim=-1)
        probability = torch.exp(score - page_max.unsqueeze(-1))
        probability = torch.where(
            torch.isfinite(score), probability, torch.zeros_like(probability)
        )
        page_sum = probability.sum(dim=-1)

        # For one S2 tile, the official golden's rowMax_A algebra reduces to
        # quantizing exp(score - page_max) with scale 448.
        probability_fp8 = (probability * FP8_MAX).clamp_(
            -FP8_MAX, FP8_MAX
        ).to(FP8_DTYPE)
        page_out = torch.matmul(
            probability_fp8.permute(0, 1, 3, 2, 4).float(),
            latent.unsqueeze(1),
        ).permute(0, 1, 3, 2, 4)

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
    return output.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)


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
    parser.add_argument(
        "--mode",
        choices=("decode", "verify", "both"),
        default="verify",
        help="decode uses Sq=1; verify uses a DSpark gamma+1 query block",
    )
    parser.add_argument(
        "--verify-width",
        type=int,
        default=8,
        help="DSpark target-verify width (Kimi-K3 gamma=7 gives 8)",
    )
    parser.add_argument(
        "--skip-stepwise-causal-check",
        action="store_true",
        help="skip comparing Sq=N verify with N independent Sq=1 FIA calls",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--reference-chunk-tokens", type=int, default=8192)
    parser.add_argument(
        "--packed-repo-check",
        choices=("off", "report", "supported", "unsupported"),
        default="off",
        help=(
            "probe FIA consumption of MLAProlog's 656-byte combined KV/rope/scale "
            "repository; report records the result without changing pass/fail, while "
            "supported/unsupported assert the expected CANN capability"
        ),
    )
    return parser.parse_args()


def run_case(args: argparse.Namespace, query_seq_len: int) -> None:
    config = CaseConfig(
        seq_len=args.seq_len,
        query_seq_len=query_seq_len,
        reference_chunk_tokens=args.reference_chunk_tokens,
    )
    config.validate()
    device = torch.device(args.device)

    print(
        "case: "
        f"mode={'decode' if query_seq_len == 1 else 'target_verify'}, "
        f"B={config.batch_size}, prefix={config.seq_len}, Sq={query_seq_len}, "
        f"Skv={config.total_kv_len}, "
        f"Nq(global/local)={config.global_query_heads}/"
        f"{config.local_query_heads}, FIA_Nq={config.padded_query_heads}, "
        f"TP={config.tensor_parallel_size}, "
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
    if query_seq_len > 1 and not args.skip_stepwise_causal_check:
        stepwise = run_fia_v2_stepwise(inputs, config)
        print("causal check: batched Sq=N versus N independent Sq=1 calls")
        compare_outputs(actual, stepwise)
    if args.packed_repo_check != "off":
        print(
            "packed repository check: MLAProlog V3 ABI "
            "(kv_cache_quant_mode=3, ckvkr_repo_mode=1, "
            "quant_scale_repo_mode=1)"
        )
        probe_fia_packed_repo(inputs, config, args.packed_repo_check)
    print("PASS")


def main() -> None:
    args = parse_args()
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("This case requires torch_npu and an Ascend NPU") from exc

    query_lens = {
        "decode": [1],
        "verify": [args.verify_width],
        "both": [1, args.verify_width],
    }[args.mode]
    for query_seq_len in query_lens:
        run_case(args, query_seq_len)


if __name__ == "__main__":
    main()
