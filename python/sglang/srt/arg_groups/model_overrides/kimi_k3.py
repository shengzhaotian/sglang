"""Config-time override declarations for kimi_k3.

Architectures: KimiK3ForConditionalGeneration.
"""

import inspect
import logging
from typing import Any

from sglang.srt.arg_groups.model_override_base import (
    _dspark_verify_on_decode_backend,
    _is_mxfp4_pack_quantized,
    _register_for,
    attention_backends_of,
    is_attention_backend_not_set,
    resolving_view,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils.common import get_device_name, is_mnnvl_fabric_device

logger = logging.getLogger(__name__)


def _require_kimi_k3_cutedsl_dcp_support() -> None:
    try:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

        parameters = inspect.signature(trtllm_batch_decode_with_kv_cache_mla).parameters
    except (ImportError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Kimi-K3 DCP with decode_attention_backend='cutedsl_mla' requires "
            "FlashInfer 0.6.17 or newer with "
            "trtllm_batch_decode_with_kv_cache_mla exposing enable_dcp."
        ) from exc

    if "enable_dcp" not in parameters:
        raise RuntimeError(
            "Kimi-K3 DCP with decode_attention_backend='cutedsl_mla' requires "
            "enable_dcp in the signature of "
            "flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla; upgrade "
            "to FlashInfer 0.6.17 or newer."
        )


def _resolve_kimi_k3_npu_dcp(cfg: Any, hf_config: Any = None) -> dict:
    """Kimi-K3 DCP on Ascend: keep the ascend attention backend, default to
    replicated Q + a2a, and reject what the NPU DCP path does not cover.

    Combinations (per-layer DCP collectives):
      --dcp-size N                                            -> 1x a2a
      --dcp-size N --no-dcp-replicate-q-proj --dcp-comm-backend a2a
                                                              -> Q AG + a2a
      --dcp-size N --no-dcp-replicate-q-proj                  -> Q AG + LSE AG + RS

    Speculative decoding: only DSPARK (linear chain, static ragged verify).
    Target verify splits attention into history (all heads over this rank's KV
    shard, LSE-merged across ranks) + current (this rank's heads over the
    verify window's own K/V); the draft model keeps its replicated KV pool.
    """
    if cfg.speculative_algorithm is not None:
        if cfg.speculative_algorithm != "DSPARK":
            raise ValueError(
                "Kimi-K3 DCP on NPU supports only speculative_algorithm "
                "'DSPARK' (history/current split target verify); got "
                f"{cfg.speculative_algorithm!r}."
            )
        from sglang.srt.speculative.ragged_verify import (
            RaggedVerifyMode,
            read_ragged_verify_mode,
        )

        ragged_mode = read_ragged_verify_mode()
        if ragged_mode is not RaggedVerifyMode.STATIC:
            raise ValueError(
                "Kimi-K3 DCP + DSPARK on NPU requires "
                "SGLANG_RAGGED_VERIFY_MODE=static: the split verify assumes a "
                "fixed window of speculative_num_draft_tokens per request "
                f"(got {ragged_mode.value!r})."
            )
        topk = cfg.speculative_eagle_topk
        if topk not in (None, 1):
            raise ValueError(
                "Kimi-K3 DCP + DSPARK on NPU requires a linear draft chain "
                f"(speculative_eagle_topk in (None, 1)), got {topk!r}."
            )
    if hf_config is not None:
        from sglang.srt.configs.model_config import is_deepseek_dsa

        if is_deepseek_dsa(hf_config):
            raise ValueError(
                "Kimi-K3 DCP on NPU does not support DSA checkpoints (index_topk "
                "set): the sparse indexer and its index-K cache are not aware of "
                "the DCP-sharded KV layout."
            )
    if envs.SGLANG_NPU_USE_MLAPO.get():
        raise ValueError(
            "Kimi-K3 DCP on NPU does not support SGLANG_NPU_USE_MLAPO=1: the "
            "fused MLA preprocess writes KV and projects Q without the DCP "
            "owner filter / full-head query."
        )
    if cfg.enable_hierarchical_cache:
        raise ValueError(
            "Kimi-K3 DCP on NPU does not support --enable-hierarchical-cache: "
            "host offload is not aware of the DCP-sharded KV layout."
        )
    if cfg.disaggregation_mode == "decode":
        raise ValueError(
            "Kimi-K3 DCP on NPU does not support --disaggregation-mode decode: "
            "KV transfer is not aware of the DCP-sharded KV layout."
        )
    for backend in attention_backends_of(cfg):
        if backend not in (None, "ascend"):
            raise ValueError(
                "Kimi-K3 DCP on NPU requires the 'ascend' attention backend, "
                f"got {backend!r}."
            )

    overrides = {}
    replicate_q_proj = cfg.dcp_replicate_q_proj
    if replicate_q_proj is None:
        logger.info("Kimi-K3 DCP on NPU enables replicated Q projection by default.")
        replicate_q_proj = True
        overrides["dcp_replicate_q_proj"] = True
    if replicate_q_proj:
        # Replicated Q needs the a2a merge. An explicit --dcp-replicate-q-proj
        # with ag_rs is already rejected by handle_dcp_validation, so this only
        # switches the backend when replication was enabled by default above.
        if cfg.dcp_comm_backend != "a2a":
            logger.info(
                "Kimi-K3 DCP on NPU with replicated Q selects communication "
                f"backend: {cfg.dcp_comm_backend!r} -> 'a2a'."
            )
        overrides["dcp_comm_backend"] = "a2a"
    return overrides


@_register_for("KimiK3ForConditionalGeneration")
def _kimi_k3_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    if cfg.dcp_size > 1 and get_platform().is_npu:
        return _resolve_kimi_k3_npu_dcp(cfg, hf_config)
    if cfg.dcp_size > 1:
        overrides = {}
        if cfg.enable_symm_mem:
            logger.warning(
                "Kimi-K3 DCP disables --enable-symm-mem due to decode CUDA "
                "graph correctness issues."
            )
            overrides["enable_symm_mem"] = False

        if cfg.speculative_algorithm == "DSPARK":
            from sglang.srt.speculative.ragged_verify import (
                RaggedVerifyMode,
                read_ragged_verify_mode,
            )

            ragged_mode = read_ragged_verify_mode()
            if ragged_mode is not RaggedVerifyMode.STATIC:
                raise ValueError(
                    "Kimi-K3 DCP + DSPARK currently requires "
                    "SGLANG_RAGGED_VERIFY_MODE=static; compact/cap-accept are "
                    f"not validated under DCP (got {ragged_mode.value!r})."
                )

            # DSPARK target-verify + draft-extend must run on the decode
            # (cutedsl_mla) backend, whose _run_decode_kernel implements the DCP
            # signature (causal_seqs / cp_world / cp_rank). The default
            # "prefill" routes verify to trtllm_mla, whose base _run_decode_kernel
            # lacks that DCP path (TypeError: unexpected kwarg 'causal_seqs').
            overrides["speculative_attention_mode"] = "decode"

        prefill_backend, decode_backend = attention_backends_of(cfg)
        if decode_backend == "cutedsl_mla" or decode_backend is None:
            _require_kimi_k3_cutedsl_dcp_support()
            logger.info(
                "Kimi-K3 DCP keeps decode attention backend 'cutedsl_mla' "
                f"(prefill={prefill_backend!r} -> 'trtllm_mla')."
            )
            overrides.update(
                prefill_attention_backend="trtllm_mla",
                decode_attention_backend="cutedsl_mla",
            )
        elif decode_backend == "tokenspeed_mla":
            logger.info(
                "Kimi-K3 DCP overrides attention backends: "
                f"prefill={prefill_backend!r}, decode={decode_backend!r} -> "
                "'tokenspeed_mla'."
            )
            logger.info(
                "Kimi-K3 DCP with tokenspeed mla backend overrides KV cache dtype: "
                f"{cfg.kv_cache_dtype!r} -> 'fp8_e4m3'."
            )
            overrides.update(
                prefill_attention_backend="tokenspeed_mla",
                decode_attention_backend="tokenspeed_mla",
                kv_cache_dtype="fp8_e4m3",
            )
        else:
            raise AssertionError(
                f"Decode attention backend for Kimi-K3 DCP must be 'cutedsl_mla' or 'tokenspeed_mla', got {decode_backend!r}."
            )

        if cfg.dcp_replicate_q_proj is None:
            logger.info("Kimi-K3 DCP enables replicated Q projection by default.")
            overrides["dcp_replicate_q_proj"] = True

        device_name = get_device_name()
        dcp_comm_backend = "fi_a2a" if is_mnnvl_fabric_device() else "a2a"
        logger.info(
            "Kimi-K3 DCP selects communication backend on "
            f"{device_name!r}: {cfg.dcp_comm_backend!r} -> "
            f"{dcp_comm_backend!r}."
        )
        overrides["dcp_comm_backend"] = dcp_comm_backend
        return overrides

    if not (get_platform().is_sm100 and get_platform().device_sm in (100, 103)):
        return {}
    backends_unset = is_attention_backend_not_set(cfg)
    if cfg.speculative_algorithm != "DSPARK":
        if not backends_unset:
            return {}
        logger.info(
            "Use trtllm_mla as the default prefill and decode attention "
            "backend for Kimi-K3 on SM100/SM103."
        )
        return {
            "decode_attention_backend": "trtllm_mla",
            "prefill_attention_backend": "trtllm_mla",
        }
    # DSPARK: verify runs on the decode backend (mode=decode below), so this
    # picks the verify kernel -- mode=prefill routes it to flashinfer, which is
    # slow and syncs, while plain decode is cold under dspark.
    q_len = cfg.speculative_num_draft_tokens or (
        cfg.speculative_dspark_block_size + 1
        if cfg.speculative_dspark_block_size is not None
        # Checkpoint auto-infer happens after overrides; K3 draft uses block 7.
        else 8
    )
    overrides = {}
    if backends_unset:
        backend = "trtllm_mla"
        overrides["decode_attention_backend"] = backend
        overrides["prefill_attention_backend"] = "trtllm_mla"
    else:
        # Explicit backend knobs keep priority, but the mode is a separate knob
        # that still needs declaring -- else verify stays on the prefill backend,
        # whose host-side plan (flashinfer by default) forces a per-step D2H.
        _, backend = attention_backends_of(cfg)
    if _dspark_verify_on_decode_backend(backend, q_len, cfg.kv_cache_dtype):
        overrides["speculative_attention_mode"] = "decode"
        logger.info(
            "Kimi-K3 DSPARK on SM100/SM103: decode/verify attention backend "
            f"{backend} (speculative_attention_mode=decode)."
        )
    else:
        logger.warning(
            f"Kimi-K3 DSPARK: decode attention backend {backend!r} cannot serve "
            f"target verify at q_len={q_len}, so verify runs on the prefill "
            "backend (speculative_attention_mode=prefill). A host-plan prefill "
            "backend costs a per-step seq_lens D2H sync; leave the attention "
            "backend knobs unset for the sync-free default."
        )
    return overrides


@_register_for("KimiK3ForConditionalGeneration")
def _kimi_k3_moe_runner_overrides(server_args: Any, hf_config: Any) -> dict:
    # MoE runner default, independent of the attention-backend gate above.
    # trtllm-gen fused MoE (flashinfer_mxfp4) beats marlin on both the decode
    # (M=bs) and the target-verify (M=bs*(gamma+1)) regimes on SM100/SM103.
    # SM107 uses the same packed-MXFP4 runner; leaving auto unresolved falls
    # back to BF16 weight materialization during model loading.
    cfg = resolving_view(server_args)
    if cfg.moe_runner_backend != "auto":
        return {}
    if not (get_platform().is_sm100 and get_platform().device_sm in (100, 103, 107)):
        return {}
    if not _is_mxfp4_pack_quantized(hf_config):
        return {}
    logger.info(
        "Kimi-K3 on SM100/SM103/SM107: moe_runner_backend=flashinfer_mxfp4 "
        "(FlashInfer SiTU kernels)."
    )
    return {"moe_runner_backend": "flashinfer_mxfp4"}
