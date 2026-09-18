"""Checkpoint scale loading for the NPU shared-latent Q8/KV8 MLA path."""

import torch
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.quantization.modelslim.schemes.modelslim_scheme import (
    ModelSlimKVSchemeBase,
)
from torch import nn


def _modelslim_kv_weight_loader(
    param: torch.Tensor, loaded_weight: torch.Tensor
) -> None:
    if param.numel() == 1 and loaded_weight.numel() == 1:
        param.data.fill_(loaded_weight.item())
        return

    loaded_weight = loaded_weight.to(param.dtype)
    if loaded_weight.shape != param.shape:
        if loaded_weight.ndim == 0:
            raise ValueError("Cannot shard a scalar ModelSlim KV scale")
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        if loaded_weight.shape[0] % tp_size:
            raise ValueError(
                f"Cannot shard ModelSlim KV weight {tuple(loaded_weight.shape)} "
                f"across tensor parallel size {tp_size}."
            )
        shard_size = loaded_weight.shape[0] // tp_size
        loaded_weight = loaded_weight.narrow(0, shard_size * tp_rank, shard_size)
    if loaded_weight.shape != param.shape:
        raise ValueError(
            f"ModelSlim KV weight shape {tuple(loaded_weight.shape)} does not "
            f"match parameter shape {tuple(param.shape)}."
        )
    param.data.copy_(loaded_weight)


class ModelSlimQFP8DynamicKVFP8Scheme(ModelSlimKVSchemeBase):
    def __init__(self, quant_config, prefix: str):
        self.quant_config = quant_config
        self.prefix = prefix

    def create_weights(self, layer: nn.Module, num_heads: int, num_kv_heads: int):
        if num_heads <= 0 or num_kv_heads <= 0:
            raise ValueError("ModelSlim attention head counts must be positive")
        # Q is dynamically quantized per token/head; this checkpoint has no fa_q.
        for name in ("fa_k", "fa_v"):
            module = nn.Module()
            for field, initial_value in (("scale", torch.nan), ("offset", 0.0)):
                param = nn.Parameter(
                    torch.full((num_kv_heads, 1), initial_value, dtype=torch.float32),
                    requires_grad=False,
                )
                param.weight_loader = _modelslim_kv_weight_loader
                module.register_parameter(field, param)
            layer.add_module(name, module)

        for name in ("fak_descale_float", "fak_descale_reciprocal"):
            layer.register_buffer(
                name,
                torch.full((1, num_kv_heads), torch.nan, dtype=torch.float32),
                persistent=False,
            )
        layer._requires_modelslim_fp8_kv_scale = True
        layer._modelslim_fp8_kv_scale_ready = False

    def process_weights_after_loading(self, layer: nn.Module):
        # Invalidate a previous successful load before checking replacement data.
        layer._modelslim_fp8_kv_scale_ready = False
        # The absorbed MLA cache quantizes the shared latent with fa_k only.
        # fa_v is loaded for checkpoint compatibility, not used to quantize or
        # descale the latent a second time (matching the ModelSlim Kimi adapter).
        scale = layer.fa_k.scale
        if not torch.isfinite(scale).all():
            raise RuntimeError(
                f"Missing or non-finite ModelSlim fa_k.scale for {self.prefix}; "
                "Q_FP8_DYNAMIC_KV_FP8 does not permit a unit-scale fallback."
            )
        if (scale <= 0).any():
            raise ValueError("ModelSlim fa_k.scale must be positive")
        offset = layer.fa_k.offset
        if not torch.isfinite(offset).all() or torch.count_nonzero(offset).item():
            raise ValueError("ModelSlim fa_k.offset must be finite and zero")

        scale = scale.reshape(1, -1)
        reciprocal = scale.reciprocal()
        if not torch.isfinite(reciprocal).all():
            raise ValueError("ModelSlim KV reciprocal scale must be finite in FP32")
        layer.fak_descale_float.copy_(scale)
        layer.fak_descale_reciprocal.copy_(reciprocal)
        layer._modelslim_fp8_kv_scale_ready = True
