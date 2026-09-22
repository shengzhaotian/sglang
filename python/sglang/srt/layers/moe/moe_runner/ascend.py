"""Ascend MoE runner backend with NPU‑specific ops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.moe.activation import (
    AllGatherActivationWrapper,
    NPUGeluAndMul,
    NPUSitu,
    NPUSituMXFP8Quant,
    NPUSwiglu,
    NPUSwigluDeepEPKernel,
    NPUSwigluOAI,
    NPUSwigluQuant,
    NPUSwigluStepAndMul,
)
from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUMXFP8MoEMethod,
    NPUW4A8Int8MoEMethod,
    NPUW4A8MXFP4MoEMethod,
    NPUW8A8Int8MoEMethod,
)
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_post_permute,
    register_pre_permute,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.ascend_tp import (
        AscendTPCombineInput,
        AscendTPDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput,
        DeepEPLLDispatchOutput,
        DeepEPNormalCombineInput,
        DeepEPNormalDispatchOutput,
    )

from sglang.srt.layers.moe.utils import (
    MoeRunnerBackend,
    get_moe_a2a_backend,
)


# ---------------------------------------------------------------------------
# Runner IO dataclasses
# ---------------------------------------------------------------------------
@dataclass
class AscendRunnerInput(RunnerInput):
    """Input bundle for the NPU runner."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]  # None for unquant
    expert_tokens: torch.Tensor
    group_list_type: int  # 0 or 1 (passed to NPU ops)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.ASCEND


@dataclass
class AscendRunnerOutput(RunnerOutput):
    """Output bundle from the NPU runner."""

    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.ASCEND


# ---------------------------------------------------------------------------
# Main runner core
# ---------------------------------------------------------------------------
class AscendRunnerCore(MoeRunnerCore):
    runner_backend = MoeRunnerBackend.ASCEND

    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)

        kernel = config.layer.w2_kernel
        self.use_fused_gmm1_situ = False

        if isinstance(kernel, NPUMXFP8MoEMethod):
            # MXFP8 fuses gate/up + swiglu + requant into gmm1, so there is no
            # separate activation step — run() skips it. Left None on purpose so
            # that reaching for it fails loudly instead of silently applying an
            # unfused swiglu to already-requantised activations. This holds for
            # both dispatchers: ascend_tp gets its activation quant fused into
            # routing, DeepEP dispatches bf16 and gmm1 quantises it itself.
            self.activation = None
        elif get_moe_a2a_backend().is_deepep():
            # DeepEP path: use a unified kernel that decides quantisation
            is_quant_kernel = isinstance(
                kernel, (NPUW4A8Int8MoEMethod, NPUW8A8Int8MoEMethod)
            )
            if config.activation == "situ":
                beta = config.gemm1_alpha if config.gemm1_alpha is not None else 4.0
                if (isinstance(kernel, NPUW4A8MXFP4MoEMethod)
                    and envs.SGLANG_NPU_MOE_GMM_SITU_QUANT_FUSED.get()):
                    self.use_fused_gmm1_situ = True
                    self.activation = None  # 融合路径不需要独立 activation
                    self.fused_beta = beta  # 已有的 beta 计算
                    self.fused_linear_beta = config.gemm1_clamp_limit
                elif (isinstance(kernel, NPUW4A8MXFP4MoEMethod)
                    and envs.SGLANG_NPU_MOE_SITU_MXFP8_FUSED.get()
                ):
                    if config.gemm1_clamp_limit is None:
                        raise ValueError(
                            "fused SiTU MXFP8 quantization requires gemm1_clamp_limit"
                        )
                    self.activation = NPUSituMXFP8Quant(
                        beta=beta,
                        linear_beta=config.gemm1_clamp_limit,
                    )
                else:
                    self.activation = NPUSitu(
                        need_quant=is_quant_kernel,
                        beta=beta,
                        linear_beta=config.gemm1_clamp_limit,
                    )
            else:
                self.activation = NPUSwigluDeepEPKernel(
                    need_quant=is_quant_kernel,
                    alpha=config.gemm1_alpha,
                    limit=config.gemm1_clamp_limit,
                )
        else:
            # Non‑DeepEP (ascend_tp) path
            # 1. Choose the base activation according to the quant method
            if isinstance(kernel, (NPUW4A8Int8MoEMethod, NPUW8A8Int8MoEMethod)):
                inner = NPUSwigluQuant()
            elif config.activation == "situ":
                # Grouped SiTU (Kimi-K3). need_quant=False: the MXFP4 / BF16
                # gmm2 requantizes the activations itself, so no quant is
                # fused here. Matches the DeepEP branch below.
                inner = NPUSitu(
                    need_quant=False,
                    beta=config.gemm1_alpha if config.gemm1_alpha is not None else 4.0,
                    linear_beta=config.gemm1_clamp_limit,
                )
            else:
                if config.activation == "npu_swiglu_oai":
                    # NPUSwigluOAI requires the runner config to pass
                    # gemm1_alpha and gemm1_clamp_limit to the triton kernel.
                    inner = NPUSwigluOAI(moe_runner_config=config)
                elif config.activation == "silu":
                    if config.gemm1_clamp_limit is not None:
                        inner = NPUSwigluStepAndMul(
                            clamp_limit=config.gemm1_clamp_limit
                        )
                    else:
                        inner = NPUSwiglu()
                else:
                    inner = NPUGeluAndMul()

            # 2. If the quant method (GGUF) needs TP all‑gather, wrap the activation
            if getattr(config, "use_tp_all_gather_activation", False):
                self.activation = AllGatherActivationWrapper(inner, dim=-1)
            else:
                self.activation = inner

    def run(
        self,
        runner_input: AscendRunnerInput,
        quant_info: AscendQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> AscendRunnerOutput:
        """
        Execute the MoE layer using NPU‑specific grouped matmul ops.
        """
        cur_stream = torch.cuda.current_stream()
        torch.npu.reset_stream_limit(cur_stream)
        x = runner_input.hidden_states
        original_dtype = torch.float16 if x.dtype == torch.float16 else torch.bfloat16
        expert_tokens = runner_input.expert_tokens
        group_list_type = runner_input.group_list_type

        w13_kernel = self.config.layer.w13_kernel

        if getattr(self, 'use_fused_gmm1_situ', False):
            # 融合 GMM1+SiTU+MXFP8 quant → 直接产出 (FP8, E8M0 scale) 给 GMM2
            hidden_states, pertoken_scale = w13_kernel.apply_fused_gmm1_situ(
                quant_info, x, expert_tokens,
                pertoken_scale=runner_input.hidden_states_scale,
                group_list_type=group_list_type,
                beta=self.fused_beta,
                linear_beta=self.fused_linear_beta,
            )
        elif isinstance(w13_kernel, NPUMXFP8MoEMethod):
            # --- w13 projection + activation, fused into one kernel ---
            # MXFP8 gmm1 returns activations already requantised for gmm2, so
            # there is no separate activation step to run.
            hidden_states, pertoken_scale = w13_kernel.apply_fused_gmm1_swiglu(
                quant_info,
                x,
                expert_tokens,
                pertoken_scale=runner_input.hidden_states_scale,
                group_list_type=group_list_type,
            )
        else:
            # --- w13 (gate & up) projection ---
            hidden_states = w13_kernel.apply(
                quant_info,
                x,
                expert_tokens,
                pertoken_scale=runner_input.hidden_states_scale,
                output_dtype=original_dtype,
                weight_prefix="w13",
                group_list_type=group_list_type,
            )

            # --- Activation ---
            # Grouped-row activations require dispatch metadata.
            if isinstance(
                self.activation,
                (NPUSwigluDeepEPKernel, NPUSitu, NPUSituMXFP8Quant),
            ):
                hidden_states, pertoken_scale = self.activation._apply_activation(
                    hidden_states,
                    group_list=expert_tokens,
                    group_list_type=group_list_type,
                )
            else:
                hidden_states, pertoken_scale = self.activation._apply_activation(
                    hidden_states
                )

        # --- w2 (down) projection ---
        hidden_states = self.config.layer.w2_kernel.apply(
            quant_info,
            hidden_states,
            expert_tokens,
            pertoken_scale=pertoken_scale,
            output_dtype=original_dtype,
            weight_prefix="w2",
            group_list_type=group_list_type,
        )
        return AscendRunnerOutput(hidden_states=hidden_states)

    def w13_proj(
        self,
        runner_input: AscendRunnerInput,
        quant_info: AscendQuantInfo,
    ):
        """Run only the w13 (gate & up) grouped matmul.

        Returns ``(hidden_states, pertoken_scale, original_dtype)``.
        ``pertoken_scale`` is ``None`` when activation is not fused.
        """
        x = runner_input.hidden_states
        original_dtype = torch.float16 if x.dtype == torch.float16 else torch.bfloat16
        expert_tokens = runner_input.expert_tokens
        group_list_type = runner_input.group_list_type

        w13_kernel = self.config.layer.w13_kernel

        if getattr(self, 'use_fused_gmm1_situ', False):
            # 融合 GMM1+SiTU+MXFP8 quant → 直接产出 (FP8, E8M0 scale) 给 GMM2
            hidden_states, pertoken_scale = w13_kernel.apply_fused_gmm1_situ(
                quant_info, x, expert_tokens,
                pertoken_scale=runner_input.hidden_states_scale,
                group_list_type=group_list_type,
                beta=self.fused_beta,
                linear_beta=self.fused_linear_beta,
            )
        elif isinstance(w13_kernel, NPUMXFP8MoEMethod):
            hidden_states, pertoken_scale = w13_kernel.apply_fused_gmm1_swiglu(
                quant_info,
                x,
                expert_tokens,
                pertoken_scale=runner_input.hidden_states_scale,
                group_list_type=group_list_type,
            )
        else:
            hidden_states = w13_kernel.apply(
                quant_info,
                x,
                expert_tokens,
                pertoken_scale=runner_input.hidden_states_scale,
                output_dtype=original_dtype,
                weight_prefix="w13",
                group_list_type=group_list_type,
            )
            pertoken_scale = None
        return hidden_states, pertoken_scale, original_dtype

    def apply_act(
        self,
        hidden_states: torch.Tensor,
        pertoken_scale: Optional[torch.Tensor],
        runner_input: AscendRunnerInput,
    ):
        """Apply the activation function between w13 and w2.

        Returns ``(hidden_states, pertoken_scale)``.  When ``pertoken_scale``
        is already set (MXFP8 fused path) this is a passthrough.
        """
        if pertoken_scale is not None:
            return hidden_states, pertoken_scale

        expert_tokens = runner_input.expert_tokens
        group_list_type = runner_input.group_list_type

        if isinstance(
            self.activation,
            (NPUSwigluDeepEPKernel, NPUSitu, NPUSituMXFP8Quant),
        ):
            return self.activation._apply_activation(
                hidden_states,
                group_list=expert_tokens,
                group_list_type=group_list_type,
            )
        return self.activation._apply_activation(hidden_states)

    def w2_proj(
        self,
        hidden_states: torch.Tensor,
        pertoken_scale: Optional[torch.Tensor],
        runner_input: AscendRunnerInput,
        quant_info: AscendQuantInfo,
        original_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Run only the w2 (down) grouped matmul."""
        expert_tokens = runner_input.expert_tokens
        group_list_type = runner_input.group_list_type
        return self.config.layer.w2_kernel.apply(
            quant_info,
            hidden_states,
            expert_tokens,
            pertoken_scale=pertoken_scale,
            output_dtype=original_dtype,
            weight_prefix="w2",
            group_list_type=group_list_type,
        )


# ---------------------------------------------------------------------------
# QuantInfo
# ---------------------------------------------------------------------------
@dataclass
class AscendQuantInfo(MoeQuantInfo):
    """Quantization payload for Ascend."""

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_weight_scale: Optional[torch.Tensor] = None
    w2_weight_scale: Optional[torch.Tensor] = None
    w13_weight_offset: Optional[torch.Tensor] = None
    w2_weight_offset: Optional[torch.Tensor] = None
    w13_weight_bias: Optional[torch.Tensor] = None
    w2_weight_bias: Optional[torch.Tensor] = None
    w13_scale_bias: Optional[torch.Tensor] = None
    w2_scale_bias: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Pre/Post permute hooks
# ---------------------------------------------------------------------------


@register_pre_permute("ascend_tp", "ascend")
def pre_permute_ascend_tp_to_ascend(
    dispatch_output: AscendTPDispatchOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AscendRunnerInput:
    return AscendRunnerInput(
        hidden_states=dispatch_output.hidden_states,
        hidden_states_scale=dispatch_output.hidden_states_scale,
        expert_tokens=dispatch_output.expert_tokens,
        group_list_type=dispatch_output.group_list_type,
    )


@register_pre_permute("deepep_normal", "ascend")
def pre_permute_deepep_normal_to_ascend(
    dispatch_output: DeepEPNormalDispatchOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AscendRunnerInput:
    (
        hidden_states,
        hidden_states_scale,
        topk_ids,
        topk_weights,
        num_recv_tokens_per_expert,
    ) = dispatch_output
    group_list = torch.tensor(
        num_recv_tokens_per_expert,
        dtype=torch.int64,
        device=hidden_states.device,
    )
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights

    return AscendRunnerInput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        expert_tokens=group_list,
        group_list_type=1,
    )


@register_pre_permute("deepep_ll", "ascend")
def pre_permute_deepep_ll_to_ascend(
    dispatch_output: DeepEPLLDispatchOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AscendRunnerInput:
    (
        hidden_states,
        hidden_states_scale,
        topk_ids,
        topk_weights,
        group_list,
        _,
    ) = dispatch_output
    group_list = group_list.to(torch.int64)
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights
    return AscendRunnerInput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        expert_tokens=group_list,
        group_list_type=1,
    )


@register_post_permute("ascend", "ascend_tp")
def post_permute_ascend_to_ascend_tp(
    runner_output: AscendRunnerOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AscendTPCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.ascend_tp import AscendTPCombineInput

    return AscendTPCombineInput(hidden_states=runner_output.hidden_states)


@register_post_permute("ascend", "deepep_normal")
def post_permute_ascend_to_deepep_normal(
    runner_output: AscendRunnerOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPNormalCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalCombineInput

    return DeepEPNormalCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )


@register_post_permute("ascend", "deepep_ll")
def post_permute_ascend_to_deepep_ll(
    runner_output: AscendRunnerOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPLLCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput

    return DeepEPLLCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )
