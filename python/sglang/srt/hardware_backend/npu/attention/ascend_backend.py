from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch
import torch_npu
from sgl_kernel_npu.attention.sinks_attention import (
    attention_sinks_prefill_triton,
    attention_sinks_triton,
)

from sglang.srt.configs.model_config import AttentionArch, is_deepseek_dsa
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.attention.ascend_torch_native_backend import (
    AscendTorchNativeAttnBackend,
)
from sglang.srt.hardware_backend.npu.attention.fp8_contracts import (
    DSA_KV_QUANT_TILE_SIZE,
    get_dsa_fp8_packed_cache_dim,
)
from sglang.srt.hardware_backend.npu.attention.mla_cache import gather_mla_cache_pages
from sglang.srt.hardware_backend.npu.attention.mla_preprocess import (
    is_fia_nz,
    is_mla_preprocess_enabled,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_kv_cache
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.distributed.parallel_state import get_attn_tp_group
from sglang.srt.runtime_context import (
    get_flags,
    get_parallel,
    get_spec,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.utils import (
    get_bool_env_var,
    get_current_device_stream_fast,
    next_power_of_2,
)
from cann_ops_transformer.ops.attention.flash_mla_with_kvcache import (
    flash_mla_with_kvcache,
    flash_mla_with_kvcache_metadata,
)
from sglang.srt.hardware_backend.npu.attention.fp8_contracts import get_dsa_fp8_packed_cache_dim
if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

import logging

import numpy as np

logger = logging.getLogger(__name__)
FULL_ATTENTION_WINDOW = 2147483647


def _is_dflash_verify(spec_info: Optional[SpecInput]) -> bool:
    return (
        spec_info is not None
        and spec_info.spec_input_type == SpecInputType.DFLASH_VERIFY
    )


def _expand_dsa_sparse_indices(topk_indices: torch.Tensor) -> torch.Tensor:
    """Expand [T, K] to [T, 1, K] for NPU sparse attention."""
    if topk_indices.dim() == 2:
        return topk_indices.unsqueeze(-2)
    return topk_indices


def _reshape_kv_for_fia_nz(
    tensor: torch.Tensor, num_heads: int, head_dim: int, page_size: int
) -> torch.Tensor:
    """Reshapes a tensor for FIA NZ format."""
    return tensor.view(-1, 1, num_heads * head_dim // 16, page_size, 16)


@dataclass
class ForwardMetadata:

    # calculated map for kv positions [bs * maxseqlen]
    block_tables: Optional[torch.Tensor] = None

    # mapped block_tables for swa
    block_tables_swa: Optional[torch.Tensor] = None

    # pre-translated full->SWA write target for SWAKVPool.set_kv_buffer
    swa_out_cache_loc: Optional[torch.Tensor] = None

    # seq len inputs
    extend_seq_lens_cpu_int: Optional[torch.Tensor] = None
    seq_lens_cpu_int: Optional[torch.Tensor] = None
    seq_lens_cpu_list: Optional[List[int]] = None
    seq_lens_list_cumsum: Optional[List[int]] = None
    seq_lens: Optional[torch.Tensor] = None
    actual_seq_lengths_q: Optional[torch.Tensor] = None
    actual_seq_lengths_q_pa: Optional[torch.Tensor] = None
    # CPU mirror of actual_seq_lengths_q_pa for the host metadata op
    # (torch.ops.npu.sparse_attn_sharedkv_metadata_host reads CPU int32 inputs).
    actual_seq_lengths_q_pa_cpu: Optional[torch.Tensor] = None
    actual_seq_lengths_kv: Optional[torch.Tensor] = None

    # swa attention mask for graph mode decode
    swa_mask: Optional[torch.Tensor] = None

    # prefix cache
    prefix_lens: Optional[torch.Tensor] = None
    flatten_prefix_block_tables: Optional[torch.Tensor] = None
    metadata_flash_mla = None
    seqused_q = None

    # A2A FIAS V2 BSND: per-rank local metadata (computed once per step,
    # reused across all layers in the forward pass)
    a2a_block_table_local: Optional[torch.Tensor] = None
    a2a_cache_seqlens_local: Optional[torch.Tensor] = None
    a2a_seqused_q_local: Optional[torch.Tensor] = None
    a2a_metadata_flash_mla: Optional[torch.Tensor] = None
    a2a_num_local_reqs: int = 0
    a2a_T_padded: int = 0
    a2a_T_local: int = 0

    # Pre-computed Ulysses all-to-all metadata for sparse attention prefill.
    # Keyed by attn_tp_rank; each entry holds the per-rank tensors needed by
    # _forward_sparse_attn_tp_a2a_prefill, computed on CPU before model.forward().
    a2a_prefill_meta: Optional[dict] = None

    # Pre-computed attn-tp token-split metadata for the DSA indexer.
    # Computed on CPU before model.forward() to avoid pipeline bubbles.
    indexer_attn_tp_meta: Optional[dict] = None

class AscendAttnMaskBuilder:
    def __init__(self, model_runner: ModelRunner, device, use_fia, use_mla):
        """
        Initialize the AscendAttnMaskBuilder class.

        :param model_runner: ModelRunner instance for model execution.
        :param device: Device to run the model on (e.g., 'cuda', 'npu').
        :param use_fia: Boolean flag to indicate if environment variable ASCEND_USE_FIA is set to 1.
        """
        self.use_fia = use_fia
        self.model_runner = model_runner
        self.device = device

        # Initialize mask
        mask_len = 128
        self.mask = self.generate_attn_mask(mask_len, "norm", model_runner.dtype).to(
            self.device
        )

        # Initialize FIA mask
        fia_mask_len = 2048
        self.fia_mask = self.generate_mask_flag(fia_mask_len).to(self.device)

        # Initialize MTP mask
        mtp_mask_len = 2048
        self.mtp_mask = self.generate_mask_flag(mtp_mask_len).to(self.device)

        # Initialize mixed chunk mask cache
        mixed_mask_len = 2048
        self.mixed_chunk_attn_mask = self.get_splitfuse_attn_mask(mixed_mask_len)

        if use_mla:
            # Initialize RingMla mask
            ringmla_mask_len = 512
            self.ringmla_mask = self.generate_attn_mask(
                ringmla_mask_len, "norm", torch.bfloat16
            ).to(self.device)

    @staticmethod
    def generate_mask_flag(max_seq_len):
        """
        Generate a mask flag for attention masks.

        :param max_seq_len: Maximum sequence length for the mask.
        :return: A boolean tensor representing the mask flag.
        """
        # Construct lower triangle matrix.
        mask_flag = torch.ones((max_seq_len, max_seq_len), dtype=torch.bool).tril_()
        # Create upper triangle matrix used to mark mask positions.
        mask_flag = ~mask_flag
        return mask_flag

    @staticmethod
    def generate_attn_mask(max_seq_len, mode, dtype=torch.float16):
        """
        Generate an attention mask.

        :param max_seq_len: Maximum sequence length for the mask.
        :param mode: Mode of the mask ('mix' or 'norm').
        :param dtype: Data type of the mask tensor.
        :return: A tensor representing the attention mask.
        """
        mask_flag = AscendAttnMaskBuilder.generate_mask_flag(max_seq_len)
        if mode == "mix":
            mask_value = (
                float("-inf") if dtype in [torch.float16, torch.bfloat16] else 1
            )
        else:
            mask_value = torch.finfo(torch.float32).min if dtype == torch.float16 else 1
        attn_mask = (
            torch.zeros(size=(max_seq_len, max_seq_len))
            .masked_fill_(mask_flag, mask_value)
            .to(dtype)
        )
        return attn_mask

    @staticmethod
    def get_attention_mask_id(seq_lens, extend_lens):
        """
        Generate attention mask IDs based on sequence lengths and extended lengths.

        :param seq_lens: Sequence lengths.
        :param extend_lens: Extended lengths.
        :return: A tensor containing the attention mask IDs.
        """
        starts = seq_lens - extend_lens
        ends = seq_lens

        # Use torch.stack to stack the start and end indices together
        ranges = torch.stack((starts, ends), dim=-1)

        # Use list comprehension to generate tensors for each range and concatenate them
        attn_mask_id = torch.cat([torch.arange(start, end) for start, end in ranges])
        return attn_mask_id

    def update_attn_cache(
        self,
        seqlen: int,
        mask_cache: torch.Tensor,
        seq_len_cached: int,
        dtype: torch.dtype,
        mode,
    ):
        """
        Update the attention mask cache.

        :param seqlen: Maximum sequence length.
        :param mask_cache: Current attention mask cache.
        :param seq_len_cached: Cached sequence length.
        :param dtype: Data type of the mask tensor.
        :param mode: Mode of the mask ('mix' or 'norm').
        :return: Updated mask cache and sequence length cache.
        """
        if seqlen > seq_len_cached:
            seq_len_cached = seqlen
            mask_cache = self.generate_attn_mask(seqlen, mode, dtype)
        if mask_cache.dtype != dtype:
            mask_cache = mask_cache.to(dtype)
        return mask_cache, seq_len_cached

    def get_splitfuse_attn_mask(
        self,
        seq_lens: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Generate a splitfuse attention mask.

        :param seq_lens: Sequence lengths.
        :return: A tensor representing the splitfuse attention mask.
        """
        attn_mask = (
            torch.triu(torch.ones(seq_lens, seq_lens), diagonal=1)
            .to(torch.int8)
            .to(self.device)
        )
        return attn_mask

    def get_swa_mask(self, seq_lens: torch.Tensor, s2: int, left_context=512):
        if seq_lens.dim() == 1:
            seq_lens = seq_lens.unsqueeze(1)
        b = seq_lens.size(0)
        device = seq_lens.device
        indices = torch.arange(s2, device=device).unsqueeze(0).expand(b, -1)
        start_indices = torch.clamp(seq_lens - left_context, min=0)
        mask = (indices < start_indices) | (indices >= seq_lens)
        return mask.unsqueeze(1).to(self.device, non_blocking=True)


def _cp_allgather_and_save_kv_npu(
    forward_batch, layer, k, v, cp_size, token_to_kv_pool, swa_loc=None
):
    """NPU-compatible CP KV all-gather with merged K/V communication.

    Merges K and V along the feature dimension so only one all-gather is
    needed instead of two, halving communication latency.

    k shape: [S_local, tp_k_head_num, qk_head_dim]
    v shape: [S_local, tp_v_head_num, v_head_dim]

    Equivalent to cp_allgather_and_save_kv_cache() in cp_utils.py, but uses
    a single all-gather for both K and V.

    swa_loc is the pre-translated full->SWA write target for hybrid SWA pools
    (None for non-SWA pools); set_kv_buffer never translates internally.
    """
    cache_loc = (
        forward_batch.out_cache_loc
        if not layer.is_cross_attention
        else forward_batch.encoder_out_cache_loc
    )
    # Save original trailing shapes for reshape after gather.
    k_tail = k.shape[1:]  # (tp_k_head_num, qk_head_dim)
    v_tail = v.shape[1:]  # (tp_v_head_num, v_head_dim)

    # Flatten trailing dims then concat → one all-gather instead of two.
    # Works for GQA where tp_k_head_num != tp_v_head_num.
    k_flat = k.contiguous().reshape(k.shape[0], -1)  # [S_local, k_feat]
    v_flat = v.contiguous().reshape(v.shape[0], -1)  # [S_local, v_feat]
    k_feat_size = k_flat.shape[-1]
    kv_flat = torch.cat([k_flat, v_flat], dim=-1)  # [S_local, k_feat + v_feat]

    kv_full = cp_all_gather_rerange_kv_cache(
        kv_flat, cp_size, forward_batch, get_current_device_stream_fast()
    )  # [S_full, k_feat + v_feat]

    key_cache_full = kv_full[..., :k_feat_size].reshape(-1, *k_tail)
    value_cache_full = kv_full[..., k_feat_size:].reshape(-1, *v_tail)

    token_to_kv_pool.set_kv_buffer(
        layer,
        KVWriteLoc(cache_loc, swa_loc),
        key_cache_full,
        value_cache_full,
    )


class AscendAttnBackend(AttentionBackend):

    def __init__(self, model_runner: ModelRunner, speculative_step_id: int = 0):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device
        self.speculative_step_id = speculative_step_id
        self.speculative_step_offset_npu = torch.tensor(
            speculative_step_id + 1, device="npu"
        )
        self.page_size = model_runner.page_size
        self.model_dtype = model_runner.model_config.dtype
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        self.use_mla_fp8 = (
            self.use_mla
            and model_runner.kv_cache_dtype_str == "fp8_e4m3"
            and not is_deepseek_dsa(model_runner.model_config.hf_text_config)
        )
        if self.use_mla:
            self.kv_lora_rank = model_runner.model_config.kv_lora_rank
            self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
            if (
                "MiniCPM3ForCausalLM"
                in model_runner.model_config.hf_config.architectures
            ):
                self.qk_nope_head_dim = (
                    model_runner.model_config.hf_config.qk_nope_head_dim
                )
            else:
                self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
            self.q_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        else:
            self.use_alibi = getattr(model_runner.model_config, "use_alibi", False)
            if (
                "Gemma2ForSequenceClassification"
                in model_runner.model_config.hf_config.architectures
            ):
                self.use_native_sdpa = True
        self.native_attn = AscendTorchNativeAttnBackend()
        self.graph_metadata = {}
        self.max_context_len = model_runner.model_config.context_len
        # Pool refs — captured at construction so they survive deletion of the
        # corresponding ForwardBatch fields.
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.graph_mode = False
        self.use_fa = get_bool_env_var("ASCEND_USE_FA", "False")
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        self.use_fias_v2_bsnd = (
            envs.SGLANG_NPU_USE_FIAS_V2_BSND.get()
            and model_runner.spec_algorithm.is_dspark()
        )
        self.use_mojo_mtp = get_bool_env_var("ASCEND_USE_MOJO_MTP", "False")
        self.use_flash_mla = get_bool_env_var("SGLANG_NPU_USE_FLASH_MLA", "False")
        self.enable_torch_compile = get_flags().capture.enable_torch_compile
        self.speculative_num_draft_tokens = get_spec().speculative_num_draft_tokens
        if (
            self.speculative_num_draft_tokens is not None
            and model_runner.is_draft_worker
        ):
            self.speculative_num_draft_tokens = (
                model_runner.spec_algorithm.get_num_tokens_per_req_for_target_verify(
                    int(self.speculative_num_draft_tokens), is_draft_worker=True
                )
            )
        self.ascend_attn_mask_builder = AscendAttnMaskBuilder(
            model_runner, self.device, self.use_fia, self.use_mla
        )
        self.mask, self.fia_mask, self.mtp_mask, self.mix_mask = (
            self.ascend_attn_mask_builder.mask,
            self.ascend_attn_mask_builder.fia_mask,
            self.ascend_attn_mask_builder.mtp_mask,
            self.ascend_attn_mask_builder.mixed_chunk_attn_mask,
        )
        if self.use_mla:
            if self.use_flash_mla and not self.use_mla_fp8:
                self.mtp_mask = self.mtp_mask.to(torch.int8)
            self.ringmla_mask = self.ascend_attn_mask_builder.ringmla_mask
        self.is_hybrid_swa = model_runner.is_hybrid_swa
        if self.is_hybrid_swa:
            self.full_to_swa_index_mapping = (
                model_runner.token_to_kv_pool.full_to_swa_index_mapping
            )
            self.sliding_window_size = model_runner.sliding_window_size
        self.use_sliding_window_kv_pool = (
            isinstance(self.token_to_kv_pool, SWAKVPool)
            and self.token_to_kv_pool.swa_layer_nums > 0
        )

        # C8 FIA v2 and NPUGraph.update consume host KV lengths, including
        # DSpark's final verify boundary. The device-only opt-out is not valid
        # for this operator, even when attention TP A2A is enabled.
        self.needs_cpu_seq_lens = (
            envs.SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS.get() # or self.use_mla_fp8
        )

        # AllToAll optimization for sparse attention across attention TP ranks
        self.use_sparse_attn_a2a = (
            envs.SGLANG_NPU_SPARSE_ATTN_A2A.get()
            and get_parallel().attn_tp_size > 1
        )

        self.use_indexer_tp = (
            envs.SGLANG_NPU_INDEXER_TP.get()
            and get_parallel().attn_tp_size > 1
        )

        # head num padding
        self.padding_size_list = [1, 2, 4, 8, 16, 32, 64, 128]
        self.tp_q_head_num = (
                model_runner.model_config.num_attention_heads
                // get_parallel().attn_tp_size
        )
        self.q_head_num_padding = None
        if not self.use_flash_mla:
            if hasattr(model_runner.model_config, "num_attention_heads") and self.use_mla:
                for num in self.padding_size_list:
                    if num >= self.tp_q_head_num:
                        self.q_head_num_padding = num
                        break

        # dllm model config
        self.dllm_config = DllmConfig.from_server_args(model_runner.server_args)
        self.is_dllm_model = False
        if self.dllm_config is not None:
            self.is_dllm_model = True
            self.dllm_block_size = self.dllm_config.block_size

        self.attn_cp_size = model_runner.ps.attn_cp_size

    def _is_swa_layer(self, layer: RadixAttention) -> bool:
        return (
            self.is_hybrid_swa
            and layer.sliding_window_size is not None
            and layer.sliding_window_size > -1
        )

    @staticmethod
    def _can_use_tnd(layer: RadixAttention) -> bool:
        """Check if TND layout is supported."""
        d = layer.qk_head_dim
        v = layer.v_head_dim
        return (d == v and d in (128, 192, 256)) or (d == 192 and v == 128)

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        pass

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        bs = forward_batch.batch_size
        if in_capture:
            self._init_cuda_graph_metadata(
                bs,
                forward_batch.forward_mode,
                forward_batch.seq_lens,
                forward_batch.out_cache_loc,
            )
        self._apply_cuda_graph_metadata(
            bs=bs,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            seq_lens_cpu=(
                forward_batch.seq_lens.cpu()
                if in_capture
                else forward_batch.seq_lens_cpu
            ),
            forward_mode=forward_batch.forward_mode,
            spec_info=forward_batch.spec_info,
            out_cache_loc=forward_batch.out_cache_loc,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        self.forward_metadata = ForwardMetadata()
        if self.needs_cpu_seq_lens:
            # Empty attention-DP ranks still participate in the target forward.
            seq_lens_max = (
                forward_batch.seq_lens.max() if forward_batch.batch_size else 0
            )
            if forward_batch.forward_mode.is_target_verify():
                spec_tokens_per_req = int(forward_batch.spec_info.draft_token_num)
                # Overlap scheduling can publish the CPU sequence length one step
                # ahead of the device tensor. FIA consumes seq_lens_cpu below, so
                # derive the block-table width from the same source. Otherwise a
                # page-aligned request can expose KV_S=N while asking FIA for N+1.
                if forward_batch.batch_size:
                    seq_lens_max = (
                        forward_batch.seq_lens_cpu.max().item() + spec_tokens_per_req
                    )
            elif (
                forward_batch.forward_mode.is_decode_or_idle()
                and forward_batch.spec_info is not None
            ):
                seq_lens_max += self.speculative_step_id + 1
            self.forward_metadata.block_tables = (
                self.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, :seq_lens_max
                ][:, :: self.page_size]
                // self.page_size
            )
            if self.is_hybrid_swa:
                self.forward_metadata.block_tables_swa = (
                    (
                        self.full_to_swa_index_mapping[
                            self.req_to_token_pool.req_to_token[
                                forward_batch.req_pool_indices, :seq_lens_max
                            ]
                        ][:, :: self.page_size]
                        // self.page_size
                    )
                    .to(torch.int32)
                    .contiguous()
                )
        else:
            self.forward_metadata.block_tables = (
                self.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, :: self.page_size
                ]
                // self.page_size
            )
        if forward_batch.extend_seq_lens is not None:
            self.forward_metadata.extend_seq_lens = forward_batch.extend_seq_lens
            self.forward_metadata.extend_seq_lens_cpu_int = (
                forward_batch.extend_seq_lens.cpu().int()
            )
        if forward_batch.seq_lens is not None:
            self.forward_metadata.seq_lens = forward_batch.seq_lens.int()
        else:
            self.forward_metadata.seq_lens = forward_batch.seq_lens_cpu.to(
                self.device
            ).int()
        if forward_batch.seq_lens_cpu is not None and self.needs_cpu_seq_lens:
            self.forward_metadata.seq_lens_cpu_int = forward_batch.seq_lens_cpu.int()

        if (
            not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_target_verify()
        ):
            seq_lens_list_cumsum = np.cumsum(forward_batch.extend_seq_lens_cpu)
            self.forward_metadata.seq_lens_list_cumsum = seq_lens_list_cumsum

        if forward_batch.forward_mode.is_target_verify():
            spec_algorithm = forward_batch.spec_algorithm
            if spec_algorithm is None or not spec_algorithm.is_dspark():
                self.forward_metadata.seq_lens_cpu_int += spec_tokens_per_req
        elif (
            forward_batch.forward_mode.is_decode_or_idle()
            and forward_batch.spec_info is not None
        ):
            self.forward_metadata.seq_lens_cpu_int += self.speculative_step_id + 1

        # Set actual_seq_lengths_q from the pre-pad batch size so that the DSA
        # indexer reads a value consistent with actual_seq_lengths_kv /
        # block_tables (which are also built from the pre-pad batch here).
        # Without this, eager decode under DP attention pads q to the global
        # max while kv stays local, causing a shape mismatch in
        # npu_lightning_indexer.
        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            spec_tokens_per_req = (
                int(forward_batch.spec_info.draft_token_num)
                if forward_batch.forward_mode.is_target_verify()
                else self.speculative_num_draft_tokens
            )
            self.forward_metadata.actual_seq_lengths_q = torch.arange(
                spec_tokens_per_req,
                spec_tokens_per_req
                + forward_batch.seq_lens.shape[0] * spec_tokens_per_req,
                spec_tokens_per_req,
                dtype=torch.int32,
                device=self.device,
            )
        elif forward_batch.forward_mode.is_decode_or_idle():
            self.forward_metadata.actual_seq_lengths_q = torch.tensor(
                [1 + i for i in range(forward_batch.seq_lens.shape[0])],
                dtype=torch.int32,
                device=self.device,
            )

        if (
            self.use_mla
            and forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_target_verify()
            and sum(forward_batch.extend_prefix_lens_cpu) > 0
        ):
            self.forward_metadata.prefix_lens = forward_batch.extend_prefix_lens.to(
                "cpu"
            )
            seq_prefix_lens = self.forward_metadata.prefix_lens.tolist()
            self.forward_metadata.flatten_prefix_block_tables = torch.empty(
                0, dtype=torch.int32
            ).to(self.device)
            for req_idx, seq_len in zip(
                forward_batch.req_pool_indices.tolist(), seq_prefix_lens
            ):
                req_indices = self.req_to_token_pool.req_to_token[req_idx]
                req_prefix_block_tables = (
                    req_indices[:seq_len][:: self.page_size] // self.page_size
                )
                self.forward_metadata.flatten_prefix_block_tables = torch.cat(
                    (
                        self.forward_metadata.flatten_prefix_block_tables,
                        torch.flatten(req_prefix_block_tables),
                    )
                )

        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            self.forward_metadata.swa_out_cache_loc = (
                self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    forward_batch.out_cache_loc
                )
            )

        self.graph_mode = False

        self.prepare_a2a_attn_metadata(forward_batch)
        self.prepare_indexer_attn_tp_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        total_context_len = self.max_context_len + self.page_size - 1
        if self.speculative_num_draft_tokens is not None:
            total_context_len += self.speculative_num_draft_tokens
        self.graph_metadata = {
            "block_tables": torch.empty(
                (max_bs, total_context_len // self.page_size),
                dtype=torch.int32,
                device=self.device,
            ),
        }
        if self.use_mojo_mtp and self.speculative_num_draft_tokens is not None:
            gamma = self.speculative_num_draft_tokens
            self.graph_metadata["mojo_cu_q_lens"] = torch.arange(
                0, max_bs * gamma + 1, gamma,
                dtype=torch.int32, device=self.device,
            )
        if self.is_hybrid_swa:
            self.graph_metadata["block_tables_swa"] = torch.empty(
                (max_bs, total_context_len // self.page_size),
                dtype=torch.int32,
                device=self.device,
            )
            # SWA mask: True = masked out (don't attend), False = attend.
            # Pre-allocated at max size, sliced per batch size during capture,
            # content updated via copy_() during replay.
            self.graph_metadata["swa_mask"] = torch.ones(
                (max_bs, 1, total_context_len),
                dtype=torch.bool,
                device=self.device,
            )
            # Pre-allocated index buffer for mask generation during replay,
            # avoids torch.arange allocation on every replay step.
            self.graph_metadata["swa_indices"] = torch.arange(
                total_context_len, device=self.device, dtype=torch.int32
            )
        if self.use_sliding_window_kv_pool:
            # refilled in place at replay; the captured graph reads this storage
            self.cuda_graph_swa_out_cache_loc = torch.zeros(
                max_num_tokens,
                dtype=torch.int64,
                device=self.device,
            )
        # V4-specific extra graph buffers. Default no-op on the base class;
        # DeepseekV4AscendAttnBackend overrides.
        self._init_dsv4_graph_buffers(max_bs=max_bs, max_num_tokens=max_num_tokens)

    def _init_dsv4_graph_buffers(self, *, max_bs: int, max_num_tokens: int) -> None:
        """Hook for V4-Flash to preallocate dsv4-specific graph buffers.

        Default no-op. Overridden by DeepseekV4AscendAttnBackend.
        """
        pass

    # ------------------------------------------------------------------
    # A2A FIAS V2 BSND metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _a2a_fias_v2_sizes(bs: int, attn_tp_size: int, query_seq_len: int):
        """Return (T_padded, T_local, num_local_reqs, num_total_reqs_padded)
        for the given captured batch size."""
        T = bs * query_seq_len
        unit = attn_tp_size * query_seq_len
        T_padded = ((T + unit - 1) // unit) * unit
        T_local = T_padded // attn_tp_size
        num_local_reqs = T_local // query_seq_len
        num_total_reqs_padded = T_padded // query_seq_len
        return T_padded, T_local, num_local_reqs, num_total_reqs_padded

    def _compute_a2a_fias_v2_local_metadata(
        self,
        metadata: ForwardMetadata,
        seq_lens: torch.Tensor,
        seqused_q: torch.Tensor,
        block_table: torch.Tensor,
        bs: int,
    ):
        """Slice per-rank local metadata for the FIAS V2 A2A path and store
        into *metadata*.

        The ``metadata.a2a_*`` buffers are pre-allocated by
        :py:meth:`_init_cuda_graph_metadata`; this method fills them in-place
        via ``copy_`` so the captured graph sees stable tensor addresses.
        """
        attn_tp_size = get_parallel().attn_tp_size
        attn_tp_rank = get_parallel().attn_tp_rank
        query_seq_len = self.speculative_num_draft_tokens
        H_total = attn_tp_size * self.tp_q_head_num
        D_total = self.kv_lora_rank + self.qk_rope_head_dim

        T_padded, T_local, num_local_reqs, num_total_reqs_padded = (
            self._a2a_fias_v2_sizes(bs, attn_tp_size, query_seq_len)
        )
        req_start = attn_tp_rank * num_local_reqs

        # --- block_table_local ---
        if num_total_reqs_padded > block_table.shape[0]:
            pad_bt = torch.zeros(
                num_total_reqs_padded - block_table.shape[0],
                block_table.shape[1],
                dtype=block_table.dtype,
                device=block_table.device,
            )
            block_table_padded = torch.cat([block_table, pad_bt], dim=0)
        else:
            block_table_padded = block_table[:num_total_reqs_padded]
        block_table_local = block_table_padded[
            req_start : req_start + num_local_reqs
        ].contiguous()

        # --- cache_seqlens_local ---
        seq_lens_i32 = seq_lens.to(torch.int32)
        if num_total_reqs_padded > seq_lens_i32.shape[0]:
            pad_sl = torch.ones(
                num_total_reqs_padded - seq_lens_i32.shape[0],
                dtype=torch.int32,
                device=seq_lens_i32.device,
            )
            seq_lens_padded = torch.cat([seq_lens_i32, pad_sl], dim=0)
        else:
            seq_lens_padded = seq_lens_i32[:num_total_reqs_padded]
        cache_seqlens_local = seq_lens_padded[
            req_start : req_start + num_local_reqs
        ].contiguous()

        # --- seqused_q_local ---
        seqused_q_i32 = seqused_q.to(torch.int32)
        if num_total_reqs_padded > seqused_q_i32.shape[0]:
            pad_sq = torch.zeros(
                num_total_reqs_padded - seqused_q_i32.shape[0],
                dtype=torch.int32,
                device=seqused_q_i32.device,
            )
            seqused_q_padded = torch.cat([seqused_q_i32, pad_sq], dim=0)
        else:
            seqused_q_padded = seqused_q_i32[:num_total_reqs_padded]
        seqused_q_local = seqused_q_padded[
            req_start : req_start + num_local_reqs
        ].contiguous()

        # --- flash_mla metadata (computed once per step, reused per layer) ---
        metadata_flash_mla_local = flash_mla_with_kvcache_metadata(
            cache_seqlens=cache_seqlens_local,
            num_heads_q=H_total,
            num_heads_kv=1,
            cu_seqlens_q=None,
            seqused_q=seqused_q_local,
            max_seqlen_q=-1,
            max_seqlen_kv=-1,
            head_dim_qk=D_total,
            head_dim_v=self.kv_lora_rank,
            mask_mode=3,
            layout_q="BSND",
        )

        # Fill pre-allocated buffers in-place (graph-safe)
        metadata.a2a_block_table_local.copy_(block_table_local)
        metadata.a2a_cache_seqlens_local.copy_(cache_seqlens_local)
        metadata.a2a_seqused_q_local.copy_(seqused_q_local)
        metadata.a2a_metadata_flash_mla.copy_(metadata_flash_mla_local)
        metadata.a2a_num_local_reqs = num_local_reqs
        metadata.a2a_T_padded = T_padded
        metadata.a2a_T_local = T_local

    def _init_cuda_graph_metadata(
        self,
        bs: int,
        forward_mode: ForwardMode,
        seq_lens: torch.Tensor,
        out_cache_loc: Optional[torch.Tensor] = None,
    ) -> ForwardMetadata:
        """Create and store the per-bs ForwardMetadata for CUDA graph capture."""
        metadata = ForwardMetadata()
        metadata.block_tables = self.graph_metadata["block_tables"][:bs, :]
        if self.is_hybrid_swa:
            metadata.block_tables_swa = self.graph_metadata["block_tables_swa"][:bs, :]
            metadata.swa_mask = self.graph_metadata["swa_mask"][:bs, :, :]
        if self.use_sliding_window_kv_pool and out_cache_loc is not None:
            num_tokens = out_cache_loc.shape[0]
            metadata.swa_out_cache_loc = self.cuda_graph_swa_out_cache_loc[:num_tokens]
        metadata.seq_lens_cpu_list = seq_lens.cpu().int().tolist()
        metadata.seq_lens = seq_lens
        if forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2():
            metadata.actual_seq_lengths_q = torch.arange(
                self.speculative_num_draft_tokens,
                self.speculative_num_draft_tokens
                + bs * self.speculative_num_draft_tokens,
                self.speculative_num_draft_tokens,
                dtype=torch.int32,
                device=seq_lens.device,
            )
        else:
            metadata.actual_seq_lengths_q = torch.tensor(
                [1 + i for i in range(bs)],
                dtype=torch.int32,
                device=seq_lens.device,
            )
        if forward_mode.is_dllm_extend():
            extend_seq_lens_cpu_int = torch.tensor(
                [self.dllm_block_size for i in range(bs)],
                dtype=torch.int32,
                device=seq_lens.device,
            )
            metadata.seq_lens_list_cumsum = (
                torch.cumsum(extend_seq_lens_cpu_int, dim=0).int().tolist()
            )
        if not self.use_flash_mla:
            if (
                self.q_head_num_padding is not None
                and self.q_head_num_padding > self.tp_q_head_num
            ):
                dtype = self.model_dtype if self.model_dtype is not None else torch.bfloat16
                metadata.nope_padding = torch.empty(
                    [
                        bs,
                        1,
                        self.q_head_num_padding - self.tp_q_head_num,
                        self.kv_lora_rank,
                    ],
                    dtype=dtype,
                    device=seq_lens.device,
                )
                metadata.rope_padding = torch.empty(
                    [
                        bs,
                        1,
                        self.q_head_num_padding - self.tp_q_head_num,
                        self.qk_rope_head_dim,
                    ],
                    dtype=dtype,
                    device=seq_lens.device,
                )
        device = seq_lens.device
        if self.use_mla and self.use_flash_mla and not self.use_mla_fp8:
            def _calculate_metadata_size(batch_size, aic_core_num, aiv_core_num):
                """计算 metadata tensor 的对齐后大小。

                MLA 硬约束 kv head num == 1（B2/D16 上界公式，禁止按 num_heads_kv>1 放大）：
                最坏容量 = ((aic + aiv) * batch_size * 1 + 1) * 16 个 int32（每核 16 word，
                外加 1 个 header），按 4096 个 INT32 元素对齐（与 flash_attn.py 一致）。
                核数从硬件获取，见 _get_core_nums。
                """
                metadata_size = ((aic_core_num + aiv_core_num) * batch_size * 1 + 1) * 16
                return ((metadata_size + 4095) // 4096) * 4096

            def _get_core_nums():
                """从硬件获取 AIC/AIV 核数（与 aclnn host 侧 GetCurrentPlatformInfo 同源）。"""
                props = torch.npu.get_device_properties()
                return props.cube_core_num, props.vector_core_num

            b_size = seq_lens.size(0)
            aic_core_num, aiv_core_num = _get_core_nums()
            metadata_size = _calculate_metadata_size(b_size, aic_core_num, aiv_core_num)
            metadata_flash_mla = torch.empty((metadata_size,), dtype=torch.int32, device=device)
            metadata.metadata_flash_mla = metadata_flash_mla

            metadata.seqused_q=torch.zeros(bs, dtype=torch.int32, device=device)
            metadata.actual_seq_lengths_q = torch.cat([torch.zeros(1, dtype=torch.int32, device=device),metadata.actual_seq_lengths_q])

            # --- Pre-allocate A2A FIAS V2 local buffers (graph capture) ---
            if self.use_fias_v2_bsnd and self.use_sparse_attn_a2a:
                attn_tp_size = get_parallel().attn_tp_size
                query_seq_len = self.speculative_num_draft_tokens
                _, _, num_local_reqs, _ = self._a2a_fias_v2_sizes(
                    bs, attn_tp_size, query_seq_len
                )
                max_pages = metadata.block_tables.shape[1]
                metadata.a2a_block_table_local = torch.empty(
                    (num_local_reqs, max_pages), dtype=torch.int32, device=device,
                )
                metadata.a2a_cache_seqlens_local = torch.empty(
                    num_local_reqs, dtype=torch.int32, device=device,
                )
                metadata.a2a_seqused_q_local = torch.empty(
                    num_local_reqs, dtype=torch.int32, device=device,
                )
                metadata_size_local = _calculate_metadata_size(
                    num_local_reqs, aic_core_num, aiv_core_num
                )
                metadata.a2a_metadata_flash_mla = torch.empty(
                    (metadata_size_local,), dtype=torch.int32, device=device,
                )
        self.graph_metadata[bs] = metadata
        return metadata

    def _apply_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        out_cache_loc: Optional[torch.Tensor] = None,
    ):
        """Shared capture+replay body for the cuda-graph init path.

        Public entry: :py:meth:`init_forward_metadata_out_graph`.
        """
        metadata = self.graph_metadata[bs]

        # refill the captured SWA write-target buffer in place from the live loc
        if self.use_sliding_window_kv_pool and out_cache_loc is not None:
            n = out_cache_loc.shape[0]
            self.cuda_graph_swa_out_cache_loc[n:].zero_()
            self.cuda_graph_swa_out_cache_loc[:n].copy_(
                self.token_to_kv_pool.translate_loc_from_full_to_swa(out_cache_loc)
            )
        if self.needs_cpu_seq_lens:
            max_len = seq_lens_cpu[:bs].max().item()
            if forward_mode.is_target_verify() and not _is_dflash_verify(spec_info):
                max_len += self.speculative_num_draft_tokens
            elif forward_mode.is_decode_or_idle() and spec_info is not None:
                max_len += self.speculative_step_id + 1
            max_seq_pages = (max_len + self.page_size - 1) // self.page_size

            if self.is_hybrid_swa:
                full_page_locs = self.req_to_token[
                    req_pool_indices[:bs],
                    0 : max_len : self.page_size,
                ]
                swa_page_table = (
                    self.full_to_swa_index_mapping[full_page_locs] // self.page_size
                )

                metadata.block_tables_swa[:bs, :max_seq_pages].copy_(swa_page_table)
                metadata.block_tables_swa[:bs, max_seq_pages:].fill_(0)
                metadata.block_tables_swa[bs:, :].fill_(0)

                # Update SWA mask: True = masked out (don't attend), False = attend
                seq_lens_int = seq_lens[:bs].int()
                starts = torch.clamp(seq_lens_int - self.sliding_window_size, min=0)
                indices = self.graph_metadata["swa_indices"]
                start_exp = starts.unsqueeze(1)
                seq_exp = seq_lens_int.unsqueeze(1)
                mask = (indices.unsqueeze(0) < start_exp) | (
                    indices.unsqueeze(0) >= seq_exp
                )
                metadata.swa_mask[:bs, 0, :].copy_(mask)
                metadata.swa_mask[bs:, :, :].fill_(True)
            metadata.block_tables[:bs, :max_seq_pages].copy_(
                self.req_to_token[req_pool_indices[:bs], 0 : max_len : self.page_size]
                // self.page_size
            )

            metadata.block_tables[:bs, max_seq_pages:].fill_(0)
        else:
            total_pages = min(
                metadata.block_tables.shape[1],
                (self.req_to_token.shape[1] + self.page_size - 1) // self.page_size,
            )
            metadata.block_tables[:bs, :total_pages].copy_(
                self.req_to_token[
                    req_pool_indices[:bs],
                    0 : total_pages * self.page_size : self.page_size,
                ]
                // self.page_size
            )
            if total_pages < metadata.block_tables.shape[1]:
                metadata.block_tables[:bs, total_pages:].fill_(0)
        metadata.block_tables[bs:, :].fill_(0)

        if self.use_mla_fp8:
            # FIA v2 reads the stable page table above; NPUGraph.update replaces
            # its host KV lengths on replay. It does not consume FlashMLA's
            # BF16 metadata or the device-side speculative length adjustment.
            self.forward_metadata = metadata
            self.graph_mode = True
            return

        if self.use_mla and self.use_flash_mla:
            query_seq_len = (
                self.speculative_num_draft_tokens
                if forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2()
                else 1
            )
            # Identify zero-length padding before adding the draft-token offset.
            seqused_q = (seq_lens[:bs] > 0).to(torch.int32) * query_seq_len

        if forward_mode.is_target_verify():
            seq_lens = seq_lens + self.speculative_num_draft_tokens
        elif forward_mode.is_decode_or_idle() and spec_info is not None:
            seq_lens = seq_lens + self.speculative_step_offset_npu
        metadata.seq_lens[:bs].copy_(seq_lens[:bs])
        if self.use_mla and self.use_flash_mla:
            metadata.seqused_q.copy_(seqused_q.to(torch.int32))
            # Compute A2A FIAS V2 local metadata once per step (reused
            # across all layers by _forward_fias_v2_bsnd_tp_a2a).
            if (
                self.use_fias_v2_bsnd
                and self.use_sparse_attn_a2a
                and (
                    forward_mode.is_target_verify()
                    or forward_mode.is_draft_extend_v2()
                )
            ):
                self._compute_a2a_fias_v2_local_metadata(
                    metadata,
                    seq_lens=metadata.seq_lens[:bs],
                    seqused_q=metadata.seqused_q[:bs],
                    block_table=metadata.block_tables,
                    bs=bs,
                )
            else:
                metadata_flash_mla = flash_mla_with_kvcache_metadata(
                    cache_seqlens=seq_lens.to(torch.int32),
                    num_heads_q=self.tp_q_head_num,
                    num_heads_kv=1,
                    cu_seqlens_q=None,
                    seqused_q=seqused_q,
                    max_seqlen_q=-1,
                    max_seqlen_kv=-1,
                    head_dim_qk=576,
                    head_dim_v=512,
                    mask_mode=3,
                    layout_q="BSND",
                )
                metadata.metadata_flash_mla.copy_(metadata_flash_mla)
        self.forward_metadata = metadata
        self.graph_mode = True

    def _pad_topk_indices(
        self, topk_indices: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        current_tokens = topk_indices.shape[0]
        if current_tokens == num_tokens:
            return topk_indices

        assert current_tokens <= num_tokens, (
            f"topk_indices rows ({current_tokens}) > num_tokens ({num_tokens}); "
            "this indicates a mismatch between indexer output and q layout."
        )

        pad_size = num_tokens - current_tokens
        padding = torch.full(
            (pad_size, topk_indices.shape[1]),
            -1,
            dtype=topk_indices.dtype,
            device=topk_indices.device,
        )
        return torch.cat([topk_indices, padding], dim=0)

    def get_cuda_graph_seq_len_fill_value(self):
        return 0

    def _generate_alibi_bias(
        self,
        seq_len: int,
        slopes: torch.Tensor,
        num_heads: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        position_point = (
            torch.arange(seq_len).view(1, 1, -1).expand(num_heads, -1, -1).to(device)
        )
        alibi = slopes.view(-1, 1, 1) * position_point
        alibi_bias = alibi.view(num_heads, 1, seq_len).to(device).to(dtype)
        return alibi_bias

    def generate_alibi_bias(
        self,
        q_seq_len: int,
        kv_seq_len: int,
        slopes: torch.Tensor,
        num_heads: int,
        device: torch.device,
        is_extend: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        MAX_LEN_ALB = 5000
        max_seq_len = max(kv_seq_len, q_seq_len, MAX_LEN_ALB)
        if getattr(self, "alibi_bias", None) is None:
            self.alibi_bias = self._generate_alibi_bias(
                max_seq_len, slopes, num_heads, device, dtype
            )

        if getattr(self, "super_mask", None) is None:
            super_mask = torch.ones(size=(1, max_seq_len, max_seq_len), dtype=dtype)
            super_mask = super_mask.float().fill_(float("-inf")).type_as(super_mask)
            super_mask = torch.triu(super_mask, 1).to(device)
            self.super_mask = super_mask
        if is_extend:
            return (
                self.alibi_bias[:, :q_seq_len, :kv_seq_len]
                + self.super_mask[:, :q_seq_len, :kv_seq_len]
            )
        else:
            return self.alibi_bias[:, :q_seq_len, :kv_seq_len]

    def attn_alibi(
        self,
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        query_lens,
        scale_value,
        num_heads,
        slopes,
        is_extend,
    ):
        curr = 0
        num_prompts = query_lens.shape[0]
        head_size = k_cache.shape[3]
        head_size_v = v_cache.shape[3]
        block_size = k_cache.shape[1]
        attn_output = []
        for i in range(num_prompts):
            seq_len = seq_lens[i].item()
            block_table = block_tables[i]

            j = torch.arange(seq_len, device=block_table.device)

            block_number = block_table[j // block_size]
            block_offset = j % block_size

            k = k_cache[block_number, block_offset]
            v = v_cache[block_number, block_offset]
            k = k.view(seq_len, num_heads, head_size)
            v = v.view(seq_len, num_heads, head_size_v)

            if is_extend:
                q_len = query_lens[i].item()
                query = q[curr : curr + q_len]
            else:
                q_len = 1
                query = q[curr : curr + 1]

            query = query.to(torch.float32)
            query = query * scale_value
            query = query.permute(1, 0, 2)
            k = k.permute(1, 2, 0)

            score = torch.bmm(query, k)
            score = score.to(torch.float32)
            if slopes is not None:
                alibi_bias = self.generate_alibi_bias(
                    q_seq_len=q_len,
                    kv_seq_len=seq_len,
                    slopes=slopes,
                    num_heads=num_heads,
                    device=q.device,
                    is_extend=is_extend,
                    dtype=query.dtype,
                )
                score = score + alibi_bias
            score = torch.max(score, torch.tensor(torch.finfo(score.dtype).min))
            p = torch.nn.functional.softmax(score, dim=-1)
            v = v.permute(1, 0, 2)
            out = torch.bmm(p, v)
            out = out.permute(1, 0, 2)
            out = out.reshape(-1, num_heads * head_size_v)
            attn_output.append(out)
            curr += q_len
        attn_output = torch.cat(attn_output, dim=0).to(q.dtype).to(q.device)
        attn_output = attn_output.view(-1, num_heads * head_size)
        return attn_output

    def do_cp_balance_attn(
        self,
        q_nope,
        k_nope,
        q_pe,
        k_pe,
        topk_indices,
        layer,
        actual_seq_qlen,
        actual_seq_lengths_kv,
    ):
        seq_len = q_nope.shape[0]
        split_len = (seq_len + 1) // 2
        q_nope_prev, q_nope_next = torch.split(q_nope, split_len, dim=0)
        q_rope_prev, q_rope_next = torch.split(q_pe, split_len, dim=0)
        q_nope_prev = q_nope_prev.contiguous()
        q_nope_next = q_nope_next.contiguous()
        q_rope_prev = q_rope_prev.contiguous()
        q_rope_next = q_rope_next.contiguous()
        topk_indices = _expand_dsa_sparse_indices(topk_indices)
        topk_indices_prev, topk_indices_next = torch.split(
            topk_indices, split_len, dim=0
        )

        actual_seq_qlen_prev, actual_seq_qlen_next = actual_seq_qlen
        actual_seq_lengths_kv_prev, actual_seq_lengths_kv_next = actual_seq_lengths_kv

        if self.token_to_kv_pool.dsa_kv_cache_store_fp8:
            if q_nope.dtype != torch.bfloat16 or q_pe.dtype != torch.bfloat16:
                raise RuntimeError(
                    "Packed FP8 DSA sparse attention requires BF16 q_nope "
                    f"and q_rope, got {q_nope.dtype} and {q_pe.dtype}."
                )
            packed_cache_dim = get_dsa_fp8_packed_cache_dim(
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
            )
            if k_nope.shape[-1] != packed_cache_dim:
                raise RuntimeError(
                    f"Unexpected packed DSA KV width {k_nope.shape[-1]}, "
                    f"expected {packed_cache_dim}."
                )
            if k_nope.dtype == torch.uint8:
                k_nope = k_nope.view(torch.float8_e4m3fn)
            if k_nope.dtype != torch.float8_e4m3fn:
                raise RuntimeError(
                    f"Unexpected packed DSA KV dtype {k_nope.dtype}, "
                    f"expected {torch.float8_e4m3fn}."
                )

            orig_num_heads = q_nope_prev.shape[1]
            if (
                self.q_head_num_padding is not None
                and self.q_head_num_padding > orig_num_heads
            ):
                pad_size = self.q_head_num_padding - orig_num_heads
                q_nope_prev = torch.cat(
                    [
                        q_nope_prev,
                        torch.zeros(
                            q_nope_prev.shape[0], pad_size, q_nope_prev.shape[2],
                            dtype=q_nope_prev.dtype, device=q_nope_prev.device,
                        ),
                    ], dim=1,
                ).contiguous()
                q_nope_next = torch.cat(
                    [
                        q_nope_next,
                        torch.zeros(
                            q_nope_next.shape[0], pad_size, q_nope_next.shape[2],
                            dtype=q_nope_next.dtype, device=q_nope_next.device,
                        ),
                    ], dim=1,
                ).contiguous()
                q_rope_prev = torch.cat(
                    [
                        q_rope_prev,
                        torch.zeros(
                            q_rope_prev.shape[0], pad_size, q_rope_prev.shape[2],
                            dtype=q_rope_prev.dtype, device=q_rope_prev.device,
                        ),
                    ], dim=1,
                ).contiguous()
                q_rope_next = torch.cat(
                    [
                        q_rope_next,
                        torch.zeros(
                            q_rope_next.shape[0], pad_size, q_rope_next.shape[2],
                            dtype=q_rope_next.dtype, device=q_rope_next.device,
                        ),
                    ], dim=1,
                ).contiguous()

            attn_out_prev = torch_npu.npu_kv_quant_sparse_flash_attention(
                query=torch.cat((q_nope_prev, q_rope_prev), dim=-1).contiguous(),
                key=k_nope.view(-1, self.page_size, 1, packed_cache_dim),
                value=k_nope.view(-1, self.page_size, 1, packed_cache_dim),
                sparse_indices=topk_indices_prev,
                scale_value=layer.scaling,
                key_quant_mode=2,
                value_quant_mode=2,
                key_dequant_scale=None,
                value_dequant_scale=None,
                actual_seq_lengths_query=actual_seq_qlen_prev.to(
                    device=q_nope.device, dtype=torch.int32
                ),
                actual_seq_lengths_kv=actual_seq_lengths_kv_prev.to(
                    device=q_nope.device, dtype=torch.int32
                ),
                block_table=self.forward_metadata.block_tables,
                sparse_block_size=1,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                attention_mode=2,
                quant_scale_repo_mode=1,
                tile_size=DSA_KV_QUANT_TILE_SIZE,
                rope_head_dim=self.qk_rope_head_dim,
            )
            attn_out_next = torch_npu.npu_kv_quant_sparse_flash_attention(
                query=torch.cat((q_nope_next, q_rope_next), dim=-1).contiguous(),
                key=k_nope.view(-1, self.page_size, 1, packed_cache_dim),
                value=k_nope.view(-1, self.page_size, 1, packed_cache_dim),
                sparse_indices=topk_indices_next,
                scale_value=layer.scaling,
                key_quant_mode=2,
                value_quant_mode=2,
                key_dequant_scale=None,
                value_dequant_scale=None,
                actual_seq_lengths_query=actual_seq_qlen_next.to(
                    device=q_nope.device, dtype=torch.int32
                ),
                actual_seq_lengths_kv=actual_seq_lengths_kv_next.to(
                    device=q_nope.device, dtype=torch.int32
                ),
                block_table=self.forward_metadata.block_tables,
                sparse_block_size=1,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                attention_mode=2,
                quant_scale_repo_mode=1,
                tile_size=DSA_KV_QUANT_TILE_SIZE,
                rope_head_dim=self.qk_rope_head_dim,
            )

            if self.q_head_num_padding is not None and self.q_head_num_padding > orig_num_heads:
                attn_out_prev = attn_out_prev[:, :orig_num_heads, :]
                attn_out_next = attn_out_next[:, :orig_num_heads, :]
        else:
            attn_out_prev, _, _ = torch_npu.npu_sparse_flash_attention(
                query=q_nope_prev,
                key=k_nope,
                value=k_nope,
                query_rope=q_rope_prev,
                key_rope=k_pe,
                sparse_indices=topk_indices_prev,
                scale_value=layer.scaling,
                actual_seq_lengths_query=actual_seq_qlen_prev.to(
                    device=q_nope.device, dtype=torch.int32
                ),
                actual_seq_lengths_kv=actual_seq_lengths_kv_prev.to(
                    device=q_nope.device, dtype=torch.int32
                ),
                block_table=self.forward_metadata.block_tables,
                sparse_block_size=1,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                attention_mode=2,
                return_softmax_lse=False,
            )
            attn_out_next, _, _ = torch_npu.npu_sparse_flash_attention(
                query=q_nope_next,
                key=k_nope,
                value=k_nope,
                query_rope=q_rope_next,
                key_rope=k_pe,
                sparse_indices=topk_indices_next,
                scale_value=layer.scaling,
                actual_seq_lengths_query=actual_seq_qlen_next.to(
                    device=q_nope.device, dtype=torch.int32
                ),
                actual_seq_lengths_kv=actual_seq_lengths_kv_next.to(
                    device=q_nope.device, dtype=torch.int32
                ),
                block_table=self.forward_metadata.block_tables,
                sparse_block_size=1,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                attention_mode=2,
                return_softmax_lse=False,
            )
        return torch.cat([attn_out_prev, attn_out_next], dim=0)

    def _a2a_q(self, x: torch.Tensor, attn_tp_size: int) -> torch.Tensor:
        num_tokens = x.shape[0]
        local_heads = x.shape[1]
        head_dim = x.shape[2]
        tokens_per_rank = num_tokens // attn_tp_size
        x_flat = x.reshape(num_tokens, -1)
        output = torch.empty_like(x_flat)
        get_attn_tp_group().all_to_all_single(output, x_flat)
        return (
            output.view(attn_tp_size, tokens_per_rank, local_heads, head_dim)
            .transpose(0, 1)
            .contiguous()
            .view(tokens_per_rank, attn_tp_size * local_heads, head_dim)
        )

    def _a2a_attn_out(
        self, x: torch.Tensor, attn_tp_size: int, local_heads: int
    ) -> torch.Tensor:
        tokens_per_rank = x.shape[0]
        head_dim = x.shape[-1]
        num_tokens = tokens_per_rank * attn_tp_size
        x_flat = (
            x.view(tokens_per_rank, attn_tp_size, local_heads, head_dim)
            .transpose(0, 1)
            .contiguous()
            .view(num_tokens, -1)
        )
        output = torch.empty_like(x_flat)
        get_attn_tp_group().all_to_all_single(output, x_flat)
        return output.view(num_tokens, local_heads, head_dim)

    def attn_a2a_split_seq_metadata(
        self,
        actual_seq_qlen: torch.Tensor,
        actual_seq_lengths_kv: torch.Tensor,
        block_tables: Optional[torch.Tensor],
        num_tokens: int,
        attn_tp_rank: int,
        attn_tp_size: int,
    ):
        tokens_per_rank = num_tokens // attn_tp_size
        start_tok = attn_tp_rank * tokens_per_rank
        end_tok = start_tok + tokens_per_rank

        npu_device = block_tables.device if block_tables is not None else torch.device("npu")
        actual_seq_qlen = actual_seq_qlen.to(device=npu_device)
        actual_seq_lengths_kv = actual_seq_lengths_kv.to(device=npu_device)

        dtype = actual_seq_qlen.dtype
        seq_starts = torch.cat(
            [torch.zeros(1, dtype=dtype, device=npu_device), actual_seq_qlen[:-1]]
        )
        seq_ends = actual_seq_qlen

        overlap_start = seq_starts.clamp(min=start_tok)
        overlap_end = seq_ends.clamp(max=end_tok)
        new_qlens = (overlap_end - overlap_start).clamp(min=0)
        mask = new_qlens > 0

        rank_seq_qlen = torch.cumsum(new_qlens[mask], dim=0).to(torch.int32)
        causal_kvlen = (actual_seq_lengths_kv - seq_ends + overlap_end).to(torch.int32)
        rank_seq_kvlen = causal_kvlen[mask].contiguous()
        rank_block_tables = (
            block_tables[mask].contiguous() if block_tables is not None else None
        )

        return rank_seq_qlen, rank_seq_kvlen, rank_block_tables

    def prepare_a2a_attn_metadata(
        self,
        forward_batch: ForwardBatch,
    ):
        if not self.use_sparse_attn_a2a:
            return

        attn_tp_size = get_parallel().attn_tp_size

        is_prefill = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_target_verify()
        )
        if not is_prefill:
            return
        if not self.token_to_kv_pool.dsa_kv_cache_store_fp8:
            return

        attn_tp_rank = get_parallel().attn_tp_rank

        if self.forward_metadata.actual_seq_lengths_q is not None:
            actual_seq_qlen_cpu = self.forward_metadata.actual_seq_lengths_q.cpu().tolist()
        else:
            actual_seq_qlen_cpu = forward_batch.extend_seq_lens.cpu().cumsum(0).tolist()

        if self.forward_metadata.actual_seq_lengths_kv is not None:
            actual_seq_kvlen_cpu = self.forward_metadata.actual_seq_lengths_kv.cpu().tolist()
        elif self.forward_metadata.seq_lens_cpu_int is not None:
            actual_seq_kvlen_cpu = self.forward_metadata.seq_lens_cpu_int.tolist()
        else:
            actual_seq_kvlen_cpu = self.forward_metadata.seq_lens.cpu().tolist()

        block_tables = self.forward_metadata.block_tables

        num_tokens = actual_seq_qlen_cpu[-1] if actual_seq_qlen_cpu else 0
        remainder = num_tokens % attn_tp_size
        if remainder > 0:
            pad_size = attn_tp_size - remainder
        else:
            pad_size = 0
        num_tokens_padded = num_tokens + pad_size
        tokens_per_rank = num_tokens_padded // attn_tp_size

        start_tok = attn_tp_rank * tokens_per_rank
        end_tok = start_tok + tokens_per_rank

        seq_starts = [0] + actual_seq_qlen_cpu[:-1]
        seq_ends = actual_seq_qlen_cpu

        rank_qlens = []
        rank_kvlen_indices = []
        for i in range(len(seq_ends)):
            ov_s = max(seq_starts[i], start_tok)
            ov_e = min(seq_ends[i], end_tok)
            qlen = ov_e - ov_s
            if qlen > 0:
                rank_qlens.append(qlen)
                rank_kvlen_indices.append(i)

        real_tokens_in_rank = sum(rank_qlens)
        rank_pad = tokens_per_rank - real_tokens_in_rank
        if rank_pad > 0:
            if rank_qlens:
                rank_qlens[-1] += rank_pad
            else:
                rank_qlens.append(rank_pad)
                rank_kvlen_indices.append(len(seq_ends) - 1)

        rank_seq_qlen_cpu = []
        acc = 0
        for ql in rank_qlens:
            acc += ql
            rank_seq_qlen_cpu.append(acc)

        rank_seq_kvlen_cpu = [
            actual_seq_kvlen_cpu[i] - seq_ends[i] + min(seq_ends[i], end_tok)
            for i in rank_kvlen_indices
        ]

        self.forward_metadata.a2a_prefill_meta = {
            "num_tokens": num_tokens,
            "num_tokens_padded": num_tokens_padded,
            "pad_size": pad_size,
            "tokens_per_rank": tokens_per_rank,
            "start_tok": start_tok,
            "rank_seq_qlen": torch.tensor(
                rank_seq_qlen_cpu, dtype=torch.int32, device=self.device
            ),
            "rank_seq_kvlen": torch.tensor(
                rank_seq_kvlen_cpu, dtype=torch.int32, device=self.device
            ),
            "rank_block_tables": (
                block_tables[rank_kvlen_indices].contiguous()
                if block_tables is not None
                else None
            ),
        }

    def prepare_indexer_attn_tp_metadata(
        self,
        forward_batch: ForwardBatch,
    ):
        if not self.use_indexer_tp:
            return

        attn_tp_size = get_parallel().attn_tp_size
        fm = self.forward_metadata

        is_prefill = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_target_verify()
        )

        if not is_prefill:
            return

        seq_lens_cpu_list = list(forward_batch.extend_seq_lens_cpu)
        total_tokens = sum(seq_lens_cpu_list)
        actual_seq_qlen_cpu = []
        acc = 0
        for s in seq_lens_cpu_list:
            acc += s
            actual_seq_qlen_cpu.append(acc)
        num_requests = len(seq_lens_cpu_list)
        seq_lens_cpu_for_overlap = (
            forward_batch.seq_lens_cpu.tolist()
            if forward_batch.seq_lens_cpu is not None
            else seq_lens_cpu_list
        )

        if fm.seq_lens_cpu_int is not None:
            actual_seq_kvlen_cpu = fm.seq_lens_cpu_int.clamp(min=1).cpu().tolist()
        elif fm.seq_lens is not None:
            actual_seq_kvlen_cpu = fm.seq_lens.clamp(min=1).cpu().tolist()
        else:
            actual_seq_kvlen_cpu = forward_batch.seq_lens_cpu.int().clamp(min=1).tolist()

        block_tables = fm.block_tables
        if is_prefill and block_tables is not None:
            block_tables = block_tables[:num_requests]

        attn_tp_rank = get_parallel().attn_tp_rank
        tokens_per_rank = (total_tokens + attn_tp_size - 1) // attn_tp_size
        padded_total = tokens_per_rank * attn_tp_size
        pad_count = padded_total - total_tokens

        token_start = attn_tp_rank * tokens_per_rank
        token_end = token_start + tokens_per_rank

        local_q_lens = []
        local_kvlen_values = []
        local_bt_indices = []

        if seq_lens_cpu_list is not None and total_tokens % num_requests != 0:
            prev_cs = 0
            for req in range(num_requests):
                cur_cs = seq_lens_cpu_for_overlap[req]
                ov_s = max(prev_cs, token_start)
                ov_e = min(cur_cs, token_end)
                if ov_e > ov_s:
                    local_q_lens.append(ov_e - ov_s)
                    local_kvlen_values.append(actual_seq_kvlen_cpu[req])
                    local_bt_indices.append(req)
                prev_cs = cur_cs
        else:
            tokens_per_req = total_tokens // num_requests if num_requests > 0 else 1
            real_end = min(token_end, total_tokens)
            if real_end > token_start:
                req_lo = token_start // tokens_per_req
                req_hi = (real_end - 1) // tokens_per_req + 1
                for req in range(req_lo, req_hi):
                    r_start = req * tokens_per_req
                    r_end = r_start + tokens_per_req
                    q_count = min(r_end, real_end) - max(r_start, token_start)
                    if q_count > 0:
                        local_q_lens.append(q_count)
                        local_kvlen_values.append(actual_seq_kvlen_cpu[req])
                        local_bt_indices.append(req)

        pad_in_rank = max(0, token_end - max(token_start, total_tokens))
        if pad_in_rank > 0:
            if local_q_lens:
                local_q_lens[-1] += pad_in_rank
            else:
                local_q_lens.append(pad_in_rank)
                local_kvlen_values.append(actual_seq_kvlen_cpu[-1])
                local_bt_indices.append(num_requests - 1)

        local_q_cumsum = []
        acc = 0
        for ql in local_q_lens:
            acc += ql
            local_q_cumsum.append(acc)

        fm.indexer_attn_tp_meta = {
            "tokens_per_rank": tokens_per_rank,
            "padded_total": padded_total,
            "pad_count": pad_count,
            "token_start": token_start,
            "token_end": token_end,
            "actual_seq_lengths_q_local": torch.tensor(
                local_q_cumsum, dtype=torch.int32, device=self.device
            ),
            "actual_seq_lengths_kv_local": torch.tensor(
                local_kvlen_values, dtype=torch.int32, device=self.device
            ),
            "block_table_local": (
                block_tables[local_bt_indices].contiguous()
                if block_tables is not None
                else None
            ),
        }

    def do_cp_attn_fia(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """CP-aware attention for standard (non-MLA) models using FIA on Ascend NPU.

        Uses npu_fused_infer_attention_score with paged KV cache (block_table).
        The KV cache must already contain the full gathered sequence
        (written by _cp_allgather_and_save_kv_npu before this call).

        Args:
            q:            Query tensor, shape [total_q_tokens, tp_q_head_num * qk_head_dim]
            k_cache:      Full key cache from token_to_kv_pool
            v_cache:      Full value cache from token_to_kv_pool
            layer:        RadixAttention layer
            forward_batch: ForwardBatch with attn_cp_metadata populated

        Returns:
            attn_output [total_q_tokens, tp_q_head_num * v_head_dim]
        """
        cp_meta = forward_batch.attn_cp_metadata

        # Local tokens are laid out [all_seqs_prev, all_seqs_next]; split at
        # total_q_prev_tokens rather than the midpoint to support bs > 1.
        split = cp_meta.total_q_prev_tokens
        q_prev = (
            q[:split].contiguous().reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
        )
        q_next = (
            q[split:].contiguous().reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
        )

        k_cache_paged = k_cache.view(
            -1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim
        )
        v_cache_paged = v_cache.view(
            -1, self.page_size, layer.tp_v_head_num * layer.v_head_dim
        )

        attn_out_prev, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q_prev,
            k_cache_paged,
            v_cache_paged,
            block_table=self.forward_metadata.block_tables,
            block_size=self.page_size,
            num_heads=layer.tp_q_head_num,
            num_key_value_heads=layer.tp_k_head_num,
            input_layout="TND",
            atten_mask=self.fia_mask,
            sparse_mode=3,
            next_tokens=0,
            scale=layer.scaling,
            actual_seq_lengths=np.cumsum(cp_meta.actual_seq_q_prev_list).tolist(),
            actual_seq_lengths_kv=cp_meta.kv_len_prev_list,
        )

        attn_out_next, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q_next,
            k_cache_paged,
            v_cache_paged,
            block_table=self.forward_metadata.block_tables,
            block_size=self.page_size,
            num_heads=layer.tp_q_head_num,
            num_key_value_heads=layer.tp_k_head_num,
            input_layout="TND",
            atten_mask=self.fia_mask,
            sparse_mode=3,
            next_tokens=0,
            scale=layer.scaling,
            actual_seq_lengths=np.cumsum(cp_meta.actual_seq_q_next_list).tolist(),
            actual_seq_lengths_kv=cp_meta.kv_len_next_list,
        )

        attn_out = torch.cat([attn_out_prev, attn_out_next], dim=0)
        return attn_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _forward_sparse_attn_tp_a2a_prefill(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        topk_indices: torch.Tensor,
        layer: RadixAttention,
        actual_seq_qlen: torch.Tensor,
        actual_seq_lengths_kv: torch.Tensor,
    ) -> torch.Tensor:
        topk_indices = _expand_dsa_sparse_indices(topk_indices)

        if q_nope.dtype != torch.bfloat16 or q_pe.dtype != torch.bfloat16:
            raise RuntimeError(
                "Packed FP8 DSA sparse attention requires BF16 q_nope "
                f"and q_rope, got {q_nope.dtype} and {q_pe.dtype}."
            )
        packed_cache_dim = get_dsa_fp8_packed_cache_dim(
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
        )
        if k_nope.shape[-1] != packed_cache_dim:
            raise RuntimeError(
                f"Unexpected packed DSA KV width {k_nope.shape[-1]}, "
                f"expected {packed_cache_dim}."
            )
        if k_nope.dtype == torch.uint8:
            k_nope = k_nope.view(torch.float8_e4m3fn)
        if k_nope.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                f"Unexpected packed DSA KV dtype {k_nope.dtype}, "
                f"expected {torch.float8_e4m3fn}."
            )

        orig_num_heads = q_nope.shape[1]
        if (
            self.q_head_num_padding is not None
            and self.q_head_num_padding > orig_num_heads
        ):
            pad_size = self.q_head_num_padding - orig_num_heads
            q_nope = torch.cat(
                [
                    q_nope,
                    torch.zeros(
                        q_nope.shape[0],
                        pad_size,
                        q_nope.shape[2],
                        dtype=q_nope.dtype,
                        device=q_nope.device,
                    ),
                ],
                dim=1,
            ).contiguous()
            q_pe = torch.cat(
                [
                    q_pe,
                    torch.zeros(
                        q_pe.shape[0],
                        pad_size,
                        q_pe.shape[2],
                        dtype=q_pe.dtype,
                        device=q_pe.device,
                    ),
                ],
                dim=1,
            ).contiguous()
        q = torch.cat((q_nope, q_pe), dim=-1).contiguous()
        k = k_nope.view(-1, self.page_size, 1, packed_cache_dim)

        padded_local_heads = q.shape[1]
        num_tokens = q.shape[0]

        meta = self.forward_metadata.a2a_prefill_meta
        attn_tp_size = get_parallel().attn_tp_size

        if meta is not None and meta["num_tokens"] == num_tokens:
            pad_size = meta["pad_size"]
            tokens_per_rank = meta["tokens_per_rank"]
            start_tok = meta["start_tok"]
            rank_seq_qlen = meta["rank_seq_qlen"]
            rank_seq_kvlen = meta["rank_seq_kvlen"]
            rank_block_tables = meta["rank_block_tables"]
        else:
            attn_tp_rank = get_parallel().attn_tp_rank
            remainder = num_tokens % attn_tp_size
            if remainder > 0:
                pad_size = attn_tp_size - remainder
            else:
                pad_size = 0
            num_tokens_padded = num_tokens + pad_size
            tokens_per_rank = num_tokens_padded // attn_tp_size
            start_tok = attn_tp_rank * tokens_per_rank

            rank_seq_qlen, rank_seq_kvlen, rank_block_tables = (
                self.attn_a2a_split_seq_metadata(
                    actual_seq_qlen,
                    actual_seq_lengths_kv,
                    self.forward_metadata.block_tables,
                    num_tokens_padded,
                    attn_tp_rank,
                    attn_tp_size,
                )
            )

            if rank_seq_qlen.shape[0] > 0:
                real_tokens_in_rank = rank_seq_qlen[-1]
            else:
                real_tokens_in_rank = 0
            rank_pad = tokens_per_rank - real_tokens_in_rank
            if rank_pad > 0:
                if rank_seq_qlen.shape[0] > 0:
                    rank_seq_qlen[-1] = rank_seq_qlen[-1] + rank_pad
                else:
                    rank_seq_qlen = torch.tensor(
                        [tokens_per_rank], dtype=torch.int32, device=self.device
                    )
                    rank_seq_kvlen = actual_seq_lengths_kv[-1:].to(
                        device=self.device, dtype=torch.int32
                    ).contiguous()
                    rank_block_tables = (
                        self.forward_metadata.block_tables[-1:].contiguous()
                        if self.forward_metadata.block_tables is not None
                        else None
                    )

        if pad_size > 0:
            q = torch.cat(
                [
                    q,
                    q.new_zeros(pad_size, q.shape[1], q.shape[2]),
                ],
                dim=0,
            )
            topk_indices = torch.cat(
                [
                    topk_indices,
                    torch.full(
                        (pad_size,) + tuple(topk_indices.shape[1:]),
                        0,
                        dtype=topk_indices.dtype,
                        device=topk_indices.device,
                    ),
                ],
                dim=0,
            )

        q = self._a2a_q(q, attn_tp_size)

        topk_indices = topk_indices[start_tok : start_tok + tokens_per_rank]
        attn_out = torch_npu.npu_kv_quant_sparse_flash_attention(
            query=q,
            key=k,
            value=k,
            sparse_indices=topk_indices,
            scale_value=layer.scaling,
            key_quant_mode=2,
            value_quant_mode=2,
            key_dequant_scale=None,
            value_dequant_scale=None,
            actual_seq_lengths_query=rank_seq_qlen,
            actual_seq_lengths_kv=rank_seq_kvlen,
            block_table=rank_block_tables,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=0,
            attention_mode=2,
            quant_scale_repo_mode=1,
            tile_size=DSA_KV_QUANT_TILE_SIZE,
            rope_head_dim=self.qk_rope_head_dim,
        )

        attn_out = self._a2a_attn_out(attn_out, attn_tp_size, padded_local_heads)
        attn_out = attn_out[:num_tokens]

        if self.q_head_num_padding is not None and self.q_head_num_padding > orig_num_heads:
            attn_out = attn_out[:, :orig_num_heads, :]

        return attn_out

    def _forward_sparse_attn_tp_a2a_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        topk_indices: torch.Tensor,
        layer: RadixAttention,
        actual_seq_lengths_kv: torch.Tensor,
        tokens_per_req: int,
    ) -> torch.Tensor:
        """AllToAll-accelerated sparse attention for target_verify with attn_tp > 1.

        Redistributes query tokens across the attn_tp group so that each
        rank processes T/tp tokens with all q heads (instead of T tokens
        with H/tp heads), improving NPU sparse attention kernel utilization.

        Supports target_verify where each request contributes
        ``tokens_per_req`` (= speculative_num_draft_tokens) query tokens.
        Padding ensures T_padded is divisible by (attn_tp_size * tokens_per_req)
        so every rank receives a whole number of complete requests.

        Only for multi-query attention (kv head_num = 1); the KV cache is
        full on every rank and needs no communication.

        Graph-safe: no CPU synchronization, all ops are capturable.
        """
        attn_tp_size = get_parallel().attn_tp_size
        attn_tp_group = get_parallel().attn_tp_group
        attn_tp_rank = get_parallel().attn_tp_rank

        T = q_nope.shape[0]  # num_tokens = batch_size * tokens_per_req
        H_local = q_nope.shape[1]
        D_nope = q_nope.shape[2]
        D_rope = q_pe.shape[2]
        D_total = D_nope + D_rope
        K_sparse = topk_indices.shape[-1]

        if T == 0:
            return q_nope.new_zeros(0, H_local, D_nope)

        # --- Padded sizes ---
        # T_padded must be divisible by (attn_tp_size * tokens_per_req) so
        # that T_local = T_padded / attn_tp_size is a whole number of requests.
        unit = attn_tp_size * tokens_per_req
        T_padded = ((T + unit - 1) // unit) * unit
        T_local = T_padded // attn_tp_size
        num_local_reqs = T_local // tokens_per_req
        num_total_reqs_padded = T_padded // tokens_per_req

        # --- Head padding (power-of-2 for NPU kernel) ---
        H_local_padded = (
            self.q_head_num_padding
            if (
                self.q_head_num_padding is not None
                and self.q_head_num_padding > H_local
            )
            else H_local
        )
        if H_local_padded > H_local:
            pad_h = H_local_padded - H_local
            q_nope = torch.cat(
                [q_nope, q_nope.new_zeros(T, pad_h, D_nope)], dim=1
            )
            q_pe = torch.cat(
                [q_pe, q_pe.new_zeros(T, pad_h, D_rope)], dim=1
            )

        # --- Pad tokens to T_padded ---
        if T_padded > T:
            pad_t = T_padded - T
            q_nope = torch.cat(
                [q_nope, q_nope.new_zeros(pad_t, H_local_padded, D_nope)],
                dim=0,
            )
            q_pe = torch.cat(
                [q_pe, q_pe.new_zeros(pad_t, H_local_padded, D_rope)],
                dim=0,
            )
            if topk_indices.dim() == 3:
                topk_indices = topk_indices.squeeze(-2)
            topk_indices = torch.cat(
                [
                    topk_indices,
                    torch.full(
                        (pad_t, K_sparse),
                        0,
                        dtype=topk_indices.dtype,
                        device=topk_indices.device,
                    ),
                ],
                dim=0,
            )

        # --- Concatenate q_nope + q_pe ---
        q_concat = torch.cat((q_nope, q_pe), dim=-1).contiguous()
        # [T_padded, H_local_padded, D_total]

        # ============================================================
        # Forward AllToAll: [T_padded, H_local, D] -> [T_local, H_total, D]
        # ============================================================
        # Direct flatten: all_to_all_single splits into tp equal chunks.
        # Chunk j = q_concat[j*T_local : (j+1)*T_local].flatten()
        #   => rank j receives tokens [j*T_local : (j+1)*T_local] (contiguous)
        q_send = q_concat.view(-1)
        q_recv = torch.empty_like(q_send)
        attn_tp_group.all_to_all_single(q_recv, q_send)

        # Reshape: [tp, T_local, H_local, D] -> [T_local, tp*H_local, D]
        q_local = (
            q_recv.view(attn_tp_size, T_local, H_local_padded, D_total)
            .transpose(0, 1)
            .contiguous()
            .view(T_local, attn_tp_size * H_local_padded, D_total)
        )

        # ============================================================
        # Adapt metadata for local token/request subset (contiguous slice)
        # ============================================================
        token_start = attn_tp_rank * T_local
        req_start = attn_tp_rank * num_local_reqs

        # sparse_indices: per-token, slice by token offset -> [T_local, 1, K]
        topk_indices_local = topk_indices[token_start : token_start + T_local]
        topk_indices_local = topk_indices_local.unsqueeze(-2)

        # actual_seq_lengths_query: cumulative [s, 2s, ..., num_local_reqs*s]
        actual_seq_qlen_local = torch.arange(
            tokens_per_req,
            tokens_per_req * (num_local_reqs + 1),
            tokens_per_req,
            dtype=torch.int32,
            device=q_nope.device,
        )

        # actual_seq_lengths_kv: per-request, slice by request offset
        actual_seq_lengths_kv_dev = actual_seq_lengths_kv.to(
            device=q_nope.device, dtype=torch.int32
        )
        if num_total_reqs_padded > actual_seq_lengths_kv_dev.shape[0]:
            pad_kv = torch.ones(
                num_total_reqs_padded - actual_seq_lengths_kv_dev.shape[0],
                dtype=torch.int32,
                device=q_nope.device,
            )
            actual_seq_lengths_kv_padded = torch.cat(
                [actual_seq_lengths_kv_dev, pad_kv], dim=0
            )
        else:
            actual_seq_lengths_kv_padded = actual_seq_lengths_kv_dev[
                :num_total_reqs_padded
            ]
        actual_seq_lengths_kv_local = actual_seq_lengths_kv_padded[
            req_start : req_start + num_local_reqs
        ]

        # block_table: per-request, slice by request offset
        block_table = self.forward_metadata.block_tables
        if num_total_reqs_padded > block_table.shape[0]:
            pad_bt = torch.zeros(
                num_total_reqs_padded - block_table.shape[0],
                block_table.shape[1],
                dtype=block_table.dtype,
                device=block_table.device,
            )
            block_table_padded = torch.cat([block_table, pad_bt], dim=0)
        else:
            block_table_padded = block_table[:num_total_reqs_padded]
        block_table_local = block_table_padded[
            req_start : req_start + num_local_reqs
        ]

        # --- KV cache dtype handling ---
        packed_cache_dim = get_dsa_fp8_packed_cache_dim(
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
        )
        if k_nope.dtype == torch.uint8:
            k_nope = k_nope.view(torch.float8_e4m3fn)
        k = k_nope.view(-1, self.page_size, 1, packed_cache_dim)
        # ============================================================
        # Kernel call: full heads on T_local tokens
        # ============================================================
        attn_out_local = torch_npu.npu_kv_quant_sparse_flash_attention(
            query=q_local.contiguous(),
            key=k,
            value=k,
            sparse_indices=topk_indices_local.contiguous(),
            scale_value=layer.scaling,
            key_quant_mode=2,
            value_quant_mode=2,
            key_dequant_scale=None,
            value_dequant_scale=None,
            actual_seq_lengths_query=actual_seq_qlen_local,
            actual_seq_lengths_kv=actual_seq_lengths_kv_local,
            block_table=block_table_local,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            attention_mode=2,
            quant_scale_repo_mode=1,
            tile_size=DSA_KV_QUANT_TILE_SIZE,
            rope_head_dim=self.qk_rope_head_dim,
        )
        # [T_local, tp * H_local_padded, D_out]

        D_out = attn_out_local.shape[-1]

        # ============================================================
        # Reverse AllToAll: [T_local, H_total, D] -> [T_padded, H_local, D]
        # ============================================================
        # Group heads by destination rank, then all_to_all.
        attn_out_send = (
            attn_out_local.view(
                T_local, attn_tp_size, H_local_padded, D_out
            )
            .transpose(0, 1)  # [tp, T_local, H_local, D_out]
            .contiguous()
            .view(-1)
        )

        attn_out_recv = torch.empty_like(attn_out_send)
        attn_tp_group.all_to_all_single(attn_out_recv, attn_out_send)

        # Direct view to [T_padded, H_local, D_out]
        attn_out_full = attn_out_recv.view(
            T_padded, H_local_padded, D_out
        )

        # --- Unpad tokens and heads ---
        attn_out = attn_out_full[:T]
        if H_local_padded > H_local:
            attn_out = attn_out[:, :H_local, :]

        return attn_out

    def _forward_fias_v2_bsnd_tp_a2a(
        self,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        kv_cache: torch.Tensor,
        layer: RadixAttention,
    ) -> torch.Tensor:
        """AllToAll-accelerated FIAS V2 BSND MLA attention for target_verify
        with attn_tp > 1.

        Redistributes query tokens across the attn_tp group so that each
        rank processes T/tp tokens with all q heads (instead of T tokens
        with H/tp heads), improving NPU flash_mla kernel utilization.

        Per-step metadata (block_table slice, cache_seqlens, seqused_q,
        flash_mla metadata) is pre-computed once in
        :py:meth:`_compute_a2a_fias_v2_local_metadata` and stored in
        ``self.forward_metadata.a2a_*``; this method only does the
        per-layer AllToAll + kernel + reverse-AllToAll.

        Only for multi-query attention (kv head_num = 1); the KV cache is
        full on every rank and needs no communication.
        """
        attn_tp_size = get_parallel().attn_tp_size
        attn_tp_group = get_parallel().attn_tp_group

        T = q_nope.shape[0]  # num_tokens = batch_size * query_seq_len
        H_local = q_nope.shape[1]
        D_nope = q_nope.shape[2]
        D_rope = q_rope.shape[2]
        D_total = D_nope + D_rope

        if T == 0:
            return q_nope.new_zeros(0, H_local, D_nope)

        # --- Read pre-computed per-step metadata ---
        md = self.forward_metadata
        T_padded = md.a2a_T_padded
        T_local = md.a2a_T_local
        num_local_reqs = md.a2a_num_local_reqs
        query_seq_len = self.speculative_num_draft_tokens
        H_total = attn_tp_size * H_local

        # --- Pad tokens to T_padded ---
        if T_padded > T:
            pad_t = T_padded - T
            q_nope = torch.cat(
                [q_nope, q_nope.new_zeros(pad_t, H_local, D_nope)], dim=0
            )
            q_rope = torch.cat(
                [q_rope, q_rope.new_zeros(pad_t, H_local, D_rope)], dim=0
            )

        # --- Concat q_nope + q_pe -> [T_padded, H_local, D_total] ---
        q_concat = torch.cat((q_nope, q_rope), dim=-1).contiguous()

        # ============================================================
        # Forward AllToAll: [T_padded, H_local, D] -> [T_local, H_total, D]
        # ============================================================
        q_send = q_concat.view(-1)
        q_recv = torch.empty_like(q_send)
        attn_tp_group.all_to_all_single(q_recv, q_send)

        q_local = (
            q_recv.view(attn_tp_size, T_local, H_local, D_total)
            .transpose(0, 1)
            .contiguous()
            .view(T_local, H_total, D_total)
        )

        # ============================================================
        # Reshape query to BSND: [B_local, query_seq_len, H_total, D_total]
        # ============================================================
        q_local_bsnd = q_local.view(
            num_local_reqs, query_seq_len, H_total, D_total
        ).contiguous()

        # ============================================================
        # Kernel call: flash_mla_with_kvcache with pre-computed local
        # metadata (block_table, cache_seqlens, seqused_q, flash_mla
        # metadata — all computed once per step)
        # ============================================================
        attn_out_local, _ = flash_mla_with_kvcache(
            q_local_bsnd,
            kv_cache,
            block_table=md.a2a_block_table_local,
            cache_seqlens=md.a2a_cache_seqlens_local,
            cu_seqlens_q=None,
            seqused_q=md.a2a_seqused_q_local,
            attn_mask=self.mtp_mask,
            metadata=md.a2a_metadata_flash_mla,
            head_dim_v=self.kv_lora_rank,
            softmax_scale=layer.scaling,
            mask_mode=3,
            max_seqlen_q=-1,
            max_seqlen_kv=-1,
            layout_q="BSND",
            layout_kv="PA_BBND",
            layout_out="BSND",
            return_softmax_lse=False,
        )
        # [num_local_reqs, query_seq_len, H_total, kv_lora_rank]

        attn_out_local = attn_out_local.reshape(T_local, H_total, D_nope)

        # ============================================================
        # Reverse AllToAll: [T_local, H_total, D] -> [T_padded, H_local, D]
        # ============================================================
        D_out = attn_out_local.shape[-1]
        attn_out_send = (
            attn_out_local.view(T_local, attn_tp_size, H_local, D_out)
            .transpose(0, 1)  # [tp, T_local, H_local, D_out]
            .contiguous()
            .view(-1)
        )
        attn_out_recv = torch.empty_like(attn_out_send)
        attn_tp_group.all_to_all_single(attn_out_recv, attn_out_send)
        attn_out_full = attn_out_recv.view(T_padded, H_local, D_out)

        # --- Unpad tokens ---
        return attn_out_full[:T]

    def forward_sparse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi_head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: torch.Tensor = None,
    ):

        is_prefill = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_target_verify()
        )

        if save_kv_cache:
            k = k.view(-1, layer.tp_k_head_num, self.kv_lora_rank)
            k_rope = k_rope.view(-1, layer.tp_k_head_num, self.qk_rope_head_dim)
            self.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, k_rope
            )
        q_nope, q_pe = q, q_rope
        if self.use_flash_mla:
            kv_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            expected_cache_dim = get_dsa_fp8_packed_cache_dim(
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
            )
            k_nope = kv_cache[..., : expected_cache_dim]
            k_pe = kv_cache[..., expected_cache_dim:]
        else:
            k_nope, k_pe = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)

        if is_prefill:
            if self.forward_metadata.actual_seq_lengths_q is not None:
                actual_seq_qlen = self.forward_metadata.actual_seq_lengths_q
            else:
                actual_seq_qlen = torch.cumsum(forward_batch.extend_seq_lens, dim=0)
        else:
            if self.forward_metadata.actual_seq_lengths_q is None:
                if (
                    forward_batch.forward_mode.is_draft_extend_v2()
                    or forward_batch.forward_mode.is_target_verify()
                ):
                    actual_seq_qlen = (
                        torch.arange(
                            self.speculative_num_draft_tokens,
                            self.speculative_num_draft_tokens + q.shape[0],
                            self.speculative_num_draft_tokens,
                            dtype=torch.int32,
                        )
                        .to(q.device)
                        .to(torch.int32)
                    )
                else:
                    actual_seq_qlen = (
                        torch.arange(1, q.shape[0] + 1).to(q.device).to(torch.int32)
                    )
            else:
                actual_seq_qlen = self.forward_metadata.actual_seq_lengths_q

        if self.forward_metadata.actual_seq_lengths_kv is not None:
            actual_seq_lengths_kv = self.forward_metadata.actual_seq_lengths_kv
        elif self.forward_metadata.seq_lens_cpu_int is not None:
            actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_int
        else:
            actual_seq_lengths_kv = self.forward_metadata.seq_lens

        if (
            is_prefill
            and is_dsa_enable_prefill_cp()
            and forward_batch.attn_cp_metadata is not None
        ):
            attn_out = self.do_cp_balance_attn(
                q_nope,
                k_nope,
                q_pe,
                k_pe,
                topk_indices,
                layer,
                actual_seq_qlen,
                actual_seq_lengths_kv,
            )
        else:
            if topk_indices is not None:
                topk_indices = self._pad_topk_indices(topk_indices, q_nope.shape[0])

            # --- AllToAll optimization for prefill and target_verify with attn_tp > 1 ---
            if (
                self.use_sparse_attn_a2a
                and not is_prefill
                and forward_batch.forward_mode.is_target_verify()
                and self.token_to_kv_pool.dsa_kv_cache_store_fp8
                and topk_indices is not None
            ):
                tokens_per_req = (
                    self.speculative_num_draft_tokens
                    if self.speculative_num_draft_tokens is not None
                    else 1
                )
                return self._forward_sparse_attn_tp_a2a_decode(
                    q_nope,
                    q_pe,
                    k_nope,
                    topk_indices,
                    layer,
                    actual_seq_lengths_kv,
                    tokens_per_req,
                )
            elif (
                self.use_sparse_attn_a2a
                and is_prefill
                and self.token_to_kv_pool.dsa_kv_cache_store_fp8
                and topk_indices is not None
            ):
                return self._forward_sparse_attn_tp_a2a_prefill(
                    q_nope,
                    q_pe,
                    k_nope,
                    topk_indices,
                    layer,
                    actual_seq_qlen,
                    actual_seq_lengths_kv,
                )

            topk_indices = _expand_dsa_sparse_indices(topk_indices)
            if self.token_to_kv_pool.dsa_kv_cache_store_fp8:
                if q_nope.dtype != torch.bfloat16 or q_pe.dtype != torch.bfloat16:
                    raise RuntimeError(
                        "Packed FP8 DSA sparse attention requires BF16 q_nope "
                        f"and q_rope, got {q_nope.dtype} and {q_pe.dtype}."
                    )
                packed_cache_dim = get_dsa_fp8_packed_cache_dim(
                    kv_lora_rank=self.kv_lora_rank,
                    qk_rope_head_dim=self.qk_rope_head_dim,
                )
                if k_nope.shape[-1] != packed_cache_dim:
                    raise RuntimeError(
                        f"Unexpected packed DSA KV width {k_nope.shape[-1]}, "
                        f"expected {packed_cache_dim}."
                    )
                if k_nope.dtype == torch.uint8:
                    k_nope = k_nope.view(torch.float8_e4m3fn)
                if k_nope.dtype != torch.float8_e4m3fn:
                    raise RuntimeError(
                        f"Unexpected packed DSA KV dtype {k_nope.dtype}, "
                        f"expected {torch.float8_e4m3fn}."
                    )

                orig_num_heads = q_nope.shape[1]
                if (
                    self.q_head_num_padding is not None
                    and self.q_head_num_padding > orig_num_heads
                ):
                    pad_size = self.q_head_num_padding - orig_num_heads
                    q_nope = torch.cat(
                        [
                            q_nope,
                            torch.zeros(
                                q_nope.shape[0],
                                pad_size,
                                q_nope.shape[2],
                                dtype=q_nope.dtype,
                                device=q_nope.device,
                            ),
                        ],
                        dim=1,
                    ).contiguous()
                    q_pe = torch.cat(
                        [
                            q_pe,
                            torch.zeros(
                                q_pe.shape[0],
                                pad_size,
                                q_pe.shape[2],
                                dtype=q_pe.dtype,
                                device=q_pe.device,
                            ),
                        ],
                        dim=1,
                    ).contiguous()
                attn_out = torch_npu.npu_kv_quant_sparse_flash_attention(
                    query=torch.cat((q_nope, q_pe), dim=-1).contiguous(),
                    key=k_nope.view(
                        -1,
                        self.page_size,
                        1,
                        packed_cache_dim,
                    ),
                    value=k_nope.view(
                        -1,
                        self.page_size,
                        1,
                        packed_cache_dim,
                    ),
                    sparse_indices=topk_indices,
                    scale_value=layer.scaling,
                    key_quant_mode=2,
                    value_quant_mode=2,
                    key_dequant_scale=None,
                    value_dequant_scale=None,
                    actual_seq_lengths_query=actual_seq_qlen.to(
                        device=q_nope.device,
                        dtype=torch.int32,
                    ),
                    actual_seq_lengths_kv=actual_seq_lengths_kv.to(
                        device=q_nope.device,
                        dtype=torch.int32,
                    ),
                    block_table=self.forward_metadata.block_tables,
                    sparse_block_size=1,
                    layout_query="TND",
                    layout_kv="PA_BSND",
                    sparse_mode=0,
                    attention_mode=2,
                    quant_scale_repo_mode=1,
                    tile_size=DSA_KV_QUANT_TILE_SIZE,
                    rope_head_dim=self.qk_rope_head_dim,
                )

                if self.q_head_num_padding is not None and self.q_head_num_padding > orig_num_heads:
                    attn_out = attn_out[:, :orig_num_heads, :]
            else:
                attn_out, _, _ = torch_npu.npu_sparse_flash_attention(
                    query=q_nope,
                    key=k_nope,
                    value=k_nope,
                    query_rope=q_pe,
                    key_rope=k_pe,
                    sparse_indices=topk_indices,
                    scale_value=layer.scaling,
                    actual_seq_lengths_query=actual_seq_qlen.to(
                        device=q_nope.device, dtype=torch.int32
                    ),
                    actual_seq_lengths_kv=actual_seq_lengths_kv.to(
                        device=q_nope.device, dtype=torch.int32
                    ),
                    block_table=self.forward_metadata.block_tables,
                    sparse_block_size=1,
                    layout_query="TND",
                    layout_kv="PA_BSND",
                    sparse_mode=3,
                    attention_mode=2,
                    return_softmax_lse=False,
                )

        return attn_out

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi_head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        slopes: Optional[torch.Tensor] = None,
        dequant_scale_q_nope: Optional[torch.Tensor] = None,
        fp8_kv_scale: Optional[torch.Tensor] = None,
    ):
        if is_mla_preprocess_enabled() and self.use_mla:
            # MLAPO and MLAPROLOG do save kv_cache
            save_kv_cache = False
        if self.is_dllm_model:
            return self.forward_dllm(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope=q_rope,
                k_rope=k_rope,
            )
        if topk_indices is not None:
            return self.forward_sparse(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope,
                k_rope,
                topk_indices,
            )
        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            return self.forward_mtp(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope=q_rope,
                k_rope=k_rope,
                sinks=sinks,
                dequant_scale_q_nope=dequant_scale_q_nope,
                fp8_kv_scale=fp8_kv_scale,
            )

        if not self.use_mla:
            # Detect CP mode for prefill (context parallel)
            is_cp_mode = (
                forward_batch.forward_mode.is_context_parallel_extend()
                and forward_batch.attn_cp_metadata is not None
                and self.attn_cp_size > 1
            )

            # In cross attention layer, when there is no vision input,the values of k and v is None
            if save_kv_cache and k is not None and v is not None:
                if is_cp_mode:
                    # All-gather K/V from all CP ranks and write full sequence to KV pool
                    _cp_allgather_and_save_kv_npu(
                        forward_batch,
                        layer,
                        k,
                        v,
                        self.attn_cp_size,
                        self.token_to_kv_pool,
                        swa_loc=self.forward_metadata.swa_out_cache_loc,
                    )
                else:
                    # support cross attention
                    cache_loc = (
                        forward_batch.out_cache_loc
                        if not layer.is_cross_attention
                        else forward_batch.encoder_out_cache_loc
                    )
                    swa_loc = (
                        self.forward_metadata.swa_out_cache_loc
                        if not layer.is_cross_attention
                        else None
                    )
                    self.token_to_kv_pool.set_kv_buffer(
                        layer, KVWriteLoc(cache_loc, swa_loc), k, v
                    )

            k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id)

            if sinks is not None or (self._is_swa_layer(layer) and self.use_fia):
                # Use SWA block tables if hybrid SWA is enabled for this layer
                if self._is_swa_layer(layer):
                    block_tables = self.forward_metadata.block_tables_swa
                else:
                    block_tables = self.forward_metadata.block_tables
                if self.use_fia:
                    if self._can_use_tnd(layer):
                        num_token_padding = q.shape[0]
                        if num_token_padding > forward_batch.num_token_non_padded_cpu:
                            q, k, v = [
                                data[: forward_batch.num_token_non_padded_cpu]
                                for data in [q, k, v]
                            ]
                        q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
                        block_size = self.page_size
                        attn_out, _ = torch_npu.npu_fused_infer_attention_score_v2(
                            query=q,
                            key=k_cache.view(
                                -1,
                                self.page_size,
                                layer.tp_k_head_num * layer.qk_head_dim,
                            ),
                            value=v_cache.view(
                                -1,
                                self.page_size,
                                layer.tp_v_head_num * layer.v_head_dim,
                            ),
                            pre_tokens=(
                                layer.sliding_window_size
                                if layer.sliding_window_size != -1
                                else FULL_ATTENTION_WINDOW
                            ),
                            next_tokens=(
                                0
                                if layer.sliding_window_size != -1
                                else FULL_ATTENTION_WINDOW
                            ),
                            atten_mask=self.fia_mask,
                            block_table=block_tables,
                            input_layout="TND",
                            block_size=block_size,
                            num_query_heads=layer.tp_q_head_num,
                            num_key_value_heads=layer.tp_k_head_num,
                            actual_seq_qlen=self.forward_metadata.seq_lens_list_cumsum,
                            actual_seq_kvlen=self.forward_metadata.seq_lens_cpu_int,
                            softmax_scale=layer.scaling,
                            sparse_mode=4 if layer.sliding_window_size != -1 else 3,
                            learnable_sink=sinks,
                        )
                        attn_out = attn_out.view(
                            -1, layer.tp_q_head_num * layer.v_head_dim
                        )
                        if num_token_padding != forward_batch.num_token_non_padded_cpu:
                            attn_out = torch.cat(
                                [
                                    attn_out,
                                    attn_out.new_zeros(
                                        num_token_padding - attn_out.shape[0],
                                        *attn_out.shape[1:],
                                    ),
                                ],
                                dim=0,
                            )
                    else:
                        q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)

                        # FIA BSND with paged KV cache (reads prefix tokens from cache)
                        seq_lens_cpu = forward_batch.seq_lens.cpu().tolist()
                        attn_out = torch.empty(
                            (q.shape[0], layer.tp_q_head_num, layer.v_head_dim),
                            device=q.device,
                            dtype=q.dtype,
                        )
                        q_len_offset = 0
                        for seq_idx, q_len in enumerate(
                            forward_batch.extend_seq_lens_cpu
                        ):
                            if q_len == 0:
                                continue
                            total_kv_len = seq_lens_cpu[seq_idx]
                            result, _ = torch_npu.npu_fused_infer_attention_score_v2(
                                query=q[None, q_len_offset : q_len_offset + q_len],
                                key=k_cache.view(
                                    -1,
                                    self.page_size,
                                    layer.tp_k_head_num * layer.qk_head_dim,
                                ),
                                value=v_cache.view(
                                    -1,
                                    self.page_size,
                                    layer.tp_v_head_num * layer.v_head_dim,
                                ),
                                num_query_heads=layer.tp_q_head_num,
                                num_key_value_heads=layer.tp_k_head_num,
                                input_layout="BSND",
                                block_table=block_tables[seq_idx : seq_idx + 1],
                                block_size=self.page_size,
                                actual_seq_qlen=[q_len],
                                actual_seq_kvlen=[total_kv_len],
                                atten_mask=self.fia_mask.unsqueeze(0),
                                sparse_mode=4,
                                softmax_scale=layer.scaling,
                                pre_tokens=layer.sliding_window_size,
                                next_tokens=0,
                            )
                            attn_out[q_len_offset : q_len_offset + q_len] = result[0]
                            q_len_offset += q_len

                        attn_out = attn_out.view(
                            -1, layer.tp_q_head_num * layer.v_head_dim
                        )

                else:
                    attn_out = attention_sinks_prefill_triton(
                        q,
                        k_cache,
                        v_cache,
                        sinks,
                        self.forward_metadata.extend_seq_lens,
                        block_tables,
                        self.forward_metadata.seq_lens,
                        layer.scaling,
                        layer.sliding_window_size,
                        layer.tp_q_head_num,
                        layer.tp_k_head_num,
                    )
                return attn_out

            if is_cp_mode:
                if self.use_fia:
                    attn_output = self.do_cp_attn_fia(
                        q, k_cache, v_cache, layer, forward_batch
                    )
                else:
                    raise NotImplementedError(
                        "CP attention for non-FIA path on Ascend is not yet implemented. "
                        "Set ASCEND_USE_FIA=1 to use FIA-based CP attention."
                    )
                return attn_output

            if self.use_fia:
                if self._can_use_tnd(layer):
                    """FIA supports multi-bs in the current version of CANN"""
                    q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
                    num_token_padding = q.shape[0]
                    if num_token_padding > forward_batch.num_token_non_padded_cpu:
                        q, k, v = [
                            data[: forward_batch.num_token_non_padded_cpu]
                            for data in [q, k, v]
                        ]
                    attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                        query=q,
                        key=k_cache.view(
                            -1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim
                        ),
                        value=v_cache.view(
                            -1, self.page_size, layer.tp_v_head_num * layer.v_head_dim
                        ),
                        block_table=self.forward_metadata.block_tables,
                        block_size=self.page_size,
                        atten_mask=self.fia_mask,
                        input_layout="TND",
                        actual_seq_lengths=self.forward_metadata.seq_lens_list_cumsum,
                        actual_seq_lengths_kv=self.forward_metadata.seq_lens_cpu_int,
                        num_key_value_heads=layer.tp_k_head_num,
                        num_heads=layer.tp_q_head_num,
                        scale=layer.scaling,
                        sparse_mode=3,
                    )
                    attn_output = attn_output.view(
                        -1, layer.tp_q_head_num * layer.v_head_dim
                    )

                    if num_token_padding != forward_batch.num_token_non_padded_cpu:
                        attn_output = torch.cat(
                            [
                                attn_output,
                                attn_output.new_zeros(
                                    num_token_padding - attn_output.shape[0],
                                    *attn_output.shape[1:],
                                ),
                            ],
                            dim=0,
                        )
                else:
                    q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)

                    # FIA BSND with paged KV cache (reads prefix tokens from cache)
                    seq_lens_cpu = forward_batch.seq_lens.cpu().tolist()
                    attn_output = torch.empty(
                        (q.shape[0], layer.tp_q_head_num, layer.v_head_dim),
                        device=q.device,
                        dtype=q.dtype,
                    )
                    q_len_offset = 0
                    for seq_idx, q_len in enumerate(forward_batch.extend_seq_lens_cpu):
                        if q_len == 0:
                            continue
                        total_kv_len = seq_lens_cpu[seq_idx]
                        result, _ = torch_npu.npu_fused_infer_attention_score_v2(
                            query=q[None, q_len_offset : q_len_offset + q_len],
                            key=k_cache.view(
                                -1,
                                self.page_size,
                                layer.tp_k_head_num * layer.qk_head_dim,
                            ),
                            value=v_cache.view(
                                -1,
                                self.page_size,
                                layer.tp_v_head_num * layer.v_head_dim,
                            ),
                            num_query_heads=layer.tp_q_head_num,
                            num_key_value_heads=layer.tp_k_head_num,
                            input_layout="BSND",
                            block_table=self.forward_metadata.block_tables[
                                seq_idx : seq_idx + 1
                            ],
                            block_size=self.page_size,
                            actual_seq_qlen=[q_len],
                            actual_seq_kvlen=[total_kv_len],
                            atten_mask=self.fia_mask.unsqueeze(0),
                            sparse_mode=3,
                            softmax_scale=layer.scaling,
                        )
                        attn_output[q_len_offset : q_len_offset + q_len] = result[0]
                        q_len_offset += q_len

                    attn_output = attn_output.view(
                        -1, layer.tp_q_head_num * layer.v_head_dim
                    )
            elif self.use_fa:
                from flash_attn_npu_v3 import flash_attn_with_kvcache

                q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
                k = k_cache.view(
                    -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                )
                v = v_cache.view(
                    -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                )
                extend_seq_lens = self.forward_metadata.extend_seq_lens_cpu_int.npu()
                cu_seqlens_q = torch.cat(
                    [
                        torch.zeros(1, dtype=torch.int32).npu(),
                        extend_seq_lens.cumsum(0).to(torch.int32),
                    ]
                )
                max_seqlen_q = extend_seq_lens.max().item()
                attn_output = flash_attn_with_kvcache(
                    q,
                    k,
                    v,
                    cache_seqlens=self.forward_metadata.seq_lens,
                    page_table=self.forward_metadata.block_tables,
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=True,
                    window_size=[-1, -1],
                    softcap=0.0,
                    rotary_interleaved=False,
                    num_splits=0,
                    sm_margin=0,
                    return_softmax_lse=False,
                )
                attn_output = attn_output.view(
                    -1, layer.tp_q_head_num * layer.v_head_dim
                )
            else:
                causal = True
                if (
                    layer.is_cross_attention
                    or layer.attn_type == AttentionType.ENCODER_ONLY
                ):
                    causal = False
                # there are some accuracy issues in cross attention scene to use torch_npu._npu_flash_attention_qlens
                # forward_batch.encoder_lens is not None in cross attention scend, we add native attn to solve accuracy issues
                # Model skywork-reward-gemma2-2-27B also suffers from precision anomalies, thus the torch native backend becomes beneficial approach.
                if (
                    layer.qk_head_dim <= 128
                    and causal
                    and forward_batch.encoder_lens is None
                    and layer.logit_cap == 0
                    and not getattr(self, "use_native_sdpa", False)
                ):
                    if not self.use_alibi:
                        query = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
                        attn_output = torch.empty(
                            (query.shape[0], layer.tp_q_head_num * layer.v_head_dim),
                            dtype=query.dtype,
                            device=query.device,
                        )
                        torch_npu._npu_flash_attention_qlens(
                            query=query,
                            key_cache=k_cache,
                            value_cache=v_cache,
                            mask=self.mask,
                            block_table=self.forward_metadata.block_tables,
                            seq_len=self.forward_metadata.extend_seq_lens_cpu_int,
                            context_lens=self.forward_metadata.seq_lens_cpu_int,
                            scale_value=layer.scaling,
                            num_heads=layer.tp_q_head_num,
                            num_kv_heads=layer.tp_k_head_num,
                            out=attn_output,
                        )
                    else:
                        attn_output = self.attn_alibi(
                            q=q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim),
                            k_cache=k_cache,
                            v_cache=v_cache,
                            block_tables=self.forward_metadata.block_tables,
                            seq_lens=self.forward_metadata.seq_lens_cpu_int,
                            query_lens=self.forward_metadata.extend_seq_lens_cpu_int,
                            scale_value=layer.scaling,
                            num_heads=layer.tp_q_head_num,
                            slopes=slopes,
                            is_extend=True,
                        )
                else:
                    if layer.qk_head_dim != layer.v_head_dim:
                        attn_output = q.new_empty(
                            (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                        )
                    else:
                        attn_output = torch.empty_like(
                            q, memory_format=torch.contiguous_format
                        )

                    use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

                    q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
                    o_ = attn_output.view(-1, layer.tp_q_head_num, layer.v_head_dim)

                    # add forward_batch.encoder_lens and is_cross_attention arguments for cross attention scene
                    attn_output = self.native_attn.run_sdpa_forward_extend(
                        q_,
                        o_,
                        k_cache.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                        v_cache.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                        self.req_to_token_pool.req_to_token,
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        forward_batch.extend_prefix_lens,
                        forward_batch.extend_seq_lens,
                        forward_batch.encoder_lens,
                        is_cross_attention=layer.is_cross_attention,
                        scaling=layer.scaling,
                        enable_gqa=use_gqa,
                        causal=causal,
                        sliding_window_size=layer.sliding_window_size,
                        full_to_swa_mapping=(
                            self.full_to_swa_index_mapping
                            if self._is_swa_layer(layer)
                            else None
                        ),
                        logit_cap=layer.logit_cap,
                        logit_capping_method=layer.logit_capping_method,
                    )
                    attn_output = attn_output.view(
                        -1, layer.tp_q_head_num * layer.v_head_dim
                    )
        elif sum(forward_batch.extend_prefix_lens_cpu) > 0:
            # This branch adds support for prefix cache for GLM-4.7-Flash.
            # When using the MLA architecture, if qk head dim equals v head dim and the head count is not a power of 2,
            # we use the FIA kernel for computation.
            q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)

            k_buffer = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_buffer = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
            kv_cached = gather_mla_cache_pages(
                k_buffer,
                self.forward_metadata.flatten_prefix_block_tables,
                is_nz=is_fia_nz(),
            )
            if self.use_mla_fp8:
                # Prefill still projects the historical latent cache in BF16.
                kv_cached = (kv_cached.float() * fp8_kv_scale).to(torch.bfloat16)
            k_rope_cached = gather_mla_cache_pages(
                v_buffer,
                self.forward_metadata.flatten_prefix_block_tables,
                is_nz=is_fia_nz(),
            ).flatten(0, 1)

            assert layer.kv_b_proj is not None
            kv = layer.kv_b_proj(kv_cached)[0].view(
                -1, layer.tp_k_head_num, self.qk_nope_head_dim + layer.v_head_dim
            )
            k_nope, v_pre = kv.split([self.qk_nope_head_dim, layer.v_head_dim], dim=-1)

            k_rope = k_rope_cached.expand(-1, layer.tp_k_head_num, -1)
            k_pre = torch.cat([k_nope, k_rope], dim=-1)

            attn_output = torch.empty(
                (q.size(0), layer.tp_q_head_num, layer.v_head_dim),
                device=q.device,
                dtype=q.dtype,
            )
            q_len_offset = 0
            prefix_len_offset = 0
            for q_len, prefix_len in zip(
                self.forward_metadata.extend_seq_lens_cpu_int,
                self.forward_metadata.prefix_lens,
            ):
                k_cur_slice = k[None, q_len_offset : q_len_offset + q_len]
                v_cur_slice = v[None, q_len_offset : q_len_offset + q_len]
                k_pre_slice = k_pre[
                    None, prefix_len_offset : prefix_len_offset + prefix_len
                ]
                v_pre_slice = v_pre[
                    None, prefix_len_offset : prefix_len_offset + prefix_len
                ]

                k_full = torch.cat([k_pre_slice, k_cur_slice], dim=1)
                v_full = torch.cat([v_pre_slice, v_cur_slice], dim=1)

                attn_output[q_len_offset : q_len_offset + q_len] = (
                    torch.ops.npu.npu_fused_infer_attention_score(
                        q[None, q_len_offset : q_len_offset + q_len],
                        k_full,
                        v_full,
                        num_heads=layer.tp_q_head_num,
                        num_key_value_heads=layer.tp_k_head_num,
                        input_layout="BSND",  # todo, TND not supports q_heads!=k_heads
                        atten_mask=self.fia_mask,
                        sparse_mode=3,
                        scale=layer.scaling,
                        next_tokens=0,
                    )[0]
                )
                q_len_offset += q_len
                prefix_len_offset += prefix_len
            attn_output = attn_output.view(-1, layer.tp_q_head_num * layer.v_head_dim)
        else:
            if layer.qk_head_dim == layer.v_head_dim:
                """FIA will support multi-bs in the later version of CANN"""
                q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
                attn_output = torch.empty(
                    (q.size(0), layer.tp_q_head_num, layer.v_head_dim),
                    device=q.device,
                    dtype=q.dtype,
                )
                q_len_offset = 0
                for q_len in forward_batch.extend_seq_lens_cpu:
                    attn_output[q_len_offset : q_len_offset + q_len] = (
                        torch.ops.npu.npu_fused_infer_attention_score(
                            q[None, q_len_offset : q_len_offset + q_len],
                            k[None, q_len_offset : q_len_offset + q_len],
                            v[None, q_len_offset : q_len_offset + q_len].contiguous(),
                            num_heads=layer.tp_q_head_num,
                            num_key_value_heads=layer.tp_k_head_num,
                            input_layout="BSND",  # todo, TND not supports q_heads!=k_heads
                            atten_mask=self.fia_mask.unsqueeze(0),
                            sparse_mode=3 if q_len != 1 else 0,
                            scale=layer.scaling,
                            next_tokens=0,
                        )[0]
                    )
                    q_len_offset += q_len
                attn_output = attn_output.view(
                    -1, layer.tp_q_head_num * layer.v_head_dim
                )
            elif layer.v_head_dim in [256]:
                """Currently, in NO_QUANT situation, qk_nope_head_dim == v_head_dim, and rope exists, v_head_dim only support 512 and 128"""
                kv_lora_rank = k.shape[-1] - self.qk_rope_head_dim
                kv_c, k_rope = k.split([kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                if save_kv_cache:
                    self.token_to_kv_pool.set_kv_buffer(
                        layer, forward_batch.out_cache_loc, kv_c, k_rope
                    )
                attn_output = q.new_empty(
                    (q.shape[0], layer.tp_q_head_num, kv_lora_rank)
                )
                use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

                k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
                v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
                kv_cache = torch.cat([k_cache, v_cache], dim=-1)
                attn_output = self.native_attn.run_sdpa_forward_extend(
                    q,
                    attn_output,
                    kv_cache.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                    k_cache.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                    self.req_to_token_pool.req_to_token,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.extend_prefix_lens,
                    forward_batch.extend_seq_lens,
                    scaling=layer.scaling,
                    enable_gqa=use_gqa,
                    causal=True,
                )
            else:
                num_token_padding = q.shape[0]
                q, k, v = [
                    data[: forward_batch.num_token_non_padded_cpu] for data in [q, k, v]
                ]

                q_nope, q_rope = q.split(
                    [layer.v_head_dim, self.qk_rope_head_dim], dim=-1
                )
                k_nope, k_rope = k.split(
                    [layer.v_head_dim, self.qk_rope_head_dim], dim=-1
                )

                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                    q_nope,
                    k_nope.contiguous(),
                    v.contiguous(),
                    query_rope=q_rope,
                    key_rope=k_rope.contiguous(),
                    num_heads=layer.tp_q_head_num,
                    input_layout="TND",
                    atten_mask=self.fia_mask,
                    sparse_mode=3,
                    actual_seq_lengths=self.forward_metadata.seq_lens_list_cumsum,
                    actual_seq_lengths_kv=self.forward_metadata.seq_lens_list_cumsum,
                    scale=layer.scaling,
                    next_tokens=0,
                )

                attn_output = attn_output.reshape(
                    -1, layer.tp_q_head_num, layer.v_head_dim
                )
                if num_token_padding != forward_batch.num_token_non_padded_cpu:
                    attn_output = torch.cat(
                        [
                            attn_output,
                            attn_output.new_zeros(
                                num_token_padding - attn_output.shape[0],
                                *attn_output.shape[1:],
                            ),
                        ],
                        dim=0,
                    )

        return attn_output

    def forward_dllm(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi_head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                KVWriteLoc(
                    forward_batch.out_cache_loc,
                    self.forward_metadata.swa_out_cache_loc,
                ),
                k,
                v,
            )

        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
        query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)

        if self.forward_metadata.seq_lens_cpu_int is None:
            # capture
            actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_list
        else:
            # eagle
            actual_seq_lengths_kv = (
                self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
            )

        if self.forward_metadata.extend_seq_lens_cpu_int is None:
            # capture & replay
            actual_seq_lengths = self.forward_metadata.seq_lens_list_cumsum
        else:
            actual_seq_lengths = (
                torch.cumsum(self.forward_metadata.extend_seq_lens_cpu_int, dim=0)
                .int()
                .tolist()
            )

        attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            query,
            k_cache.view(-1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim),
            v_cache.view(-1, self.page_size, layer.tp_v_head_num * layer.v_head_dim),
            block_table=self.forward_metadata.block_tables,
            block_size=self.page_size,
            num_heads=layer.tp_q_head_num,
            num_key_value_heads=layer.tp_k_head_num,
            input_layout="TND",
            atten_mask=None,
            scale=layer.scaling,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
        )
        attn_output = attn_output.view(-1, layer.tp_q_head_num * layer.v_head_dim)

        return attn_output

    def _forward_mla_fp8(
        self,
        q,
        q_rope,
        layer,
        forward_batch,
        dequant_scale_q_nope,
        fp8_kv_scale,
        *,
        is_verify,
    ):
        # Same FIA v2 C8 contract for eager and graph decode/static verify.
        if self.attn_cp_size > 1:
            raise NotImplementedError("MLA C8 currently requires non-CP execution")
        use_a2a = self.use_sparse_attn_a2a
        if use_a2a and layer.tp_k_head_num != 1:
            raise NotImplementedError("MLA C8 A2A requires a replicated single KV head")
        if dequant_scale_q_nope is None or fp8_kv_scale is None:
            raise ValueError(
                "MLA C8 requires Q per-token-head and KV per-tensor descales"
            )
        if q.dtype != torch.float8_e4m3fn or q_rope.dtype != torch.bfloat16:
            raise ValueError("MLA C8 expects FP8 E4M3 Q and BF16 side features")

        padded_tokens = q.shape[0]
        # Capture the full bucket. Replay updates the KV lengths (zero for
        # padded requests), not this Python slicing decision or tensor shapes.
        num_tokens = (
            padded_tokens
            if self.graph_mode
            else forward_batch.num_token_non_padded_cpu
        )
        if num_tokens is None:
            num_tokens = padded_tokens
        output = torch.zeros(
            (padded_tokens, layer.tp_q_head_num, self.kv_lora_rank),
            dtype=torch.bfloat16,
            device=q.device,
        )
        if num_tokens == 0:
            return output.flatten(1)
        q = q.view(-1, layer.tp_q_head_num, self.kv_lora_rank)[:num_tokens]
        q_rope = q_rope.view(-1, layer.tp_q_head_num, self.qk_rope_head_dim)[:num_tokens]
        q_scale = dequant_scale_q_nope.view(-1, layer.tp_q_head_num)[:num_tokens]
        kv_scale = fp8_kv_scale.reshape(-1).to(device=q.device, dtype=torch.float32)

        width = 1
        if is_verify:
            spec_info = forward_batch.spec_info
            if getattr(spec_info, "ragged_verify_layout", None) is not None:
                raise NotImplementedError(
                    "MLA C8 target verify currently requires static DSpark blocks"
                )
            if forward_batch.forward_mode.is_draft_extend_v2():
                if use_a2a:
                    raise NotImplementedError(
                        "MLA C8 A2A requires decode or static target-verify blocks"
                    )
                query_lens = forward_batch.extend_seq_lens_cpu
                actual_seq_qlen = np.cumsum(query_lens).tolist()
                batch_size = len(query_lens)
            else:
                width = int(spec_info.draft_token_num)
                if num_tokens % width:
                    raise ValueError(
                        "MLA C8 verify requires complete request-major draft blocks"
                    )
                batch_size = num_tokens // width
                actual_seq_qlen = list(range(width, num_tokens + 1, width))
            input_layout = "TND"
        else:
            batch_size = num_tokens
            actual_seq_qlen = None
            input_layout = "BSND"

        # DSpark's CPU metadata already includes the verify block. Do not add
        # its width again; trim DP padding in Q, Q scale and block tables alike.
        fm = self.forward_metadata
        kv_lens = fm.seq_lens_cpu_int
        if kv_lens is None:
            kv_lens = fm.seq_lens_cpu_list
        else:
            kv_lens = kv_lens.tolist()
        kv_lens = kv_lens[:batch_size]
        block_table = fm.block_tables[:batch_size]
        num_query_heads = layer.tp_q_head_num
        live_output = output[:num_tokens]
        if use_a2a:
            parallel = get_parallel()
            tp_size = parallel.attn_tp_size
            tp_group = parallel.attn_tp_group
            t_padded, t_local, local_bs, padded_bs = self._a2a_fias_v2_sizes(
                batch_size, tp_size, width
            )
            req_start = parallel.attn_tp_rank * local_bs
            # As in BF16 A2A, trade request rows for all Q heads. Transport
            # mixed dtypes as bytes: no FP8 collective or numerical cast, and
            # the per-token-head scale travels with its exact Q/side pair.
            q_bytes = self.kv_lora_rank * q.element_size()
            side_bytes = self.qk_rope_head_dim * q_rope.element_size()
            packed = torch.cat(
                (
                    q.contiguous().view(torch.uint8),
                    q_rope.contiguous().view(torch.uint8),
                    q_scale.unsqueeze(-1).contiguous().view(torch.uint8),
                ),
                dim=-1,
            )
            if t_padded > num_tokens:
                padding = packed.new_zeros(
                    t_padded - num_tokens, num_query_heads, packed.shape[-1]
                )
                padding[..., q_bytes + side_bytes :] = torch.ones(
                    1, dtype=torch.float32, device=q.device
                ).view(torch.uint8)
                packed = torch.cat((packed, padding), dim=0)
            received = torch.empty_like(packed)
            tp_group.all_to_all_single(received.view(-1), packed.view(-1))
            packed = (
                received.view(tp_size, t_local, num_query_heads, -1)
                .transpose(0, 1)
                .contiguous()
                .view(t_local, tp_size * num_query_heads, -1)
            )
            num_query_heads *= tp_size
            q = packed[..., :q_bytes].contiguous().view(torch.float8_e4m3fn)
            q_rope = (
                packed[..., q_bytes : q_bytes + side_bytes]
                .contiguous()
                .view(torch.bfloat16)
            )
            q_scale = (
                packed[..., q_bytes + side_bytes :]
                .contiguous()
                .view(torch.float32)
                .squeeze(-1)
            )
            kv_lens = (kv_lens + [0] * (padded_bs - batch_size))[
                req_start : req_start + local_bs
            ]
            if padded_bs > batch_size:
                block_table = torch.cat(
                    (
                        block_table,
                        block_table.new_zeros(
                            padded_bs - batch_size, block_table.shape[1]
                        ),
                    ),
                    dim=0,
                )
            # These are captured tensor operations, so replay reads the
            # refreshed full page table rather than a stale Python-side copy.
            block_table = block_table[req_start : req_start + local_bs]
            if is_verify:
                actual_seq_qlen = list(range(width, t_local + 1, width))
            live_output = output.new_empty(t_local, num_query_heads, self.kv_lora_rank)

        if not is_verify:
            q = q.unsqueeze(1)
            q_rope = q_rope.unsqueeze(1)
            q_scale = q_scale.unsqueeze(1)
            live_output = live_output.unsqueeze(1)
        c_kv = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        k_rope = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
        if is_fia_nz():
            c_kv = _reshape_kv_for_fia_nz(
                c_kv, layer.tp_k_head_num, self.kv_lora_rank, self.page_size
            )
            k_rope = _reshape_kv_for_fia_nz(
                k_rope, layer.tp_k_head_num, self.qk_rope_head_dim, self.page_size
            )
        else:
            c_kv = c_kv.view(-1, layer.tp_k_head_num, self.page_size, self.kv_lora_rank)
            k_rope = k_rope.view(
                -1, layer.tp_k_head_num, self.page_size, self.qk_rope_head_dim
            )
        from fia_decode_c8_tilelang_h24 import fia_decode_c8, workspace_numel
        workspace = torch.empty(workspace_numel(batch_size, 4), dtype=torch.float32, device=q.device)
        fia_decode_c8(
            q.contiguous(),
            c_kv.squeeze(1),
            q_rope.contiguous(),
            k_rope.squeeze(1),
            block_table,
            self.forward_metadata.seq_lens.to(torch.int64),
            q_scale.contiguous(),
            kv_scale,
            out = live_output,
            workspace = workspace,
            seq_len = None,
            num_cores = 32,
            splits = 4
        )

        # torch_npu.npu_fused_infer_attention_score_v2.out(
        #     q.contiguous(),
        #     c_kv,
        #     c_kv,
        #     query_rope=q_rope.contiguous(),
        #     key_rope=k_rope,
        #     num_query_heads=num_query_heads,
        #     num_key_value_heads=layer.tp_k_head_num,
        #     input_layout=input_layout,
        #     softmax_scale=layer.scaling,
        #     block_table=block_table,
        #     block_size=self.page_size,
        #     actual_seq_qlen=actual_seq_qlen,
        #     actual_seq_kvlen=kv_lens,
        #     sparse_mode=3 if is_verify else 0,
        #     atten_mask=self.mtp_mask if is_verify else None,
        #     dequant_scale_query=q_scale.contiguous(),
        #     dequant_scale_key=kv_scale,
        #     dequant_scale_value=kv_scale,
        #     key_quant_mode=0,
        #     value_quant_mode=0,
        #     query_quant_mode=3,
        #     out=[live_output, torch.empty(1, dtype=torch.bfloat16, device=q.device)],
        # )
        if use_a2a:
            # Return each source rank's head shard in the original token order.
            send_output = (
                live_output.view(
                    t_local, tp_size, layer.tp_q_head_num, self.kv_lora_rank
                )
                .transpose(0, 1)
                .contiguous()
                .view(-1)
            )
            recv_output = torch.empty_like(send_output)
            tp_group.all_to_all_single(recv_output, send_output)
            output[:num_tokens].copy_(
                recv_output.view(t_padded, layer.tp_q_head_num, self.kv_lora_rank)[
                    :num_tokens
                ]
            )
        return output.flatten(1)

    def forward_mtp(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        dequant_scale_q_nope: Optional[torch.Tensor] = None,
        fp8_kv_scale: Optional[torch.Tensor] = None,
    ):
        if self.use_mla_fp8:
            if save_kv_cache:
                raise ValueError(
                    "MLA C8 cache must be written by the quantized prepare path"
                )
            return self._forward_mla_fp8(
                q,
                q_rope,
                layer,
                forward_batch,
                dequant_scale_q_nope,
                fp8_kv_scale,
                is_verify=True,
            )
        if save_kv_cache:
            if self.use_mla:
                k = k.view(-1, layer.tp_k_head_num, self.kv_lora_rank)
                k_rope = k_rope.view(-1, layer.tp_k_head_num, self.qk_rope_head_dim)
                self.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, k_rope
                )
            else:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(
                        forward_batch.out_cache_loc,
                        self.forward_metadata.swa_out_cache_loc,
                    ),
                    k,
                    v,
                )

        if not self.use_mla:
            k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id).view(
                -1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim
            )
            v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id).view(
                -1, self.page_size, layer.tp_v_head_num * layer.v_head_dim
            )
            query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim).contiguous()

            if not self.graph_mode:
                num_token_padding = query.shape[0]
                query = query[: forward_batch.num_token_non_padded_cpu]

            if self.forward_metadata.seq_lens_cpu_int is None:
                actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_list
            else:
                actual_seq_lengths_kv = (
                    self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                )

            if forward_batch.forward_mode.is_draft_extend_v2():
                actual_seq_lengths = (
                    np.array(forward_batch.extend_seq_lens_cpu).cumsum().tolist()
                )
            else:
                actual_seq_lengths = np.arange(
                    self.speculative_num_draft_tokens,
                    self.speculative_num_draft_tokens + query.shape[0],
                    self.speculative_num_draft_tokens,
                )

            is_swa_layer = layer.sliding_window_size != -1
            if (
                is_swa_layer
                and self.is_hybrid_swa
                and hasattr(self.forward_metadata, "block_tables_swa")
            ):
                block_table = self.forward_metadata.block_tables_swa
            else:
                block_table = self.forward_metadata.block_tables

            if layer.attn_type == AttentionType.ENCODER_ONLY:
                mask = None
                sparse_mode = 0
            else:
                mask = self.mtp_mask
                sparse_mode = 4 if is_swa_layer else 3

            # ----- mojo_opset Triton MTP decode path -----------------------
            # Bypass FIA for non-MLA, non-SWA, non-encoder draft attention.
            # Enabled via ASCEND_USE_MOJO_MTP=1.  Only applies when the draft
            # model uses the ascend backend (DFlashAttention → forward_mtp).
            # if (
            #     self.use_mojo_mtp
            #     and not self.use_mla
            #     and not is_swa_layer
            #     and layer.attn_type != AttentionType.ENCODER_ONLY
            #     and not forward_batch.forward_mode.is_draft_extend_v2()
            # ):
            #     bs = block_table.shape[0]
            #     gamma = self.speculative_num_draft_tokens
            #     seq_lens_t = self.forward_metadata.seq_lens[:bs].to(torch.int32)
            #     cu_q_lens = torch.arange(
            #         0, bs * gamma + 1, gamma, dtype=torch.int32, device=query.device
            #     )
            #     attn_output = mojo_paged_mtp_decode(
            #         query=query,
            #         k_cache=k_cache,
            #         v_cache=v_cache,
            #         seq_lens=seq_lens_t,
            #         cu_q_lens=cu_q_lens,
            #         block_tables=block_table[:bs],
            #         page_size=self.page_size,
            #         num_q_heads=layer.tp_q_head_num,
            #         num_kv_heads=layer.tp_k_head_num,
            #         head_dim=layer.qk_head_dim,
            #         gamma=gamma,
            #         softmax_scale=layer.scaling,
            #     )
            #     attn_output = attn_output.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            #     if (
            #         not self.graph_mode
            #         and forward_batch.num_token_non_padded_cpu is not None
            #         and forward_batch.num_token_non_padded_cpu != num_token_padding
            #     ):
            #         attn_output = torch.cat(
            #             [
            #                 attn_output,
            #                 attn_output.new_zeros(
            #                     num_token_padding - forward_batch.num_token_non_padded_cpu,
            #                     *attn_output.shape[1:],
            #                 ),
            #             ],
            #             dim=0,
            #         )
            #     return attn_output

            if self.is_hybrid_swa or self.use_fias_v2_bsnd:
                if self.use_mojo_mtp:
                    bs = block_table.shape[0]
                    gamma = self.speculative_num_draft_tokens
                    seq_lens_t = self.forward_metadata.seq_lens[:bs].to(torch.int32)
                    if self.graph_mode:
                        cu_q_lens = self.graph_metadata["mojo_cu_q_lens"][: bs + 1]
                        max_kv_len = self.max_context_len + gamma
                    else:
                        cu_q_lens = torch.arange(
                            0, bs * gamma + 1, gamma,
                            dtype=torch.int32, device=query.device,
                        )
                        max_kv_len = int(seq_lens_t.max().item())
                    attn_output = mojo_paged_mtp_decode(
                        query=query,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        seq_lens=seq_lens_t,
                        cu_q_lens=cu_q_lens,
                        block_tables=block_table[:bs],
                        page_size=self.page_size,
                        num_q_heads=layer.tp_q_head_num,
                        num_kv_heads=layer.tp_k_head_num,
                        head_dim=layer.qk_head_dim,
                        gamma=gamma,
                        softmax_scale=layer.scaling,
                        max_kv_len=max_kv_len,
                    )
                    attn_output = attn_output.view(-1, layer.tp_q_head_num * layer.v_head_dim)
                    if (
                        not self.graph_mode
                        and forward_batch.num_token_non_padded_cpu is not None
                        and forward_batch.num_token_non_padded_cpu != num_token_padding
                    ):
                        attn_output = torch.cat(
                            [
                                attn_output,
                                attn_output.new_zeros(
                                    num_token_padding - forward_batch.num_token_non_padded_cpu,
                                    *attn_output.shape[1:],
                                ),
                            ],
                            dim=0,
                        )
                    return attn_output
                else:
                    attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
                        query,
                        k_cache,
                        v_cache,
                        block_table=block_table,
                        block_size=self.page_size,
                        num_query_heads=layer.tp_q_head_num,
                        num_key_value_heads=layer.tp_k_head_num,
                        input_layout="TND",
                        atten_mask=mask,
                        softmax_scale=layer.scaling,
                        actual_seq_qlen=actual_seq_lengths,
                        actual_seq_kvlen=actual_seq_lengths_kv,
                        sparse_mode=sparse_mode,
                        pre_tokens=(
                            layer.sliding_window_size
                            if is_swa_layer
                            else FULL_ATTENTION_WINDOW
                        ),
                        next_tokens=0 if is_swa_layer else FULL_ATTENTION_WINDOW,
                        learnable_sink=sinks,
                    )
            else:
                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                    query,
                    k_cache,
                    v_cache,
                    block_table=self.forward_metadata.block_tables,
                    block_size=self.page_size,
                    num_heads=layer.tp_q_head_num,
                    num_key_value_heads=layer.tp_k_head_num,
                    input_layout="TND",
                    atten_mask=mask,
                    scale=layer.scaling,
                    actual_seq_lengths=actual_seq_lengths,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    sparse_mode=sparse_mode,
                )
            attn_output = attn_output.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            if (
                not self.graph_mode
                and forward_batch.num_token_non_padded_cpu is not None
                and forward_batch.num_token_non_padded_cpu != num_token_padding
            ):
                attn_output = torch.cat(
                    [
                        attn_output,
                        attn_output.new_zeros(
                            num_token_padding - forward_batch.num_token_non_padded_cpu,
                            *attn_output.shape[1:],
                        ),
                    ],
                    dim=0,
                )
            return attn_output
        else:
            if self.use_flash_mla:
                kv_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            else:
                c_kv, k_rope = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
                if is_fia_nz():
                    k_rope_cache = _reshape_kv_for_fia_nz(
                        k_rope, layer.tp_k_head_num, self.qk_rope_head_dim, self.page_size
                    )
                    c_kv_cache = _reshape_kv_for_fia_nz(
                        c_kv, layer.tp_v_head_num, self.kv_lora_rank, self.page_size
                    )
                else:
                    k_rope_cache = k_rope.view(
                        -1, layer.tp_k_head_num, self.page_size, self.qk_rope_head_dim
                    )
                    c_kv_cache = c_kv.view(
                        -1, layer.tp_v_head_num, self.page_size, self.kv_lora_rank
                    )

            q_nope = q.view(-1, layer.tp_q_head_num, self.kv_lora_rank).contiguous()
            q_rope = q_rope.view(-1, layer.tp_q_head_num, self.qk_rope_head_dim)
            if not self.graph_mode:
                num_token_padding = q.shape[0]
                q_nope = q_nope[: forward_batch.num_token_non_padded_cpu]
                q_rope = q_rope[: forward_batch.num_token_non_padded_cpu]
            if self.forward_metadata.seq_lens_cpu_int is None:
                actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_list
            else:
                actual_seq_lengths_kv = (
                    self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                )
            actual_seq_lengths = np.arange(
                self.speculative_num_draft_tokens,
                self.speculative_num_draft_tokens + q_nope.shape[0],
                self.speculative_num_draft_tokens,
            )

            # When not in graph_mode, query is sliced to num_token_non_padded
            # which may drop finished requests. The FIA TND kernel requires
            # block_table.shape[0] == len(actual_seq_lengths); slice to match.
            if not self.graph_mode:
                actual_bs = len(actual_seq_lengths)
                block_table = self.forward_metadata.block_tables[:actual_bs]
                actual_seq_lengths_kv = actual_seq_lengths_kv[:actual_bs]
            else:
                block_table = self.forward_metadata.block_tables

            if not self.use_flash_mla:
                if (
                    self.q_head_num_padding is not None
                    and self.q_head_num_padding > self.tp_q_head_num
                ):
                    nope_padding = torch.empty(
                        [
                            q_nope.shape[0],
                            self.q_head_num_padding - self.tp_q_head_num,
                            self.kv_lora_rank,
                        ],
                        dtype=(
                            self.model_dtype
                            if self.model_dtype is not None
                            else torch.bfloat16
                        ),
                        device=q_nope.device,
                    )
                    rope_padding = torch.empty(
                        [
                            q_rope.shape[0],
                            self.q_head_num_padding - self.tp_q_head_num,
                            self.qk_rope_head_dim,
                        ],
                        dtype=(
                            self.model_dtype
                            if self.model_dtype is not None
                            else torch.bfloat16
                        ),
                        device=q_rope.device,
                    )
                    q_nope = torch.cat([q_nope, nope_padding], dim=1).contiguous()
                    q_rope = torch.cat([q_rope, rope_padding], dim=1).contiguous()

            num_query_heads = q_nope.shape[1]
            if self.use_fias_v2_bsnd:
                # The existing paged MLA cache is [block, KV_N, page, D].
                # V2 consumes it with BNSD queries; keep the cache unchanged.
                batch_size = len(actual_seq_lengths_kv)
                query_seq_len = self.speculative_num_draft_tokens
                assert (
                    q_nope.shape[0] == batch_size * query_seq_len
                ), "FIAS V2 target verify requires one fixed draft block per request"
                if batch_size == 0:
                    attn_output = torch.empty_like(q_nope)
                elif self.use_sparse_attn_a2a:
                    attn_output = self._forward_fias_v2_bsnd_tp_a2a(
                        q_nope, q_rope, kv_cache, layer,
                    )
                else:
                    if self.use_flash_mla:
                        q_nope_bsnd = (
                            q_nope.view(
                                batch_size,
                                query_seq_len,
                                num_query_heads,
                                self.kv_lora_rank,
                            ).contiguous()
                        )
                        q_rope_bsnd = (
                            q_rope.view(
                                batch_size,
                                query_seq_len,
                                num_query_heads,
                                self.qk_rope_head_dim,
                            ).contiguous()
                        )
                        attn_output, _ = flash_mla_with_kvcache(
                            torch.cat([q_nope_bsnd, q_rope_bsnd], dim=-1),
                            kv_cache,
                            block_table=block_table,
                            cache_seqlens=self.forward_metadata.seq_lens.to(torch.int32),
                            cu_seqlens_q=None,
                            seqused_q=self.forward_metadata.seqused_q.to(torch.int32),
                            attn_mask=self.mtp_mask,
                            metadata=self.forward_metadata.metadata_flash_mla,
                            head_dim_v=512,
                            softmax_scale=layer.scaling,
                            mask_mode=3,
                            max_seqlen_q = -1,
                            max_seqlen_kv= -1,
                            layout_q= "BSND",
                            layout_kv= "PA_BBND",
                            layout_out= "BSND",
                            return_softmax_lse= False,
                        )
                        attn_output = (
                            attn_output
                            .reshape(-1, num_query_heads, self.kv_lora_rank)
                        )
                    else:
                        q_nope_bnsd = (
                            q_nope.view(
                                batch_size,
                                query_seq_len,
                                num_query_heads,
                                self.kv_lora_rank,
                            )
                            .transpose(1, 2)
                            .contiguous()
                        )
                        q_rope_bnsd = (
                            q_rope.view(
                                batch_size,
                                query_seq_len,
                                num_query_heads,
                                self.qk_rope_head_dim,
                            )
                            .transpose(1, 2)
                            .contiguous()
                        )
                        attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
                            q_nope_bnsd,
                            c_kv_cache,
                            c_kv_cache,
                            query_rope=q_rope_bnsd,
                            key_rope=k_rope_cache,
                            num_query_heads=num_query_heads,
                            num_key_value_heads=layer.tp_k_head_num,
                            input_layout="BNSD",
                            softmax_scale=layer.scaling,
                            block_table=block_table,
                            block_size=self.page_size,
                            sparse_mode=3,
                            atten_mask=self.mtp_mask,
                            actual_seq_qlen=[query_seq_len] * batch_size,
                            actual_seq_kvlen=actual_seq_lengths_kv,
                            pre_tokens=FULL_ATTENTION_WINDOW,
                            next_tokens=0,
                        )
                        attn_output = (
                            attn_output.transpose(1, 2)
                            .contiguous()
                            .reshape(-1, num_query_heads, self.kv_lora_rank)
                        )
            else:
                workspace = (
                    torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                        q_nope,
                        c_kv_cache,
                        c_kv_cache,
                        query_rope=q_rope,
                        key_rope=k_rope_cache,
                        num_heads=num_query_heads,
                        num_key_value_heads=layer.tp_k_head_num,
                        input_layout="TND",
                        scale=layer.scaling,
                        antiquant_mode=0,
                        antiquant_scale=None,
                        block_table=block_table,
                        block_size=self.page_size,
                        sparse_mode=3,
                        atten_mask=self.mtp_mask,
                        actual_seq_lengths=actual_seq_lengths,
                        actual_seq_lengths_kv=actual_seq_lengths_kv,
                    )
                )
                attn_output = torch.empty_like(q_nope, dtype=q.dtype, device=q.device)
                softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)
                torch_npu.npu_fused_infer_attention_score.out(
                    q_nope,
                    c_kv_cache,
                    c_kv_cache,
                    query_rope=q_rope,
                    key_rope=k_rope_cache,
                    num_heads=num_query_heads,
                    num_key_value_heads=layer.tp_k_head_num,
                    input_layout="TND",
                    scale=layer.scaling,
                    antiquant_mode=0,
                    antiquant_scale=None,
                    block_table=block_table,
                    block_size=self.page_size,
                    sparse_mode=3,
                    atten_mask=self.mtp_mask,
                    actual_seq_lengths=actual_seq_lengths,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    workspace=workspace,
                    out=[attn_output, softmax_lse],
                )
            attn_output = attn_output[:, : layer.tp_q_head_num, :]
            attn_output = attn_output.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            if (
                not self.graph_mode
                and forward_batch.num_token_non_padded_cpu != num_token_padding
            ):
                attn_output = torch.cat(
                    [
                        attn_output,
                        attn_output.new_zeros(
                            num_token_padding - attn_output.shape[0],
                            *attn_output.shape[1:],
                        ),
                    ],
                    dim=0,
                )
            return attn_output

    def forward_decode_graph(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ):
        if save_kv_cache:
            if self.use_mla:
                k = k.view(-1, layer.tp_k_head_num, self.kv_lora_rank)
                k_rope = k_rope.view(-1, layer.tp_k_head_num, self.qk_rope_head_dim)
                self.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, k_rope
                )
            else:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(
                        forward_batch.out_cache_loc,
                        self.forward_metadata.swa_out_cache_loc,
                    ),
                    k,
                    v,
                )

        if sinks is not None or self.is_hybrid_swa:
            # Use SWA block tables if hybrid SWA is enabled for this layer
            if self._is_swa_layer(layer):
                block_tables = self.forward_metadata.block_tables_swa
            else:
                block_tables = self.forward_metadata.block_tables
            if self.use_fia:
                k_cache = (
                    self.token_to_kv_pool.get_key_buffer(layer.layer_id)
                    .view(-1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim)
                    .contiguous()
                )
                v_cache = (
                    self.token_to_kv_pool.get_value_buffer(layer.layer_id)
                    .view(-1, self.page_size, layer.tp_v_head_num * layer.v_head_dim)
                    .contiguous()
                )
                query = q.reshape(
                    -1, layer.tp_q_head_num, layer.qk_head_dim
                ).contiguous()

                if self.forward_metadata.seq_lens_cpu_int is None:
                    actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_list
                else:
                    actual_seq_lengths_kv = (
                        self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                    )
                seq_lens_list = (
                    self.forward_metadata.seq_lens_cpu_list
                    if self.forward_metadata.seq_lens_cpu_int is None
                    else self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                )
                actual_seq_lengths = (
                    torch.tensor([1] * len(seq_lens_list), dtype=torch.int32)
                    .cumsum(dim=0)
                    .tolist()
                )
                if layer.sliding_window_size != -1:
                    sparse_mode = 4
                else:
                    sparse_mode = 3

                attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
                    query,
                    k_cache,
                    v_cache,
                    num_query_heads=layer.tp_q_head_num,
                    num_key_value_heads=layer.tp_k_head_num,
                    input_layout="TND",
                    pre_tokens=(
                        layer.sliding_window_size
                        if layer.sliding_window_size != -1
                        else FULL_ATTENTION_WINDOW
                    ),
                    next_tokens=(
                        0 if layer.sliding_window_size == -1 else FULL_ATTENTION_WINDOW
                    ),
                    atten_mask=self.fia_mask.to(torch.int8),
                    sparse_mode=sparse_mode,
                    softmax_scale=layer.scaling,
                    block_table=block_tables,
                    block_size=self.page_size,
                    actual_seq_qlen=actual_seq_lengths,
                    actual_seq_kvlen=actual_seq_lengths_kv,
                    learnable_sink=sinks,
                )
                attn_output = attn_output.view(
                    -1, layer.tp_q_head_num * layer.v_head_dim
                )
                return attn_output
            else:
                k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
                v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
                attn_out = attention_sinks_triton(
                    q,
                    k_cache,
                    v_cache,
                    sinks,
                    block_tables,
                    self.forward_metadata.seq_lens,
                    layer.scaling,
                    layer.sliding_window_size,
                    layer.tp_q_head_num,
                    layer.tp_k_head_num,
                )
                return attn_out

        if not self.use_mla:
            seq_lens_cpu_int = self.forward_metadata.seq_lens_cpu_int
            seq_lens_cpu_list = self.forward_metadata.seq_lens_cpu_list
            if self._is_swa_layer(layer):
                # CUDA/NPU graph capture uses seq_len fill value 0 on Ascend.
                # Avoid dynamic window block-table construction during capture,
                # because it can create a zero-width block table and break tiling.
                block_tables = self.forward_metadata.block_tables_swa
                attn_mask = self.forward_metadata.swa_mask
            else:
                block_tables = self.forward_metadata.block_tables
                attn_mask = None
            k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id).view(
                -1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim
            )
            v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id).view(
                -1, self.page_size, layer.tp_v_head_num * layer.v_head_dim
            )
            query = q.reshape(-1, 1, layer.tp_q_head_num * layer.qk_head_dim)
            if seq_lens_cpu_int is None:
                actual_seq_len_kv = seq_lens_cpu_list
            else:
                actual_seq_len_kv = seq_lens_cpu_int.cpu().int().tolist()

            num_tokens = query.shape[0]
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query,
                k_cache,
                v_cache,
                block_table=block_tables,
                block_size=self.page_size,
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                input_layout="BSH",
                scale=layer.scaling,
                actual_seq_lengths_kv=actual_seq_len_kv,
                atten_mask=attn_mask,
                sparse_mode=0,
            )
            output = torch.empty(
                (num_tokens, 1, layer.tp_q_head_num * layer.v_head_dim),
                dtype=q.dtype,
                device=q.device,
            )
            softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)
            torch_npu.npu_fused_infer_attention_score.out(
                query,
                k_cache,
                v_cache,
                block_table=block_tables,
                block_size=self.page_size,
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                input_layout="BSH",
                scale=layer.scaling,
                actual_seq_lengths_kv=actual_seq_len_kv,
                atten_mask=attn_mask,
                sparse_mode=0,
                workspace=workspace,
                out=[output, softmax_lse],
            )
            return output.view(num_tokens, layer.tp_q_head_num * layer.v_head_dim)
        else:
            c_kv, k_rope = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            if is_fia_nz():
                k_rope_cache = _reshape_kv_for_fia_nz(
                    k_rope, layer.tp_k_head_num, self.qk_rope_head_dim, self.page_size
                )
                c_kv_cache = _reshape_kv_for_fia_nz(
                    c_kv, layer.tp_v_head_num, self.kv_lora_rank, self.page_size
                )
            else:
                k_rope_cache = k_rope.view(
                    -1, self.page_size, layer.tp_k_head_num * self.qk_rope_head_dim
                )
                c_kv_cache = c_kv.view(
                    -1, self.page_size, layer.tp_k_head_num * self.kv_lora_rank
                )

            q_nope = q.view(-1, 1, layer.tp_q_head_num, self.kv_lora_rank).contiguous()
            q_rope = q_rope.view(-1, 1, layer.tp_q_head_num, self.qk_rope_head_dim)

            assert (
                self.q_head_num_padding is None
                or self.q_head_num_padding >= layer.tp_q_head_num
            )

            if (
                self.q_head_num_padding is not None
                and self.q_head_num_padding > layer.tp_q_head_num
            ):
                # The FIA kernel only supports head counts that are powers of 2.
                # Therefore, we pad the head dimension when it is not a power of 2.
                q_nope = torch.cat(
                    [q_nope, self.forward_metadata.nope_padding], dim=2
                ).contiguous()
                q_rope = torch.cat(
                    [q_rope, self.forward_metadata.rope_padding], dim=2
                ).contiguous()

            if self.forward_metadata.seq_lens_cpu_int is None:
                actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_list
            else:
                actual_seq_len_kv = (
                    self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                )

            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                q_nope,
                c_kv_cache,
                c_kv_cache,
                query_rope=q_rope,
                key_rope=k_rope_cache,
                num_heads=self.q_head_num_padding,
                num_key_value_heads=layer.tp_k_head_num,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                input_layout="BSND",
                scale=layer.scaling,
                actual_seq_lengths_kv=actual_seq_len_kv,
                antiquant_mode=0,
                antiquant_scale=None,
                sparse_mode=0,
            )
            output = torch.empty_like(q_nope, dtype=q.dtype, device=q.device)
            softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)

            torch_npu.npu_fused_infer_attention_score.out(
                q_nope,
                c_kv_cache,
                c_kv_cache,
                query_rope=q_rope,
                key_rope=k_rope_cache,
                num_heads=self.q_head_num_padding,
                num_key_value_heads=layer.tp_k_head_num,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                input_layout="BSND",
                scale=layer.scaling,
                actual_seq_lengths_kv=actual_seq_len_kv,
                antiquant_mode=0,
                antiquant_scale=None,
                sparse_mode=0,
                workspace=workspace,
                out=[output, softmax_lse],
            )

            output = output[:, :, : layer.tp_q_head_num, :]
            return output.view(-1, layer.tp_q_head_num * self.kv_lora_rank)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        slopes: Optional[torch.Tensor] = None,
        dequant_scale_q_nope: Optional[torch.Tensor] = None,
        fp8_kv_scale: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if is_mla_preprocess_enabled() and self.use_mla:
            # MLAPO does saving kv_cache
            save_kv_cache = False
        if topk_indices is not None:
            return self.forward_sparse(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope,
                k_rope,
                topk_indices,
            )

        if self.use_mla_fp8:
            if save_kv_cache:
                raise ValueError(
                    "MLA C8 cache must be written by the quantized prepare path"
                )
            return self._forward_mla_fp8(
                q,
                q_rope,
                layer,
                forward_batch,
                dequant_scale_q_nope,
                fp8_kv_scale,
                is_verify=False,
            )

        if self.graph_mode and (not self.enable_torch_compile):
            return self.forward_decode_graph(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope=q_rope,
                k_rope=k_rope,
                sinks=sinks,
            )

        if not self.use_mla:
            # In cross attention layer, when there is no vision input,the values of k and v is None
            if save_kv_cache and k is not None and v is not None:
                # support cross attention
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                # swa_out_cache_loc is the full->SWA write target, derived from
                # out_cache_loc; it must not be applied to cross-attention writes
                # (which target encoder_out_cache_loc) and is None for non-SWA pools.
                swa_loc = (
                    self.forward_metadata.swa_out_cache_loc
                    if not layer.is_cross_attention
                    else None
                )
                self.token_to_kv_pool.set_kv_buffer(
                    layer, KVWriteLoc(cache_loc, swa_loc), k, v
                )
            num_tokens = q.shape[0]
            k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id)

            if sinks is not None or (self._is_swa_layer(layer) and self.use_fia):
                # Use SWA block tables if hybrid SWA is enabled for this layer
                if self._is_swa_layer(layer):
                    block_tables = self.forward_metadata.block_tables_swa
                else:
                    block_tables = self.forward_metadata.block_tables
                if self.use_fia:
                    if self.forward_metadata.seq_lens_cpu_int is None:
                        actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_list
                    else:
                        actual_seq_len_kv = (
                            self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                        )
                    block_size = self.page_size

                    if sinks is not None:
                        mask = self.fia_mask
                    else:
                        max_model_len = block_tables.shape[-1] * block_size
                        mask = self.ascend_attn_mask_builder.get_swa_mask(
                            self.forward_metadata.seq_lens,
                            max_model_len,
                            layer.sliding_window_size,
                        )

                    attn_out, _ = torch_npu.npu_fused_infer_attention_score_v2(
                        q.view(
                            forward_batch.batch_size,
                            -1,
                            layer.tp_q_head_num,
                            layer.qk_head_dim,
                        ),
                        k_cache.view(
                            -1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim
                        ),
                        v_cache.view(
                            -1, self.page_size, layer.tp_v_head_num * layer.v_head_dim
                        ),
                        num_query_heads=layer.tp_q_head_num,
                        num_key_value_heads=layer.tp_k_head_num,
                        input_layout="BSND",
                        block_size=block_size,
                        atten_mask=(mask if layer.sliding_window_size != -1 else None),
                        sparse_mode=4 if layer.sliding_window_size != -1 else 0,
                        softmax_scale=layer.scaling,
                        block_table=block_tables,
                        actual_seq_qlen=[1] * len(self.forward_metadata.seq_lens),
                        actual_seq_kvlen=actual_seq_len_kv,
                        pre_tokens=layer.sliding_window_size,
                        next_tokens=0,
                        learnable_sink=sinks,
                    )
                    attn_out = attn_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
                else:
                    attn_out = attention_sinks_triton(
                        q,
                        k_cache,
                        v_cache,
                        sinks,
                        block_tables,
                        self.forward_metadata.seq_lens,
                        layer.scaling,
                        layer.sliding_window_size,
                        layer.tp_q_head_num,
                        layer.tp_k_head_num,
                    )
                return attn_out

            if self.use_fia:
                if self.forward_metadata.seq_lens_cpu_int is None:
                    actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_list
                else:
                    actual_seq_len_kv = (
                        self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                    )
                num_token_padding = q.shape[0]
                actual_bs = self.forward_metadata.block_tables.shape[0]
                q = q[:actual_bs]
                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                    q.view(
                        -1,
                        1,
                        layer.tp_q_head_num,
                        layer.qk_head_dim,
                    ),
                    k_cache.view(
                        -1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim
                    ),
                    v_cache.view(
                        -1, self.page_size, layer.tp_v_head_num * layer.v_head_dim
                    ),
                    num_heads=layer.tp_q_head_num,
                    num_key_value_heads=layer.tp_k_head_num,
                    input_layout="BSND",
                    atten_mask=None,
                    block_size=self.page_size,
                    block_table=self.forward_metadata.block_tables,
                    actual_seq_lengths_kv=actual_seq_len_kv,
                    scale=layer.scaling,
                )
                if actual_bs != num_token_padding:
                    attn_output = torch.cat(
                        [
                            attn_output,
                            attn_output.new_zeros(
                                num_token_padding - actual_bs,
                                *attn_output.shape[1:],
                            ),
                        ],
                        dim=0,
                    )
            elif self.use_fa:
                from flash_attn_npu_v3 import flash_attn_with_kvcache

                q = q.view(
                    forward_batch.batch_size, -1, layer.tp_q_head_num, layer.qk_head_dim
                )
                k = k_cache.view(
                    -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                )
                v = v_cache.view(
                    -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                )
                attn_output = flash_attn_with_kvcache(
                    q,
                    k,
                    v,
                    page_table=self.forward_metadata.block_tables,
                    cache_seqlens=self.forward_metadata.seq_lens,
                    softmax_scale=layer.scaling,
                )
            # there are some accuracy issues in cross attention scene to use torch_npu._npu_flash_attention_qlens
            # forward_batch.encoder_lens is not None in cross attention scend, we add native attn to solve accuracy issues
            elif forward_batch.encoder_lens is None and layer.logit_cap == 0:
                query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
                num_tokens = query.shape[0]
                if not self.use_alibi:
                    attn_output = torch.empty(
                        (num_tokens, layer.tp_q_head_num, layer.v_head_dim),
                        dtype=query.dtype,
                        device=query.device,
                    )

                    torch_npu._npu_paged_attention(
                        query=query,
                        key_cache=k_cache,
                        value_cache=v_cache,
                        num_heads=layer.tp_q_head_num,
                        num_kv_heads=layer.tp_k_head_num,
                        scale_value=layer.scaling,
                        block_table=self.forward_metadata.block_tables,
                        context_lens=self.forward_metadata.seq_lens_cpu_int,
                        out=attn_output,
                    )
                else:
                    attn_output = self.attn_alibi(
                        q=query,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        block_tables=self.forward_metadata.block_tables,
                        seq_lens=self.forward_metadata.seq_lens_cpu_int,
                        query_lens=torch.ones(num_tokens, dtype=torch.int32),
                        scale_value=layer.scaling,
                        num_heads=layer.tp_q_head_num,
                        slopes=slopes,
                        is_extend=False,
                    )
            else:
                if layer.qk_head_dim != layer.v_head_dim:
                    attn_output = q.new_empty(
                        (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                    )
                else:
                    attn_output = torch.empty_like(
                        q, memory_format=torch.contiguous_format
                    )

                use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

                q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
                o_ = attn_output.view(-1, layer.tp_q_head_num, layer.v_head_dim)

                attn_output = self.native_attn.run_sdpa_forward_decode(
                    q_,
                    o_,
                    k_cache.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                    v_cache.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                    self.req_to_token_pool.req_to_token,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.encoder_lens,
                    is_cross_attention=layer.is_cross_attention,
                    scaling=layer.scaling,
                    enable_gqa=use_gqa,
                    causal=False,
                    sliding_window_size=layer.sliding_window_size,
                    full_to_swa_mapping=(
                        self.full_to_swa_index_mapping
                        if self._is_swa_layer(layer)
                        else None
                    ),
                    logit_cap=layer.logit_cap,
                    logit_capping_method=layer.logit_capping_method,
                )
            return attn_output.view(num_tokens, layer.tp_q_head_num * layer.v_head_dim)
        else:
            if save_kv_cache:
                self.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, k_rope
                )
            num_tokens = q.shape[0]
            kv_c = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
            k_pe = self.token_to_kv_pool.get_value_buffer(layer.layer_id)

            if self.use_fia and (layer.tp_q_head_num // layer.tp_k_head_num) >= 8:
                """layer.tp_q_head_num // layer.tp_k_head_num < 8 will support in the later version of CANN"""
                if is_fia_nz():
                    kv_c = _reshape_kv_for_fia_nz(
                        kv_c, layer.tp_k_head_num, self.kv_lora_rank, self.page_size
                    )
                    k_pe = _reshape_kv_for_fia_nz(
                        k_pe, layer.tp_k_head_num, self.qk_rope_head_dim, self.page_size
                    )
                else:
                    kv_c = kv_c.view(
                        -1, self.page_size, layer.tp_k_head_num * self.kv_lora_rank
                    )
                    k_pe = k_pe.view(
                        -1, self.page_size, layer.tp_k_head_num * self.qk_rope_head_dim
                    )
                q = q.view(
                    forward_batch.batch_size, -1, layer.tp_q_head_num, self.kv_lora_rank
                )
                q_rope = q_rope.view(
                    forward_batch.batch_size,
                    -1,
                    layer.tp_q_head_num,
                    self.qk_rope_head_dim,
                )
                if (layer.tp_q_head_num & (layer.tp_q_head_num - 1)) != 0:
                    power_of_2_head = next_power_of_2(layer.tp_q_head_num)
                    padding_head = power_of_2_head - layer.tp_q_head_num
                    q_padding_tensor = torch.zeros(
                        [num_tokens, q.shape[1], padding_head, q.shape[-1]],
                        dtype=q.dtype,
                        device=q.device,
                    )
                    q = torch.cat((q, q_padding_tensor), dim=-2)
                    q_rope_padding_tensor = torch.zeros(
                        [num_tokens, q_rope.shape[1], padding_head, q_rope.shape[-1]],
                        dtype=q_rope.dtype,
                        device=q_rope.device,
                    )
                    q_rope = torch.cat((q_rope, q_rope_padding_tensor), dim=-2)
                    tp_q_head_num = power_of_2_head
                else:
                    tp_q_head_num = layer.tp_q_head_num

                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                    q,
                    kv_c,
                    kv_c,
                    query_rope=q_rope,
                    key_rope=k_pe,
                    num_heads=tp_q_head_num,
                    num_key_value_heads=layer.tp_k_head_num,
                    input_layout="BSND",
                    atten_mask=None,
                    sparse_mode=0,
                    scale=layer.scaling,
                    antiquant_mode=0,
                    antiquant_scale=None,
                    block_table=self.forward_metadata.block_tables,
                    block_size=self.page_size,
                    actual_seq_lengths_kv=self.forward_metadata.seq_lens_cpu_int,
                )
                attn_output = attn_output[:, :, : layer.tp_q_head_num, :]
            else:
                assert (
                    self.graph_mode == False
                )  # _npu_paged_attention_mla not support graph mode
                if q_rope is not None:
                    q = torch.cat([q, q_rope], dim=-1)
                query = q.view(-1, layer.tp_q_head_num, layer.head_dim)
                kv_c_and_k_pe_cache = torch.cat([kv_c, k_pe], dim=-1)
                kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
                    -1,
                    self.page_size,
                    layer.tp_k_head_num,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                )
                attn_output = torch.empty(
                    [num_tokens, layer.tp_q_head_num, self.kv_lora_rank],
                    dtype=q.dtype,
                    device=q.device,
                )
                torch_npu._npu_paged_attention_mla(
                    query=query,
                    key_cache=kv_c_and_k_pe_cache,
                    num_kv_heads=layer.tp_k_head_num,
                    num_heads=layer.tp_q_head_num,
                    scale_value=layer.scaling,
                    block_table=self.forward_metadata.block_tables,
                    context_lens=self.forward_metadata.seq_lens_cpu_int,
                    mla_vheadsize=self.kv_lora_rank,
                    out=attn_output,
                )
            return attn_output.view(num_tokens, layer.tp_q_head_num * self.kv_lora_rank)

    def forward_mixed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if (
            topk_indices is not None
            or self.use_mla
            or (not self.use_fia and layer.qk_head_dim > 128)
        ):
            raise NotImplementedError(
                "The 'enable-mixed-chunk' feature is currently unsupported in the following scenarios: "
                "1. When using the MLA backend on Ascend NPU devices, "
                "2. When using the deepseekv3.2 model on Ascend NPU devices, "
                "3. When the environment variable ASCEND_USE_FIA is set to 0 and qk_head_dim exceeds 128 on Ascend NPU devices."
            )
        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                KVWriteLoc(
                    forward_batch.out_cache_loc,
                    self.forward_metadata.swa_out_cache_loc,
                ),
                k,
                v,
            )
        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
        num_block, block_size, _, _ = k_cache.shape
        key = k_cache.view(num_block, block_size, -1)
        value = v_cache.view(num_block, block_size, -1)

        query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)

        attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            query,
            key,
            value,
            num_heads=layer.tp_q_head_num,
            num_key_value_heads=layer.tp_k_head_num,
            input_layout="TND",
            block_size=block_size,
            block_table=self.forward_metadata.block_tables,
            atten_mask=self.mix_mask,
            sparse_mode=3,
            actual_seq_lengths=self.forward_metadata.seq_lens_list_cumsum,
            actual_seq_lengths_kv=self.forward_metadata.seq_lens_cpu_int,
            scale=layer.scaling,
        )

        return attn_output.view(
            attn_output.shape[0], layer.tp_q_head_num * layer.v_head_dim
        )


class AscendAttnMultiStepDraftBackend:
    """
    Wrap multiple Ascend attention backends as one for multiple consecutive
    draft decoding steps
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps

        self.attn_backends = []
        for step_id in range(self.speculative_num_steps):
            self.attn_backends.append(
                AscendAttnBackend(model_runner, speculative_step_id=step_id)
            )
        self.needs_cpu_seq_lens = self.attn_backends[0].needs_cpu_seq_lens

    def common_template(self, forward_batch: ForwardBatch, call_fn: int):
        assert forward_batch.spec_info is not None

        for i in range(self.speculative_num_steps - 1):
            call_fn(i, forward_batch)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        from sglang.srt.model_executor.forward_batch_info import build_inner_fb_view

        inner_fb = build_inner_fb_view(
            forward_batch,
            bs=forward_batch.batch_size,
            forward_mode=ForwardMode.DECODE,
        )

        def call_fn(i, _forward_batch):
            self.attn_backends[i].init_forward_metadata_out_graph(
                inner_fb, in_capture=in_capture
            )

        self.common_template(forward_batch, call_fn)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        def call_fn(i, _forward_batch):
            self.attn_backends[i].init_forward_metadata_in_graph(forward_batch)

        self.common_template(forward_batch, call_fn)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            assert forward_batch.spec_info is not None
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, call_fn)

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)


# ---------------------------------------------------------------------------
# mojo_opset MTP paged-attention Triton kernels
#
# Replaces npu_fused_infer_attention_score in forward_mtp for the DSpark
# draft model.  Each program processes one (batch, kv_head) pair and all
# GAMMA query tokens × GROUP_SIZE heads simultaneously (Q_TILE rows).
#
# KV cache layout: FIA 3D  [num_blocks, page_size, Hkv*D]
#   stride_k_block = page_size * Hkv * D
#   stride_k_blksz = Hkv * D
#   stride_k_head  = D
#   stride_k_dim   = 1
# ---------------------------------------------------------------------------

import triton
import triton.language as tl


@triton.jit
def _mojo_paged_decode_fd_mtp_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr,
    seqlens_ptr, cu_q_lens_ptr, block_tables_ptr,
    acc_ws_ptr, lse_ws_ptr,
    stride_qb, stride_qh, stride_qd,
    stride_k_block, stride_k_head, stride_k_blksz, stride_k_dim,
    stride_v_block, stride_v_head, stride_v_blksz, stride_v_dim,
    stride_bt_batch, stride_bt_block,
    stride_aws_task, stride_aws_q, stride_aws_d,
    stride_lse_task, stride_lse_q,
    softmax_scale,
    BATCH_SIZE,
    KV_SPLIT_PARTS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    GAMMA: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    GROUP_SIZE: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    Q_TILE: tl.constexpr = GAMMA * GROUP_SIZE

    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)
    total_fd_tasks = BATCH_SIZE * NUM_KV_HEADS * KV_SPLIT_PARTS

    for fd_task_id in range(pid, total_fd_tasks, n_progs):
        split_idx = fd_task_id % KV_SPLIT_PARTS
        kv_task = fd_task_id // KV_SPLIT_PARTS
        b_id = kv_task // NUM_KV_HEADS
        kv_head_id = kv_task % NUM_KV_HEADS

        kv_seq_len = tl.load(seqlens_ptr + b_id)
        q_start = tl.load(cu_q_lens_ptr + b_id)

        raw_chunk = tl.cdiv(kv_seq_len, KV_SPLIT_PARTS)
        chunk_size = tl.cdiv(raw_chunk, PAGE_SIZE) * PAGE_SIZE
        kv_start = split_idx * chunk_size
        kv_end = tl.minimum(kv_start + chunk_size, kv_seq_len)

        ws_task_idx = (b_id * NUM_KV_HEADS + kv_head_id) * KV_SPLIT_PARTS + split_idx

        offs_qg = tl.arange(0, Q_TILE)
        gamma_idx = offs_qg // GROUP_SIZE
        g_idx = offs_qg % GROUP_SIZE

        if GQA_INTERLEAVE:
            q_head_ids = kv_head_id + g_idx * NUM_KV_HEADS
        else:
            q_head_ids = kv_head_id * GROUP_SIZE + g_idx

        q_token_pos = q_start + gamma_idx
        offs_d = tl.arange(0, BLOCK_SIZE_D)

        q_ptrs = (
            q_ptr
            + q_token_pos[:, None] * stride_qb
            + q_head_ids[:, None] * stride_qh
            + offs_d[None, :] * stride_qd
        )
        q = tl.load(q_ptrs, mask=offs_d[None, :] < HEAD_DIM, other=0.0)

        m_i = tl.zeros((Q_TILE,), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((Q_TILE,), dtype=tl.float32)
        acc = tl.zeros((Q_TILE, BLOCK_SIZE_D), dtype=tl.float32)

        num_kv_blocks = tl.cdiv(kv_end - kv_start, BLOCK_SIZE_N)

        for kv_block_id in range(num_kv_blocks):
            kv_block_start = kv_start + kv_block_id * BLOCK_SIZE_N
            kv_block_end = tl.minimum(kv_block_start + BLOCK_SIZE_N, kv_end)
            kv_block_len = kv_block_end - kv_block_start

            logical_page_id = kv_block_start // PAGE_SIZE
            kv_block_start_in_page = kv_block_start % PAGE_SIZE
            physical_page_id = tl.load(
                block_tables_ptr
                + b_id * stride_bt_batch
                + logical_page_id * stride_bt_block
            )

            K_T_block_ptr = tl.make_block_ptr(
                base=(
                    k_cache_ptr
                    + physical_page_id * stride_k_block
                    + kv_head_id * stride_k_head
                    + kv_block_start_in_page * stride_k_blksz
                ),
                shape=(HEAD_DIM, kv_block_len),
                strides=(stride_k_dim, stride_k_blksz),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
                order=(0, 1),
            )
            V_block_ptr = tl.make_block_ptr(
                base=(
                    v_cache_ptr
                    + physical_page_id * stride_v_block
                    + kv_head_id * stride_v_head
                    + kv_block_start_in_page * stride_v_blksz
                ),
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_v_blksz, stride_v_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )

            mask = tl.arange(0, BLOCK_SIZE_N) < kv_block_len

            k_T = tl.load(K_T_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            qk = tl.dot(q, k_T)
            qk = qk * softmax_scale
            qk = tl.where(mask[None, :], qk, float("-inf"))

            m_ij = tl.maximum(
                m_i, tl.max(qk, 1, propagate_nan=True),
                propagate_nan=tl.PropagateNan.ALL,
            )
            qk = qk - m_ij[:, None]
            p = tl.math.exp(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp(m_i - m_ij)

            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None] + tl.dot(p.to(k_T.dtype), v)
            m_i = m_ij

        l_i_safe = tl.where(l_i > 0, l_i, 1.0)
        acc = acc / l_i_safe[:, None]
        lse_i = tl.where(l_i > 0, m_i + tl.math.log(l_i), float("-inf"))

        lse_ptrs = (
            lse_ws_ptr
            + ws_task_idx * stride_lse_task
            + offs_qg * stride_lse_q
        )
        tl.store(lse_ptrs, lse_i)

        acc_ptrs = (
            acc_ws_ptr
            + ws_task_idx * stride_aws_task
            + offs_qg[:, None] * stride_aws_q
            + offs_d[None, :] * stride_aws_d
        )
        tl.store(acc_ptrs, acc, mask=offs_d[None, :] < HEAD_DIM)


@triton.jit
def _mojo_paged_decode_fd_mtp_reduce_kernel(
    acc_ws_ptr, lse_ws_ptr, o_ptr,
    seqlens_ptr, cu_q_lens_ptr,
    stride_aws_task, stride_aws_q, stride_aws_d,
    stride_lse_task, stride_lse_q,
    stride_ob, stride_oh, stride_od,
    BATCH_SIZE,
    KV_SPLIT_PARTS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GAMMA: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    GROUP_SIZE: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    Q_TILE: tl.constexpr = GAMMA * GROUP_SIZE

    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)
    total_reduce_tasks = BATCH_SIZE * NUM_KV_HEADS

    for reduce_task_id in range(pid, total_reduce_tasks, n_progs):
        b_id = reduce_task_id // NUM_KV_HEADS
        kv_head_id = reduce_task_id % NUM_KV_HEADS

        q_start = tl.load(cu_q_lens_ptr + b_id)

        offs_qg = tl.arange(0, Q_TILE)
        gamma_idx = offs_qg // GROUP_SIZE
        g_idx = offs_qg % GROUP_SIZE

        if GQA_INTERLEAVE:
            q_head_ids = kv_head_id + g_idx * NUM_KV_HEADS
        else:
            q_head_ids = kv_head_id * GROUP_SIZE + g_idx

        q_token_pos = q_start + gamma_idx
        offs_d = tl.arange(0, BLOCK_SIZE_D)

        lse_max = tl.zeros((Q_TILE,), dtype=tl.float32) - float("inf")
        for split_idx in tl.static_range(KV_SPLIT_PARTS):
            ws_task_idx = (b_id * NUM_KV_HEADS + kv_head_id) * KV_SPLIT_PARTS + split_idx
            lse_ptrs = (
                lse_ws_ptr
                + ws_task_idx * stride_lse_task
                + offs_qg * stride_lse_q
            )
            lse_max = tl.maximum(lse_max, tl.load(lse_ptrs))

        out = tl.zeros((Q_TILE, BLOCK_SIZE_D), dtype=tl.float32)
        exp_sum = tl.zeros((Q_TILE,), dtype=tl.float32)

        for split_idx in tl.static_range(KV_SPLIT_PARTS):
            ws_task_idx = (b_id * NUM_KV_HEADS + kv_head_id) * KV_SPLIT_PARTS + split_idx

            lse_ptrs = (
                lse_ws_ptr
                + ws_task_idx * stride_lse_task
                + offs_qg * stride_lse_q
            )
            lse = tl.load(lse_ptrs)
            w = tl.math.exp(lse - lse_max)
            exp_sum += w

            acc_ptrs = (
                acc_ws_ptr
                + ws_task_idx * stride_aws_task
                + offs_qg[:, None] * stride_aws_q
                + offs_d[None, :] * stride_aws_d
            )
            acc_split = tl.load(acc_ptrs, mask=offs_d[None, :] < HEAD_DIM, other=0.0)
            out += w[:, None] * acc_split

        exp_sum_safe = tl.where(exp_sum > 0, exp_sum, 1.0)
        out = out / exp_sum_safe[:, None]

        o_ptrs = (
            o_ptr
            + q_token_pos[:, None] * stride_ob
            + q_head_ids[:, None] * stride_oh
            + offs_d[None, :] * stride_od
        )
        tl.store(o_ptrs, out.to(o_ptr.dtype.element_ty), mask=offs_d[None, :] < HEAD_DIM)


@triton.jit
def _mojo_paged_decode_mtp_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, o_ptr,
    seqlens_ptr, cu_q_lens_ptr, block_tables_ptr,
    BATCH_SIZE, NUM_TOTAL_BLOCKS, MAX_NUM_BLOCKS_PER_SEQ,
    stride_qb, stride_qh, stride_qd,
    stride_k_block, stride_k_head, stride_k_blksz, stride_k_dim,
    stride_v_block, stride_v_head, stride_v_blksz, stride_v_dim,
    stride_ob, stride_oh, stride_od,
    stride_bt_batch, stride_bt_block,
    softmax_scale,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    GAMMA: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    GROUP_SIZE: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    Q_TILE: tl.constexpr = GAMMA * GROUP_SIZE
    tl.static_assert(HEAD_DIM <= BLOCK_SIZE_D)
    tl.static_assert(PAGE_SIZE % BLOCK_SIZE_N == 0)

    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)
    num_tasks = BATCH_SIZE * NUM_KV_HEADS

    for kv_task_id in range(pid, num_tasks, n_progs):
        kv_head_id = kv_task_id % NUM_KV_HEADS
        b_id = kv_task_id // NUM_KV_HEADS

        kv_seq_len = tl.load(seqlens_ptr + b_id)
        q_start = tl.load(cu_q_lens_ptr + b_id)

        offs_qg = tl.arange(0, Q_TILE)
        gamma_idx = offs_qg // GROUP_SIZE
        g_idx = offs_qg % GROUP_SIZE

        if GQA_INTERLEAVE:
            q_head_ids = kv_head_id + g_idx * NUM_KV_HEADS
        else:
            q_head_ids = kv_head_id * GROUP_SIZE + g_idx

        q_token_pos = q_start + gamma_idx

        offs_d = tl.arange(0, BLOCK_SIZE_D)
        q_ptrs = (
            q_ptr
            + q_token_pos[:, None] * stride_qb
            + q_head_ids[:, None] * stride_qh
            + offs_d[None, :] * stride_qd
        )
        q = tl.load(q_ptrs, mask=offs_d[None, :] < HEAD_DIM, other=0.0)

        m_i = tl.zeros((Q_TILE,), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((Q_TILE,), dtype=tl.float32)
        acc = tl.zeros((Q_TILE, BLOCK_SIZE_D), dtype=tl.float32)

        num_kv_blocks = tl.cdiv(kv_seq_len, BLOCK_SIZE_N)

        for kv_block_id in range(num_kv_blocks):
            kv_block_start = kv_block_id * BLOCK_SIZE_N
            kv_block_end = min(kv_block_start + BLOCK_SIZE_N, kv_seq_len)
            kv_block_len = kv_block_end - kv_block_start

            logical_page_id = kv_block_start // PAGE_SIZE
            kv_block_start_in_page = kv_block_start % PAGE_SIZE
            physical_page_id = tl.load(
                block_tables_ptr
                + b_id * stride_bt_batch
                + logical_page_id * stride_bt_block
            )

            K_T_block_ptr = tl.make_block_ptr(
                base=(
                    k_cache_ptr
                    + physical_page_id * stride_k_block
                    + kv_head_id * stride_k_head
                    + kv_block_start_in_page * stride_k_blksz
                ),
                shape=(HEAD_DIM, kv_block_len),
                strides=(stride_k_dim, stride_k_blksz),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
                order=(0, 1),
            )
            V_block_ptr = tl.make_block_ptr(
                base=(
                    v_cache_ptr
                    + physical_page_id * stride_v_block
                    + kv_head_id * stride_v_head
                    + kv_block_start_in_page * stride_v_blksz
                ),
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_v_blksz, stride_v_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )

            mask = tl.arange(0, BLOCK_SIZE_N) < kv_block_len

            k_T = tl.load(K_T_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            qk = tl.dot(q, k_T)
            qk *= softmax_scale
            qk = tl.where(mask[None, :], qk, float("-inf"))

            m_ij = tl.maximum(
                m_i, tl.max(qk, 1, propagate_nan=True),
                propagate_nan=tl.PropagateNan.ALL,
            )
            qk = qk - m_ij[:, None]
            p = tl.math.exp(qk)
            p_cast = p.to(k_T.dtype)
            pv = tl.dot(p_cast, v)

            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp(m_i - m_ij)

            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None] + pv
            m_i = m_ij

        if kv_seq_len > 0:
            acc = acc / l_i[:, None]

        o_ptrs = (
            o_ptr
            + q_token_pos[:, None] * stride_ob
            + q_head_ids[:, None] * stride_oh
            + offs_d[None, :] * stride_od
        )
        tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=offs_d[None, :] < HEAD_DIM)


def _mojo_get_num_cores(op_type: str = "vector") -> int:
    props = triton.runtime.driver.active.utils.get_device_properties("npu")
    if op_type == "vector":
        return props["num_vectorcore"]
    return props["num_aicore"]


def _mojo_should_use_flash_decode(
    batch_size: int, num_kv_heads: int, group_size: int,
    max_kv_len: int, cube_num: int,
) -> bool:
    if max_kv_len < 256:
        return False
    loop_times = batch_size * num_kv_heads
    if loop_times >= cube_num:
        return False
    if group_size == 1:
        return True
    return max_kv_len >= 2048


def _mojo_compute_kv_split_parts(
    batch_size: int, num_kv_heads: int, max_kv_len: int, cube_num: int,
) -> int:
    KV_SPLIT_LIMIT = 256
    loop_times = batch_size * num_kv_heads
    max_by_cores = cube_num // max(1, loop_times)
    max_by_len = max_kv_len // KV_SPLIT_LIMIT
    return max(1, min(max_by_cores, max_by_len))


def mojo_paged_mtp_decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_q_lens: torch.Tensor,
    block_tables: torch.Tensor,
    page_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    gamma: int,
    softmax_scale: float,
    max_kv_len: int,
) -> torch.Tensor:
    """
    mojo_opset multi-token paged decode attention.

    query:        [total_q, Hq, D]
    k_cache:      [num_blocks, page_size, Hkv*D]  (FIA 3D)
    v_cache:      [num_blocks, page_size, Hkv*D]
    seq_lens:     [B] int32
    cu_q_lens:    [B+1] int32
    block_tables: [B, max_blocks] int32
    max_kv_len:   int — upper bound on max KV seq length (pre-computed by caller)
    returns:      [total_q, Hq, D]
    """
    total_q, _, _ = query.shape
    batch_size = seq_lens.shape[0]
    num_total_blocks = k_cache.shape[0]
    max_num_blocks_per_seq = block_tables.shape[1]
    group_size = num_q_heads // num_kv_heads
    q_tile = gamma * group_size

    stride_k_block = k_cache.stride(0)
    stride_k_blksz = k_cache.stride(1)
    stride_k_head = head_dim
    stride_k_dim = 1

    stride_v_block = v_cache.stride(0)
    stride_v_blksz = v_cache.stride(1)
    stride_v_head = head_dim
    stride_v_dim = 1

    o = torch.empty_like(query)

    cube_num = _mojo_get_num_cores("cube")
    vector_num = _mojo_get_num_cores("vector")
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    BLOCK_SIZE_N = min(128, triton.next_power_of_2(page_size))

    if _mojo_should_use_flash_decode(
        batch_size, num_kv_heads, group_size, max_kv_len, cube_num
    ):
        kv_split_parts = _mojo_compute_kv_split_parts(
            batch_size, num_kv_heads, max_kv_len, cube_num
        )
        num_ws_tasks = batch_size * num_kv_heads * kv_split_parts
        acc_ws = torch.empty(
            (num_ws_tasks, q_tile, head_dim),
            dtype=torch.float32, device=query.device,
        )
        lse_ws = torch.full(
            (num_ws_tasks, q_tile),
            float("-inf"), dtype=torch.float32, device=query.device,
        )

        _mojo_paged_decode_fd_mtp_kernel[(cube_num,)](
            query, k_cache, v_cache, seq_lens, cu_q_lens, block_tables,
            acc_ws, lse_ws,
            query.stride(0), query.stride(1), query.stride(2),
            stride_k_block, stride_k_head, stride_k_blksz, stride_k_dim,
            stride_v_block, stride_v_head, stride_v_blksz, stride_v_dim,
            block_tables.stride(0), block_tables.stride(1),
            acc_ws.stride(0), acc_ws.stride(1), acc_ws.stride(2),
            lse_ws.stride(0), lse_ws.stride(1),
            softmax_scale,
            batch_size,
            KV_SPLIT_PARTS=kv_split_parts,
            NUM_Q_HEADS=num_q_heads,
            NUM_KV_HEADS=num_kv_heads,
            GQA_INTERLEAVE=False,
            HEAD_DIM=head_dim,
            PAGE_SIZE=page_size,
            GAMMA=gamma,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

        _mojo_paged_decode_fd_mtp_reduce_kernel[(vector_num,)](
            acc_ws, lse_ws, o, seq_lens, cu_q_lens,
            acc_ws.stride(0), acc_ws.stride(1), acc_ws.stride(2),
            lse_ws.stride(0), lse_ws.stride(1),
            o.stride(0), o.stride(1), o.stride(2),
            batch_size,
            KV_SPLIT_PARTS=kv_split_parts,
            NUM_Q_HEADS=num_q_heads,
            NUM_KV_HEADS=num_kv_heads,
            GQA_INTERLEAVE=False,
            HEAD_DIM=head_dim,
            GAMMA=gamma,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
        )
        return o

    _mojo_paged_decode_mtp_kernel[(cube_num,)](
        query, k_cache, v_cache, o,
        seq_lens, cu_q_lens, block_tables,
        batch_size, num_total_blocks, max_num_blocks_per_seq,
        query.stride(0), query.stride(1), query.stride(2),
        stride_k_block, stride_k_head, stride_k_blksz, stride_k_dim,
        stride_v_block, stride_v_head, stride_v_blksz, stride_v_dim,
        o.stride(0), o.stride(1), o.stride(2),
        block_tables.stride(0), block_tables.stride(1),
        softmax_scale,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        GQA_INTERLEAVE=False,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        GAMMA=gamma,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return o
