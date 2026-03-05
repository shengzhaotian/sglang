# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Custom configuration for GLM4 MoE Lite model."""

from transformers import PretrainedConfig


class Glm4MoeLiteConfig(PretrainedConfig):
    """Configuration for GLM4 MoE Lite model (GLM-4.7-flash)."""

    model_type = "glm4_moe_lite"

    def __init__(
        self,
        vocab_size=154880,
        hidden_size=2048,
        intermediate_size=10240,
        moe_intermediate_size=1536,
        num_hidden_layers=47,
        num_attention_heads=20,
        num_key_value_heads=20,
        n_routed_experts=64,
        n_shared_experts=1,
        num_experts_per_tok=4,
        first_k_dense_replace=1,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.8,
        q_lora_rank=768,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        hidden_act="silu",
        max_position_embeddings=202752,
        rms_norm_eps=1e-5,
        rope_theta=1000000,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        partial_rotary_factor=1.0,
        tie_word_embeddings=False,
        pad_token_id=154820,
        eos_token_id=None,
        num_nextn_predict_layers=None,
        topk_method="noaux_tc",
        norm_topk_prob=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.partial_rotary_factor = partial_rotary_factor
        self.tie_word_embeddings = tie_word_embeddings
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob

        if eos_token_id is None:
            eos_token_id = [154820, 154827, 154829]

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
