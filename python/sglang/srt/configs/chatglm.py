# Adapted from
# https://github.com/THUDM/ChatGLM2-6B
# https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/configs/chatglm.py

# ChatGLM2 and ChatGLM3 share the same config.
# ChatGLM4 is officially supported by Huggingface
# transformers >= 4.46.0 is required
# https://huggingface.co/docs/transformers/en/model_doc/glm
from transformers import PretrainedConfig


class Glm4MoeLiteConfig(PretrainedConfig):
    """Configuration class for GLM-4.7-Flash (Glm4MoeLite) model.

    This model uses MLA (Multi-Latent Attention) and MoE (Mixture of Experts)
    architecture, similar to DeepSeek V2/V3.
    """

    model_type = "glm4_moe_lite"

    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 10240,
        moe_intermediate_size: int = 1536,
        num_hidden_layers: int = 47,
        num_attention_heads: int = 20,
        num_key_value_heads: int = 20,
        n_routed_experts: int = 64,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 4,
        first_k_dense_replace: int = 1,
        moe_layer_freq: int = 1,
        n_group: int = 1,
        topk_group: int = 1,
        topk_method: str = "noaux_tc",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.8,
        q_lora_rank: int = 768,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 192,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 256,
        hidden_act: str = "silu",
        max_position_embeddings: int = 202752,
        rms_norm_eps: float = 1e-5,
        vocab_size: int = 154880,
        rope_theta: float = 1000000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        num_nextn_predict_layers: int = 1,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
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
        self.moe_layer_freq = moe_layer_freq
        self.n_group = n_group
        self.topk_group = topk_group
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.vocab_size = vocab_size
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)


class ChatGLMConfig(PretrainedConfig):
    model_type = "chatglm"
    attribute_map = {
        "num_hidden_layers": "num_layers",
        "n_head_kv": "multi_query_group_num",
    }

    def __init__(
        self,
        num_layers=28,
        padded_vocab_size=65024,
        hidden_size=4096,
        ffn_hidden_size=13696,
        kv_channels=128,
        num_attention_heads=32,
        seq_length=2048,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        layernorm_epsilon=1e-5,
        rmsnorm=True,
        apply_residual_connection_post_layernorm=False,
        post_layer_norm=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        interleaved_qkv=False,
        bias_dropout_fusion=True,
        multi_query_attention=False,
        multi_query_group_num=1,
        apply_query_key_layer_scaling=True,
        attention_softmax_in_fp32=True,
        fp32_residual_connection=False,
        quantization_bit=0,
        pre_seq_len=None,
        prefix_projection=False,
        **kwargs
    ):
        self.num_layers = num_layers
        self.vocab_size = padded_vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        # It is to be compatible with long lora.
        self.max_position_embeddings = seq_length
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.layernorm_epsilon = layernorm_epsilon
        self.rmsnorm = rmsnorm
        self.apply_residual_connection_post_layernorm = (
            apply_residual_connection_post_layernorm
        )
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        self.interleaved_qkv = interleaved_qkv
        super().__init__(**kwargs)
