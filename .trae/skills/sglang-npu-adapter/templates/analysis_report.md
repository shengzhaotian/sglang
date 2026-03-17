# Model Architecture Analysis Report

## 1. Basic Information
- Model Name: [Required]
- Architecture Type: [Dense/MoE/MoE-MLA/VLM]
- Architecture Name: [Exactly match the architectures field in HF config]

## 2. Model Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| hidden_size | | |
| num_hidden_layers | | |
| num_attention_heads | | |
| num_key_value_heads | | |
| intermediate_size | | |
| vocab_size | | |
| max_position_embeddings | | |

## 3. MoE Configuration (Required for MoE models)
| Parameter | Value | Description |
|-----------|-------|-------------|
| n_routed_experts | | Number of routed experts |
| n_shared_experts | | Number of shared experts |
| num_experts_per_tok | | Number of experts activated per token |
| moe_intermediate_size | | Expert intermediate layer size |

## 4. MLA Configuration (Required for MLA models)
| Parameter | Value | Description |
|-----------|-------|-------------|
| q_lora_rank | | Query LoRA rank |
| kv_lora_rank | | KV LoRA rank |
| qk_nope_head_dim | | QK non-RoPE dimension |
| qk_rope_head_dim | | QK RoPE dimension |
| v_head_dim | | Value dimension |

## 5. Parallel Configuration Derivation
### Derivation Process
1. Derived min_tp=xxx based on hidden_size=xxx
2. Based on n_experts=xxx, optional EP values are [...]
3. Selected TP=xxx, EP=xxx that satisfies TP % EP == 0
4. Verified device count: need xxx, available xxx

### Configuration Results
| Parameter | Recommended Value | Description |
|-----------|-------------------|-------------|
| TP | | Tensor Parallel |
| EP | | Expert Parallel |
| PP | | Pipeline Parallel |
| Total Devices | | TP × PP |

### Constraint Validation
| Check Item | Result | Details |
|------------|--------|---------|
| Device Count | ✅/❌ | |
| TP/EP Divisibility | ✅/❌ | |
| Expert Distribution | ✅/❌ | |
| Attention Head | ✅/❌ | |

## 6. Resource Evaluation
- Weight Size: xxx GB
- Minimum Devices: xxx
- Recommended Devices: xxx
- Memory Per Device: xxx GB

## 7. NPU Compatibility
- Compatibility: ✅/⚠️/❌
- Attention Backend: ascend
- MoE Backend: fused_moe
- Known Issues: [...]
- Workarounds: [...]

## 8. Risk Assessment
| Level | Category | Description | Mitigation |
|-------|----------|-------------|------------|

## 9. Reference Model
- SGLang Implementation: [File path]
- Similarity: high/medium/low
- Main Differences: [...]

## 10. Next Action
- proceed: Enter code modification phase
- call_agent2: Need debugging support