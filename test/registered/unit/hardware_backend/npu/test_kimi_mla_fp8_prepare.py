"""CPU contracts for the real NPU MLA preparation functions (no NPU import)."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import pytest
import torch
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

SOURCE = (
    Path(__file__).resolve().parents[5]
    / "python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py"
)


def _functions(ns):
    names = {
        "_get_fp8_kv_runtime_scale",
        "forward_mha_prepare_npu",
        "forward_mha_core_npu",
        "forward_mla_prepare_npu",
        "forward_mla_core_npu",
    }
    nodes = [
        node
        for node in ast.parse(SOURCE.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    ns.update(torch=torch, Optional=Optional)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)  # noqa: S102
    return ns


def _case(tokens, fp8=True):
    torch.manual_seed(11)
    heads, q_dim, kv_dim, side_dim, v_dim = 2, 4, 8, 2, 3
    latent = torch.randn(tokens, q_dim + kv_dim + side_dim).bfloat16()
    q_weight = torch.randn(q_dim, heads * (q_dim + side_dim)).bfloat16()
    kv_weight = torch.randn(kv_dim, heads * (q_dim + v_dim)).bfloat16()
    layer = SimpleNamespace(
        kv_cache_dtype="fp8_e4m3" if fp8 else "auto",
        use_dsa=False,
        _modelslim_fp8_kv_scale_ready=True,
        fak_descale_float=torch.tensor([[0.125]], dtype=torch.float32),
        fak_descale_reciprocal=torch.tensor([[8.0]], dtype=torch.float32),
        q_lora_rank=q_dim,
        kv_lora_rank=kv_dim,
        qk_nope_head_dim=q_dim,
        qk_rope_head_dim=side_dim,
        qk_head_dim=q_dim + side_dim,
        v_head_dim=v_dim,
        num_local_heads=heads,
        q_a_layernorm=lambda x: x,
        kv_a_layernorm=lambda x: x,
        q_b_proj=lambda x: (x @ q_weight, None),
        kv_b_proj=lambda x: (x @ kv_weight, None),
        w_kc=torch.randn(heads, q_dim, kv_dim).bfloat16(),
        w_vc=torch.randn(heads, kv_dim, v_dim).bfloat16(),
        rotary_emb=None,
        use_deepseek_yarn_rope=False,
        _disable_npu_fused_split_qk_norm=True,
        layer_id=3,
        o_proj=lambda x: (x, None),
        _concat_and_cast_mha_k=lambda k, side, batch: torch.cat(
            (k, side.expand(-1, heads, -1)), dim=-1
        ),
    )
    quant_inputs = []

    def dynamic_quant(x, dst_type):
        quant_inputs.append(x.clone())
        x = x.float()
        scale = x.abs().amax(dim=-1).clamp_min(1e-6) / 448.0
        return (x / scale[:, None]).clamp(-448, 448).to(dst_type), scale

    pool = SimpleNamespace(set_kv_buffer=Mock())
    ns = _functions(
        {
            "torch_npu": SimpleNamespace(npu_dynamic_quant=dynamic_quant),
            "_use_ag_after_qlora": False,
            "get_attn_tp_context": lambda: SimpleNamespace(
                fetch_qkv_latent=lambda: latent
            ),
            "get_token_to_kv_pool": lambda: pool,
            "dsa_use_prefill_cp": lambda batch: False,
            "is_mla_preprocess_enabled": lambda: False,
        }
    )
    batch = SimpleNamespace(out_cache_loc=torch.arange(tokens) + 17)
    args = (layer, torch.arange(tokens), latent, batch, None, None)
    return ns, args, layer, pool, latent, quant_inputs


@pytest.mark.parametrize("tokens", [1, 6])
def test_nope_decode_and_multitoken_verify_quantize_q_and_write_kv_once(tokens):
    ns, args, layer, pool, latent, quant_inputs = _case(tokens)
    state = ns["forward_mla_prepare_npu"](*args)
    q_side, k_side, q, k, _, _, _, topk, scale = state
    assert q.dtype == torch.float8_e4m3fn
    assert q.shape == (tokens, 2, 8)
    assert scale.shape == (tokens, 2, 1)
    assert scale.dtype == torch.float32
    assert q_side.dtype == k_side.dtype == torch.bfloat16
    assert topk is None
    assert len(quant_inputs) == 1
    assert quant_inputs[0].shape == (tokens * 2, 8)
    original_q = layer.q_b_proj(latent[:, :4])[0].view(tokens, 2, 6)
    expected_side = (original_q[..., 4:] / scale / layer.fak_descale_float).bfloat16()
    torch.testing.assert_close(q_side, expected_side, rtol=0, atol=0)
    # NoPE is an independent unrotated branch; never quantize the stored side.
    torch.testing.assert_close(k_side[:, 0], latent[:, 12:], rtol=0, atol=0)
    pool.set_kv_buffer.assert_called_once()
    saved_layer, saved_slots, saved_k, saved_side = pool.set_kv_buffer.call_args.args
    assert saved_layer is layer
    torch.testing.assert_close(saved_slots, args[3].out_cache_loc)
    torch.testing.assert_close(saved_k, k)
    torch.testing.assert_close(saved_side, k_side)


def test_c8_nope_bypasses_rope_dependent_mlaprolog():
    ns, args, _, _, _, _ = _case(2)
    ns["is_mla_preprocess_enabled"] = lambda: True
    ns["NPUFusedMLAPreprocess"] = Mock(side_effect=AssertionError("No RoPE cache"))
    assert ns["forward_mla_prepare_npu"](*args)[2].dtype == torch.float8_e4m3fn
    ns["NPUFusedMLAPreprocess"].assert_not_called()


def test_side_compensation_preserves_the_unquantized_attention_term():
    # CPU algebra check of the PR's FIA contract, not an NPU kernel accuracy test.
    ns, args, layer, _, latent, _ = _case(6)
    side, k_side, q, k, *_, q_scale = ns["forward_mla_prepare_npu"](*args)
    kv_scale = layer.fak_descale_float.reshape(())
    k8 = (k.float() / kv_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    dot = torch.einsum("thd,sd->ths", q.float(), k8[:, 0].float())
    side_dot = torch.einsum("thd,sd->ths", side.float(), k_side[:, 0].float())
    fused_logits = (dot + side_dot) * q_scale * kv_scale
    original_side = layer.q_b_proj(latent[:, :4])[0].view(6, 2, 6)[..., 4:]
    expected_logits = dot * q_scale * kv_scale + torch.einsum(
        "thd,sd->ths", original_side.float(), k_side[:, 0].float()
    )
    # Only rounding of the compensated BF16 side differs from the algebra.
    side_rounding = (side.float() * q_scale * kv_scale - original_side.float()).abs()
    bound = torch.einsum("thd,sd->ths", side_rounding, k_side[:, 0].float().abs())
    assert torch.all((fused_logits - expected_logits).abs() <= bound + 2e-5)


def test_prefill_keeps_bf16_compute_and_passes_scale_for_prefix_reads():
    ns, args, layer, pool, _, quant_inputs = _case(4)
    state = ns["forward_mha_prepare_npu"](*args)
    q, k, v, _, scale = state
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert scale is layer.fak_descale_float
    assert quant_inputs == []
    pool.set_kv_buffer.assert_called_once()
    layer.attn_mha = Mock(return_value=torch.zeros(4, 2, 3, dtype=torch.bfloat16))
    ns["forward_mha_core_npu"](layer, *state)
    assert layer.attn_mha.call_args.kwargs == {
        "save_kv_cache": False,
        "fp8_kv_scale": scale,
    }


def test_core_propagates_scales_and_disables_second_cache_write():
    ns, args, layer, pool, _, _ = _case(6)
    state = ns["forward_mla_prepare_npu"](*args)
    layer.attn_mqa = Mock(return_value=torch.zeros(6, 2, 8, dtype=torch.bfloat16))
    ns["torch_npu"].npu_transpose_batchmatmul = lambda x, w, **_: torch.einsum(
        "thd,hdv->thv", x.float(), w.float()
    ).bfloat16()
    result = ns["forward_mla_core_npu"](layer, *state)
    assert result.dtype == torch.bfloat16
    assert result.shape == (6, 6)
    kwargs = layer.attn_mqa.call_args.kwargs
    assert kwargs["save_kv_cache"] is False
    assert kwargs["fp8_kv_scale"] is layer.fak_descale_float
    assert kwargs["dequant_scale_q_nope"] is state[-1]
    pool.set_kv_buffer.assert_called_once()


def test_bf16_prepare_does_not_quantize_or_change_cache_ownership():
    ns, args, _, pool, _, quant_inputs = _case(2, fp8=False)
    state = ns["forward_mla_prepare_npu"](*args)
    assert state[2].dtype == torch.bfloat16
    assert state[-1] is None
    assert quant_inputs == []
    pool.set_kv_buffer.assert_not_called()


def test_missing_checkpoint_scale_is_not_replaced_with_one():
    ns, args, layer, pool, _, _ = _case(2)
    layer._modelslim_fp8_kv_scale_ready = False
    with pytest.raises(RuntimeError, match="loaded ModelSlim"):
        ns["forward_mla_prepare_npu"](*args)
    pool.set_kv_buffer.assert_not_called()


def test_dsa_keeps_its_separate_quantization_contract():
    ns, _, layer, _, _, _ = _case(1)
    layer.use_dsa = True
    layer._modelslim_fp8_kv_scale_ready = False
    assert ns["_get_fp8_kv_runtime_scale"](layer, "fak_descale_float") is None
