"""CPU contracts for K3's mixed MXFP8 QKV / BF16 G projection.

Execute real model construction, dispatch and weight-loader methods via AST;
mock only unrelated modules and NPU allocation/compute. No device-kernel or
full-checkpoint numerical validation is implied by these tests.
"""

import ast
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "python/sglang").is_dir()
)
MODEL = ROOT / "python/sglang/srt/models/kimi_k3.py"
LINEAR = ROOT / "python/sglang/srt/layers/linear.py"
QUANT = ROOT / "python/sglang/srt/layers/quantization/modelslim/modelslim.py"
SCHEME = QUANT.parent / "schemes/modelslim_mxfp8.py"


def _class(source, name, methods, namespace, base="object"):
    original = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == name
    )
    cls = ast.ClassDef(
        name=name,
        bases=[ast.parse(base, mode="eval").body],
        keywords=[],
        body=[
            n
            for n in original.body
            if isinstance(n, ast.FunctionDef) and n.name in methods
        ],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                cls,
            ],
            type_ignores=[],
        )
    )
    exec(compile(module, str(source), "exec"), namespace)  # noqa: S102
    return namespace[name]


def _parameter(data, **attrs):
    param = torch.nn.Parameter(data, requires_grad=False)
    for name, value in attrs.items():
        setattr(param, name, value)
    return param


def _case(*, npu=True, kinds=None, full_rank=True, rank=0, tp=32, attn_tp=8):
    ns = {
        "torch": torch,
        "nn": torch.nn,
        "MappingProxyType": MappingProxyType,
        "_is_npu": npu,
        # Exercise the NPU loader's simple narrow/copy branch on CPU tensors.
        "_is_cpu": False,
        "_disable_hip_linear_quant": False,
        "divide": lambda x, n: x // n,
    }
    quant_cls = _class(
        QUANT,
        "ModelSlimConfig",
        {"is_layer_skipped", "_resolve_quant_prefix", "_quant_prefix_candidates"},
        ns,
    )
    quant = None
    if kinds is not None:
        quant = quant_cls()
        quant.quant_description = {
            f"language_model.model.layers.0.self_attn.{name}_proj.weight": kind
            for name, kind in zip(("q", "k", "v", "g"), kinds)
        }

    # Only allocate fake module weights here. Projection sizing/sharding for
    # QKV and the actual copy of checkpoint shards use the real linear code.
    column_loader = _class(LINEAR, "ColumnParallelLinear", {"weight_loader"}, ns)

    class Column(torch.nn.Module):
        weight_loader = column_loader.weight_loader

        def __init__(self, input_size, output_size, **kw):
            super().__init__()
            self.tp_rank = kw.get("tp_rank", 0)
            self.tp_size = kw.get("tp_size", 1)
            self.use_presharded_weights = kw.get("use_presharded_weights", False)
            self.prefix = kw.get("prefix", "")
            self.weight = _parameter(
                torch.zeros(
                    output_size // self.tp_size,
                    input_size,
                    dtype=kw.get("params_dtype") or torch.bfloat16,
                ),
                output_dim=0,
                weight_loader=self.weight_loader,
            )
            self.bias = None

        def forward(self, x):
            return torch.nn.functional.linear(x, self.weight), None

    class Merged(Column):
        def __init__(self, input_size, output_sizes, **kw):
            super().__init__(input_size, sum(output_sizes), **kw)

    ns["ColumnParallelLinear"] = Column
    _class(
        LINEAR,
        "QKVParallelLinear",
        {"__init__", "weight_loader"},
        ns,
        "ColumnParallelLinear",
    )
    ns.update(
        get_parallel=lambda: SimpleNamespace(
            tp_size=tp, attn_tp_size=attn_tp, attn_tp_rank=rank
        ),
        _uses_modelopt_fp8_pb_wo=lambda *args: False,
        MergedColumnParallelLinear=Merged,
        ReplicatedLinear=Column,
        RowParallelLinear=Column,
        MergedColumnParallelRepeatedLinear=lambda *args, **kw: SimpleNamespace(),
        ColumnParallelBatchedLinear=lambda *args, **kw: SimpleNamespace(),
        set_weight_attrs=lambda p, attrs: [setattr(p, k, v) for k, v in attrs.items()],
        sharded_weight_loader=lambda *args: Mock(),
        FusedRMSNormGated=lambda *args, **kw: SimpleNamespace(),
        k3_gemm_ar=SimpleNamespace(maybe_wrap_o_proj=lambda *args: None),
        RadixLinearAttention=lambda **kw: SimpleNamespace(**kw),
    )
    cls = _class(
        MODEL,
        "KimiK3DeltaAttention",
        {"__init__", "forward", "forward_qkvbfg"},
        ns,
        "nn.Module",
    )
    config = SimpleNamespace(
        # Small tensors, but the production TP32 / attention-TP8 split.
        linear_attn_config={
            "head_dim": 4,
            "num_heads": 16,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": full_rank,
        },
        v_head_dim=4,
        dtype=torch.bfloat16,
        is_moe=False,
        is_linear_attn=True,
        num_hidden_layers=1,
        is_kda_layer=lambda layer: layer == 0,
    )
    layer = cls(
        0,
        64,
        config,
        quant_config=quant,
        prefix="language_model.model.layers.0.self_attn",
    )
    return layer, config, ns


MIXED = ("W8A8_MXFP8", "W8A8_MXFP8", "W8A8_MXFP8", "FLOAT")


@pytest.mark.parametrize("rank", [0, 7])
def test_mixed_qkv_and_g_construct_separately_with_attention_tp(rank):
    layer, _, _ = _case(kinds=MIXED, rank=rank)
    assert layer.use_full_rank_gate and not layer.do_fuse_qkvg
    assert not hasattr(layer, "fused_qkvg_proj")
    assert hasattr(layer, "qkv_proj") and hasattr(layer, "g_proj")
    assert not hasattr(layer, "g_a_proj")
    for module in (layer.qkv_proj, layer.g_proj, layer.b_proj, layer.f_b_proj):
        assert (module.tp_rank, module.tp_size) == (rank, 8)
    assert layer.qkv_proj.num_heads == layer.qkv_proj.num_kv_heads == 2
    assert layer.qkv_proj.weight.shape == (24, 64)


@pytest.mark.parametrize(
    "npu,kinds",
    [(True, ("FLOAT",) * 4), (True, ("W8A8_MXFP8",) * 4), (False, MIXED), (True, None)],
)
def test_homogeneous_quantization_gpu_and_unquantized_keep_full_rank_fusion(npu, kinds):
    layer, _, _ = _case(npu=npu, kinds=kinds)
    assert layer.do_fuse_qkvg and layer.use_full_rank_gate
    assert hasattr(layer, "fused_qkvg_proj") and not hasattr(layer, "g_proj")


@pytest.mark.parametrize("quantized,tp,attn_tp", [(True, 32, 8), (False, 8, 8)])
def test_low_rank_gate_keeps_existing_projection_path(quantized, tp, attn_tp):
    layer, _, _ = _case(
        kinds=MIXED if quantized else None, full_rank=False, tp=tp, attn_tp=attn_tp
    )
    assert not layer.use_full_rank_gate and not layer.do_fuse_qkvg
    if quantized:
        assert not layer.do_fuse_qkvbfg
        assert hasattr(layer, "g_a_proj") and hasattr(layer, "g_b_proj")
    else:
        assert layer.do_fuse_qkvbfg and hasattr(layer, "fused_qkvbfg_a_proj")


def test_forward_dispatch_uses_split_and_keeps_full_rank_g_independent():
    layer, _, _ = _case(kinds=MIXED)

    class StopAfterProjection(Exception):
        pass

    # Stop at the projection boundary; no KDA backend needs to be imported.
    layer.forward_qkvbfg = Mock(side_effect=StopAfterProjection)
    layer.forward_qkvbfg_fused = Mock(side_effect=AssertionError("unexpected fusion"))
    x = torch.randn(2, 64)
    with pytest.raises(StopAfterProjection):
        layer(x, None, None, None)
    layer.forward_qkvbfg.assert_called_once_with(x)
    layer.forward_qkvbfg_fused.assert_not_called()

    # Also execute the real split projection function (including G dispatch).
    for name in ("qkv_proj", "b_proj", "f_a_proj", "f_b_proj", "g_proj"):
        module = getattr(layer, name)
        module.forward = Mock(return_value=(torch.full((2, 3), len(name)), None))
    qkv, beta, forget, gate = type(layer).forward_qkvbfg(layer, x)
    assert qkv is layer.qkv_proj.forward.return_value[0]
    assert beta is layer.b_proj.forward.return_value[0]
    assert forget is layer.f_b_proj.forward.return_value[0]
    assert gate is layer.g_proj.forward.return_value[0]
    layer.g_proj.forward.assert_called_once_with(x)
    layer.f_b_proj.forward.assert_called_once_with(
        layer.f_a_proj.forward.return_value[0]
    )


@pytest.mark.parametrize("rank", [0, 7])
def test_real_loader_maps_qkv_weight_and_scale_and_keeps_g_name(rank):
    layer, config, ns = _case(kinds=MIXED, rank=rank)
    ns.update(
        ModelWeightParameter=_parameter,
        GroupQuantScaleParameter=_parameter,
        MXFP8_BLOCK_SIZE=32,
    )
    scheme = _class(SCHEME, "ModelSlimMXFP8Scheme", {"create_weights"}, ns)()
    # Real ModelSlim allocation contract, real QKV loader; no NPU post-load
    # layout conversion or matmul is mocked as though it were validated.
    del layer.qkv_proj.weight
    scheme.create_weights(
        layer.qkv_proj,
        64,
        [8, 8, 8],
        64,
        192,
        torch.bfloat16,
        weight_loader=layer.qkv_proj.weight_loader,
    )
    ns.update(
        get_layer_id=lambda name: int(name.split(".")[2]),
        maybe_remap_kv_scale_name=lambda name, params: name,
        default_weight_loader=lambda param, value, **kw: param.data.copy_(value),
    )
    scale_mapper = next(
        n
        for n in ast.parse(MODEL.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_maybe_map_fp8_pb_scale_name"
    )
    exec(  # noqa: S102
        compile(ast.Module(body=[scale_mapper], type_ignores=[]), str(MODEL), "exec"),
        ns,
    )
    model_cls = _class(MODEL, "KimiK3LinearForCausalLM", {"load_weights"}, ns)
    model = model_cls()
    model.config = config
    model.model = SimpleNamespace(
        layers=[SimpleNamespace(self_attn=layer)], start_layer=0, end_layer=1
    )
    params = {
        f"model.layers.0.self_attn.{name}": param
        for name, param in layer.named_parameters()
    }
    model.named_parameters = lambda: params.items()
    model.post_load_weights = Mock()

    weights, expected_weight, expected_scale = [], [], []
    for i, name in enumerate(("q", "k", "v")):
        base = f"model.layers.0.self_attn.{name}_proj"
        weight = ((torch.arange(64 * 64).reshape(64, 64) // 64 + i) % 16).to(
            torch.float8_e4m3fn
        )
        scale = ((torch.arange(64 * 2).reshape(64, 2) + 31 * i) % 256).to(torch.uint8)
        weights.extend(
            [
                (base + ".weight", weight),
                (base + ".weight_scale", scale[:, None, :, None]),
            ]
        )
        expected_weight.append(weight[rank * 8 : (rank + 1) * 8])
        expected_scale.append(scale[rank * 8 : (rank + 1) * 8])
    g_weight = torch.arange(64 * 64).reshape(64, 64).bfloat16()
    weights.append(("model.layers.0.self_attn.g_proj.weight", g_weight))

    loaded = model.load_weights(weights)
    assert loaded == {
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.self_attn.qkv_proj.weight_scale",
        "model.layers.0.self_attn.g_proj.weight",
    }
    assert layer.qkv_proj.weight.dtype == torch.float8_e4m3fn
    assert layer.qkv_proj.weight_scale.dtype == torch.uint8
    assert layer.g_proj.weight.dtype == torch.bfloat16
    torch.testing.assert_close(
        layer.qkv_proj.weight.float(), torch.cat(expected_weight).float()
    )
    torch.testing.assert_close(layer.qkv_proj.weight_scale, torch.cat(expected_scale))
    torch.testing.assert_close(layer.g_proj.weight, g_weight[rank * 8 : (rank + 1) * 8])
    model.post_load_weights.assert_called_once()
