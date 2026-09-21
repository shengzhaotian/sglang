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


def _packed_mapping(class_name):
    cls = next(
        n
        for n in ast.parse(MODEL.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    return ast.literal_eval(
        next(
            n.value
            for n in cls.body
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "packed_modules_mapping"
                for t in n.targets
            )
        )
    )


def _case(
    *, npu=True, kinds=None, full_rank=True, rank=0, tp=32, attn_tp=8, split_qkvg=True
):
    ns = {
        "torch": torch,
        "nn": torch.nn,
        "MappingProxyType": MappingProxyType,
        "_is_npu": npu,
        # Exercise the NPU loader's simple narrow/copy branch on CPU tensors.
        "_is_cpu": False,
        "_disable_hip_linear_quant": False,
        "divide": lambda x, n: x // n,
        "envs": SimpleNamespace(
            SGLANG_K3_SPLIT_QKVG=SimpleNamespace(get=lambda: split_qkvg),
            SGLANG_NPU_W4A4_NEW_PACKING=SimpleNamespace(get=lambda: False),
        ),
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
    packed_mapping = _packed_mapping("KimiK3ForConditionalGeneration")

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
            quant_config = kw.get("quant_config")
            if npu and isinstance(quant_config, quant_cls):
                self.quant_skipped = quant_config.is_layer_skipped(
                    self.prefix, packed_mapping
                )
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

    ns["ColumnParallelLinear"] = Column
    _class(
        LINEAR,
        "MergedColumnParallelLinear",
        {"__init__", "weight_loader"},
        ns,
        "ColumnParallelLinear",
    )
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
        {"__init__", "forward", "forward_qkvbfg", "forward_qkvbfg_fused"},
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


def test_split_flag_defaults_on():
    tree = ast.parse((ROOT / "python/sglang/srt/environ.py").read_text())
    setting = next(
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "SGLANG_K3_SPLIT_QKVG"
            for t in n.targets
        )
    )
    assert isinstance(setting, ast.Call) and setting.func.id == "EnvBool"
    assert ast.literal_eval(setting.args[0]) is True


@pytest.mark.parametrize("rank", [0, 7])
def test_mixed_qkv_and_g_construct_separately_with_attention_tp(rank):
    layer, _, _ = _case(kinds=MIXED, rank=rank)
    assert layer.use_full_rank_gate
    assert not hasattr(layer, "fused_qkvg_proj")
    assert hasattr(layer, "fused_qkv_proj") and hasattr(layer, "g_proj")
    assert not hasattr(layer, "g_a_proj")
    for module in (layer.fused_qkv_proj, layer.g_proj, layer.b_proj, layer.f_b_proj):
        assert (module.tp_rank, module.tp_size) == (rank, 8)
    assert layer.fused_qkv_proj.output_sizes == [64, 64, 64]
    assert layer.fused_qkv_proj.weight.shape == (24, 64)
    assert not layer.fused_qkv_proj.quant_skipped and layer.g_proj.quant_skipped


@pytest.mark.parametrize("rank", [0, 7])
def test_disabled_split_rejects_mixed_modelslim_shards(rank):
    with pytest.raises(ValueError, match="some but not all shards"):
        _case(kinds=MIXED, rank=rank, split_qkvg=False)


@pytest.mark.parametrize(
    "class_name", ["KimiK3LinearForCausalLM", "KimiK3ForConditionalGeneration"]
)
def test_packed_mappings_describe_both_full_rank_layouts(class_name):
    mapping = _packed_mapping(class_name)
    assert mapping["fused_qkv_proj"] == ["q_proj", "k_proj", "v_proj"]
    assert mapping["fused_qkvg_proj"] == ["q_proj", "k_proj", "v_proj", "g_proj"]


@pytest.mark.parametrize(
    "npu,kinds",
    [(True, ("FLOAT",) * 4), (True, ("W8A8_MXFP8",) * 4), (False, None), (True, None)],
)
@pytest.mark.parametrize("split_qkvg", [True, False])
@pytest.mark.parametrize("rank", [0, 7])
def test_full_rank_layout_follows_split_flag(npu, kinds, split_qkvg, rank):
    layer, _, _ = _case(npu=npu, kinds=kinds, split_qkvg=split_qkvg, rank=rank)
    assert layer.use_full_rank_gate
    projection = layer.fused_qkv_proj if split_qkvg else layer.fused_qkvg_proj
    assert (projection.tp_rank, projection.tp_size) == (rank, 8)
    assert projection.weight.shape == (24 if split_qkvg else 32, 64)
    assert hasattr(layer, "g_proj") is split_qkvg
    assert hasattr(layer, "fused_qkvg_proj") is not split_qkvg


@pytest.mark.parametrize("quantized,tp,attn_tp", [(True, 32, 8), (False, 8, 8)])
@pytest.mark.parametrize("split_qkvg", [True, False])
def test_low_rank_gate_keeps_existing_projection_path(
    quantized, tp, attn_tp, split_qkvg
):
    layer, _, _ = _case(
        kinds=MIXED if quantized else None,
        full_rank=False,
        tp=tp,
        attn_tp=attn_tp,
        split_qkvg=split_qkvg,
    )
    assert not layer.use_full_rank_gate
    if quantized:
        assert not layer.do_fuse_qkvbfg
        assert hasattr(layer, "g_a_proj") and hasattr(layer, "g_b_proj")
    else:
        assert layer.do_fuse_qkvbfg and hasattr(layer, "fused_qkvbfg_a_proj")
    x = torch.randn(2, 64)
    layer.forward_qkvbfg = Mock(side_effect=StopAfterProjection)
    layer.forward_qkvbfg_fused = Mock(side_effect=StopAfterProjection)
    with pytest.raises(StopAfterProjection):
        layer(x, None, None, None)
    expected, unused = (
        (layer.forward_qkvbfg, layer.forward_qkvbfg_fused)
        if quantized
        else (layer.forward_qkvbfg_fused, layer.forward_qkvbfg)
    )
    expected.assert_called_once_with(x)
    unused.assert_not_called()


class StopAfterProjection(Exception):
    pass


@pytest.mark.parametrize("split_qkvg", [True, False])
def test_full_rank_forward_dispatch_and_projection_follow_split_flag(split_qkvg):
    layer, _, _ = _case(
        kinds=MIXED if split_qkvg else ("W8A8_MXFP8",) * 4, split_qkvg=split_qkvg
    )
    # Stop at the projection boundary; no KDA backend needs to be imported.
    layer.forward_qkvbfg = Mock(side_effect=AssertionError("unexpected fallback"))
    layer.forward_qkvbfg_fused = Mock(side_effect=StopAfterProjection)
    x = torch.randn(2, 64)
    with pytest.raises(StopAfterProjection):
        layer(x, None, None, None)
    layer.forward_qkvbfg_fused.assert_called_once_with(x)
    layer.forward_qkvbfg.assert_not_called()

    # Execute the real full-rank helper without the optional device BFA GEMM.
    for name in ("b_proj", "f_a_proj", "f_b_proj"):
        module = getattr(layer, name)
        module.forward = Mock(return_value=(torch.full((2, 3), len(name)), None))
    projection = layer.fused_qkv_proj if split_qkvg else layer.fused_qkvg_proj
    projected = torch.arange(2 * (24 if split_qkvg else 32)).reshape(2, -1)
    projection.forward = Mock(return_value=(projected, None))
    if split_qkvg:
        layer.g_proj.forward = Mock(return_value=(torch.ones(2, 8), None))
    qkv, beta, forget, gate = type(layer).forward_qkvbfg_fused(layer, x)
    projection.forward.assert_called_once_with(x)
    torch.testing.assert_close(qkv, projected[:, :24])
    if split_qkvg:
        assert gate is layer.g_proj.forward.return_value[0]
        layer.g_proj.forward.assert_called_once_with(x)
    else:
        torch.testing.assert_close(gate, projected[:, 24:])
    assert beta is layer.b_proj.forward.return_value[0]
    assert forget is layer.f_b_proj.forward.return_value[0]
    layer.f_b_proj.forward.assert_called_once_with(
        layer.f_a_proj.forward.return_value[0]
    )


@pytest.mark.parametrize("rank", [0, 7])
@pytest.mark.parametrize("split_qkvg", [True, False])
def test_real_loader_maps_projection_weights_and_scales_for_both_layouts(
    rank, split_qkvg
):
    layer, config, ns = _case(
        kinds=MIXED if split_qkvg else ("W8A8_MXFP8",) * 4,
        rank=rank,
        split_qkvg=split_qkvg,
    )
    projection_name = "fused_qkv_proj" if split_qkvg else "fused_qkvg_proj"
    projection = getattr(layer, projection_name)
    shard_names = ("q", "k", "v") if split_qkvg else ("q", "k", "v", "g")
    ns.update(
        ModelWeightParameter=_parameter,
        GroupQuantScaleParameter=_parameter,
        MXFP8_BLOCK_SIZE=32,
    )
    scheme = _class(SCHEME, "ModelSlimMXFP8Scheme", {"create_weights"}, ns)()
    # Real ModelSlim allocation contract and merged-column shard loader;
    # no NPU post-load layout conversion or matmul is claimed as validated.
    del projection.weight
    scheme.create_weights(
        projection,
        64,
        [8] * len(shard_names),
        64,
        64 * len(shard_names),
        torch.bfloat16,
        weight_loader=projection.weight_loader,
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
    for i, name in enumerate(shard_names):
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
    if split_qkvg:
        g_weight = torch.arange(64 * 64).reshape(64, 64).bfloat16()
        weights.append(("model.layers.0.self_attn.g_proj.weight", g_weight))

    loaded = model.load_weights(weights)
    expected_loaded = {
        f"model.layers.0.self_attn.{projection_name}.weight",
        f"model.layers.0.self_attn.{projection_name}.weight_scale",
    }
    if split_qkvg:
        expected_loaded.add("model.layers.0.self_attn.g_proj.weight")
        assert layer.g_proj.weight.dtype == torch.bfloat16
        torch.testing.assert_close(
            layer.g_proj.weight, g_weight[rank * 8 : (rank + 1) * 8]
        )
    assert loaded == expected_loaded
    assert projection.weight.dtype == torch.float8_e4m3fn
    assert projection.weight_scale.dtype == torch.uint8
    torch.testing.assert_close(
        projection.weight.float(), torch.cat(expected_weight).float()
    )
    torch.testing.assert_close(projection.weight_scale, torch.cat(expected_scale))
    model.post_load_weights.assert_called_once()
