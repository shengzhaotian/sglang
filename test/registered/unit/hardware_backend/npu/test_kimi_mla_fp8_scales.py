"""CPU contracts for checkpoint-derived Kimi MLA FP8 scales.

Execute the real scale/dispatch methods without importing NPU kernels or building
the model. This checks checkpoint loading, not CANN numerical execution.
"""

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_SRT = Path(__file__).resolve().parents[5] / "python/sglang/srt"
_QUANT = "layers/quantization/modelslim/"


def _execute(nodes, namespace, filename):
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            )
        ]
        + nodes,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), filename, "exec"), namespace)  # noqa: S102


def _load_definitions(relative, names, namespace):
    path = _SRT / relative
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert len(nodes) == len(names)
    _execute(nodes, namespace, str(path))


def _load_methods(relative, owner, names, namespace):
    path = _SRT / relative
    cls = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == owner
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(methods) == len(names)
    _execute(methods, namespace, str(path))
    return {name: namespace[name] for name in names}


def _contracts():
    namespace = {
        "torch": torch,
        "nn": torch.nn,
        "ModelSlimKVSchemeBase": object,
        "QuantizeMethodBase": object,
        "get_tensor_model_parallel_rank": lambda: 0,
        "get_tensor_model_parallel_world_size": lambda: 1,
        "_is_npu": True,
    }
    _load_definitions(
        _QUANT + "schemes/modelslim_q_fp8_dynamic_kv_fp8.py",
        {"_modelslim_kv_weight_loader", "ModelSlimQFP8DynamicKVFP8Scheme"},
        namespace,
    )
    _load_definitions(
        _QUANT + "modelslim.py", {"ModelSlimQFP8DynamicKVFP8Method"}, namespace
    )
    methods = _load_methods(
        _QUANT + "modelslim.py",
        "ModelSlimConfig",
        {
            "_quant_prefix_candidates",
            "_resolve_kv_prefix",
            "_maybe_get_kv_method",
            "get_quant_method",
        },
        namespace,
    )
    namespace["ModelSlimConfig"] = type("ModelSlimConfig", (), methods)
    _load_methods(
        "models/kimi_k3.py",
        "KimiK3MLAAttention",
        {"_init_npu_modelslim_kv_quant_override"},
        namespace,
    )
    _load_methods(
        "models/deepseek_v2.py",
        "DeepseekV2AttentionMLA",
        {"_init_kv_quant_weights", "refresh_fa_k_scale_params"},
        namespace,
    )
    return namespace


class TestKimiMLAFP8Scales(unittest.TestCase):
    def setUp(self):
        # Only class checks for non-attention layers need these heavy imports.
        # Restore only these keys: rolling back all of sys.modules would unload
        # modules lazily imported by torch.testing during another test.
        for name, module in {
            "sglang.srt.layers.linear": SimpleNamespace(
                LinearBase=type("LinearBase", (), {})
            ),
            "sglang.srt.layers.moe.fused_moe_triton": SimpleNamespace(
                FusedMoE=type("FusedMoE", (), {})
            ),
        }.items():
            if name in sys.modules:
                self.addCleanup(sys.modules.__setitem__, name, sys.modules[name])
            else:
                self.addCleanup(sys.modules.pop, name, None)
            sys.modules[name] = module
        self.ns = _contracts()
        self.prefix = "model.layers.3.self_attn"
        self.checkpoint_prefix = "language_model." + self.prefix

    def config(self, quant_type="Q_FP8_DYNAMIC_KV_FP8", *, declared=True):
        config = self.ns["ModelSlimConfig"]()
        config.quant_description = {
            f"{self.checkpoint_prefix}.quant_type": quant_type,
        }
        if declared:
            config.quant_description.update(
                {
                    "fa_quant_type": "FAKQuant",
                    f"{self.checkpoint_prefix}.fa_k.scale": "FAQuant",
                    f"{self.checkpoint_prefix}.fa_v.scale": "FAQuant",
                }
            )
        return config

    def layer(self, config=None, *, override=False):
        config = self.config() if config is None else config
        layer = torch.nn.Module()
        layer.num_local_heads = 12
        layer.kv_quant_method = None
        if override:
            self.ns["_init_npu_modelslim_kv_quant_override"](layer, config, self.prefix)
        self.ns["_init_kv_quant_weights"](layer, config, self.prefix)
        return layer

    def load(self, layer, scale=0.25):
        # Exercise the parameter-level callback used by Kimi's generic loader.
        for name, param in layer.named_parameters():
            param.weight_loader(
                param, torch.tensor(scale if name.endswith(".scale") else 0.0)
            )

    def refresh(self, layer):
        self.ns["refresh_fa_k_scale_params"](layer)

    def test_register_load_refresh(self):
        layer = self.layer()
        self.assertEqual(
            set(dict(layer.named_parameters())),
            {"fa_k.scale", "fa_k.offset", "fa_v.scale", "fa_v.offset"},
        )
        self.assertFalse(layer._modelslim_fp8_kv_scale_ready)
        self.assertEqual(layer.fa_k.scale.dtype, torch.float32)
        self.assertEqual(layer.fa_k.scale.shape, (1, 1))
        self.load(layer)
        self.refresh(layer)
        self.assertTrue(layer._modelslim_fp8_kv_scale_ready)
        torch.testing.assert_close(layer.fak_descale_float, torch.tensor([[0.25]]))
        torch.testing.assert_close(layer.fak_descale_reciprocal, torch.tensor([[4.0]]))

    def test_missing_scale_does_not_fall_back_to_one(self):
        layer = self.layer()
        layer.fa_v.scale.weight_loader(layer.fa_v.scale, torch.tensor([0.25]))
        with self.assertRaisesRegex(RuntimeError, "unit-scale fallback"):
            self.refresh(layer)
        self.assertFalse(layer._modelslim_fp8_kv_scale_ready)

    def test_invalid_scales_invalidate_previous_success(self):
        for value in (0.0, -0.25, float("inf"), float("nan"), 1e-45):
            with self.subTest(value=value):
                layer = self.layer()
                self.load(layer)
                self.refresh(layer)
                layer.fa_k.scale.data.fill_(value)
                with self.assertRaises((ValueError, RuntimeError)):
                    self.refresh(layer)
                self.assertFalse(layer._modelslim_fp8_kv_scale_ready)

    def test_checkpoint_distinct_k_v_scales_use_k_for_shared_latent(self):
        layer = self.layer()
        k_scale = torch.tensor([[0.00885009765625]], dtype=torch.float32)
        v_scale = torch.tensor([[0.01129150390625]], dtype=torch.float32)
        for role, scale in (("k", k_scale), ("v", v_scale)):
            module = getattr(layer, f"fa_{role}")
            module.scale.weight_loader(module.scale, scale)
            module.offset.weight_loader(module.offset, torch.zeros_like(scale))
        self.refresh(layer)
        self.assertTrue(layer._modelslim_fp8_kv_scale_ready)
        torch.testing.assert_close(layer.fak_descale_float, k_scale, rtol=0, atol=0)
        torch.testing.assert_close(
            layer.fak_descale_reciprocal, k_scale.reciprocal(), rtol=0, atol=0
        )
        torch.testing.assert_close(layer.fa_k.scale, k_scale, rtol=0, atol=0)
        torch.testing.assert_close(layer.fa_v.scale, v_scale, rtol=0, atol=0)

    def test_unused_v_parameters_do_not_control_runtime_scale(self):
        layer = self.layer()
        layer.fa_k.scale.weight_loader(layer.fa_k.scale, torch.tensor([[0.25]]))
        # V is registered for the checkpoint loader but absent from this runtime
        # contract: its initial NaN scale must not replace or invalidate K.
        self.refresh(layer)
        self.assertTrue(layer._modelslim_fp8_kv_scale_ready)
        torch.testing.assert_close(layer.fak_descale_float, torch.tensor([[0.25]]))
        torch.testing.assert_close(layer.fak_descale_reciprocal, torch.tensor([[4.0]]))

    def test_nonzero_or_nonfinite_offset_is_rejected(self):
        for value in (1.0, float("nan"), float("inf")):
            layer = self.layer()
            self.load(layer)
            layer.fa_k.offset.data.fill_(value)
            with self.assertRaisesRegex(ValueError, "finite and zero"):
                self.refresh(layer)

    def test_kimi_override_ignores_bad_labels_without_mutating_description(self):
        for label in (
            "V_FP8",
            "K_FP8",
            "Q_FP8_DYNAMIC_V_FP8",
            "Q_FP8_DYNAMIC_K_FP8",
            "unreliable-export-label",
            None,
        ):
            with self.subTest(label=label):
                config = self.config(label)
                before = dict(config.quant_description)
                layer = self.layer(config, override=True)
                self.assertIsNotNone(layer.kv_quant_method)
                self.assertEqual(config.quant_description, before)
        config = self.config("unused")
        del config.quant_description[f"{self.checkpoint_prefix}.quant_type"]
        self.assertIsNotNone(self.layer(config, override=True).kv_quant_method)

    def test_other_models_keep_exact_scheme_dispatch(self):
        # No Kimi override: FAQuant scale declarations alone do not select C8.
        for label in ("V_FP8", "K_FP8", "unknown"):
            self.assertIsNone(self.layer(self.config(label)).kv_quant_method)

    def test_override_requires_npu_modelslim_and_both_fak_scales(self):
        hook = self.ns["_init_npu_modelslim_kv_quant_override"]
        for config in (
            None,
            SimpleNamespace(quant_description={}),
            self.config(declared=False),
        ):
            layer = torch.nn.Module()
            hook(layer, config, self.prefix)
            self.assertFalse(hasattr(layer, "_npu_modelslim_kv_quant_type"))
        for key in ("fa_quant_type", f"{self.checkpoint_prefix}.fa_v.scale"):
            config = self.config("V_FP8")
            del config.quant_description[key]
            self.assertIsNone(self.layer(config, override=True).kv_quant_method)
        self.ns["_is_npu"] = False
        layer = self.layer(self.config("V_FP8"), override=True)
        self.assertIsNone(layer.kv_quant_method)
        self.assertFalse(hasattr(layer, "_npu_modelslim_kv_quant_type"))

    def test_head_scale_loader_shards_and_rejects_bad_shapes(self):
        loader = self.ns["_modelslim_kv_weight_loader"]
        self.ns["get_tensor_model_parallel_rank"] = lambda: 1
        self.ns["get_tensor_model_parallel_world_size"] = lambda: 2
        param = torch.nn.Parameter(torch.empty((2, 1)), requires_grad=False)
        loader(param, torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
        torch.testing.assert_close(param, torch.tensor([[3.0], [4.0]]))
        for invalid in (torch.tensor(1.0), torch.ones((3, 1)), torch.ones((4, 2))):
            with self.assertRaises(ValueError):
                loader(param, invalid)


if __name__ == "__main__":
    unittest.main()
