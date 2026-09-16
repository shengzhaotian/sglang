"""CPU unit tests for ``ModelRunner._prepare_replicated_q_proj``.

``--dcp-replicate-q-proj`` all-gathers each MLA layer's attn-TP head shard of
``q_b_proj`` (or ``q_proj``) and ``w_kc`` over the DCP group into
``q_b_proj_qrep_weight`` / ``w_kc_qrep``. Layers with a quantized q-proj, a
non-bf16/fp16 q-proj weight / ``w_kc``, or no materialized ``w_kc`` are
skipped and keep the per-layer Q all-gather; the skips are summarized in one
warning keyed by reason with the layer ids.

No model is built: fake MLA modules are ``DeepseekV2AttentionMLA`` instances
created without ``__init__``, and ``get_parallel()`` is patched with a fake
DCP group whose ``all_gather`` concatenates prepared per-rank shards.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.test.test_utils import CustomTestCase

LOGGER = "sglang.srt.model_executor.model_runner"

# K3-shaped tiny layer: full heads split evenly across attn-TP == DCP ranks.
DCP = 4
HEADS_PER_RANK = 2
Q_LORA = 6
QK_HEAD = 5  # qk_nope + qk_rope
QK_NOPE = 3
KV_LORA = 7


class FakeQuantMethod:
    """Stand-in for a non-unquantized linear method (e.g. W8A8)."""


class FakeDcpGroup:
    """Returns the full tensor registered for the local shard's identity."""

    def __init__(self, world_size, rank, full_by_shard_ptr):
        self.world_size = world_size
        self.rank_in_group = rank
        self._full = full_by_shard_ptr
        self.calls = 0

    def all_gather(self, t, dim=0):
        assert dim == 0
        self.calls += 1
        return self._full[t.data_ptr()].clone()


def _unquant():
    return object.__new__(UnquantizedLinearMethod)


def _make_layer(layer_id, *, rank, registry, q_dtype=torch.bfloat16,
                kc_dtype=torch.bfloat16, quant=None, no_w_kc=False,
                has_q_b_proj=True, seed=0):
    g = torch.Generator().manual_seed(1000 + layer_id + seed)
    full_q = torch.randn(DCP * HEADS_PER_RANK * QK_HEAD, Q_LORA, generator=g)
    full_kc = torch.randn(DCP * HEADS_PER_RANK, QK_NOPE, KV_LORA, generator=g)
    if q_dtype.is_floating_point:
        full_q = full_q.to(q_dtype)
    else:
        full_q = (full_q * 10).to(q_dtype)
    full_kc = full_kc.to(kc_dtype)
    q_shard = full_q.chunk(DCP, dim=0)[rank].contiguous()
    kc_shard = full_kc.chunk(DCP, dim=0)[rank].contiguous()
    registry[q_shard.data_ptr()] = full_q
    registry[kc_shard.data_ptr()] = full_kc

    m = object.__new__(DeepseekV2AttentionMLA)
    # nn.Module attribute storage without running Module.__init__.
    torch.nn.Module.__init__(m)
    m.layer_id = layer_id
    m.has_q_b_proj = has_q_b_proj
    proj = SimpleNamespace(
        quant_method=quant if quant is not None else _unquant(),
        weight=torch.nn.Parameter(q_shard, requires_grad=False),
    )
    if has_q_b_proj:
        m.q_b_proj = proj
    else:
        m.q_proj = proj
    m.w_kc = None if no_w_kc else kc_shard
    m.q_b_proj_qrep_weight = None
    m.w_kc_qrep = None
    return m, full_q, full_kc


class _FakeModel:
    def __init__(self, layers):
        self.layers = layers

    def modules(self):
        yield self
        yield torch.nn.Linear(2, 2)  # non-MLA module is ignored
        yield from self.layers


class TestPrepareReplicatedQProj(CustomTestCase):
    def _run(self, layers, registry, *, world_size=DCP, rank=0, npu=False):
        group = FakeDcpGroup(world_size, rank, registry)
        runner = SimpleNamespace(model=_FakeModel(layers))
        with patch(
            "sglang.srt.model_executor.model_runner.get_parallel",
            return_value=SimpleNamespace(dcp_group=group),
        ), patch(
            "sglang.srt.model_executor.model_runner.is_npu", return_value=npu
        ):
            ModelRunner._prepare_replicated_q_proj(runner)
        return group

    def test_npu_backs_q_shard_by_view_of_full_head_weight(self):
        for rank in range(DCP):
            registry = {}
            layers = [
                _make_layer(i, rank=rank, registry=registry, has_q_b_proj=i != 1)[0]
                for i in range(2)
            ]
            before = []
            x = torch.randn(3, Q_LORA, dtype=torch.bfloat16)
            for m in layers:
                qp = m.q_b_proj if m.has_q_b_proj else m.q_proj
                before.append(
                    (qp.weight, qp.weight.detach().clone(), F.linear(x, qp.weight))
                )
            with self.assertLogs(LOGGER, level="INFO") as cm:
                self._run(layers, registry, rank=rank, npu=True)
            for m, (param, orig, y) in zip(layers, before):
                qp = m.q_b_proj if m.has_q_b_proj else m.q_proj
                full = m.q_b_proj_qrep_weight
                self.assertIs(qp.weight, param)  # same Parameter object
                self.assertTrue(torch.equal(qp.weight, orig))
                rows = orig.shape[0]
                row_numel = full[0].numel()
                self.assertEqual(
                    qp.weight.data_ptr(),
                    full.data_ptr() + rank * rows * row_numel * full.element_size(),
                )
                self.assertTrue(torch.equal(F.linear(x, qp.weight), y))
                # in-place update of the shard writes into the full-head slice
                if m.layer_id == 0:
                    qp.weight.data[0, 0] += 1
                    self.assertTrue(
                        torch.equal(full[rank * rows], qp.weight.data[0])
                    )
            mib = 2 * orig.numel() * orig.element_size() / (1 << 20)
            self.assertIn(
                f"freed duplicate TP q-proj shards: 2 layers, {mib:.2f} MiB",
                "\n".join(cm.output),
            )

    def test_cuda_path_keeps_separate_q_shard(self):
        registry = {}
        m, _, _ = _make_layer(0, rank=2, registry=registry)
        param = m.q_b_proj.weight
        ptr = param.data_ptr()
        with self.assertLogs(LOGGER, level="INFO") as cm:
            self._run([m], registry, rank=2, npu=False)
        self.assertIs(m.q_b_proj.weight, param)
        self.assertEqual(m.q_b_proj.weight.data_ptr(), ptr)
        full = m.q_b_proj_qrep_weight
        self.assertFalse(
            full.data_ptr()
            <= ptr
            < full.data_ptr() + full.numel() * full.element_size()
        )
        self.assertIn("freed duplicate TP q-proj shards: 0 layers, 0.00 MiB",
                      "\n".join(cm.output))

    def test_npu_mismatched_rows_keep_shard_and_warn(self):
        registry = {}
        m, full_q, _ = _make_layer(7, rank=1, registry=registry)
        ptr = m.q_b_proj.weight.data_ptr()
        registry[ptr] = torch.flip(full_q, dims=[0])  # wrong row order
        with self.assertLogs(LOGGER, level="INFO") as cm:
            self._run([m], registry, rank=1, npu=True)
        self.assertEqual(m.q_b_proj.weight.data_ptr(), ptr)
        out = "\n".join(cm.output)
        self.assertIn("layer 7 full-head rows [10, 20) do not match", out)
        self.assertIn("freed duplicate TP q-proj shards: 0 layers", out)

    def test_prepared_layers_get_full_heads_in_rank_order(self):
        for rank in range(DCP):
            registry = {}
            m0, q0, kc0 = _make_layer(0, rank=rank, registry=registry)
            m1, q1, kc1 = _make_layer(
                1, rank=rank, registry=registry, q_dtype=torch.float16,
                kc_dtype=torch.float16, has_q_b_proj=False,
            )
            with self.assertLogs(LOGGER, level="INFO") as cm:
                self._run([m0, m1], registry, rank=rank)
            for m, q, kc in ((m0, q0, kc0), (m1, q1, kc1)):
                self.assertTrue(torch.equal(m.q_b_proj_qrep_weight, q))
                self.assertTrue(torch.equal(m.w_kc_qrep, kc))
                # head group j of the gathered weight is rank j's shard.
                own = (m.q_b_proj if m.has_q_b_proj else m.q_proj).weight
                self.assertTrue(
                    torch.equal(m.q_b_proj_qrep_weight.chunk(DCP)[rank], own)
                )
                self.assertTrue(torch.equal(m.w_kc_qrep.chunk(DCP)[rank], m.w_kc))
            out = "\n".join(cm.output)
            self.assertIn("prepared full-head Q weights for 2/2 MLA layers", out)
            self.assertNotIn("WARNING", out)

    def test_forward_replicated_equals_concat_of_rank_shards(self):
        registry = {}
        m, full_q, _ = _make_layer(3, rank=1, registry=registry)
        self._run([m], registry, rank=1)
        x = torch.randn(4, Q_LORA, dtype=torch.bfloat16)
        rep = F.linear(x, m.q_b_proj_qrep_weight).view(
            4, DCP * HEADS_PER_RANK, QK_HEAD
        )
        per_rank = [
            F.linear(x, shard).view(4, HEADS_PER_RANK, QK_HEAD)
            for shard in full_q.chunk(DCP, dim=0)
        ]
        self.assertTrue(torch.equal(rep, torch.cat(per_rank, dim=1)))

    def test_skips_are_summarized_by_reason_with_layer_ids(self):
        registry = {}
        ok, _, _ = _make_layer(0, rank=0, registry=registry)
        quant, _, _ = _make_layer(1, rank=0, registry=registry, quant=FakeQuantMethod())
        quant2, _, _ = _make_layer(4, rank=0, registry=registry, quant=FakeQuantMethod())
        fp32, _, _ = _make_layer(2, rank=0, registry=registry, q_dtype=torch.float32)
        int8, _, _ = _make_layer(3, rank=0, registry=registry, q_dtype=torch.int8)
        kc32, _, _ = _make_layer(5, rank=0, registry=registry, kc_dtype=torch.float32)
        nokc, _, _ = _make_layer(6, rank=0, registry=registry, no_w_kc=True)
        layers = [ok, quant, fp32, int8, quant2, kc32, nokc]
        with self.assertLogs(LOGGER, level="INFO") as cm:
            group = self._run(layers, registry)
        self.assertEqual(group.calls, 2)  # only the prepared layer gathers
        self.assertIsNotNone(ok.q_b_proj_qrep_weight)
        for m in layers[1:]:
            self.assertIsNone(m.q_b_proj_qrep_weight)
            self.assertIsNone(m.w_kc_qrep)

        infos = [r for r in cm.records if r.levelno == logging.INFO]
        warns = [r for r in cm.records if r.levelno == logging.WARNING]
        self.assertEqual(len(warns), 1)  # one summary, no per-layer spam
        self.assertIn("1/7 MLA layers", infos[-1].getMessage())
        w = warns[0].getMessage()
        self.assertIn("skipped 6/7 MLA layers", w)
        self.assertIn("per-layer Q all-gather", w)
        self.assertIn("quantized q-proj (FakeQuantMethod): layers [1, 4]", w)
        self.assertIn("q-proj weight dtype torch.float32 (bf16/fp16 only): layers [2]", w)
        self.assertIn("q-proj weight dtype torch.int8 (bf16/fp16 only): layers [3]", w)
        self.assertIn("w_kc dtype torch.float32 (bf16/fp16 only): layers [5]", w)
        self.assertIn(
            "w_kc not materialized (e.g. split GGUF kv_b): layers [6]", w
        )
        self.assertNotIn("effectively disabled", w)

    def test_all_skipped_warns_effectively_disabled(self):
        registry = {}
        layers = [
            _make_layer(i, rank=0, registry=registry, quant=FakeQuantMethod())[0]
            for i in range(3)
        ]
        with self.assertLogs(LOGGER, level="INFO") as cm:
            self._run(layers, registry)
        out = "\n".join(cm.output)
        self.assertIn("0/3 MLA layers", out)
        self.assertIn("replication is effectively disabled", out)
        self.assertIn("all 3 MLA layers skipped", out)
        self.assertIn("layers [0, 1, 2]", out)

    def test_world_size_one_returns_early(self):
        registry = {}
        m, _, _ = _make_layer(0, rank=0, registry=registry)
        runner = SimpleNamespace(model=SimpleNamespace(modules=None))  # untouched
        group = FakeDcpGroup(1, 0, registry)
        with patch(
            "sglang.srt.model_executor.model_runner.get_parallel",
            return_value=SimpleNamespace(dcp_group=group),
        ), self.assertNoLogs(LOGGER, level="INFO"):
            ModelRunner._prepare_replicated_q_proj(runner)
        self.assertEqual(group.calls, 0)
        self.assertIsNone(m.q_b_proj_qrep_weight)
        self.assertIsNone(m.w_kc_qrep)


if __name__ == "__main__":
    unittest.main()
