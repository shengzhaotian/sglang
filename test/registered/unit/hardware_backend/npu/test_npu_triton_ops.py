"""CPU unit tests for the in-tree Triton-Ascend kernels and their call sites.

The kernels themselves are checked numerically by ``triton_ops_check.py``,
which this file runs in a subprocess under ``TRITON_INTERPRET=1``: Triton binds
``@triton.jit`` to the compiler or the interpreter when the decorator runs, so
the switch has to be in the environment of a fresh process and must not leak
into the rest of a pytest run.

The remaining tests cover the wiring that decides whether a kernel is used at
all, with the kernel itself stubbed out.

Usage:
    python -m pytest test_npu_triton_ops.py -v
"""

import os
import subprocess
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

_CHECK_SCRIPT = os.path.join(os.path.dirname(__file__), "triton_ops_check.py")


class TestTritonKernelNumerics(CustomTestCase):
    # triton_ops_check.py returns this when triton cannot run a kernel here
    # (no triton, or sglang's platform stub standing in for it).
    SKIP_EXIT_CODE = 77

    def test_kernels_match_the_torch_references(self):
        env = dict(os.environ, TRITON_INTERPRET="1")
        proc = subprocess.run(
            [sys.executable, _CHECK_SCRIPT],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if proc.returncode == self.SKIP_EXIT_CODE:
            raise unittest.SkipTest(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("checks passed", proc.stdout)


class TestSplitQkNormGating(CustomTestCase):
    """``_use_triton_split_qk_norm``: the fused split + RMSNorm kernel only runs
    for a plain weighted RMSNorm pair over a row-contiguous latent."""

    def _module(self):
        mocked = {
            name: MagicMock()
            for name in (
                "torch_npu",
                "sgl_kernel_npu",
                "sgl_kernel_npu.norm",
                "sgl_kernel_npu.norm.fused_split_qk_norm",
            )
            if name not in sys.modules
        }
        with patch.dict(sys.modules, mocked):
            from sglang.srt.hardware_backend.npu.modules import (
                deepseek_v2_attention_mla_npu as mla_npu,
            )
        return mla_npu

    @staticmethod
    def _norm(hidden):
        return SimpleNamespace(
            has_weight=True,
            variance_size_override=None,
            cast_x_before_out_mul=False,
            override_orig_dtype=None,
            variance_epsilon=1e-6,
            weight=torch.ones(hidden),
        )

    def _attention(self):
        return SimpleNamespace(
            q_a_layernorm=self._norm(6),
            kv_a_layernorm=self._norm(4),
            q_lora_rank=6,
            kv_lora_rank=4,
            qk_rope_head_dim=2,
        )

    def test_gating(self):
        mla_npu = self._module()
        m = self._attention()
        latent = torch.randn(3, 12)
        batch = SimpleNamespace()
        with patch.object(mla_npu, "dsa_use_prefill_cp", return_value=False):
            # On by default on this branch; switching it off restores the
            # split + two npu_rms_norm path.
            self.assertTrue(mla_npu._use_triton_split_qk_norm(m, latent, batch))
            with envs.SGLANG_NPU_FUSED_SPLIT_QK_NORM_TRITON.override(False):
                self.assertFalse(mla_npu._use_triton_split_qk_norm(m, latent, batch))
            with envs.SGLANG_NPU_FUSED_SPLIT_QK_NORM_TRITON.override(True):
                self.assertTrue(mla_npu._use_triton_split_qk_norm(m, latent, batch))
                # A column-major latent has no contiguous reduction dim.
                self.assertFalse(
                    mla_npu._use_triton_split_qk_norm(m, latent.t(), batch)
                )
                self.assertFalse(
                    mla_npu._use_triton_split_qk_norm(m, latent.unsqueeze(0), batch)
                )
                # Norm variants the kernel does not reproduce.
                for field, value in (
                    ("has_weight", False),
                    ("variance_size_override", 2),
                    ("cast_x_before_out_mul", True),
                    ("override_orig_dtype", torch.float32),
                ):
                    m = self._attention()
                    setattr(m.kv_a_layernorm, field, value)
                    self.assertFalse(
                        mla_npu._use_triton_split_qk_norm(m, latent, batch), field
                    )
        # The prefill-CP path still needs the un-split latent_cache.
        m = self._attention()
        with envs.SGLANG_NPU_FUSED_SPLIT_QK_NORM_TRITON.override(True), patch.object(
            mla_npu, "dsa_use_prefill_cp", return_value=True
        ):
            self.assertFalse(mla_npu._use_triton_split_qk_norm(m, latent, batch))


class TestDcpKvStoreDispatch(CustomTestCase):
    """NPUMLATokenToKVPool.set_kv_buffer routes to the fused kernel only when
    SGLANG_NPU_DCP_KV_STORE_TRITON is on and the cache layout is the plain one,
    and the fused write lands in the same places as the two-scatter path."""

    C, SLOTS, D_C, D_R = 4, 12, 3, 2

    def _pool(self, use_triton, fia_nz=False, dsa_fp8=False):
        from sglang.srt.hardware_backend.npu.memory_pool_npu import (
            NPUMLATokenToKVPool,
        )

        pool = object.__new__(NPUMLATokenToKVPool)
        pool.start_layer = 0
        pool.kv_lora_rank, pool.qk_rope_head_dim = self.D_C, self.D_R
        pool.dtype = pool.store_dtype = torch.float32
        pool.dsa_kv_cache_store_fp8 = dsa_fp8
        pool.kv_cache_dim = self.D_C
        pool.use_fia_nz = fia_nz
        pool.use_triton_dcp_kv_store = use_triton
        pool.k_buffer = [torch.zeros(self.SLOTS, 1, self.D_C)]
        pool.v_buffer = [torch.zeros(self.SLOTS, 1, self.D_R)]
        return pool

    @staticmethod
    def _stub_kernel(recorder):
        """A stand-in for the Triton kernel with the same contract."""

        def dcp_store_mla_kv(k_buf, v_buf, cache_k, cache_v, loc, dcp_size, dcp_rank):
            recorder.append(dcp_rank)
            owned = (loc >= 0) & (loc % dcp_size == dcp_rank)
            slots = (loc // dcp_size)[owned]
            k_buf[slots] = cache_k[owned]
            v_buf[slots] = cache_v[owned]

        module = ModuleType("sglang.srt.hardware_backend.npu.triton_ops.kv_store")
        module.dcp_store_mla_kv = dcp_store_mla_kv
        return module

    def _run(self, pool, loc, cache_k, cache_v, rank, calls):
        from sglang.srt.hardware_backend.npu import memory_pool_npu

        parallel = SimpleNamespace(
            dcp_enabled=True, attn_dcp_size=self.C, attn_dcp_rank=rank
        )
        fake_npu = MagicMock()

        def scatter(buffer, indices, values):
            buffer[indices.view(-1)] = values

        fake_npu.npu_scatter_nd_update_.side_effect = scatter
        with patch.object(
            memory_pool_npu, "get_parallel", return_value=parallel
        ), patch.object(
            memory_pool_npu, "torch_npu", fake_npu, create=True
        ), patch.dict(
            sys.modules,
            {
                "sglang.srt.hardware_backend.npu.triton_ops.kv_store": self._stub_kernel(
                    calls
                )
            },
        ):
            pool.set_kv_buffer(
                SimpleNamespace(layer_id=0), loc, cache_k.clone(), cache_v.clone()
            )
        return fake_npu.npu_scatter_nd_update_.call_count

    def test_fused_write_matches_the_scatter_path(self):
        torch.manual_seed(0)
        # Virtual locs of one page, plus a padding row. The allocator never
        # hands out the first page (physical slot 0 is the reserved pad slot
        # the torch path writes unowned tokens to), so every loc here is >= C.
        loc = torch.tensor([-1, 5, 6, 9, 10, 7, 4, 11])
        cache_k = torch.randn(loc.numel(), self.D_C)
        cache_v = torch.randn(loc.numel(), self.D_R)
        for rank in range(self.C):
            calls = []
            fused = self._pool(use_triton=True)
            scatters = self._run(fused, loc, cache_k, cache_v, rank, calls)
            self.assertEqual(calls, [rank])
            self.assertEqual(scatters, 0)

            plain = self._pool(use_triton=False)
            self.assertEqual(self._run(plain, loc, cache_k, cache_v, rank, []), 2)
            fused_k = fused.k_buffer[0].view(self.SLOTS, self.D_C)
            plain_k = plain.k_buffer[0].view(self.SLOTS, self.D_C)
            fused_v = fused.v_buffer[0].view(self.SLOTS, self.D_R)
            plain_v = plain.v_buffer[0].view(self.SLOTS, self.D_R)
            # Real slots agree; the pad slot is written by the torch path only.
            self.assertTrue(torch.equal(fused_k[1:], plain_k[1:]), rank)
            self.assertTrue(torch.equal(fused_v[1:], plain_v[1:]), rank)
            self.assertTrue(torch.all(fused_k[0] == 0), rank)
            self.assertTrue(torch.any(plain_k[0] != 0), rank)
            owned = [l for l in loc.tolist() if l >= 0 and l % self.C == rank]
            self.assertTrue(owned, rank)
            for l in owned:
                i = loc.tolist().index(l)
                self.assertTrue(torch.equal(fused_k[l // self.C], cache_k[i]))
                self.assertTrue(torch.equal(fused_v[l // self.C], cache_v[i]))

    def test_layouts_the_kernel_does_not_handle_keep_the_torch_path(self):
        loc = torch.tensor([0, 1, 2, 3])
        cache_k = torch.randn(4, self.D_C)
        cache_v = torch.randn(4, self.D_R)
        for kwargs in ({"fia_nz": True}, {"dsa_fp8": True}):
            calls = []
            pool = self._pool(use_triton=True, **kwargs)
            pool._set_fia_nz_kv_buffer = MagicMock()
            pool._pack_dsa_fp8_kv_cache = MagicMock(return_value=torch.zeros(4, 1, 1))
            self._run(pool, loc, cache_k, cache_v, 0, calls)
            self.assertEqual(calls, [], kwargs)


class TestTritonSplitMergeDispatch(CustomTestCase):
    """``dcp_a2a_exchange`` / ``dcp_merge_shards`` reach the Triton pack and
    combine kernels when the merge implementation is "triton".

    This is the one switch combination -- SGLANG_NPU_DCP_VERIFY_FUSED_MERGE
    with SGLANG_NPU_DCP_MERGE_IMPL=triton -- that no numeric case reaches: the
    target-verify exactness test runs in-process, where triton cannot, so it
    only covers the npu / torch merges. The kernels' own math is checked by
    triton_ops_check.py; what is checked here is that the split-merge API
    hands them the right arguments and passes their results through, with the
    kernels stubbed by their torch equivalents.
    """

    N, B, H, D = 4, 3, 2, 8

    def _shards(self):
        torch.manual_seed(0)
        outs = torch.randn(self.N, self.B, self.H, self.D)
        lses = torch.randn(self.N, self.B, self.H)
        extra_out = torch.randn(self.B, self.H, self.D)
        extra_lse = torch.randn(self.B, self.H)
        return outs, lses, extra_out, extra_lse

    @staticmethod
    def _stub_kernels(calls):
        """A stand-in triton_ops.dcp_merge with the same contract."""
        from sglang.srt.hardware_backend.npu.dcp import ops as dcp_ops

        def dcp_exchange_a2a_triton(out, lse, group):
            calls.append(("exchange", tuple(out.shape), tuple(lse.shape)))
            return dcp_ops._dcp_a2a_packed_exchange(out, lse, group)

        def lse_combine_shards(
            outs, lses, extra_out=None, extra_lse=None, return_lse=False
        ):
            calls.append(("combine", outs.shape[0], extra_out is not None))
            return dcp_ops.dcp_merge_shards(
                outs,
                lses,
                "torch",
                extra_out=extra_out,
                extra_lse=extra_lse,
                return_lse=return_lse,
            )

        module = ModuleType("sglang.srt.hardware_backend.npu.triton_ops.dcp_merge")
        module.dcp_exchange_a2a_triton = dcp_exchange_a2a_triton
        module.lse_combine_shards = lse_combine_shards
        return module

    def test_merge_shards_routes_the_extra_shard_to_the_kernel(self):
        from sglang.srt.hardware_backend.npu.dcp import ops as dcp_ops

        outs, lses, extra_out, extra_lse = self._shards()
        calls = []
        key = "sglang.srt.hardware_backend.npu.triton_ops.dcp_merge"
        with patch.dict(sys.modules, {key: self._stub_kernels(calls)}):
            got = dcp_ops.dcp_merge_shards(
                outs, lses, "triton", extra_out=extra_out, extra_lse=extra_lse
            )
        # The kernel is called once, over the N shards, with the extra one.
        self.assertEqual(calls, [("combine", self.N, True)])
        want = dcp_ops.dcp_merge_shards(
            outs, lses, "torch", extra_out=extra_out, extra_lse=extra_lse
        )
        self.assertTrue(torch.allclose(got, want, atol=1e-6), (got - want).abs().max())

    def test_the_other_implementations_do_not_import_the_kernels(self):
        from sglang.srt.hardware_backend.npu.dcp import ops as dcp_ops

        outs, lses, extra_out, extra_lse = self._shards()
        key = "sglang.srt.hardware_backend.npu.triton_ops.dcp_merge"
        for impl in ("torch", "npu"):
            calls = []
            with patch.dict(
                sys.modules, {key: self._stub_kernels(calls)}
            ), patch.object(
                dcp_ops, "_attention_update_op", _reference_attention_update
            ):
                dcp_ops.dcp_merge_shards(
                    outs, lses, impl, extra_out=extra_out, extra_lse=extra_lse
                )
            self.assertEqual(calls, [], impl)

    def test_exchange_routes_to_the_pack_kernel(self):
        from sglang.srt.hardware_backend.npu.dcp import ops as dcp_ops

        out = torch.randn(self.B, 1 * self.H, self.D, dtype=torch.bfloat16)
        lse = torch.randn(self.B, 1 * self.H)
        group = _SingleRankGroup()
        key = "sglang.srt.hardware_backend.npu.triton_ops.dcp_merge"
        calls = []
        with patch.dict(sys.modules, {key: self._stub_kernels(calls)}):
            recv_out, recv_lse = dcp_ops.dcp_a2a_exchange(out, lse, group, "triton")
        self.assertEqual(calls, [("exchange", tuple(out.shape), tuple(lse.shape))])
        self.assertTrue(torch.equal(recv_out[0], out))
        self.assertTrue(torch.allclose(recv_lse[0], lse, atol=1e-6))
        # The other implementations keep the torch packing.
        calls = []
        with patch.dict(sys.modules, {key: self._stub_kernels(calls)}):
            dcp_ops.dcp_a2a_exchange(out, lse, group, "npu")
        self.assertEqual(calls, [])


class _SingleRankGroup:
    """A one-rank DCP group: all_to_all_single is a copy."""

    world_size = 1
    rank_in_group = 0

    @staticmethod
    def all_to_all_single(output, input):
        output.copy_(input)


def _reference_attention_update(lse_list, out_list, update_type=0):
    """Pure-torch torch_npu.npu_attention_update, as test_npu_dcp_ops.py uses."""
    lses = torch.stack([l.float() for l in lse_list])
    outs = torch.stack([o.float() for o in out_list])
    lse_max = lses.amax(dim=0)
    lse = lse_max + torch.log(torch.exp(lses - lse_max).sum(dim=0))
    out = (outs * torch.exp(lses - lse).unsqueeze(-1)).sum(dim=0)
    return out, (lse if update_type == 1 else None)


if __name__ == "__main__":
    unittest.main()
