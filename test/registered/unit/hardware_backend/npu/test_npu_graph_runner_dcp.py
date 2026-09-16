"""CPU unit tests for NPUGraphRunner.execute seq-len rebinding under decode
context parallel (DCP), incl. DSPARK target verify.

Usage:
    python -m pytest test_npu_graph_runner_dcp.py -v
"""

import unittest
import unittest.mock
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.dcp.ops import dcp_verify_history_local_lens
from sglang.srt.hardware_backend.npu.graph_runner import npu_graph_runner as ngr
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeBackend:
    def __init__(self):
        self.calls = []

    def replay_with_input_update(self, graph_key, seq_lens, attr_name, attr_type):
        self.calls.append((list(seq_lens), attr_name))
        return ngr.PPProxyTensors({"x": torch.zeros(4)})


class TestNpuGraphRunnerDcpSeqLens(CustomTestCase):
    W = 8

    def _runner(self, capture_mode, is_draft_worker, use_fias_v2_bsnd=False):
        runner = object.__new__(ngr.NPUGraphRunner)
        runner.is_dllm = False
        runner._init_arch_map()
        runner.model_runner = SimpleNamespace(
            is_draft_worker=is_draft_worker,
            spec_algorithm=SimpleNamespace(
                is_dspark=lambda: True, is_dflash=lambda: False
            ),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["KimiK3ForConditionalGeneration"]
                )
            ),
        )
        runner.capture_forward_mode = capture_mode
        runner.captured_req_width = self.W
        runner.if_use_v2 = False
        runner.use_fias_v2_bsnd = use_fias_v2_bsnd
        runner.bs, runner.raw_bs = 4, 3
        runner.raw_num_token = 3 * self.W
        runner.buffers = SimpleNamespace(
            input_ids=torch.zeros(64, dtype=torch.long),
            positions=torch.zeros(64, dtype=torch.long),
        )
        runner.backend = _FakeBackend()
        return runner

    def _execute(self, runner, mode, seq_lens_cpu, dcp):
        parallel = SimpleNamespace(
            dcp_enabled=dcp, attn_dcp_size=2 if dcp else 1, attn_dcp_rank=1 if dcp else 0
        )
        fb = SimpleNamespace(
            needs_forward_metadata_init=lambda: False,
            input_ids=torch.zeros(runner.raw_num_token, dtype=torch.long),
            positions=torch.zeros(runner.raw_num_token, dtype=torch.long),
            input_embeds=None,
            mrope_positions=None,
            forward_mode=mode,
            seq_lens=torch.tensor(seq_lens_cpu),
            seq_lens_cpu=torch.tensor(seq_lens_cpu),
        )
        with patch.object(ngr, "get_parallel", return_value=parallel):
            runner.execute(fb)
        return runner.backend.calls[-1]

    def test_dcp_target_verify_history_lens_v1_key(self):
        seq_lens = [8, 9, 30]  # include the verify window
        for use_v2 in (False, True):
            runner = self._runner(ForwardMode.TARGET_VERIFY, False, use_v2)
            lens, key = self._execute(
                runner, ForwardMode.TARGET_VERIFY, seq_lens, dcp=True
            )
            self.assertEqual(key, "actual_seq_lengths_kv")
            self.assertEqual(
                lens, dcp_verify_history_local_lens(seq_lens + [0], self.W, 2, 1)
            )
            self.assertEqual(lens, [0, 0, 11, 0])

    def test_non_dcp_target_verify_unchanged(self):
        seq_lens = [8, 9, 30]
        runner = self._runner(ForwardMode.TARGET_VERIFY, False, True)
        lens, key = self._execute(runner, ForwardMode.TARGET_VERIFY, seq_lens, dcp=False)
        self.assertEqual((lens, key), (seq_lens + [0], "actual_seq_kvlen"))
        runner = self._runner(ForwardMode.TARGET_VERIFY, False, False)
        lens, key = self._execute(runner, ForwardMode.TARGET_VERIFY, seq_lens, dcp=False)
        self.assertEqual((lens, key), (seq_lens + [0], "actual_seq_lengths_kv"))

    def test_dcp_draft_worker_uses_global_lens(self):
        seq_lens = [8, 9, 30]
        runner = self._runner(ForwardMode.TARGET_VERIFY, True, True)
        lens, key = self._execute(runner, ForwardMode.TARGET_VERIFY, seq_lens, dcp=True)
        self.assertEqual((lens, key), (seq_lens + [0], "actual_seq_kvlen"))
        runner = self._runner(ForwardMode.DECODE, True)
        lens, key = self._execute(runner, ForwardMode.DECODE, seq_lens, dcp=True)
        self.assertEqual((lens, key), (seq_lens + [0], "actual_seq_lengths_kv"))

    def test_dspark_without_dcp_keeps_global_lens(self):
        # dcp_size == 1: no DCP length rewriting anywhere; a DSPARK FIAS v2
        # target-verify graph keeps updating actual_seq_kvlen with global lens.
        seq_lens = [8, 9, 30]
        boom = unittest.mock.MagicMock(side_effect=AssertionError("DCP lens"))
        cases = [
            (ForwardMode.TARGET_VERIFY, False, True, "actual_seq_kvlen"),
            (ForwardMode.TARGET_VERIFY, False, False, "actual_seq_lengths_kv"),
            (ForwardMode.TARGET_VERIFY, True, True, "actual_seq_kvlen"),
            (ForwardMode.DECODE, False, False, "actual_seq_lengths_kv"),
            (ForwardMode.DECODE, True, False, "actual_seq_lengths_kv"),
        ]
        with patch.object(ngr, "dcp_verify_history_local_lens", boom), patch.object(
            ngr, "dcp_local_seq_lens", boom
        ):
            for mode, is_draft, use_v2, expect_key in cases:
                runner = self._runner(mode, is_draft, use_v2)
                with patch.object(
                    ngr,
                    "get_parallel",
                    return_value=SimpleNamespace(dcp_enabled=False),
                ):
                    self.assertFalse(runner._is_dcp_target_verify_graph())
                lens, key = self._execute(runner, mode, seq_lens, dcp=False)
                self.assertEqual(
                    (lens, key), (seq_lens + [0], expect_key), (mode, is_draft)
                )
        boom.assert_not_called()

    def test_dcp_decode_local_lens(self):
        seq_lens = [8, 9, 30]
        runner = self._runner(ForwardMode.DECODE, False)
        lens, key = self._execute(runner, ForwardMode.DECODE, seq_lens, dcp=True)
        self.assertEqual((lens, key), ([4, 4, 15, 0], "actual_seq_lengths_kv"))


if __name__ == "__main__":
    unittest.main()
