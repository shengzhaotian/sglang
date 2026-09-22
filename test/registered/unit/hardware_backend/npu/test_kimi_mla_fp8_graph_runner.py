"""CPU contracts for C8 graph replay, using the actual NPU runner methods."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[5]
SOURCE = (
    ROOT / "python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py"
)
TREE = ast.parse(SOURCE.read_text())
RUNNER = next(
    n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "NPUGraphRunner"
)


class Output(SimpleNamespace):
    pass


def _mode(name):
    return SimpleNamespace(
        is_target_verify=lambda: name == "verify",
        is_idle=lambda: name == "idle",
    )


def _runner(
    *,
    dtype="fp8_e4m3",
    arch="MLA",
    dsa=False,
    draft=False,
    capture_mode="decode",
    bsnd=False,
    dspark=True,
    architectures=None,
    a2a=False,
    attn_tp_size=8,
    attn_tp_rank=0,
):
    names = {
        "_init_arch_map",
        "_uses_v2_seq_len_update",
        "_get_update_attr_name",
        "_get_update_attr_type",
        "execute",
    }
    cls = ast.ClassDef(
        name="Runner",
        bases=[],
        keywords=[],
        decorator_list=[],
        body=[
            n for n in RUNNER.body if isinstance(n, ast.FunctionDef) and n.name in names
        ],
    )
    namespace = {
        "torch": torch,
        "AttentionArch": SimpleNamespace(MLA="MLA", MHA="MHA"),
        "is_deepseek_dsa": lambda c: c.dsa,
        "is_deepseek_v4": lambda c: False,
        "LogitsProcessorOutput": Output,
        "envs": SimpleNamespace(
            SGLANG_NPU_SPARSE_ATTN_A2A=SimpleNamespace(get=lambda: a2a)
        ),
    }
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
    exec(compile(module, str(SOURCE), "exec"), namespace)  # noqa: S102
    runner = namespace["Runner"]()
    runner.attn_tp_size = attn_tp_size
    runner.attn_tp_rank = attn_tp_rank
    runner.model_runner = SimpleNamespace(
        model_config=SimpleNamespace(
            attention_arch=arch,
            hf_config=SimpleNamespace(
                architectures=architectures or ["KimiK3ForConditionalGeneration"],
                dsa=dsa,
            ),
        ),
        kv_cache_dtype_str=dtype,
        # Deliberately differ for BF16 draft cases: this is not the runner dtype.
        server_args=SimpleNamespace(kv_cache_dtype="fp8_e4m3"),
        is_draft_worker=draft,
        spec_algorithm=SimpleNamespace(is_dspark=lambda: dspark),
    )
    init = next(
        n
        for n in RUNNER.body
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    assignments = [
        n
        for n in init.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Attribute)
            and t.attr in ("use_mla_fp8", "use_mla_fp8_a2a", "if_use_v2")
            for t in n.targets
        )
    ]
    exec(  # noqa: S102
        compile(ast.Module(body=assignments, type_ignores=[]), str(SOURCE), "exec"),
        dict(namespace, self=runner, model_runner=runner.model_runner),
    )
    runner.is_dllm = False
    runner.use_fias_v2_bsnd = bsnd
    runner.capture_forward_mode = _mode(capture_mode)
    runner.captured_req_width = 3 if capture_mode == "verify" else 1
    runner.bs = 4
    runner.raw_bs = 2
    runner.raw_num_token = 2 * runner.captured_req_width
    runner._init_arch_map()
    runner.load_batch = Mock()
    runner._make_graph_key = Mock(side_effect=lambda bs: bs)
    output = Output(
        next_token_logits=torch.ones((12, 2)),
        full_logits=None,
        hidden_states=torch.ones((12, 3)),
    )
    runner.backend = SimpleNamespace(
        replay_with_input_update=Mock(return_value=output),
        replay=Mock(return_value=output),
    )
    return runner


def _batch(mode="decode", seq_lens=(127, 255), cpu_lens=None):
    return SimpleNamespace(
        forward_mode=_mode(mode),
        seq_lens=torch.tensor(seq_lens),
        seq_lens_cpu=torch.tensor(cpu_lens) if cpu_lens is not None else None,
        needs_forward_metadata_init=lambda: True,
    )


def test_c8_decode_replay_updates_v2_lengths_each_step_with_bucket_padding():
    runner = _runner()
    for lengths in ([127, 255], [128, 256], [260, 129]):
        result = runner.execute(_batch(seq_lens=lengths))
        kw = runner.backend.replay_with_input_update.call_args.kwargs
        assert kw["attr_name"] == "actual_seq_kvlen"
        assert kw["seq_lens"] == lengths + [0, 0]
        assert kw["attr_type"] == []
        assert result.next_token_logits.shape[0] == 2
    assert runner.backend.replay_with_input_update.call_count == 3
    runner.backend.replay.assert_not_called()


@pytest.mark.parametrize("bsnd", [False, True])
def test_dspark_verify_replay_uses_final_cpu_boundary_without_adding_width(bsnd):
    runner = _runner(capture_mode="verify", bsnd=bsnd)
    for prefix, final in (([127, 255], [130, 258]), ([128, 258], [131, 261])):
        result = runner.execute(_batch("verify", prefix, final))
        kw = runner.backend.replay_with_input_update.call_args.kwargs
        assert kw["attr_name"] == "actual_seq_kvlen"
        assert kw["seq_lens"] == final + [0, 0]
        assert result.hidden_states.shape[0] == 6


def test_c8_verify_idle_replay_clears_stale_lengths_then_resumes_active():
    runner = _runner(capture_mode="verify", bsnd=False)
    runner.execute(_batch("verify", [127, 255], [130, 258]))
    # IDLE must not consume stale request lengths left in reusable inputs.
    runner.raw_num_token = 0
    runner.execute(_batch("idle", [130, 258], [130, 258]))
    kw = runner.backend.replay_with_input_update.call_args.kwargs
    assert kw["attr_name"] == "actual_seq_kvlen" and kw["seq_lens"] == [0, 0, 0, 0]
    runner.raw_bs, runner.raw_num_token = 1, 3
    runner.execute(_batch("verify", [258], [261]))
    kw = runner.backend.replay_with_input_update.call_args.kwargs
    assert kw["attr_name"] == "actual_seq_kvlen" and kw["seq_lens"] == [261, 0, 0, 0]


def test_empty_c8_verify_rank_updates_the_captured_v2_graph():
    runner = _runner(capture_mode="verify", bsnd=False)
    runner.raw_bs = runner.raw_num_token = 0
    output = runner.execute(_batch("idle", []))
    assert (
        runner.backend.replay_with_input_update.call_args.kwargs["seq_lens"] == [0] * 4
    )
    assert output.next_token_logits.shape[0] == 0


@pytest.mark.parametrize("capture_mode,width", [("decode", 1), ("verify", 4)])
@pytest.mark.parametrize(
    "bucket,rank,full_lengths",
    [
        (8, 0, [100]),
        (8, 7, [107]),
        (16, 0, [100, 101]),
        (16, 7, [114, 115]),
        (10, 0, [100, 101]),
        (10, 7, [0, 0]),
    ],
)
def test_c8_a2a_replay_uses_fixed_bucket_request_shards(
    capture_mode, width, bucket, rank, full_lengths
):
    runner = _runner(capture_mode=capture_mode, a2a=True, attn_tp_rank=rank)
    runner.bs = bucket
    runner.captured_req_width = width
    assert runner.use_mla_fp8_a2a
    for live in (bucket, 1, bucket):
        runner.raw_bs = live
        runner.raw_num_token = live * width
        final = list(range(100, 100 + live))
        prefix = [length - width for length in final]
        runner.execute(
            _batch(capture_mode, prefix, final)
            if capture_mode == "verify"
            else _batch(seq_lens=final)
        )
        kw = runner.backend.replay_with_input_update.call_args.kwargs
        expected = full_lengths if live == bucket else [0] * len(full_lengths)
        if live == 1 and rank == 0:
            expected[0] = 100
        assert kw["attr_name"] == "actual_seq_kvlen"
        assert kw["seq_lens"] == expected
        assert kw["attr_type"] == []


@pytest.mark.parametrize(
    "bucket,rank,local_bs", [(8, 0, 1), (8, 7, 1), (16, 0, 2), (16, 7, 2), (10, 7, 2)]
)
def test_c8_a2a_verify_idle_clears_local_lengths_then_resumes(bucket, rank, local_bs):
    runner = _runner(capture_mode="verify", a2a=True, attn_tp_rank=rank)
    runner.bs = bucket
    runner.captured_req_width = 4
    runner.raw_bs, runner.raw_num_token = bucket, bucket * 4
    prefix = list(range(127, 127 + bucket))
    final = [length + 4 for length in prefix]
    runner.execute(_batch("verify", prefix, final))
    first = runner.backend.replay_with_input_update.call_args.kwargs["seq_lens"]
    runner.raw_bs = runner.raw_num_token = 0
    runner.execute(_batch("idle", final, final))
    assert (
        runner.backend.replay_with_input_update.call_args.kwargs["seq_lens"]
        == [0] * local_bs
    )
    runner.raw_bs, runner.raw_num_token = bucket, bucket * 4
    runner.execute(_batch("verify", prefix, final))
    assert runner.backend.replay_with_input_update.call_args.kwargs["seq_lens"] == first


@pytest.mark.parametrize(
    "kwargs",
    [
        {"a2a": False},
        {"a2a": True, "attn_tp_size": 1},
        {"a2a": True, "dtype": "bf16"},
        {"a2a": True, "dtype": "bf16", "draft": True},
        {"a2a": True, "arch": "MHA", "draft": True},
    ],
)
def test_a2a_length_sharding_only_applies_to_enabled_dense_c8_tp(kwargs):
    runner = _runner(**kwargs)
    assert not runner.use_mla_fp8_a2a
    runner.execute(_batch(seq_lens=[10, 20]))
    assert runner.backend.replay_with_input_update.call_args.kwargs["seq_lens"] == [
        10,
        20,
        0,
        0,
    ]


@pytest.mark.parametrize(
    "arch,dtype,draft,dsa,expected",
    [
        ("MLA", "fp8_e4m3", False, False, True),
        ("MLA", "bf16", False, False, False),
        ("MLA", "bf16", True, False, False),
        ("MHA", "bf16", True, False, False),
        ("MHA", "fp8_e4m3", True, False, False),
        ("MLA", "fp8_e4m3", False, True, False),
    ],
)
def test_v2_c8_selection_uses_own_runner_dtype_not_target_args(
    arch, dtype, draft, dsa, expected
):
    runner = _runner(arch=arch, dtype=dtype, draft=draft, dsa=dsa)
    assert runner.use_mla_fp8 is expected
    assert runner._uses_v2_seq_len_update() is expected


def test_bf16_decode_and_existing_v2_architecture_selection_are_unchanged():
    runner = _runner(dtype="bf16")
    runner.execute(_batch(seq_lens=[10, 20]))
    kw = runner.backend.replay_with_input_update.call_args.kwargs
    assert kw["attr_name"] == "actual_seq_lengths_kv" and kw["seq_lens"] == [
        10,
        20,
        0,
        0,
    ]
    assert _runner(
        dtype="bf16", architectures=["Step3p5ForCausalLM"]
    )._uses_v2_seq_len_update()


@pytest.mark.parametrize(
    "bsnd,key", [(False, "actual_seq_lengths_kv"), (True, "actual_seq_kvlen")]
)
def test_bf16_dspark_verify_and_idle_retain_existing_selection(bsnd, key):
    runner = _runner(dtype="bf16", capture_mode="verify", bsnd=bsnd)
    runner.execute(_batch("verify", [10, 20], [13, 23]))
    kw = runner.backend.replay_with_input_update.call_args.kwargs
    assert kw["attr_name"] == key and kw["seq_lens"] == [13, 23, 0, 0]
    runner.execute(_batch("idle", [10, 20]))
    kw = runner.backend.replay_with_input_update.call_args.kwargs
    assert kw["attr_name"] == key and kw["seq_lens"] == [10, 20, 0, 0]


def test_dsa_keeps_its_replay_without_host_length_update():
    runner = _runner(dsa=True, a2a=True)
    assert not runner.use_mla_fp8_a2a
    runner.execute(_batch())
    runner.backend.replay.assert_called_once()
    runner.backend.replay_with_input_update.assert_not_called()


@pytest.mark.parametrize("has_previous_replay", [False, True])
def test_existing_graph_update_waits_before_replay(has_previous_replay):
    source = SOURCE.with_name("npu_cudagraph_backend.py")
    cls = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "NPUCudaGraphBackend"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "replay_with_input_update"
    )
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                method,
            ],
            type_ignores=[],
        )
    )
    ns = {"torch": torch, "np": np}
    exec(compile(module, str(source), "exec"), ns)  # noqa: S102
    events = []
    graph = SimpleNamespace(
        update=Mock(side_effect=lambda **kw: events.append(("update", kw))),
        replay=Mock(side_effect=lambda: events.append(("replay", None))),
    )

    def submit(fn, **kwargs):
        fn(**kwargs)
        return SimpleNamespace(result=lambda: events.append(("wait", None)))

    previous_fence = SimpleNamespace(
        synchronize=lambda: events.append(("fence_wait", None))
    )
    next_fence = SimpleNamespace(record=lambda: events.append(("fence_record", None)))
    backend = SimpleNamespace(
        _graphs={4: graph},
        _outputs={4: "output"},
        _update_executor=SimpleNamespace(submit=submit),
        _rebind_fence=previous_fence if has_previous_replay else None,
        _device_module=SimpleNamespace(Event=lambda: next_fence),
    )
    result = ns["replay_with_input_update"](
        backend,
        4,
        [130, 258, 0, 0],
        attr_name="actual_seq_kvlen",
        attr_type=[],
    )
    assert result == "output"
    expected = ["fence_wait"] if has_previous_replay else []
    assert [event[0] for event in events] == expected + [
        "update", "wait", "replay", "fence_record"
    ]
    assert backend._rebind_fence is next_fence
    assert graph.update.call_args.kwargs == {
        "cpu_update_input": [{"actual_seq_kvlen": [130, 258, 0, 0]}],
    }
