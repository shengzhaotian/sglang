"""CPU contracts for TileLang's runtime KV lengths and TP8 request sharding.

Execute the backend's actual AST with simulated collectives and a sentinel
kernel. This does not validate attention numerics, NPU graphs, or HCCL.
"""

import ast
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from test_kimi_mla_fp8_a2a import (
    HEADS,
    LATENT,
    SIDE,
    TP,
    _assert_bits,
    _Collective,
    _padded_payload,
    _rank_inputs,
    _sentinel,
)
from test_kimi_mla_fp8_attention import BACKEND, SOURCE, TREE, _case

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

WIDTH = 4


def _mode(name):
    return SimpleNamespace(
        is_idle=lambda: name == "idle",
        is_decode_or_idle=lambda: name in ("decode", "idle"),
        is_target_verify=lambda: name == "verify",
        is_draft_extend_v2=lambda: name == "draft_extend",
        is_extend=lambda: name in ("prefill", "verify", "draft_extend"),
        is_dllm_extend=lambda: False,
    )


def _length_case(bs, *, mode="verify", needs_cpu=False):
    backend, namespace, fia, args = _case(tokens=bs * WIDTH, width=WIDTH)
    names = {"_update_mla_fp8_kv_lengths", "init_forward_metadata_out_graph"}
    functions = [
        node
        for node in BACKEND.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    is_dflash = next(
        node
        for node in TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == "_is_dflash_verify"
    )
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                is_dflash,
                *functions,
            ],
            type_ignores=[],
        )
    )
    namespace["SpecInputType"] = SimpleNamespace(DFLASH_VERIFY="dflash")
    exec(compile(module, str(SOURCE), "exec"), namespace)  # noqa: S102
    for name in names:
        setattr(type(backend), name, namespace[name])
    backend.device = "cpu"
    backend.needs_cpu_seq_lens = needs_cpu
    backend.is_hybrid_swa = backend.use_sliding_window_kv_pool = False
    backend.speculative_num_draft_tokens = WIDTH
    backend.speculative_step_id = 2
    backend.req_to_token = torch.arange((bs + 1) * 512).view(bs + 1, 512)
    backend.req_to_token[0].zero_()
    backend.req_to_token_pool = SimpleNamespace(req_to_token=backend.req_to_token)
    backend.graph_metadata = {"block_tables": torch.zeros(bs, 4, dtype=torch.int32)}
    batch = args["forward_batch"]
    batch.forward_mode = batch.actual_forward_mode = _mode(mode)
    batch.batch_size = bs
    batch.seq_lens = 125 + torch.arange(bs, dtype=torch.int32) * 3
    # Overlap can leave the device prefix one step behind: not prefix + WIDTH.
    batch.seq_lens_cpu = batch.seq_lens + WIDTH + 1
    batch.spec_info.spec_input_type = "dflash"
    batch.req_pool_indices = torch.arange(1, bs + 1)
    batch.out_cache_loc = None
    batch.num_padding = 0
    batch.extend_seq_lens = None
    batch.extend_seq_lens_cpu = [WIDTH] * bs
    batch.extend_prefix_lens_cpu = [0] * bs
    return backend, namespace, fia, args


@pytest.mark.parametrize(
    "bs,graph,live",
    [
        (8, False, 8),
        (16, False, 16),
        (10, False, 10),
        (8, True, 8),
        (16, True, 16),
        (10, True, 10),
        (10, True, 5),
        (10, True, 0),
    ],
)
def test_tilelang_local_lengths_pages_payload_and_workspace(monkeypatch, bs, graph, live):
    local_bs = (bs + TP - 1) // TP
    local_tokens = local_bs * WIDTH
    padded_bs = local_bs * TP
    inputs = [_rank_inputs(rank, bs * WIDTH) for rank in range(TP)]
    packed = [
        _padded_payload(value, bs * WIDTH, padded_bs * WIDTH) for value in inputs
    ]
    expected_lengths = torch.zeros(padded_bs, dtype=torch.int64)
    expected_lengths[:live] = 130 + torch.arange(live) * 3
    expected_pages = torch.zeros(padded_bs, 4, dtype=torch.int64)
    expected_pages[:live] = (
        torch.arange(1, live + 1).unsqueeze(1) * 4 + torch.arange(4)
    )
    kernel_outputs = []
    for rank in range(TP):
        value = _sentinel(rank * local_tokens, local_tokens)
        for request in range(local_bs):
            if expected_lengths[rank * local_bs + request] == 0:
                value[request * WIDTH : (request + 1) * WIDTH].zero_()
        kernel_outputs.append(value)

    for rank in range(TP):
        backend, namespace, fia, args = _length_case(bs)
        backend.use_sparse_attn_a2a = True
        group = _Collective(rank, packed, kernel_outputs)
        namespace["get_parallel"] = lambda: SimpleNamespace(
            attn_tp_size=TP, attn_tp_rank=rank, attn_tp_group=group
        )
        batch = args["forward_batch"]
        batch.num_token_non_padded_cpu = live * WIDTH
        batch.num_padding = bs - live
        batch.req_pool_indices[live:] = 0
        if not live:
            batch.actual_forward_mode = _mode("idle")
        before = batch.seq_lens.clone()
        if graph:
            backend.init_forward_metadata_out_graph(batch, in_capture=True)
            backend.init_forward_metadata_out_graph(batch)
        else:
            backend.init_forward_metadata(batch)
        torch.testing.assert_close(batch.seq_lens, before)
        assert backend.forward_metadata.seq_lens_cpu_int is None
        args["q"], args["q_rope"], args["dequant_scale_q_nope"] = inputs[rank]
        start = rank * local_tokens
        expected_packed = torch.cat(
            [value[start : start + local_tokens] for value in packed], dim=1
        )

        def kernel(q, key, rope, key_rope, pages, lengths, dsq, dsk, **kw):
            # The supplied adapter fixes [local B, 4, 96, 512]. No kernel math.
            assert q.shape == (local_tokens, 96, LATENT)
            assert q.view(local_bs, WIDTH, 96, LATENT).shape[1] == WIDTH
            _assert_bits(q, expected_packed[..., :LATENT].contiguous().view(q.dtype))
            _assert_bits(
                rope,
                expected_packed[..., LATENT : LATENT + SIDE * 2]
                .contiguous()
                .view(torch.bfloat16),
            )
            _assert_bits(
                dsq,
                expected_packed[..., -4:].contiguous().view(torch.float32).squeeze(-1),
            )
            req = slice(rank * local_bs, (rank + 1) * local_bs)
            assert lengths.dtype == torch.int64 and lengths.is_contiguous()
            torch.testing.assert_close(lengths, expected_lengths[req])
            torch.testing.assert_close(pages, expected_pages[req], check_dtype=False)
            torch.testing.assert_close(dsk, args["fp8_kv_scale"])
            assert key.shape == (4, 128, LATENT) and key_rope.shape == (4, 128, SIDE)
            assert kw["workspace"].dtype == torch.float32
            assert kw["workspace"].numel() == local_bs * 17 + 4
            assert kw["seq_len"] is None and kw["splits"] == 4
            kw["out"].copy_(kernel_outputs[rank])

        stub = ModuleType("fia_decode_c8_tilelang_h24")
        stub.fia_decode_c8 = Mock(side_effect=kernel)
        stub.workspace_numel = Mock(side_effect=lambda size, splits: size * 17 + splits)
        monkeypatch.setitem(sys.modules, stub.__name__, stub)
        output = backend.forward_extend(**args)
        expected = torch.cat(kernel_outputs)[
            : bs * WIDTH, rank * HEADS : (rank + 1) * HEADS
        ]
        _assert_bits(output, expected.flatten(1))
        assert group.calls == 2
        stub.fia_decode_c8.assert_called_once()
        stub.workspace_numel.assert_called_once_with(local_bs, 4)
        fia.assert_not_called()


@pytest.mark.parametrize("needs_cpu", [False, True])
def test_graph_refresh_uses_final_cpu_lengths_and_stable_storage(needs_cpu):
    backend, _, _, args = _length_case(8, needs_cpu=needs_cpu)
    batch = args["forward_batch"]
    batch.num_padding = 6
    batch.req_pool_indices[2:] = 0
    before = batch.seq_lens.clone()
    backend.init_forward_metadata_out_graph(batch, in_capture=True)
    metadata = backend.forward_metadata
    pointer = metadata.actual_seq_lengths_kv.data_ptr()
    page_pointer = metadata.block_tables.data_ptr()
    torch.testing.assert_close(batch.seq_lens, before)
    torch.testing.assert_close(
        metadata.actual_seq_lengths_kv[:2], (before[:2] + WIDTH).long()
    )
    assert not metadata.actual_seq_lengths_kv[2:].any()

    for finals, idle in [
        ([129, 257], False),
        ([257, 258], False),
        ([257, 258], True),
        ([130, 259], False),
    ]:
        batch.seq_lens_cpu = torch.tensor([*finals, *([7] * 6)], dtype=torch.int32)
        batch.seq_lens.copy_(batch.seq_lens_cpu - 5)
        before = batch.seq_lens.clone()
        batch.actual_forward_mode = _mode("idle" if idle else "verify")
        backend.init_forward_metadata_out_graph(batch)
        assert backend.forward_metadata is metadata
        assert metadata.actual_seq_lengths_kv.data_ptr() == pointer
        assert metadata.block_tables.data_ptr() == page_pointer
        assert metadata.seq_lens.data_ptr() == batch.seq_lens.data_ptr()
        expected = torch.zeros(8, dtype=torch.int64)
        if not idle:
            expected[:2] = torch.tensor(finals)
        torch.testing.assert_close(metadata.actual_seq_lengths_kv, expected)
        torch.testing.assert_close(batch.seq_lens, before)


@pytest.mark.parametrize("graph", [False, True])
def test_dspark_missing_final_cpu_lengths_fails(graph):
    backend, _, _, args = _length_case(8)
    batch = args["forward_batch"]
    batch.seq_lens_cpu = None
    if graph:
        backend.init_forward_metadata_out_graph(batch, in_capture=True)
    with pytest.raises(ValueError, match="DSpark verify requires final CPU KV lengths"):
        if graph:
            backend.init_forward_metadata_out_graph(batch)
        else:
            backend.init_forward_metadata(batch)


@pytest.mark.parametrize("mode", ["decode", "verify", "draft_extend", "prefill"])
def test_eager_non_dspark_offsets_and_prefill_guard(mode):
    backend, _, _, args = _length_case(2, mode=mode)
    batch = args["forward_batch"]
    batch.spec_info.spec_input_type = "other"
    batch.seq_lens_cpu = None
    before = batch.seq_lens.clone()
    backend.init_forward_metadata(batch)
    lengths = backend.forward_metadata.actual_seq_lengths_kv
    if mode == "prefill":
        assert lengths is None  # Preserve ordinary prefill's existing source priority.
    else:
        offset = {"decode": 3, "verify": WIDTH, "draft_extend": 0}[mode]
        torch.testing.assert_close(lengths, (before + offset).long())
    torch.testing.assert_close(batch.seq_lens, before)
