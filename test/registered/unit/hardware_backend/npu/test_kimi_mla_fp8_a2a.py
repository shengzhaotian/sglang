"""CPU byte-layout contracts for dense MLA C8 attention all-to-all.

The simulated collectives check every rank's send payload and provide the exact
other-rank chunks. FIA writes token/head sentinels, not numerical attention;
these tests do not certify NPU FIA, NPUGraph or HCCL device support.
"""

from types import SimpleNamespace

import pytest
import torch
from sglang.test.ci.ci_register import register_cpu_ci
from test_kimi_mla_fp8_attention import _case

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

TP = 8
HEADS = 12
LATENT = 512
SIDE = 64
PACKED_BYTES = LATENT + SIDE * 2 + 4


def _rank_inputs(rank, tokens):
    token = torch.arange(tokens).view(tokens, 1, 1)
    head = torch.arange(HEADS).view(1, HEADS, 1)
    feature = torch.arange(LATENT).view(1, 1, LATENT)
    # Exclude the two NaN encodings, but include signs and signed zero so a
    # numerical cast instead of a byte view cannot pass this contract.
    q = ((rank * 17 + token * 3 + head * 5 + feature) % 127).to(torch.uint8)
    q = q | (((rank + token + head) % 2) * 128).to(torch.uint8)
    q = q.view(torch.float8_e4m3fn)
    side = (rank * 16 + token + head / 4 + feature[..., :SIDE] / 16).to(torch.bfloat16)
    scale = (rank * 8192 + token * HEADS + head + 1).float() / 1024
    return q, side, scale


def _padded_payload(inputs, tokens, total_tokens):
    q, side, scale = inputs
    packed = torch.cat(
        [
            q[:tokens].contiguous().view(torch.uint8),
            side[:tokens].contiguous().view(torch.uint8),
            scale[:tokens].contiguous().view(torch.uint8),
        ],
        dim=-1,
    )
    if tokens != total_tokens:
        padding = torch.zeros(
            total_tokens - tokens, HEADS, PACKED_BYTES, dtype=torch.uint8
        )
        padding[..., -4:] = torch.ones(total_tokens - tokens, HEADS, 1).view(
            torch.uint8
        )
        packed = torch.cat((packed, padding))
    return packed


def _sentinel(token_start, token_count):
    token = torch.arange(token_start, token_start + token_count).view(-1, 1, 1)
    head = torch.arange(TP * HEADS).view(1, -1, 1)
    feature = torch.arange(LATENT).view(1, 1, -1)
    result = ((token * 13 + head * 3 + feature) % 127 - 63).to(torch.bfloat16)
    result[..., 0] = token[..., 0]
    result[..., 1] = head[..., 0]
    return result


def _assert_bits(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


class _Collective:
    def __init__(self, rank, packed_inputs, fia_outputs):
        self.rank_in_group = rank
        self.world_size = TP
        self.packed_inputs = packed_inputs
        self.fia_outputs = fia_outputs
        self.calls = 0

    def all_to_all_single(self, output, value):
        rank = self.rank_in_group
        if self.calls == 0:
            assert value.dtype == output.dtype == torch.uint8
            expected_send = self.packed_inputs[rank].flatten()
            local_tokens = self.packed_inputs[rank].shape[0] // TP
            received = torch.cat(
                [
                    source[rank * local_tokens : (rank + 1) * local_tokens].flatten()
                    for source in self.packed_inputs
                ]
            )
        elif self.calls == 1:
            assert value.dtype == output.dtype == torch.bfloat16
            local = self.fia_outputs[rank]
            expected_send = (
                local.view(local.shape[0], TP, HEADS, LATENT)
                .transpose(0, 1)
                .contiguous()
                .flatten()
            )
            received = torch.cat(
                [
                    source[:, rank * HEADS : (rank + 1) * HEADS].flatten()
                    for source in self.fia_outputs
                ]
            )
        else:
            raise AssertionError(
                "C8 A2A must use one forward and one reverse collective"
            )
        _assert_bits(value.flatten(), expected_send)
        output.copy_(received.view_as(output))
        self.calls += 1
        return output


def _exercise(bs, width, *, graph=False, live_requests=None, dp_pad_requests=0):
    live_requests = bs if live_requests is None else live_requests
    bucket_tokens = (bs + dp_pad_requests) * width
    active_requests = bs if graph else live_requests
    active_tokens = active_requests * width
    local_requests = (active_requests + TP - 1) // TP
    local_tokens = local_requests * width
    total_requests = local_requests * TP
    all_inputs = [_rank_inputs(rank, bucket_tokens) for rank in range(TP)]
    packed = [
        _padded_payload(inputs, active_tokens, total_requests * width)
        for inputs in all_inputs
    ]
    lengths = [
        129 + request * 3 if request < live_requests else 0 for request in range(bs)
    ]
    padded_lengths = lengths[:active_requests] + [0] * (
        total_requests - active_requests
    )
    pages = torch.arange(bs * 3, dtype=torch.int32).view(bs, 3) + 1
    if live_requests < bs:
        pages[live_requests:] = 0
    padded_pages = torch.cat(
        (
            pages[:active_requests],
            torch.zeros(total_requests - active_requests, 3, dtype=torch.int32),
        )
    )
    fia_outputs = []
    for rank in range(TP):
        expected = _sentinel(rank * local_tokens, local_tokens)
        for request in range(local_requests):
            if padded_lengths[rank * local_requests + request] == 0:
                expected[request * width : (request + 1) * width] = 0
        fia_outputs.append(expected)

    for rank in range(TP):
        backend, namespace, fia, args = _case(
            tokens=bucket_tokens, live=live_requests * width, width=width
        )
        backend.use_sparse_attn_a2a = True
        backend.graph_mode = graph
        group = _Collective(rank, packed, fia_outputs)
        namespace["get_parallel"] = lambda rank=rank, group=group: SimpleNamespace(
            attn_tp_size=TP, attn_tp_rank=rank, attn_tp_group=group
        )
        namespace["get_attn_tp_group"] = lambda group=group: group
        args["q"], args["q_rope"], args["dequant_scale_q_nope"] = all_inputs[rank]
        metadata = backend.forward_metadata
        metadata.seq_lens_cpu_int = None if graph else torch.tensor(lengths)
        metadata.seq_lens_cpu_list = lengths
        metadata.block_tables = pages
        req_start = rank * local_requests
        token_start = rank * local_tokens
        # Expected order is token-major, then source-rank-major query heads.
        expected_packed = torch.cat(
            [source[token_start : token_start + local_tokens] for source in packed],
            dim=1,
        )

        def check_fia(
            q,
            key,
            value,
            *,
            _rank=rank,
            _packed=expected_packed,
            _req=req_start,
            _kv_scale=args["fp8_kv_scale"],
            **kw,
        ):
            assert q.dtype == torch.float8_e4m3fn
            assert key is value and key.dtype == torch.float8_e4m3fn
            assert kw["num_query_heads"] == TP * HEADS
            assert kw["num_key_value_heads"] == 1
            expected_q = _packed[..., :LATENT].contiguous().view(torch.float8_e4m3fn)
            expected_side = (
                _packed[..., LATENT : LATENT + SIDE * 2]
                .contiguous()
                .view(torch.bfloat16)
            )
            expected_scale = (
                _packed[..., -4:].contiguous().view(torch.float32).squeeze(-1)
            )
            if width == 1:
                expected_q, expected_side, expected_scale = (
                    expected_q.unsqueeze(1),
                    expected_side.unsqueeze(1),
                    expected_scale.unsqueeze(1),
                )
                assert kw["input_layout"] == "BSND" and kw["actual_seq_qlen"] is None
            else:
                assert kw["input_layout"] == "TND"
                assert kw["actual_seq_qlen"] == list(
                    range(width, local_tokens + 1, width)
                )
            _assert_bits(q, expected_q)
            _assert_bits(kw["query_rope"], expected_side)
            _assert_bits(kw["dequant_scale_query"], expected_scale)
            assert (
                kw["actual_seq_kvlen"] == padded_lengths[_req : _req + local_requests]
            )
            torch.testing.assert_close(
                kw["block_table"], padded_pages[_req : _req + local_requests]
            )
            assert kw["dequant_scale_key"] is kw["dequant_scale_value"]
            torch.testing.assert_close(kw["dequant_scale_key"].flatten(), _kv_scale)
            assert kw["query_quant_mode"] == 3
            assert kw["key_quant_mode"] == kw["value_quant_mode"] == 0
            kw["out"][0].copy_(fia_outputs[_rank].view_as(kw["out"][0]))

        fia.side_effect = check_fia
        output = (
            backend.forward_decode(**args)
            if width == 1
            else backend.forward_extend(**args)
        )
        expected_output = torch.zeros(
            bucket_tokens, HEADS, LATENT, dtype=torch.bfloat16
        )
        if active_tokens:
            full_output = torch.cat(fia_outputs)
            expected_output[:active_tokens] = full_output[
                :active_tokens, rank * HEADS : (rank + 1) * HEADS
            ]
            assert group.calls == 2
            fia.assert_called_once()
        else:
            assert group.calls == 0
            fia.assert_not_called()
        _assert_bits(output, expected_output.flatten(1))
        backend.token_to_kv_pool.set_kv_buffer.assert_not_called()


@pytest.mark.parametrize("bs", [1, 2, 8, 9, 10, 16])
@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("graph", [False, True])
def test_c8_a2a_payload_local_metadata_and_reverse_layout(bs, width, graph):
    _exercise(bs, width, graph=graph)


@pytest.mark.parametrize("width", [1, 4])
def test_eager_a2a_trims_dp_padding_before_exchange(width):
    _exercise(2, width, dp_pad_requests=2)


@pytest.mark.parametrize("live_requests", [0, 2])
@pytest.mark.parametrize("width", [1, 4])
def test_graph_a2a_keeps_bucket_with_idle_or_dp_padding(width, live_requests):
    _exercise(10, width, graph=True, live_requests=live_requests)


@pytest.mark.parametrize("width", [1, 4])
def test_eager_a2a_idle_rank_needs_no_collective(width):
    _exercise(2, width, live_requests=0)


def test_a2a_rejects_variable_length_draft_extend():
    backend, namespace, fia, args = _case(tokens=4, width=4)
    backend.use_sparse_attn_a2a = True
    namespace["get_parallel"] = lambda: SimpleNamespace(attn_tp_size=TP, attn_tp_rank=0)
    args["forward_batch"].forward_mode.is_draft_extend_v2 = lambda: True
    args["forward_batch"].extend_seq_lens_cpu = [1, 3]
    with pytest.raises(NotImplementedError, match="draft|static"):
        backend.forward_extend(**args)
    fia.assert_not_called()


def test_a2a_rejects_multiple_kv_heads_before_exchange():
    backend, namespace, fia, args = _case()
    backend.use_sparse_attn_a2a = True
    namespace["get_parallel"] = lambda: SimpleNamespace(attn_tp_size=TP, attn_tp_rank=0)
    args["layer"].tp_k_head_num = 2
    with pytest.raises(NotImplementedError, match="single KV head"):
        backend.forward_decode(**args)
    fia.assert_not_called()
