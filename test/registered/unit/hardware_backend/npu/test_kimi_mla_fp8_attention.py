"""CPU contract tests for the NPU dense MLA C8 attention adapter.

Load the actual backend methods without importing the NPU runtime. The mocked
FIA only records its arguments and writes a sentinel; these are not numerical
accuracy or device-kernel tests.
"""

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from sglang.test.ci.ci_register import register_cpu_ci
from test_kimi_mla_fp8_prepare import _case as _prepare_case

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "python/sglang").is_dir()
)
SOURCE = ROOT / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
TREE = ast.parse(SOURCE.read_text())
BACKEND = next(
    n
    for n in TREE.body
    if isinstance(n, ast.ClassDef) and n.name == "AscendAttnBackend"
)


def _load_backend(fia_v2):
    methods = {
        "_forward_mla_fp8",
        "forward_decode",
        "forward_extend",
        "forward_mtp",
        "_init_cuda_graph_metadata",
        "_apply_cuda_graph_metadata",
    }
    cls = ast.ClassDef(
        name="Backend",
        bases=[],
        keywords=[],
        body=[
            n
            for n in BACKEND.body
            if isinstance(n, ast.FunctionDef) and n.name in methods
        ],
        decorator_list=[],
    )
    reshape = next(
        n
        for n in TREE.body
        if isinstance(n, ast.FunctionDef) and n.name == "_reshape_kv_for_fia_nz"
    )
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                next(
                    n
                    for n in TREE.body
                    if isinstance(n, ast.ClassDef) and n.name == "ForwardMetadata"
                ),
                reshape,
                cls,
            ],
            type_ignores=[],
        )
    )
    namespace = {
        "torch": torch,
        "dataclass": dataclass,
        "np": np,
        "torch_npu": SimpleNamespace(
            npu_fused_infer_attention_score_v2=SimpleNamespace(out=fia_v2)
        ),
        "is_mla_preprocess_enabled": lambda: False,
        "is_fia_nz": lambda: False,
        "gather_mla_cache_pages": lambda cache, ids, **kw: cache.index_select(0, ids),
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)  # noqa: S102
    return namespace["Backend"](), namespace


def _case(tokens=2, live=None, width=1):
    def call(q, k, v, **kwargs):
        kwargs["out"][0].fill_(7)

    fia = Mock(side_effect=call)
    backend, namespace = _load_backend(fia)
    backend.use_mla = backend.use_mla_fp8 = True
    backend.graph_mode = backend.use_sparse_attn_a2a = False
    backend.attn_cp_size = 1
    backend.is_dllm_model = False
    backend.kv_lora_rank = 512
    backend.qk_rope_head_dim = 64
    backend.page_size = 128
    backend.speculative_num_draft_tokens = 99  # Must use the actual spec_info width.
    backend.mtp_mask = torch.ones((16, 16), dtype=torch.bool).triu(1)
    kv = torch.ones((4, 128, 1, 512)).to(torch.float8_e4m3fn)
    side = torch.ones((4, 128, 1, 64), dtype=torch.bfloat16)
    backend.token_to_kv_pool = SimpleNamespace(
        get_key_buffer=Mock(return_value=kv),
        get_value_buffer=Mock(return_value=side),
        set_kv_buffer=Mock(),
    )
    backend.forward_metadata = SimpleNamespace(
        seq_lens_cpu_int=torch.tensor([130, 259, 777]),
        seq_lens_cpu_list=None,
        block_tables=torch.tensor([[1, 2, 0], [2, 3, 1], [0, 0, 0]], dtype=torch.int32),
    )
    mode = SimpleNamespace(
        is_target_verify=lambda: width > 1, is_draft_extend_v2=lambda: False
    )
    batch = SimpleNamespace(
        num_token_non_padded_cpu=tokens if live is None else live,
        forward_mode=mode,
        spec_info=SimpleNamespace(draft_token_num=width, ragged_verify_layout=None),
    )
    layer = SimpleNamespace(
        layer_id=3, tp_q_head_num=12, tp_k_head_num=1, v_head_dim=512, scaling=192**-0.5
    )
    q = torch.ones((tokens, 12, 512)).to(torch.float8_e4m3fn)
    q_rope = torch.ones((tokens, 12, 64), dtype=torch.bfloat16)
    q_scale = torch.arange(1, tokens * 12 + 1, dtype=torch.float32).view(tokens, 12, 1)
    kv_scale = torch.tensor([0.25], dtype=torch.float32)
    args = {
        "q": q,
        "k": None,
        "v": None,
        "layer": layer,
        "forward_batch": batch,
        "save_kv_cache": False,
        "q_rope": q_rope,
        "dequant_scale_q_nope": q_scale,
        "fp8_kv_scale": kv_scale,
    }
    return backend, namespace, fia, args


def test_decode_fp8_contract_and_padding():
    backend, _, fia, args = _case(tokens=4, live=2)
    output = backend.forward_decode(**args)
    q, k, v = fia.call_args.args
    kw = fia.call_args.kwargs
    assert q.shape == (2, 1, 12, 512) and q.dtype == torch.float8_e4m3fn
    assert k is v and k.shape == (4, 1, 128, 512)
    assert k.dtype == torch.float8_e4m3fn
    assert kw["query_rope"].shape == (2, 1, 12, 64)
    assert kw["key_rope"].dtype == torch.bfloat16
    assert kw["input_layout"] == "BSND" and kw["sparse_mode"] == 0
    assert kw["actual_seq_qlen"] is None and kw["actual_seq_kvlen"] == [130, 259]
    assert kw["block_table"].shape[0] == 2
    assert kw["dequant_scale_query"].dtype == torch.float32
    torch.testing.assert_close(
        kw["dequant_scale_query"], args["dequant_scale_q_nope"][:2, :, 0].unsqueeze(1)
    )
    assert (
        kw["query_quant_mode"] == 3
        and kw["key_quant_mode"] == kw["value_quant_mode"] == 0
    )
    assert kw["dequant_scale_key"] is kw["dequant_scale_value"]
    assert kw["softmax_scale"] == 192**-0.5
    assert output.dtype == torch.bfloat16 and output.shape == (4, 12 * 512)
    assert (output[:2] == 7).all() and (output[2:] == 0).all()
    backend.token_to_kv_pool.set_kv_buffer.assert_not_called()


def test_dspark_verify_propagates_scales_and_does_not_add_length_twice():
    backend, _, fia, args = _case(tokens=8, live=6, width=3)
    output = backend.forward_extend(**args)
    q = fia.call_args.args[0]
    kw = fia.call_args.kwargs
    assert q.shape == (6, 12, 512)
    assert kw["input_layout"] == "TND" and kw["actual_seq_qlen"] == [3, 6]
    assert kw["actual_seq_kvlen"] == [130, 259]
    assert kw["sparse_mode"] == 3 and kw["atten_mask"] is backend.mtp_mask
    torch.testing.assert_close(
        kw["dequant_scale_query"], args["dequant_scale_q_nope"][:6, :, 0]
    )
    assert output.shape == (8, 12 * 512) and (output[6:] == 0).all()


def test_empty_dp_rank_skips_fia():
    backend, _, fia, args = _case(tokens=2, live=0)
    output = backend.forward_decode(**args)
    assert output.dtype == torch.bfloat16 and (output == 0).all()
    fia.assert_not_called()


@pytest.mark.parametrize(
    "attribute,value",
    [("use_sparse_attn_a2a", True), ("attn_cp_size", 2)],
)
def test_unsupported_c8_parallel_paths_fail_explicitly(attribute, value):
    backend, _, fia, args = _case()
    setattr(backend, attribute, value)
    with pytest.raises(NotImplementedError, match="non-CP/non-A2A"):
        backend.forward_decode(**args)
    fia.assert_not_called()


@pytest.mark.parametrize("bucket", [8, 16])
@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("live_requests", [0, 5])
def test_c8_graph_keeps_bucket_shape_for_q_scale_and_output(
    bucket, width, live_requests
):
    backend, _, fia, args = _case(
        tokens=bucket * width, live=live_requests * width, width=width
    )
    backend.graph_mode = True
    fm = backend.forward_metadata
    fm.seq_lens_cpu_int = None
    # Capture-time lengths are placeholders. The runner updates the captured
    # FIA node's host lengths on replay, not Python's live-token slicing.
    fm.seq_lens_cpu_list = [0] * bucket
    fm.block_tables = torch.zeros(bucket, 3, dtype=torch.int32)
    if width == 1:
        output = backend.forward_decode(**args)
        shape, scale_shape, qlen = (bucket, 1, 12, 512), (bucket, 1, 12), None
    else:
        output = backend.forward_extend(**args)
        shape, scale_shape = (bucket * width, 12, 512), (bucket * width, 12)
        qlen = list(range(width, bucket * width + 1, width))
    q, k, v = fia.call_args.args
    kw = fia.call_args.kwargs
    assert q.shape == shape and q.dtype == torch.float8_e4m3fn
    assert k is v and k.dtype == torch.float8_e4m3fn
    assert kw["dequant_scale_query"].shape == scale_shape
    assert kw["dequant_scale_query"].dtype == torch.float32
    assert kw["actual_seq_qlen"] == qlen
    assert kw["actual_seq_kvlen"] == [0] * bucket
    assert kw["block_table"].data_ptr() == fm.block_tables.data_ptr()
    assert kw["out"][0].dtype == torch.bfloat16
    assert output.shape == (bucket * width, 12 * 512)
    assert kw["out"][0].data_ptr() == output.data_ptr()


def _graph_metadata_case(width):
    backend, ns, _fia, args = _case(tokens=8 * width, width=width)
    backend.speculative_num_draft_tokens = width
    backend.is_hybrid_swa = backend.use_sliding_window_kv_pool = False
    backend.needs_cpu_seq_lens = True
    backend.use_fias_v2_bsnd = False
    backend.speculative_step_id = 0
    backend.speculative_step_offset_npu = torch.tensor(1, dtype=torch.int32)
    backend.tp_q_head_num = 12
    backend.graph_metadata = {"block_tables": torch.zeros(16, 4, dtype=torch.int32)}
    backend.req_to_token = torch.arange(9 * 512).view(9, 512)
    backend.req_to_token[0].zero_()
    ns["_is_dflash_verify"] = lambda spec: spec is not None
    ns["flash_mla_with_kvcache_metadata"] = Mock(
        return_value=torch.zeros(4096, dtype=torch.int32)
    )
    mode = args["forward_batch"].forward_mode
    mode.is_dllm_extend = lambda: False
    mode.is_decode_or_idle = lambda: width == 1
    device_lengths = torch.zeros(8, dtype=torch.int32)
    return backend, ns, mode, device_lengths


@pytest.mark.parametrize("width", [1, 3])
def test_c8_graph_metadata_updates_pages_without_mutating_lengths(width):
    backend, ns, mode, device_lengths = _graph_metadata_case(width)
    fm = backend._init_cuda_graph_metadata(8, mode, device_lengths)
    table_ptr = fm.block_tables.data_ptr()
    assert fm.actual_seq_lengths_q.tolist() == list(range(width, 8 * width + 1, width))
    assert fm.metadata_flash_mla is None and fm.seqused_q is None
    req = torch.tensor([1, 2, 0, 0, 0, 0, 0, 0])
    # Across page boundaries, accepted/rejected verify advances, an idle step
    # and a subsequent live step, the graph's table address must stay fixed.
    for final_lengths in ([128, 129], [129, 257], [128, 130], [0, 0], [257, 128]):
        cpu_lengths = torch.tensor(
            [*final_lengths, 0, 0, 0, 0, 0, 0], dtype=torch.int32
        )
        device_lengths.copy_(cpu_lengths)
        if width > 1:
            device_lengths.sub_(torch.where(cpu_lengths > 0, width, 0))
        before = device_lengths.clone()
        backend._apply_cuda_graph_metadata(
            bs=8,
            req_pool_indices=req,
            seq_lens=device_lengths,
            seq_lens_cpu=cpu_lengths,
            forward_mode=mode,
            spec_info=SimpleNamespace() if width > 1 else None,
        )
        assert backend.forward_metadata is fm and backend.graph_mode
        assert fm.block_tables.data_ptr() == table_ptr
        torch.testing.assert_close(device_lengths, before)
        max_len = max(final_lengths)
        pages = (max_len + 127) // 128
        expected = torch.zeros_like(fm.block_tables)
        expected[:, :pages] = backend.req_to_token[req, 0:max_len:128] // 128
        torch.testing.assert_close(fm.block_tables, expected)
    ns["flash_mla_with_kvcache_metadata"].assert_not_called()


def test_bf16_graph_metadata_still_uses_flashmla(monkeypatch):
    backend, ns, mode, lengths = _graph_metadata_case(width=3)
    backend.use_mla_fp8 = False
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            get_device_properties=lambda: SimpleNamespace(
                cube_core_num=2, vector_core_num=2
            )
        ),
        raising=False,
    )
    fm = backend._init_cuda_graph_metadata(8, mode, lengths)
    assert fm.actual_seq_lengths_q.tolist() == list(range(0, 25, 3))
    assert fm.metadata_flash_mla is not None and fm.seqused_q.shape == (8,)
    backend._apply_cuda_graph_metadata(
        bs=8,
        req_pool_indices=torch.zeros(8, dtype=torch.int64),
        seq_lens=lengths,
        seq_lens_cpu=lengths.clone(),
        forward_mode=mode,
        spec_info=SimpleNamespace(),
    )
    ns["flash_mla_with_kvcache_metadata"].assert_called_once()


def test_ragged_verify_does_not_reuse_static_lengths():
    backend, _, fia, args = _case(tokens=6, width=3)
    args["forward_batch"].spec_info.ragged_verify_layout = object()
    with pytest.raises(NotImplementedError, match="static DSpark"):
        backend.forward_extend(**args)
    fia.assert_not_called()


@pytest.mark.parametrize("name", ["dequant_scale_q_nope", "fp8_kv_scale"])
def test_c8_requires_real_descales(name):
    backend, _, fia, args = _case()
    args[name] = None
    with pytest.raises(ValueError, match="descales"):
        backend.forward_decode(**args)
    fia.assert_not_called()


def test_non_c8_decode_keeps_existing_dispatch():
    backend, _, fia, args = _case()
    backend.use_mla_fp8 = False
    backend.graph_mode = True
    backend.enable_torch_compile = False
    backend.forward_decode_graph = Mock(return_value="existing BF16/GQA path")
    assert backend.forward_decode(**args) == "existing BF16/GQA path"
    fia.assert_not_called()


@pytest.mark.parametrize(
    "mla,dsa,dtype,expected",
    [
        (True, False, "fp8_e4m3", True),
        (True, False, "auto", False),
        (True, True, "fp8_e4m3", False),
        (False, False, "fp8_e4m3", False),
    ],
)
def test_c8_gate_excludes_dsa_and_draft_gqa(mla, dsa, dtype, expected):
    init = next(
        n
        for n in BACKEND.body
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    assignment = next(
        n
        for n in init.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "use_mla_fp8" for t in n.targets
        )
    )
    backend = SimpleNamespace(use_mla=mla)
    runner = SimpleNamespace(
        kv_cache_dtype_str=dtype, model_config=SimpleNamespace(hf_text_config=object())
    )
    exec(  # noqa: S102
        compile(ast.Module(body=[assignment], type_ignores=[]), str(SOURCE), "exec"),
        {"self": backend, "model_runner": runner, "is_deepseek_dsa": lambda _: dsa},
    )
    assert backend.use_mla_fp8 is expected


def test_prefix_reader_dequantizes_only_historical_latent(monkeypatch):
    backend, _, fia, args = _case(tokens=1)
    layer = args["layer"]
    layer.tp_k_head_num = layer.tp_q_head_num
    layer.qk_head_dim, layer.v_head_dim = 192, 128
    backend.qk_nope_head_dim = 128
    layer.kv_b_proj = Mock(
        return_value=(torch.zeros((128, 12 * 256), dtype=torch.bfloat16),)
    )
    fm = backend.forward_metadata
    fm.flatten_prefix_block_tables = torch.tensor([1])
    fm.prefix_lens = torch.tensor([2])
    fm.extend_seq_lens_cpu_int = [1]
    backend.fia_mask = torch.ones((4, 4), dtype=torch.bool).triu(1)
    args["forward_batch"].extend_prefix_lens_cpu = [2]
    args.update(
        q=torch.ones((1, 12, 192), dtype=torch.bfloat16),
        k=torch.ones((1, 12, 192), dtype=torch.bfloat16),
        v=torch.ones((1, 12, 128), dtype=torch.bfloat16),
        dequant_scale_q_nope=None,
    )
    # CPU has no FP8 index_select, so gather the FP8 page through a float view.
    backend.forward_extend.__func__.__globals__["gather_mla_cache_pages"] = (
        lambda cache, ids, **kw: cache.float().index_select(0, ids).to(cache.dtype)
    )
    eager_fia = Mock(
        return_value=(torch.zeros((1, 1, 12, 128), dtype=torch.bfloat16), None)
    )
    monkeypatch.setattr(
        torch.ops.npu, "npu_fused_infer_attention_score", eager_fia, raising=False
    )
    backend.forward_extend(**args)
    latent = layer.kv_b_proj.call_args.args[0]
    assert latent.dtype == torch.bfloat16
    torch.testing.assert_close(latent, torch.full_like(latent, 0.25))
    assert eager_fia.call_args.kwargs["input_layout"] == "BSND"
    fia.assert_not_called()


@pytest.mark.parametrize("mode", ["mha", "mla"])
def test_bf16_prepare_core_preserves_ling_trailing_gate(mode):
    ns, args, model, _, _, _ = _prepare_case(2, fp8=False)
    state = ns[f"forward_{mode}_prepare_npu"](*args)
    gate = torch.full((2, 6), 0.75, dtype=torch.bfloat16)
    model._apply_gated = Mock(side_effect=lambda x, g: x * g)
    model.attn_mha = Mock(return_value=torch.ones((2, 2, 3), dtype=torch.bfloat16))
    model.attn_mqa = Mock(return_value=torch.ones((2, 2, 8), dtype=torch.bfloat16))
    ns["torch_npu"].npu_transpose_batchmatmul = lambda x, w, **_: x.new_ones(
        (x.shape[0], x.shape[1], w.shape[-1])
    )
    # DsV3MLA.forward appends the gate to the prepare tuple before core dispatch.
    result = ns[f"forward_{mode}_core_npu"](model, *(state + (gate,)))
    assert model._apply_gated.call_args.args[1] is gate
    torch.testing.assert_close(result, gate)
    attention = model.attn_mha if mode == "mha" else model.attn_mqa
    assert "fp8_kv_scale" not in attention.call_args.kwargs
    assert "dequant_scale_q_nope" not in attention.call_args.kwargs


def test_real_prepare_core_reaches_dspark_c8_verify_backend():
    ns, args, model, prepare_pool, _, _ = _prepare_case(6)
    backend, _, fia, backend_args = _case(tokens=6, width=3)
    backend.kv_lora_rank = model.kv_lora_rank
    backend.qk_rope_head_dim = model.qk_rope_head_dim
    backend.token_to_kv_pool.get_key_buffer.return_value = torch.zeros(
        (4, 128, 1, model.kv_lora_rank)
    ).to(torch.float8_e4m3fn)
    backend.token_to_kv_pool.get_value_buffer.return_value = torch.zeros(
        (4, 128, 1, model.qk_rope_head_dim), dtype=torch.bfloat16
    )
    layer = SimpleNamespace(
        layer_id=model.layer_id,
        tp_q_head_num=model.num_local_heads,
        tp_k_head_num=1,
        v_head_dim=model.kv_lora_rank,
        scaling=model.qk_head_dim**-0.5,
    )
    batch = args[3]
    batch.forward_mode = backend_args["forward_batch"].forward_mode
    batch.spec_info = backend_args["forward_batch"].spec_info
    batch.num_token_non_padded_cpu = 6
    # Stand in only for the RadixAttention wrapper; execute both real core and
    # real backend dispatch methods, including forward_extend -> forward_mtp.
    model.attn_mqa = lambda q, k, v, fb, **kw: backend.forward_extend(
        q, k, v, layer, fb, **kw
    )
    ns["torch_npu"].npu_transpose_batchmatmul = Mock(
        side_effect=lambda x, w, **_: x.new_ones((x.shape[0], x.shape[1], w.shape[-1]))
    )
    state = ns["forward_mla_prepare_npu"](*args)
    result = ns["forward_mla_core_npu"](model, *state)
    q, k, v = fia.call_args.args
    kw = fia.call_args.kwargs
    assert q.dtype == k.dtype == v.dtype == torch.float8_e4m3fn
    assert q.shape == (6, model.num_local_heads, model.kv_lora_rank)
    assert kw["input_layout"] == "TND"
    assert kw["actual_seq_qlen"] == [3, 6]
    assert kw["actual_seq_kvlen"] == [130, 259]
    torch.testing.assert_close(kw["dequant_scale_query"], state[-1].squeeze(-1))
    torch.testing.assert_close(
        kw["dequant_scale_key"], model.fak_descale_float.flatten()
    )
    projection_input = ns["torch_npu"].npu_transpose_batchmatmul.call_args.args[0]
    assert projection_input.dtype == torch.bfloat16
    assert result.dtype == torch.bfloat16 and result.shape == (6, 6)
    prepare_pool.set_kv_buffer.assert_called_once()
    backend.token_to_kv_pool.set_kv_buffer.assert_not_called()
