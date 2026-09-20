"""CPU contract tests for the NPU ordinary MLA C8 cache.

Extract the pool class so these tests do not require torch_npu or an NPU. The
scatter mock checks the storage contract, not the device operator's support.
"""

import ast
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional

import pytest
import torch
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _KVCache:
    def __init__(
        self,
        size,
        page_size,
        dtype,
        layer_num,
        device,
        enable_memory_saver,
        start_layer=None,
        end_layer=None,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.store_dtype = torch.uint8 if dtype == torch.float8_e4m3fn else dtype
        self.layer_num = layer_num
        self.device = device
        self.start_layer = start_layer or 0
        self.end_layer = end_layer
        self.layer_transfer_counter = None
        self.memory_saver_adapter = SimpleNamespace(region=lambda _: nullcontext())

    def _finalize_allocation_log(self, _):
        self.allocated_bytes = self.get_kv_size_bytes()

    def get_kv_buffer_shape(self):
        key, value = self.get_kv_buffer(self.start_layer)
        return key.shape, value.shape


class _MLATokenToKVPool(_KVCache):
    def _clear_buffers(self):
        del self.kv_buffer

    def move_kv_cache(self, tgt_loc, src_loc):
        self.parent_move = (tgt_loc, src_loc)


def _scatter(buffer, indices, values):
    assert buffer.dtype == values.dtype
    if buffer.dtype == torch.float8_e4m3fn:
        buffer, values = buffer.view(torch.uint8), values.view(torch.uint8)
    buffer[indices.flatten().long()] = values
    return buffer


def _pool(*, dtype=torch.float8_e4m3fn, nz=False, **kwargs):
    path = (
        Path(__file__).resolve().parents[5]
        / "python/sglang/srt/hardware_backend/npu/memory_pool_npu.py"
    )
    source = ast.parse(path.read_text())
    wanted = {"_mla_fia_nz_scatter_indices", "NPUMLATokenToKVPool"}
    module = ast.Module(
        body=[node for node in source.body if getattr(node, "name", None) in wanted],
        type_ignores=[],
    )
    namespace = {
        "torch": torch,
        "Optional": Optional,
        "MLATokenToKVPool": _MLATokenToKVPool,
        "get_bool_env_var": lambda _: nz,
        "get_tensor_size_bytes": lambda value: value.numel() * value.element_size(),
        "GPU_MEMORY_TYPE_KV_CACHE": "kv_cache",
        "unwrap_write_loc": lambda loc: (loc, None, None),
        "torch_npu": SimpleNamespace(npu_scatter_nd_update_=_scatter),
        "DSA_KV_QUANT_TILE_SIZE": 128,
        "get_dsa_fp8_packed_cache_dim": lambda **_: 656,
    }
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - trusted repo AST
    return namespace["NPUMLATokenToKVPool"](
        size=16,
        page_size=4,
        dtype=dtype,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_num=2,
        device="cpu",
        enable_memory_saver=False,
        start_layer=3,
        end_layer=5,
        **kwargs,
    )


def _layer(layer_id=3, scale=0.25):
    return SimpleNamespace(
        layer_id=layer_id, fak_descale_float=torch.tensor([scale], dtype=torch.float32)
    )


def _inputs(n):
    # Include saturation, signs and non-FP8-exact values.
    latent = torch.linspace(-160, 160, n * 512).to(torch.bfloat16).reshape(n, 1, 512)
    rope = torch.linspace(-3, 3, n * 64).to(torch.bfloat16).reshape(n, 1, 64)
    return latent, rope


def _quantized(latent, scale):
    return (latent.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()


def _logical(buffer, pool):
    buffer = buffer.float()
    if pool.use_fia_nz:
        dim = buffer.shape[-1]
        buffer = (
            buffer.view(-1, 1, dim // 16, pool.page_size, 16)
            .permute(0, 3, 1, 2, 4)
            .reshape(-1, 1, dim)
        )
    return buffer.reshape(-1, 1, buffer.shape[-1])


def test_split_allocation_dtype_size_shape_and_layer_offset():
    pool = _pool()
    assert pool.mla_kv_cache_store_fp8
    assert not pool.dsa_kv_cache_store_fp8
    assert not hasattr(pool, "kv_buffer")
    key, rope = pool.get_kv_buffer(4)
    assert key.dtype == torch.float8_e4m3fn
    assert rope.dtype == torch.bfloat16
    assert key.is_contiguous() and rope.is_contiguous()
    assert key.data_ptr() == pool.k_buffer[1].data_ptr()
    assert rope.data_ptr() == pool.v_buffer[1].data_ptr()
    assert pool.get_kv_buffer_shape() == ((5, 4, 1, 512), (5, 4, 1, 64))
    assert pool.allocated_bytes == 2 * 20 * (512 + 64 * 2)
    assert not bool(key.float().any())
    assert not bool(rope.any())


@pytest.mark.parametrize("nz", [False, True])
def test_write_quantizes_only_latent_with_per_layer_scale(nz):
    pool = _pool(nz=nz)
    loc = torch.tensor([4, 7, 8])
    latent, rope = _inputs(3)
    original = latent.clone(), rope.clone()
    for layer_id, scale in ((3, 0.25), (4, 0.5)):
        pool.set_kv_buffer(_layer(layer_id, scale), loc, latent, rope)
        key_buffer, rope_buffer = pool.get_kv_buffer(layer_id)
        torch.testing.assert_close(
            _logical(key_buffer, pool)[loc], _quantized(latent, scale)
        )
        torch.testing.assert_close(_logical(rope_buffer, pool)[loc], rope.float())
    torch.testing.assert_close(latent, original[0])
    torch.testing.assert_close(rope, original[1])


@pytest.mark.parametrize("nz", [False, True])
def test_chunk_prefill_then_decode_preserves_existing_quantized_cache(nz):
    pool = _pool(nz=nz)
    latent, rope = _inputs(5)
    for start, end in ((0, 2), (2, 4), (4, 5)):
        pool.set_kv_buffer(
            _layer(),
            torch.arange(start + 4, end + 4),
            latent[start:end],
            rope[start:end],
        )
    torch.testing.assert_close(
        _logical(pool.get_key_buffer(3), pool)[4:9], _quantized(latent, 0.25)
    )
    torch.testing.assert_close(
        _logical(pool.get_value_buffer(3), pool)[4:9], rope.float()
    )


def test_merged_input_is_split_and_noncontiguous_input_is_supported():
    pool = _pool()
    latent, rope = _inputs(2)
    merged = torch.cat((latent, rope), dim=-1)
    assert not merged[..., :512].is_contiguous()
    pool.set_kv_buffer(_layer(), torch.tensor([4, 5]), merged, None)
    torch.testing.assert_close(
        pool.get_key_buffer(3).float().reshape(-1, 1, 512)[4:6],
        _quantized(latent, 0.25),
    )
    torch.testing.assert_close(pool.get_value_buffer(3).reshape(-1, 1, 64)[4:6], rope)


@pytest.mark.parametrize("scale", [None, torch.ones(2)])
def test_invalid_scale_shape_fails_before_writing(scale):
    pool = _pool()
    layer = SimpleNamespace(layer_id=3, fak_descale_float=scale)
    with pytest.raises(ValueError, match="per-tensor KV descale"):
        pool.set_kv_buffer(layer, torch.tensor([4]), *_inputs(1))
    assert not bool(pool.k_buffer.float().any())
    assert not bool(pool.v_buffer.any())


@pytest.mark.parametrize("nz", [False, True])
def test_accepted_draft_relocation_copies_both_parts_and_handles_overlap(nz):
    pool = _pool(nz=nz)
    latent, rope = _inputs(4)
    for layer_id in (3, 4):
        pool.set_kv_buffer(_layer(layer_id), torch.arange(4, 8), latent, rope)
    source, target = torch.tensor([4, 5, 6]), torch.tensor([5, 6, 8])
    before = [
        (part.clone(), _logical(part, pool).clone())
        for part in (pool.k_buffer[0], pool.v_buffer[0])
    ]
    pool.move_kv_cache(target, source)
    for layer_id in (3, 4):
        for part, (_, original) in zip(pool.get_kv_buffer(layer_id), before):
            torch.testing.assert_close(_logical(part, pool)[target], original[source])
    pool.move_kv_cache(
        torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    )


def test_split_transfer_and_offload_fail_clearly_and_clear_releases_both():
    pool = _pool()
    with pytest.raises(NotImplementedError, match="cache transfer is not supported"):
        pool.get_contiguous_buf_infos()
    with pytest.raises(NotImplementedError, match="cache offload is not supported"):
        pool.get_cpu_copy(torch.tensor([4]))
    with pytest.raises(NotImplementedError, match="cache offload is not supported"):
        pool.load_cpu_copy([], torch.tensor([4]))
    pool._clear_buffers()
    assert not hasattr(pool, "k_buffer") and not hasattr(pool, "v_buffer")


def test_bf16_preserves_merged_buffer_contract_and_parent_move():
    pool = _pool(dtype=torch.bfloat16)
    assert not pool.mla_kv_cache_store_fp8
    latent, rope = _inputs(2)
    pool.set_kv_buffer(SimpleNamespace(layer_id=3), torch.tensor([4, 5]), latent, rope)
    merged = pool.get_kv_buffer(3)
    assert isinstance(merged, torch.Tensor)
    assert merged.dtype == torch.bfloat16
    assert merged.shape == (5, 4, 1, 576)
    torch.testing.assert_close(
        merged.reshape(-1, 1, 576)[4:6], torch.cat((latent, rope), dim=-1)
    )
    pool.move_kv_cache(torch.tensor([4]), torch.tensor([5]))
    assert hasattr(pool, "parent_move")
    pool._clear_buffers()
    assert not hasattr(pool, "kv_buffer")


def test_indexed_fp8_cache_does_not_enable_ordinary_mla_split():
    pool = _pool(index_head_dim=128)
    assert not pool.mla_kv_cache_store_fp8
    assert not pool.dsa_kv_cache_store_fp8
    assert pool.kv_buffer.shape[-1] == 576
    assert pool.index_k_buffer.dtype == torch.bfloat16


def test_dsa_packed_cache_keeps_its_existing_storage_and_write_path(monkeypatch):
    module_name = "sglang.srt.layers.attention.dsa.dsa_npu_indexer"
    module = ModuleType(module_name)
    module.create_npu_hadamard_128 = lambda size, device: torch.eye(size, device=device)
    monkeypatch.setitem(sys.modules, module_name, module)
    pool = _pool(
        index_head_dim=128, enable_npu_quant_lightning_indexer=True, kv_cache_dim=656
    )
    assert pool.dsa_kv_cache_store_fp8
    assert not pool.mla_kv_cache_store_fp8
    assert not hasattr(pool, "k_buffer") and not hasattr(pool, "v_buffer")
    assert pool.get_kv_buffer(3).shape == (5, 4, 1, 656)
    assert pool.index_k_buffer.dtype == torch.float8_e4m3fn
    assert pool.index_k_scale_buffer.dtype == torch.float32
    packed = torch.ones(1, 1, 656, dtype=torch.float8_e4m3fn)
    pool._pack_dsa_fp8_kv_cache = lambda key, rope, n: packed
    # DSA writes do not require the ordinary MLA per-layer descale attribute.
    pool.set_kv_buffer(SimpleNamespace(layer_id=3), torch.tensor([4]), *_inputs(1))
    torch.testing.assert_close(
        pool.get_kv_buffer(3).float().reshape(-1, 1, 656)[4:5], packed.float()
    )


@pytest.mark.parametrize("nz", [False, True])
@pytest.mark.parametrize("loc_dtype", [torch.int32, torch.int64])
def test_fixed_bucket_live_idle_writes_preserve_cache_addresses_and_live_slots(
    nz, loc_dtype
):
    """Replay the cache-write contract, without claiming NPU graph support.

    Dummy rows can race when they all target slot zero. Only its isolation from
    live slots is required; its final contents are deliberately not asserted.
    """
    pool = _pool(nz=nz)
    layer = _layer()
    bucket = 4
    latent, rope = _inputs(bucket)
    loc = torch.zeros(bucket, dtype=loc_dtype)
    original_ptrs = (
        pool.k_buffer.data_ptr(),
        pool.v_buffer.data_ptr(),
        latent.data_ptr(),
        rope.data_ptr(),
        loc.data_ptr(),
        layer.fak_descale_float.data_ptr(),
    )
    expected_k = torch.zeros(20, 1, 512)
    expected_rope = torch.zeros(20, 1, 64)

    for step, locations in enumerate(
        ([4, 5, 0, 0], [8, 9, 0, 0], [0] * 4, [5, 12, 0, 0])
    ):
        # Change values in-place, as the graph runner refreshes fixed buffers.
        loc.copy_(torch.tensor(locations, dtype=loc_dtype))
        next_latent, next_rope = _inputs(bucket)
        latent.copy_(next_latent / (step + 1))
        rope.copy_(next_rope + step)
        pool.set_kv_buffer(layer, loc, latent, rope)

        for row, slot in enumerate(locations):
            if slot:
                expected_k[slot] = _quantized(latent[row], 0.25)
                expected_rope[slot] = rope[row].float()
        torch.testing.assert_close(
            _logical(pool.get_key_buffer(3), pool)[1:], expected_k[1:]
        )
        torch.testing.assert_close(
            _logical(pool.get_value_buffer(3), pool)[1:], expected_rope[1:]
        )
        assert original_ptrs == (
            pool.k_buffer.data_ptr(),
            pool.v_buffer.data_ptr(),
            latent.data_ptr(),
            rope.data_ptr(),
            loc.data_ptr(),
            layer.fak_descale_float.data_ptr(),
        )
        assert not bool(pool.get_key_buffer(4).float().any())
        assert not bool(pool.get_value_buffer(4).any())


@pytest.mark.parametrize("nz", [False, True])
def test_cache_writer_fx_graph_uses_runtime_values_with_fixed_shapes(nz):
    """CPU FX checks tensor dependencies, not torch_npu graph compatibility."""
    from torch.fx.experimental.proxy_tensor import make_fx

    pool = _pool(nz=nz)
    layer = _layer()
    latent, rope = _inputs(4)
    loc = torch.tensor([4, 5, 0, 0], dtype=torch.int32)

    def write(key, side, indices):
        pool.set_kv_buffer(layer, indices, key, side)
        return pool.get_key_buffer(3), pool.get_value_buffer(3)

    # The mock scatter lowers to native tensor operations. Trace with fixed
    # shapes, then reuse the graph with different locations and values.
    graph = make_fx(write)(latent, rope, loc)
    assert not any(
        "_local_scalar_dense" in str(node.target) or "nonzero" in str(node.target)
        for node in graph.graph.nodes
    )
    original_ptrs = pool.k_buffer.data_ptr(), pool.v_buffer.data_ptr()
    next_latent, next_rope = latent / 2, rope + 1
    next_loc = torch.tensor([8, 9, 0, 0], dtype=torch.int32)
    key, side = graph(next_latent, next_rope, next_loc)
    torch.testing.assert_close(
        _logical(key, pool)[8:10], _quantized(next_latent[:2], 0.25)
    )
    torch.testing.assert_close(_logical(side, pool)[8:10], next_rope[:2].float())
    torch.testing.assert_close(_logical(key, pool)[4:6], _quantized(latent[:2], 0.25))
    assert original_ptrs == (pool.k_buffer.data_ptr(), pool.v_buffer.data_ptr())
