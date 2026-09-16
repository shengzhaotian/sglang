"""CPU unit tests for the Ascend NPU decode-context-parallel (DCP) helpers in
``sglang.srt.hardware_backend.npu.dcp.ops``.

Collectives are simulated in one process: every DCP rank runs in its own
thread against a barrier-synchronised fake ``GroupCoordinator``.

Usage:
    python -m pytest test_npu_dcp_ops.py -v
"""

import math
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.arg_groups.model_overrides import kimi_k3 as kimi_k3_overrides
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.dcp import ops as dcp_ops
from sglang.srt.hardware_backend.npu.dcp.ops import (
    dcp_block_tables,
    dcp_gather_chunk_rows,
    dcp_interleave_pages,
    dcp_local_seq_lens,
    dcp_merge_a2a,
    dcp_merge_a2a_npu,
    dcp_merge_a2a_vllm,
    dcp_merge_ag_rs,
    dcp_merge,
    dcp_physical_write_loc,
    dcp_prefix_chunk_plan,
    dcp_verify_history_local_lens,
    lse_combine,
    mla_decode_with_lse_torch,
    npu_attention_update,
    npu_attention_update_stacked,
)
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _SharedGroupState:
    def __init__(self, world_size: int):
        self.world_size = world_size
        self.barrier = threading.Barrier(world_size, timeout=60)
        self.inputs = {}


class _FakeGroup:
    """Per-rank view of a simulated GroupCoordinator."""

    def __init__(self, state: _SharedGroupState, rank: int):
        self.state = state
        self.world_size = state.world_size
        self.rank_in_group = rank
        self._call = 0

    def _exchange(self, tensor: torch.Tensor):
        call = self._call
        self._call += 1
        self.state.inputs[(call, self.rank_in_group)] = tensor.clone()
        self.state.barrier.wait()
        return [self.state.inputs[(call, r)] for r in range(self.world_size)]

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return torch.cat(self._exchange(input_), dim=dim)

    def all_to_all_single(self, output: torch.Tensor, input: torch.Tensor) -> None:
        inputs = self._exchange(input)
        n = self.world_size
        chunk = input.numel() // n
        recv = [
            inp.view(-1)[self.rank_in_group * chunk : (self.rank_in_group + 1) * chunk]
            for inp in inputs
        ]
        output.view(-1).copy_(torch.cat(recv))

    def reduce_scatter_along_dim(self, input_: torch.Tensor, dim: int = -1):
        total = torch.stack(self._exchange(input_)).sum(dim=0)
        return total.chunk(self.world_size, dim=dim)[self.rank_in_group].contiguous()


def _run_ranks(world_size: int, fn):
    """Run fn(rank, group) on one thread per rank; return per-rank results."""
    state = _SharedGroupState(world_size)
    results = [None] * world_size
    errors = []

    def target(rank):
        try:
            results[rank] = fn(rank, _FakeGroup(state, rank))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
            state.barrier.abort()

    threads = [threading.Thread(target=target, args=(r,)) for r in range(world_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return results


def attention_update_reference(lse_list, out_list, update_type=0):
    """Pure-torch ``torch_npu.npu_attention_update`` (documented formula).

    lse_i [T] float32, out_i [T, D]; natural log:
    lsemax = max_i lse_i; lse = lsemax + log sum_i exp(lse_i - lsemax);
    out = sum_i out_i * exp(lse_i - lse). Returns (out, lse or None).
    """
    lses = torch.stack([l.float() for l in lse_list])
    outs = torch.stack([o.float() for o in out_list])
    lse_max = lses.amax(dim=0)
    lse = lse_max + torch.log(torch.exp(lses - lse_max).sum(dim=0))
    out = (outs * torch.exp(lses - lse).unsqueeze(-1)).sum(dim=0)
    return out, (lse if update_type == 1 else None)


def _build_layout(seq_lens, page_size, dcp_size, seed=0):
    """Allocate virtual KV slots like the DCP scheduler does (allocator page
    size = page_size * dcp_size) and return (req_to_token, num_pages)."""
    stride = page_size * dcp_size
    max_len = max(seq_lens)
    total_pages = sum((n + stride - 1) // stride for n in seq_lens) + 3
    allocator = PagedTokenToKVPoolAllocator(
        size=total_pages * stride,
        page_size=stride,
        dtype=torch.float32,
        device="cpu",
        kvcache=object(),
        need_sort=False,
    )
    # Scramble the free list so requests get non-contiguous pages.
    g = torch.Generator().manual_seed(seed)
    allocator.free_pages = allocator.free_pages[
        torch.randperm(len(allocator.free_pages), generator=g)
    ]
    req_to_token = torch.zeros(
        (len(seq_lens), (max_len + stride - 1) // stride * stride), dtype=torch.int32
    )
    for i, n in enumerate(seq_lens):
        need = (n + stride - 1) // stride * stride
        if need == 0:
            continue
        locs = allocator.alloc(need)
        assert locs is not None
        req_to_token[i, :need] = locs.to(torch.int32)
    return req_to_token, total_pages + 1


class TestWriteLoc(CustomTestCase):
    def test_matches_triton_kernel_math(self):
        g = torch.Generator().manual_seed(0)
        for c in (1, 2, 4, 8):
            loc = torch.randint(0, 4096, (257,), generator=g)
            loc[:3] = torch.tensor([0, -1, c])
            for r in range(c):
                got = dcp_physical_write_loc(loc, c, r)
                # set_mla_kv_buffer_kernel: valid = loc % c == r -> loc // c.
                for x, y in zip(loc.tolist(), got.tolist()):
                    if c == 1:
                        self.assertEqual(y, x)
                    elif x >= 0 and x % c == r:
                        self.assertEqual(y, x // c)
                    else:
                        self.assertEqual(y, 0)
                self.assertEqual(got.shape, loc.shape)
                self.assertEqual(got.dtype, loc.dtype)

    def test_every_token_owned_once(self):
        loc = torch.arange(64, 512)
        c = 4
        owned = sum((dcp_physical_write_loc(loc, c, r) != 0).long() for r in range(c))
        self.assertTrue(torch.all(owned == 1))


class TestLocalSeqLens(CustomTestCase):
    def test_list_and_tensor(self):
        lens = list(range(0, 40))
        for c in (1, 2, 3, 8):
            for r in range(c):
                as_list = dcp_local_seq_lens(lens, c, r)
                as_tensor = dcp_local_seq_lens(torch.tensor(lens), c, r).tolist()
                ref = [sum(1 for p in range(n) if p % c == r) for n in lens]
                self.assertEqual(as_list, ref)
                self.assertEqual(as_tensor, ref)


class TestVerifyHistoryLocalLens(CustomTestCase):
    def test_list_and_tensor(self):
        seq_lens = [0, 1, 7, 8, 9, 10, 17, 100, 3]
        for c, w in [(2, 8), (3, 4), (8, 8), (4, 1)]:
            for r in range(c):
                got = dcp_verify_history_local_lens(seq_lens, w, c, r)
                t = dcp_verify_history_local_lens(torch.tensor(seq_lens), w, c, r)
                self.assertEqual(t.tolist(), got)
                expect = [
                    sum(1 for p in range(max(n - w, 0)) if p % c == r) for n in seq_lens
                ]
                self.assertEqual(got, expect, (c, w, r))
            # Every history token is counted on exactly one rank.
            totals = [
                sum(col)
                for col in zip(
                    *[
                        dcp_verify_history_local_lens(seq_lens, w, c, r)
                        for r in range(c)
                    ]
                )
            ]
            self.assertEqual(totals, [max(n - w, 0) for n in seq_lens])


class TestLayout(CustomTestCase):
    def test_block_table_reads_owned_tokens_in_order(self):
        for c in (2, 4, 8):
            for page_size in (1, 4, 16):
                seq_lens = [1, 3, page_size * c, page_size * c * 2 + 5, 37]
                req_to_token, num_pages = _build_layout(seq_lens, page_size, c)
                block_table = dcp_block_tables(
                    req_to_token, max(seq_lens), page_size, c
                )
                self.assertEqual(block_table.dtype, torch.int32)
                for r in range(c):
                    # Physical pool of this rank; each slot stores (req, pos).
                    pool = torch.full((num_pages * page_size, 2), -1)
                    for i, n in enumerate(seq_lens):
                        v = req_to_token[i, :n].long()
                        phys = dcp_physical_write_loc(v, c, r)
                        mine = v % c == r
                        pool[phys[mine], 0] = i
                        pool[phys[mine], 1] = torch.arange(n)[mine]
                    pages = pool.view(num_pages, page_size, 2)
                    local = dcp_local_seq_lens(seq_lens, c, r)
                    for i, n in enumerate(seq_lens):
                        npg = (local[i] + page_size - 1) // page_size
                        rows = pages[block_table[i, :npg].long()].reshape(-1, 2)
                        rows = rows[: local[i]]
                        expect = [p for p in range(n) if p % c == r]
                        self.assertTrue(torch.all(rows[:, 0] == i))
                        self.assertEqual(rows[:, 1].tolist(), expect)


class TestInterleave(CustomTestCase):
    def test_restores_global_order(self):
        for c in (2, 4, 8):
            for page_size in (1, 4, 16):
                npages = 3
                total = npages * page_size * c
                pos = torch.arange(total).view(npages, page_size, c)
                gathered = pos.permute(2, 0, 1).unsqueeze(-1).float()
                for total_len in (0, 1, total // 2 + 1, total):
                    rows = dcp_interleave_pages(gathered, total_len)
                    self.assertEqual(rows.shape, (total_len, 1))
                    self.assertEqual(rows[:, 0].long().tolist(), list(range(total_len)))

    def test_gather_chunk_rows(self):
        c, page_size, chunk_tokens = 4, 4, 7
        prefix_lens = [5, 16, 33, 0]
        req_to_token, num_pages = _build_layout(prefix_lens, page_size, c, seed=3)
        tail = 3

        def rank_fn(r, group):
            pool = _owned_pool(prefix_lens, req_to_token, num_pages, page_size, c, r)
            per_req = []
            for i, n in enumerate(prefix_lens):
                rows = []
                for ch in dcp_prefix_chunk_plan(n, chunk_tokens, page_size, c):
                    pages = _chunk_block_ids(req_to_token[i], ch, page_size, c)
                    rows.append(
                        dcp_gather_chunk_rows(
                            pool[pages], ch.row_offset, ch.end - ch.start, group
                        )
                    )
                per_req.append(torch.cat(rows) if rows else torch.empty(0, tail))
            return per_req

        for per_req in _run_ranks(c, rank_fn):
            for i, n in enumerate(prefix_lens):
                rows = per_req[i]
                self.assertEqual(rows.shape, (n, tail))
                self.assertTrue(torch.all(rows[:, 0] == i))
                self.assertEqual(rows[:, 1].long().tolist(), list(range(n)))


def _owned_pool(prefix_lens, req_to_token, num_pages, page_size, c, r, vals=None):
    """Rank r's physical pages [num_pages, P, tail]; default rows (req, pos, -pos)."""
    tail = 3 if vals is None else vals[0].shape[-1]
    pool = torch.zeros(num_pages * page_size, tail)
    for i, n in enumerate(prefix_lens):
        v = req_to_token[i, :n].long()
        mine = v % c == r
        if vals is None:
            rows = torch.stack(
                [
                    torch.full((n,), float(i)),
                    torch.arange(n).float(),
                    -torch.arange(n).float(),
                ],
                dim=-1,
            )
        else:
            rows = vals[i]
        pool[(v // c)[mine]] = rows[mine]
    return pool.view(num_pages, page_size, tail)


def _chunk_block_ids(req_row, chunk, page_size, c):
    stride = page_size * c
    first = chunk.first_page * stride
    return (req_row[first : first + chunk.num_pages * stride : stride] // stride).long()


class TestPrefixChunkPlan(CustomTestCase):
    def test_covers_prefix(self):
        for page_size, c, chunk_tokens in [
            (1, 2, 3),
            (4, 4, 16),
            (4, 4, 7),
            (16, 8, 100),
            (128, 8, 65536),
            (2, 3, 5),
        ]:
            stride = page_size * c
            for prefix_len in (
                0,
                1,
                chunk_tokens,
                chunk_tokens + 1,
                3 * stride + 5,
                257,
            ):
                plan = dcp_prefix_chunk_plan(prefix_len, chunk_tokens, page_size, c)
                self.assertEqual(len(plan), math.ceil(prefix_len / chunk_tokens))
                pos = 0
                for idx, ch in enumerate(plan):
                    self.assertEqual(ch.start, pos)
                    self.assertGreater(ch.end, ch.start)
                    if idx < len(plan) - 1:
                        self.assertEqual(ch.end - ch.start, chunk_tokens)
                    self.assertEqual(ch.first_page, ch.start // stride)
                    self.assertEqual(
                        ch.first_page + ch.num_pages, math.ceil(ch.end / stride)
                    )
                    self.assertEqual(ch.row_offset, ch.start - ch.first_page * stride)
                    self.assertTrue(0 <= ch.row_offset < stride)
                    # The selected pages hold every row of the chunk.
                    self.assertLessEqual(
                        ch.row_offset + ch.end - ch.start, ch.num_pages * stride
                    )
                    pos = ch.end
                self.assertEqual(pos, prefix_len)

    def test_unaligned_example(self):
        # P=4, c=2 -> 8 tokens per page; chunks of 5 over 13 prefix tokens.
        plan = dcp_prefix_chunk_plan(13, 5, 4, 2)
        self.assertEqual(
            [tuple(ch) for ch in plan],
            [(0, 5, 0, 1, 0), (5, 10, 0, 2, 5), (10, 13, 1, 1, 2)],
        )


def _attn_with_lse(q, k, v, scale, causal):
    """q [T, H, d], k [S, d], v [S, dv] -> (lse [T, H], out [T, H, dv]).

    causal: q token i sees k[: S - T + i + 1] (bottom-right aligned)."""
    t, s = q.shape[0], k.shape[0]
    scores = torch.einsum("thd,sd->ths", q, k) * scale
    if causal:
        allowed = torch.arange(s)[None, :] <= torch.arange(t)[:, None] + (s - t)
        scores = scores.masked_fill(~allowed[:, None, :], -math.inf)
    return torch.logsumexp(scores, dim=-1), torch.softmax(scores, dim=-1) @ v


class TestChunkedPrefixExactness(CustomTestCase):
    """c ranks: per-chunk (no mask) + current (causal) attention merged with
    npu_attention_update == single-rank causal attention over prefix+current."""

    HEADS, D_K, D_V = 3, 12, 10

    def test_exact(self):
        cases = [
            # (c, page_size, chunk_tokens, prefix_lens, extend_lens)
            (2, 4, 11, [37, 0, 8], [5, 3, 1]),
            (4, 4, 16, [100, 3], [7, 1]),
            (3, 2, 7, [50], [4]),
        ]
        for seed, (c, page_size, chunk_tokens, prefix_lens, extend_lens) in enumerate(
            cases
        ):
            g = torch.Generator().manual_seed(seed)
            scale = 1.0 / math.sqrt(self.D_K)
            pk = [torch.randn(n, self.D_K, generator=g) for n in prefix_lens]
            pv = [torch.randn(n, self.D_V, generator=g) for n in prefix_lens]
            ck = [torch.randn(n, self.D_K, generator=g) for n in extend_lens]
            cv = [torch.randn(n, self.D_V, generator=g) for n in extend_lens]
            qs = [
                torch.randn(n, self.HEADS, self.D_K, generator=g) for n in extend_lens
            ]
            req_to_token, num_pages = _build_layout(prefix_lens, page_size, c, seed)
            vals = [torch.cat([pk[i], pv[i]], dim=-1) for i in range(len(prefix_lens))]
            self.assertGreater(max(prefix_lens), 2 * chunk_tokens)

            def rank_fn(r, group):
                pool = _owned_pool(
                    prefix_lens, req_to_token, num_pages, page_size, c, r, vals
                )
                outs = []
                for i, n in enumerate(prefix_lens):
                    lse_list, out_list = [], []
                    for ch in dcp_prefix_chunk_plan(n, chunk_tokens, page_size, c):
                        pages = _chunk_block_ids(req_to_token[i], ch, page_size, c)
                        rows = dcp_gather_chunk_rows(
                            pool[pages], ch.row_offset, ch.end - ch.start, group
                        )
                        k_rows, v_rows = rows.split([self.D_K, self.D_V], dim=-1)
                        lse, out = _attn_with_lse(qs[i], k_rows, v_rows, scale, False)
                        lse_list.append(lse.reshape(-1))
                        out_list.append(out.reshape(-1, self.D_V))
                    lse, out = _attn_with_lse(qs[i], ck[i], cv[i], scale, True)
                    lse_list.append(lse.reshape(-1))
                    out_list.append(out.reshape(-1, self.D_V))
                    outs.append(
                        npu_attention_update(lse_list, out_list).view(
                            -1, self.HEADS, self.D_V
                        )
                    )
                return outs

            with patch.object(
                dcp_ops, "_attention_update_op", attention_update_reference
            ):
                per_rank = _run_ranks(c, rank_fn)
            for i in range(len(prefix_lens)):
                _, ref = _attn_with_lse(
                    qs[i],
                    torch.cat([pk[i], ck[i]]),
                    torch.cat([pv[i], cv[i]]),
                    scale,
                    True,
                )
                for r in range(c):
                    err = (per_rank[r][i] - ref).abs().max().item()
                    self.assertLess(err, 1e-5, (seed, i, r))


def _mla_attn_with_lse(q_nope, q_rope, kv, kr, scale, causal):
    """q_* [T, H, d], kv [S, Dc], kr [S, Dr] -> (lse [T, H], out [T, H, Dc]).

    causal: query i sees keys [: S - T + i + 1] (bottom-right aligned)."""
    t, s = q_nope.shape[0], kv.shape[0]
    scores = (
        torch.einsum("thd,sd->ths", q_nope, kv)
        + torch.einsum("thd,sd->ths", q_rope, kr)
    ) * scale
    if causal:
        allowed = torch.arange(s)[None, :] <= torch.arange(t)[:, None] + (s - t)
        scores = scores.masked_fill(~allowed[:, None, :], -math.inf)
    return torch.logsumexp(scores, dim=-1), torch.softmax(scores, dim=-1) @ kv


class TestVerifySplitExactness(CustomTestCase):
    """DSPARK target verify under DCP: per rank, history (all heads x local KV
    shard before the window, no mask) merged across ranks with its LSE, then
    current (own heads x the window's own K/V, causal) merged locally ==
    single-rank causal attention over prefix + window."""

    D_C, D_R, H = 12, 4, 2  # H heads per rank

    def _run(self, c, page_size, prefix_lens, w, merge_impl, fia_sentinel, seed):
        g = torch.Generator().manual_seed(seed)
        bsz = len(prefix_lens)
        heads = self.H * c
        scale = 1.0 / math.sqrt(self.D_C + self.D_R)
        seq_lens = [n + w for n in prefix_lens]  # includes the verify window
        kv = [torch.randn(n, self.D_C, generator=g) for n in seq_lens]
        kr = [torch.randn(n, self.D_R, generator=g) for n in seq_lens]
        q_nope = torch.randn(bsz * w, heads, self.D_C, generator=g)
        q_rope = torch.randn(bsz * w, heads, self.D_R, generator=g)
        req_to_token, num_pages = _build_layout(seq_lens, page_size, c, seed)
        block_table = dcp_block_tables(req_to_token, max(seq_lens), page_size, c)
        saw_empty = [False]

        def rank_fn(r, group):
            # Owner-filtered write of prefix AND window tokens (the verify KV is
            # written before attention).
            c_kv = torch.zeros(num_pages * page_size, 1, self.D_C)
            k_rope = torch.zeros(num_pages * page_size, 1, self.D_R)
            for i, n in enumerate(seq_lens):
                loc = dcp_physical_write_loc(req_to_token[i, :n].long(), c, r)
                mine = req_to_token[i, :n].long() % c == r
                c_kv[loc[mine], 0] = kv[i][mine]
                k_rope[loc[mine], 0] = kr[i][mine]
            hist_lens = dcp_verify_history_local_lens(seq_lens, w, c, r)
            saw_empty[0] |= any(
                n == 0 and p > 0 for n, p in zip(hist_lens, prefix_lens)
            )
            # History: every query token of request i reads request i's shard.
            hist_out, hist_lse = mla_decode_with_lse_torch(
                q_nope,
                q_rope,
                c_kv.view(num_pages, page_size, 1, self.D_C),
                k_rope.view(num_pages, page_size, 1, self.D_R),
                block_table.repeat_interleave(w, dim=0),
                [n for n in hist_lens for _ in range(w)],
                scale,
            )
            if fia_sentinel:
                empty = torch.isneginf(hist_lse)
                hist_lse = torch.where(
                    empty, torch.full_like(hist_lse, math.inf), hist_lse
                )
                hist_out = torch.where(
                    empty[..., None], torch.full_like(hist_out, float("nan")), hist_out
                )
            hist_out, hist_lse = dcp_merge(
                hist_out,
                hist_lse,
                group,
                "a2a" if merge_impl != "ag_rs" else "ag_rs",
                merge_impl if merge_impl != "ag_rs" else "npu",
                return_lse=True,
            )
            # Current: this rank's heads x the window's own K/V, causal.
            sl = slice(r * self.H, (r + 1) * self.H)
            outs = []
            for i in range(bsz):
                tok = slice(i * w, (i + 1) * w)
                cur_lse, cur_out = _mla_attn_with_lse(
                    q_nope[tok, sl],
                    q_rope[tok, sl],
                    kv[i][-w:],
                    kr[i][-w:],
                    scale,
                    True,
                )
                merged = npu_attention_update(
                    [hist_lse[tok].reshape(-1), cur_lse.reshape(-1)],
                    [
                        hist_out[tok].reshape(-1, self.D_C),
                        cur_out.reshape(-1, self.D_C),
                    ],
                )
                outs.append(merged.view(w, self.H, self.D_C))
            return torch.cat(outs)

        with patch.object(dcp_ops, "_attention_update_op", attention_update_reference):
            per_rank = _run_ranks(c, rank_fn)
        for i in range(bsz):
            tok = slice(i * w, (i + 1) * w)
            _, ref = _mla_attn_with_lse(
                q_nope[tok], q_rope[tok], kv[i], kr[i], scale, True
            )
            for r in range(c):
                got = per_rank[r][tok]
                exp = ref[:, r * self.H : (r + 1) * self.H]
                self.assertLess((got - exp).abs().max().item(), 1e-5, (seed, i, r))
        return saw_empty[0]

    def test_exact(self):
        cases = [
            # (c, page_size, prefix_lens, w)
            (2, 4, [0, 1, 9, 30], 3),
            (4, 2, [2, 0, 17, 5], 8),
            (3, 1, [1, 40], 4),
            (2, 8, [100], 1),
        ]
        for seed, (c, page_size, prefix_lens, w) in enumerate(cases):
            for merge_impl in ("npu", "vllm", "torch", "ag_rs"):
                saw_empty = self._run(
                    c, page_size, prefix_lens, w, merge_impl, False, seed
                )
                if min(p for p in prefix_lens if p > 0) < c:
                    self.assertTrue(saw_empty)
            # FIA's +inf / NaN sentinel for an empty shard.
            self._run(c, page_size, prefix_lens, w, "npu", True, seed)


class TestExactness(CustomTestCase):
    """Single-rank full softmax attention == c local shards + lse_combine."""

    D_C, D_R, HEADS = 32, 8, 6

    def _run(self, c, page_size, seq_lens, seed):
        g = torch.Generator().manual_seed(seed)
        bsz = len(seq_lens)
        scale = 1.0 / math.sqrt(self.D_C + self.D_R)
        q_nope = torch.randn(bsz, self.HEADS, self.D_C, generator=g)
        q_rope = torch.randn(bsz, self.HEADS, self.D_R, generator=g)
        kv = [torch.randn(n, self.D_C, generator=g) for n in seq_lens]
        kr = [torch.randn(n, self.D_R, generator=g) for n in seq_lens]

        ref_out, ref_lse = [], []
        for i in range(bsz):
            scores = (q_nope[i] @ kv[i].T + q_rope[i] @ kr[i].T) * scale
            ref_lse.append(torch.logsumexp(scores, dim=-1))
            ref_out.append(torch.softmax(scores, dim=-1) @ kv[i])
        ref_out, ref_lse = torch.stack(ref_out), torch.stack(ref_lse)

        req_to_token, num_pages = _build_layout(seq_lens, page_size, c, seed)
        block_table = dcp_block_tables(req_to_token, max(seq_lens), page_size, c)
        outs, lses = [], []
        saw_empty = False
        for r in range(c):
            c_kv = torch.zeros(num_pages * page_size, 1, self.D_C)
            k_rope = torch.zeros(num_pages * page_size, 1, self.D_R)
            for i, n in enumerate(seq_lens):
                v = req_to_token[i, :n].long()
                mine = v % c == r
                c_kv[(v // c)[mine], 0] = kv[i][mine]
                k_rope[(v // c)[mine], 0] = kr[i][mine]
            local = dcp_local_seq_lens(torch.tensor(seq_lens), c, r)
            saw_empty |= bool((local == 0).any())
            out, lse = mla_decode_with_lse_torch(
                q_nope,
                q_rope,
                c_kv.view(num_pages, page_size, 1, self.D_C),
                k_rope.view(num_pages, page_size, 1, self.D_R),
                block_table,
                local,
                scale,
            )
            self.assertTrue(torch.all(out[local == 0] == 0))
            self.assertTrue(torch.all(torch.isneginf(lse[local == 0])))
            outs.append(out)
            lses.append(lse)
        merged = lse_combine(torch.stack(outs), torch.stack(lses))
        merged_lse = torch.logsumexp(torch.stack(lses), dim=0)
        return merged, merged_lse, ref_out, ref_lse, saw_empty, outs, lses

    def test_exact_merge(self):
        cases = [
            (2, 1, [1, 2, 7, 30]),
            (4, 4, [1, 3, 16, 50, 97]),
            (8, 16, [1, 5, 8, 129, 300]),
            (3, 2, [2, 11, 12]),
        ]
        for seed, (c, page_size, seq_lens) in enumerate(cases):
            merged, merged_lse, ref_out, ref_lse, saw_empty, _, _ = self._run(
                c, page_size, seq_lens, seed
            )
            if min(seq_lens) < c:
                self.assertTrue(saw_empty)
            self.assertLess((merged - ref_out).abs().max().item(), 1e-5)
            self.assertLess((merged_lse - ref_lse).abs().max().item(), 1e-4)

    def test_exact_merge_a2a_vllm(self):
        # Full heads split into c groups of h; rank r keeps head group r.
        cases = [(2, 1, [1, 2, 7, 30]), (3, 2, [2, 11, 12]), (6, 4, [1, 5, 40])]
        for seed, (c, page_size, seq_lens) in enumerate(cases):
            h = self.HEADS // c
            _, _, ref_out, _, saw_empty, outs, lses = self._run(
                c, page_size, seq_lens, seed
            )
            self.assertTrue(saw_empty)
            with patch.object(
                dcp_ops, "_attention_update_op", attention_update_reference
            ):
                got = _run_ranks(
                    c, lambda r, grp: dcp_merge_a2a_vllm(outs[r], lses[r], grp)
                )
            for r in range(c):
                self.assertEqual(got[r].shape, (len(seq_lens), h, self.D_C))
                expect = ref_out[:, r * h : (r + 1) * h]
                self.assertLess((got[r] - expect).abs().max().item(), 1e-5)

    def test_exact_merge_a2a_npu(self):
        # Packed bf16/fp32 exchange + npu_attention_update, incl. empty ranks.
        cases = [(2, 1, [1, 2, 7, 30]), (3, 2, [2, 11, 12]), (6, 4, [1, 5, 40])]
        for seed, (c, page_size, seq_lens) in enumerate(cases):
            h = self.HEADS // c
            _, _, ref_out, _, saw_empty, outs, lses = self._run(
                c, page_size, seq_lens, seed
            )
            self.assertTrue(saw_empty)
            with patch.object(
                dcp_ops, "_attention_update_op", attention_update_reference
            ):
                got = _run_ranks(
                    c, lambda r, grp: dcp_merge_a2a_npu(outs[r], lses[r], grp)
                )
            for r in range(c):
                self.assertEqual(got[r].shape, (len(seq_lens), h, self.D_C))
                expect = ref_out[:, r * h : (r + 1) * h]
                self.assertLess((got[r] - expect).abs().max().item(), 1e-5)

    def test_all_empty_is_zero(self):
        outs = torch.full((3, 2, 4, 5), float("nan"))
        lses = torch.full((3, 2, 4), -math.inf)
        self.assertTrue(torch.all(lse_combine(outs, lses) == 0))


class TestMerge(CustomTestCase):
    def _inputs(self, n, bsz, h, d, dtype, seed=0):
        g = torch.Generator().manual_seed(seed)
        outs = [torch.randn(bsz, n * h, d, generator=g).to(dtype) for _ in range(n)]
        lses = [torch.randn(bsz, n * h, generator=g) * 3 for _ in range(n)]
        # An empty shard for one (request, head) on rank 0.
        lses[0][0, 0] = -math.inf
        return outs, lses

    def _expected(self, outs, lses, rank, h):
        o = torch.stack(outs)[:, :, rank * h : (rank + 1) * h]
        l = torch.stack(lses)[:, :, rank * h : (rank + 1) * h]
        return lse_combine(o, l)

    def test_a2a_matches_combine(self):
        n, bsz, h, d = 4, 3, 2, 16
        for dtype in (torch.bfloat16, torch.float32):
            outs, lses = self._inputs(n, bsz, h, d, dtype)
            got = _run_ranks(n, lambda r, grp: dcp_merge_a2a(outs[r], lses[r], grp))
            for r in range(n):
                self.assertEqual(got[r].shape, (bsz, h, d))
                self.assertEqual(got[r].dtype, dtype)
                self.assertTrue(
                    torch.equal(got[r], self._expected(outs, lses, r, h)), dtype
                )

    def test_attention_update_reference_matches_combine(self):
        g = torch.Generator().manual_seed(5)
        n, t, d = 4, 13, 9
        outs = torch.randn(n, t, d, generator=g)
        lses = torch.randn(n, t, generator=g) * 4
        ref, ref_lse = attention_update_reference(list(lses), list(outs), 1)
        expect = lse_combine(outs.view(n, t, 1, d), lses.view(n, t, 1)).view(t, d)
        self.assertLess((ref - expect).abs().max().item(), 1e-5)
        self.assertLess(
            (ref_lse - torch.logsumexp(lses, dim=0)).abs().max().item(), 1e-4
        )
        self.assertIsNone(attention_update_reference(list(lses), list(outs), 0)[1])

    def test_attention_update_sanitises_invalid_lse(self):
        g = torch.Generator().manual_seed(6)
        n, t, d = 3, 8, 5
        outs = torch.randn(n, t, d, generator=g)
        lses = torch.randn(n, t, generator=g)
        lses[0, 0] = math.inf  # FIA sentinel for an empty local KV
        outs[0, 0] = float("nan")
        lses[1, 1] = -math.inf
        lses[2, 2] = float("nan")
        lses[:, 3] = math.inf  # no valid shard -> 0
        outs[:, 3] = float("nan")
        with patch.object(dcp_ops, "_attention_update_op", attention_update_reference):
            got = npu_attention_update(list(lses), list(outs))
        expect = lse_combine(outs.view(n, t, 1, d), lses.view(n, t, 1)).view(t, d)
        self.assertTrue(torch.all(torch.isfinite(got)))
        self.assertTrue(torch.all(got[3] == 0))
        self.assertLess((got - expect).abs().max().item(), 1e-5)

    def test_attention_update_return_lse(self):
        g = torch.Generator().manual_seed(8)
        n, t, d = 3, 9, 4
        outs = torch.randn(n, t, d, generator=g)
        lses = torch.randn(n, t, generator=g) * 2
        lses[0, 0] = math.inf  # empty local KV (FIA sentinel)
        outs[0, 0] = float("nan")
        lses[1, 1] = -math.inf
        lses[2, 2] = float("nan")
        lses[:, 3] = math.inf  # no valid shard
        outs[:, 3] = float("nan")
        with patch.object(dcp_ops, "_attention_update_op", attention_update_reference):
            out_only = npu_attention_update(list(lses), list(outs))
            got, got_lse = npu_attention_update(list(lses), list(outs), return_lse=True)
        self.assertTrue(torch.equal(got, out_only))
        self.assertEqual(got_lse.shape, (t,))
        self.assertEqual(got_lse.dtype, torch.float32)
        valid = torch.isfinite(lses)
        expect_lse = torch.logsumexp(
            torch.where(valid, lses, torch.full_like(lses, -math.inf)), dim=0
        )
        self.assertTrue(torch.isneginf(got_lse[3]))
        keep = torch.arange(t) != 3
        self.assertLess((got_lse[keep] - expect_lse[keep]).abs().max().item(), 1e-5)
        # Merging the merged result again is associative.
        with patch.object(dcp_ops, "_attention_update_op", attention_update_reference):
            a_out, a_lse = npu_attention_update(
                list(lses[:2]), list(outs[:2]), return_lse=True
            )
            two = npu_attention_update([a_lse, lses[2]], [a_out, outs[2]])
        self.assertLess((two - got).abs().max().item(), 1e-5)

    def _invalid_shards(self, seed, n, t, d):
        g = torch.Generator().manual_seed(seed)
        outs = torch.randn(n, t, d, generator=g)
        lses = torch.randn(n, t, generator=g) * 2
        lses[0, 0] = math.inf  # FIA sentinel for an empty local KV
        outs[0, 0] = float("nan")
        lses[1, 1] = -math.inf
        lses[2, 2] = float("nan")
        lses[:, 3] = math.inf  # no valid shard
        outs[:, 3] = float("nan")
        return outs, lses

    def test_attention_update_stacked_matches_list(self):
        n, b, h, d = 3, 4, 2, 5
        outs, lses = self._invalid_shards(9, n, b * h, d)
        outs4, lses3 = outs.view(n, b, h, d), lses.view(n, b, h)
        with patch.object(dcp_ops, "_attention_update_op", attention_update_reference):
            for return_lse in (False, True):
                ref = npu_attention_update(
                    list(lses), list(outs), return_lse=return_lse
                )
                got = npu_attention_update_stacked(lses3, outs4, return_lse=return_lse)
                ref, got = (ref, got) if return_lse else ((ref,), (got,))
                for r, g_ in zip(ref, got):
                    self.assertEqual(g_.shape, r.shape)
                    self.assertTrue(torch.equal(g_, r))

    def test_attention_update_head_slices_and_mask_out(self):
        # Non-contiguous [T, H] / [T, H, D] head slices flatten like their
        # contiguous copies; with finite outputs mask_out=False is identical.
        g = torch.Generator().manual_seed(10)
        t, pad_h, h, d = 5, 4, 3, 6
        lses = [torch.randn(t, pad_h, generator=g)[:, :h] for _ in range(2)]
        outs = [torch.randn(t, pad_h, d, generator=g)[:, :h] for _ in range(2)]
        lses[0][0, 0] = -math.inf
        self.assertFalse(lses[1].is_contiguous())
        with patch.object(dcp_ops, "_attention_update_op", attention_update_reference):
            ref = npu_attention_update(
                [l.reshape(-1) for l in lses], [o.reshape(-1, d) for o in outs]
            )
            got = npu_attention_update(lses, outs)
            unmasked = npu_attention_update(lses, outs, mask_out=False)
        self.assertEqual(got.shape, (t * h, d))
        self.assertTrue(torch.equal(got, ref))
        self.assertTrue(torch.equal(unmasked, ref))

    def test_merge_fp32_flag(self):
        # By default the op gets model-dtype outputs (fp32 LSE); merge_fp32
        # casts them to float32 first.
        seen = []

        def spy(lse_list, out_list, update_type=0):
            seen.append((lse_list[0].dtype, out_list[0].dtype))
            return attention_update_reference(lse_list, out_list, update_type)

        n, t, d = 2, 3, 4
        outs = torch.randn(n, t, d).to(torch.bfloat16)
        lses = torch.randn(n, t)
        with patch.object(dcp_ops, "_attention_update_op", spy):
            npu_attention_update(list(lses), list(outs))
            npu_attention_update(list(lses), list(outs), merge_fp32=True)
            npu_attention_update_stacked(lses, outs)
            npu_attention_update_stacked(lses, outs, merge_fp32=True)
        f32, bf16 = torch.float32, torch.bfloat16
        self.assertEqual(seen, [(f32, bf16), (f32, f32), (f32, bf16), (f32, f32)])

    def test_merge_a2a_npu_head_slices(self):
        # Padded FIA outputs are passed as non-contiguous head slices; packing
        # them must equal packing contiguous copies, for bf16 and fp32 outputs.
        c, b, heads, pad, d = 4, 3, 8, 12, 6
        for dtype in (torch.bfloat16, torch.float32):
            g = torch.Generator().manual_seed(11)
            padded_out = [
                torch.randn(b, pad, d, generator=g).to(dtype) for _ in range(c)
            ]
            padded_lse = [torch.randn(b, pad, generator=g) for _ in range(c)]
            padded_lse[1][0, :] = math.inf
            with patch.object(
                dcp_ops, "_attention_update_op", attention_update_reference
            ):
                ref = _run_ranks(
                    c,
                    lambda r, grp: dcp_merge_a2a_npu(
                        padded_out[r][:, :heads].contiguous(),
                        padded_lse[r][:, :heads].contiguous(),
                        grp,
                        return_lse=True,
                    ),
                )
                got = _run_ranks(
                    c,
                    lambda r, grp: dcp_merge_a2a_npu(
                        padded_out[r][:, :heads],
                        padded_lse[r][:, :heads],
                        grp,
                        return_lse=True,
                    ),
                )
            for r in range(c):
                self.assertEqual(got[r][0].dtype, dtype)
                self.assertTrue(torch.equal(got[r][0], ref[r][0]))
                self.assertTrue(torch.equal(got[r][1], ref[r][1]))

    def test_merges_return_lse(self):
        n, bsz, h, d = 4, 3, 2, 16
        merges = dict(
            npu=dcp_merge_a2a_npu,
            vllm=dcp_merge_a2a_vllm,
            torch=dcp_merge_a2a,
            ag_rs=dcp_merge_ag_rs,
        )
        outs, lses = self._inputs(n, bsz, h, d, torch.float32, seed=4)
        lses[2][1] = math.inf
        outs[2][1] = float("nan")
        # Every rank empty for (request 2, head 0).
        for r in range(n):
            lses[r][2, 0] = math.inf
        stacked = torch.stack(lses)
        valid = torch.isfinite(stacked)
        ref_lse = torch.logsumexp(
            torch.where(valid, stacked, torch.full_like(stacked, -math.inf)), dim=0
        )
        with patch.object(dcp_ops, "_attention_update_op", attention_update_reference):
            for name, merge in merges.items():
                plain = _run_ranks(n, lambda r, grp: merge(outs[r], lses[r], grp))
                with_lse = _run_ranks(
                    n,
                    lambda r, grp: merge(outs[r], lses[r], grp, return_lse=True),
                )
                if name != "ag_rs":
                    for r in range(n):
                        self.assertTrue(torch.equal(with_lse[r][0], plain[r]), name)
                for r in range(n):
                    got_out, got_lse = with_lse[r]
                    self.assertEqual(got_lse.shape, (bsz, h), name)
                    exp = ref_lse[:, r * h : (r + 1) * h]
                    finite = torch.isfinite(exp)
                    self.assertTrue(torch.equal(torch.isneginf(got_lse), ~finite), name)
                    diff = (got_lse[finite] - exp[finite]).abs().max().item()
                    self.assertLess(diff, 1e-4, name)
                    if name != "ag_rs":
                        self.assertTrue(torch.all(torch.isfinite(got_out)), name)

    def test_merge_selects_path(self):
        calls = []

        def fake(name):
            def f(out, lse, group, return_lse=False, merge_fp32=False):
                calls.append((name, return_lse, merge_fp32))
                return out

            return f

        out, lse = torch.zeros(1, 2, 3), torch.zeros(1, 2)
        impls = ("npu", "vllm", "torch", "triton")
        fakes = {k: fake(k) for k in impls}
        with patch.dict(dcp_ops._A2A_MERGES, fakes), patch.object(
            dcp_ops, "dcp_merge_ag_rs", fake("ag_rs")
        ):
            for impl in impls:
                dcp_merge(out, lse, None, "a2a", impl)
            dcp_merge(out, lse, None, "ag_rs", "npu", return_lse=True, merge_fp32=True)
        self.assertEqual(dcp_ops.DCP_MERGE_IMPLS, impls)
        self.assertEqual(
            calls,
            [(impl, False, False) for impl in impls] + [("ag_rs", True, True)],
        )

    def test_a2a_vllm_matches_combine(self):
        n, bsz, h, d = 4, 3, 2, 16
        for dtype in (torch.bfloat16, torch.float32):
            outs, lses = self._inputs(n, bsz, h, d, dtype)
            # rank 2 has no local KV for request 1: FIA's +inf sentinel.
            lses[2][1] = math.inf
            outs[2][1] = float("nan")
            with patch.object(
                dcp_ops, "_attention_update_op", attention_update_reference
            ):
                got = _run_ranks(
                    n, lambda r, grp: dcp_merge_a2a_vllm(outs[r], lses[r], grp)
                )
            tol = 1e-5 if dtype == torch.float32 else 2e-2
            for r in range(n):
                self.assertEqual(got[r].shape, (bsz, h, d))
                self.assertEqual(got[r].dtype, dtype)
                expect = self._expected(outs, lses, r, h)
                diff = (got[r].float() - expect.float()).abs().max().item()
                self.assertLess(diff, tol, dtype)

    def test_a2a_npu_matches_combine(self):
        n, bsz, h, d = 4, 3, 2, 16
        for dtype in (torch.bfloat16, torch.float32):
            outs, lses = self._inputs(n, bsz, h, d, dtype)
            # rank 2 has no local KV for request 1: FIA's +inf sentinel.
            lses[2][1] = math.inf
            outs[2][1] = float("nan")
            with patch.object(
                dcp_ops, "_attention_update_op", attention_update_reference
            ):
                got = _run_ranks(
                    n, lambda r, grp: dcp_merge_a2a_npu(outs[r], lses[r], grp)
                )
                vllm = _run_ranks(
                    n, lambda r, grp: dcp_merge_a2a_vllm(outs[r], lses[r], grp)
                )
            for r in range(n):
                self.assertEqual(got[r].shape, (bsz, h, d))
                self.assertEqual(got[r].dtype, dtype)
                self.assertTrue(torch.all(torch.isfinite(got[r])))
                expect = self._expected(outs, lses, r, h)
                # Same inputs (bf16 out is exact in fp32), so bit-equal to vllm.
                self.assertTrue(torch.equal(got[r], vllm[r]), dtype)
                tol = 1e-5 if dtype == torch.float32 else 2e-2
                diff = (got[r].float() - expect.float()).abs().max().item()
                self.assertLess(diff, tol, dtype)

    def test_bf16_lse_pack_roundtrip(self):
        lse = torch.tensor([0.123456789, -1e30, 3.4e38, -math.inf, 7.0])
        buf = torch.empty(5, 2, dtype=torch.bfloat16)
        buf.view(torch.float32)[:, 0] = lse
        self.assertTrue(torch.equal(buf.clone().view(torch.float32)[:, 0], lse))

    def test_ag_rs_matches_a2a(self):
        n, bsz, h, d = 4, 3, 3, 8
        outs, lses = self._inputs(n, bsz, h, d, torch.float32, seed=1)
        a2a = _run_ranks(n, lambda r, grp: dcp_merge_a2a(outs[r], lses[r], grp))
        ag_rs = _run_ranks(n, lambda r, grp: dcp_merge_ag_rs(outs[r], lses[r], grp))
        for r in range(n):
            self.assertEqual(ag_rs[r].shape, (bsz, h, d))
            self.assertLess((a2a[r] - ag_rs[r]).abs().max().item(), 1e-5)


def _import_ascend_backend():
    mocked = {
        name: MagicMock()
        for name in (
            "torch_npu",
            "torch_npu.contrib",
            "sgl_kernel_npu",
            "sgl_kernel_npu.attention",
            "sgl_kernel_npu.attention.sinks_attention",
            "sglang.srt.speculative",
            "sglang.srt.speculative.decoupled_spec_io",
            "sglang.srt.speculative.spec_info",
            "sglang.srt.speculative.eagle_info",
        )
        if name not in sys.modules
    }
    with patch.dict(sys.modules, mocked):
        from sglang.srt.hardware_backend.npu.attention import ascend_backend
    return ascend_backend


def _fake_fia_bsnd(query, key, value, *, num_heads, num_key_value_heads, **kw):
    """BSND FIA stand-in: (out [1, T, H, Dv], lse [1, H, T, 1])."""
    assert kw["softmax_lse_flag"] and kw["input_layout"] == "BSND"
    assert num_heads == num_key_value_heads == query.shape[2]
    if kw["sparse_mode"] == 0:
        assert kw["atten_mask"] is None
        causal = False
    else:
        assert kw["sparse_mode"] == 3 and kw["atten_mask"] is not None
        causal = True
    t, s = query.shape[1], key.shape[1]
    scores = torch.einsum("thd,shd->hts", query[0], key[0]) * kw["scale"]
    if causal:
        allowed = torch.arange(s)[None, :] <= torch.arange(t)[:, None] + (s - t)
        scores = scores.masked_fill(~allowed, -math.inf)
    lse = torch.logsumexp(scores, dim=-1)  # [H, T]
    out = torch.einsum("hts,shd->thd", torch.softmax(scores, dim=-1), value[0])
    return out[None], lse[None, :, :, None]


_REAL_TORCH_TENSOR = torch.tensor


def _cpu_tensor(*args, **kwargs):
    kwargs.pop("device", None)
    return _REAL_TORCH_TENSOR(*args, **kwargs)


def _real_ascend_backend(
    *,
    dcp_size,
    allocator_page_size,
    is_draft_worker=False,
    page=4,
    heads=8,
    comm_backend="a2a",
):
    """Run the real AscendAttnBackend.__init__ on CPU with NPU deps mocked."""
    backend_mod = _import_ascend_backend()
    req_to_token = torch.arange(2 * 64, dtype=torch.int32).view(2, 64) + 8
    model_runner = SimpleNamespace(
        device="cpu",
        page_size=page,
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            attention_arch=backend_mod.AttentionArch.MLA,
            kv_lora_rank=6,
            qk_rope_head_dim=2,
            qk_nope_head_dim=6,
            hf_config=SimpleNamespace(architectures=["KimiK3ForConditionalGeneration"]),
            context_len=64,
            num_attention_heads=heads,
        ),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        token_to_kv_pool=object(),
        spec_algorithm=SimpleNamespace(
            is_dspark=lambda: False,
            get_num_tokens_per_req_for_target_verify=lambda n, is_draft_worker: n,
        ),
        is_draft_worker=is_draft_worker,
        is_hybrid_swa=False,
        server_args=None,
        ps=SimpleNamespace(attn_cp_size=1),
    )
    if allocator_page_size is not None:
        model_runner.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=allocator_page_size
        )
    parallel = SimpleNamespace(
        attn_tp_size=1,
        attn_dcp_size=dcp_size,
        attn_dcp_rank=0,
        dcp_comm_backend=comm_backend,
    )
    with patch.object(torch, "tensor", _cpu_tensor), patch.object(
        backend_mod, "get_parallel", return_value=parallel
    ), patch.object(
        backend_mod,
        "get_spec",
        return_value=SimpleNamespace(speculative_num_draft_tokens=None),
    ), patch.object(
        backend_mod,
        "get_flags",
        return_value=SimpleNamespace(
            capture=SimpleNamespace(enable_torch_compile=False)
        ),
    ), patch.object(
        backend_mod, "AscendAttnMaskBuilder", MagicMock()
    ), patch.object(
        backend_mod, "AscendTorchNativeAttnBackend", MagicMock()
    ), patch.object(
        backend_mod, "DllmConfig", MagicMock(from_server_args=lambda _: None)
    ), patch.object(
        backend_mod, "is_fia_nz", return_value=False
    ):
        backend = backend_mod.AscendAttnBackend(model_runner)
    return backend_mod, backend, model_runner


class TestDcpTargetBackendInit(CustomTestCase):
    """Real AscendAttnBackend with DCP on: __init__ refuses an allocator that
    was not widened to page_size * dcp_size."""

    PAGE = 4

    def test_requires_widened_allocator(self):
        dcp = 4
        _, backend, _ = _real_ascend_backend(
            dcp_size=dcp, allocator_page_size=self.PAGE * dcp, page=self.PAGE
        )
        self.assertEqual((backend.dcp_size, backend.page_size), (dcp, self.PAGE))
        # DCP metadata is built from the host lengths.
        self.assertTrue(backend.needs_cpu_seq_lens)
        for bad in (self.PAGE, self.PAGE * 2, None):
            with self.assertRaisesRegex(RuntimeError, r"page_size \* dcp_size"):
                _real_ascend_backend(
                    dcp_size=dcp, allocator_page_size=bad, page=self.PAGE
                )
        # DCP off: no allocator requirement.
        _, plain, _ = _real_ascend_backend(dcp_size=1, allocator_page_size=None)
        self.assertEqual(plain.dcp_size, 1)
        # The replicated DSPARK draft attends with the dcp=1 layout (physical
        # page P over the virtual locs), so it is not held to P * dcp.
        _, draft, _ = _real_ascend_backend(
            dcp_size=4, allocator_page_size=None, is_draft_worker=True
        )
        self.assertEqual((draft.dcp_size, draft.dcp_rank), (1, 0))
        # Outside DCP the flag follows SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS
        # (default True); only dcp_size > 1 forces it on regardless of the env.
        self.assertTrue(draft.needs_cpu_seq_lens)

    def test_dcp_forces_host_seq_lens_even_when_env_disables_it(self):
        """DCP builds rank-local lengths and block tables from the host mirror,
        so SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS=0 must not switch it off.
        Without DCP the same env value is honoured."""
        with envs.SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS.override(False):
            _, dcp_backend, _ = _real_ascend_backend(
                dcp_size=4, allocator_page_size=self.PAGE * 4, page=self.PAGE
            )
            self.assertEqual(dcp_backend.dcp_size, 4)
            self.assertTrue(dcp_backend.needs_cpu_seq_lens)

            _, plain, _ = _real_ascend_backend(dcp_size=1, allocator_page_size=None)
            self.assertEqual(plain.dcp_size, 1)
            self.assertFalse(plain.needs_cpu_seq_lens)

    def test_eager_decode_block_table_width_from_host_lens(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        dcp = 2
        backend_mod, backend, mr = _real_ascend_backend(
            dcp_size=dcp, allocator_page_size=self.PAGE * dcp, page=self.PAGE
        )
        seq_lens_cpu = [5, 19]
        widths = []
        real_block_tables = backend_mod.dcp_block_tables

        def spy(rows, max_len, page_size, dcp_size):
            widths.append(max_len)
            return real_block_tables(rows, max_len, page_size, dcp_size)

        for spec_info, extra in ((None, 0), (SimpleNamespace(), 2)):
            backend.speculative_step_id = extra - 1
            fb = SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                batch_size=2,
                # Stale device lengths: using them (a device-scalar slice bound,
                # i.e. a host sync) would change the width.
                seq_lens=torch.tensor([5, 3]),
                seq_lens_cpu=torch.tensor(seq_lens_cpu),
                spec_info=spec_info,
                spec_algorithm=None,
                req_pool_indices=torch.tensor([1, 0]),
                extend_seq_lens=None,
                extend_seq_lens_cpu=[1, 1],
                out_cache_loc=None,
            )
            widths.clear()
            with patch.object(backend_mod, "dcp_block_tables", spy), patch.object(
                torch, "tensor", _cpu_tensor
            ):
                backend.init_forward_metadata(fb)
            max_len = max(seq_lens_cpu) + extra
            self.assertEqual(widths, [max_len])
            self.assertIsInstance(widths[0], int)
            stride = self.PAGE * dcp
            expect = (
                mr.req_to_token_pool.req_to_token[[1, 0], 0:max_len:stride] // stride
            )
            self.assertTrue(
                torch.equal(backend.forward_metadata.block_tables, expect.int())
            )


class TestNpuMlaPoolDcpHelpers(CustomTestCase):
    """NPUMLATokenToKVPool CPU backup / restore translates virtual locs under
    DCP; KV relocation refuses to run."""

    C, P, D_C, D_R, LAYERS = 2, 2, 3, 2, 2

    def _pool(self, pages):
        from sglang.srt.hardware_backend.npu.memory_pool_npu import (
            NPUMLATokenToKVPool,
        )

        pool = object.__new__(NPUMLATokenToKVPool)
        pool.layer_num = self.LAYERS
        pool.start_layer = 0
        pool.index_head_dim = None
        pool.kv_lora_rank = self.D_C
        pool.qk_rope_head_dim = self.D_R
        pool.kv_cache_dim, pool.kr_cache_dim = self.D_C, self.D_R
        pool.dsa_kv_cache_store_fp8 = False
        pool.cpu_offloading_chunk_size = 3
        shape = (self.LAYERS, pages + 1, self.P, 1)
        pool.k_buffer = torch.zeros(*shape, self.D_C)
        pool.v_buffer = torch.zeros(*shape, self.D_R)
        return pool

    def test_cpu_copy_roundtrip_and_move_guard(self):
        from sglang.srt.hardware_backend.npu import memory_pool_npu

        c, seq_len, pages = self.C, 7, 4
        stride = self.P * c
        # Request on virtual page 1, restored onto virtual page 3.
        old_virtual = torch.arange(seq_len) + 1 * stride
        new_virtual = torch.arange(seq_len) + 3 * stride
        fake_npu = SimpleNamespace(synchronize=lambda: None)
        for rank in range(c):
            parallel = SimpleNamespace(
                dcp_enabled=True, attn_dcp_size=c, attn_dcp_rank=rank
            )
            pool = self._pool(pages)
            owned = old_virtual % c == rank
            k_rows = torch.randn(self.LAYERS, seq_len, self.D_C)
            v_rows = torch.randn(self.LAYERS, seq_len, self.D_R)
            for layer in range(self.LAYERS):
                phys = old_virtual[owned] // c
                pool.k_buffer[layer].view(-1, self.D_C)[phys] = k_rows[layer][owned]
                pool.v_buffer[layer].view(-1, self.D_R)[phys] = v_rows[layer][owned]
            with patch.object(
                memory_pool_npu, "get_parallel", return_value=parallel
            ), patch.object(torch, "npu", fake_npu, create=True):
                backup = pool.get_cpu_copy(old_virtual)
                pool.k_buffer.zero_()
                pool.v_buffer.zero_()
                pool.load_cpu_copy(backup, new_virtual)
                with self.assertRaisesRegex(NotImplementedError, "decode context"):
                    pool.move_kv_cache(new_virtual, old_virtual)
            for layer in range(self.LAYERS):
                k_view = pool.k_buffer[layer].view(-1, self.D_C)
                v_view = pool.v_buffer[layer].view(-1, self.D_R)
                phys = new_virtual[owned] // c
                self.assertTrue(torch.equal(k_view[phys], k_rows[layer][owned]))
                self.assertTrue(torch.equal(v_view[phys], v_rows[layer][owned]))
                # Only this rank's physical rows (and pad slot 0) are written.
                touched = torch.zeros(k_view.shape[0], dtype=torch.bool)
                touched[phys] = True
                touched[0] = True
                self.assertTrue(torch.all(k_view[~touched] == 0))


class TestAscendBackendChunkedPrefix(CustomTestCase):
    """AscendAttnBackend._forward_extend_mla_prefix_dcp on c simulated ranks
    (fake BSND FIA, reference npu_attention_update) == full causal attention."""

    HEADS, NOPE, ROPE, V, RANK = 2, 6, 4, 5, 8

    def test_matches_full_attention(self):
        backend_mod = _import_ascend_backend()
        c, page_size, chunk_tokens = 2, 4, 11
        prefix_lens, extend_lens = [29, 0, 16], [4, 3, 1]
        g = torch.Generator().manual_seed(7)
        h, dk = self.HEADS, self.NOPE + self.ROPE
        kv_b = torch.randn(h * (self.NOPE + self.V), self.RANK, generator=g) * 0.3
        latent = [torch.randn(n, self.RANK, generator=g) for n in prefix_lens]
        rope = [torch.randn(n, self.ROPE, generator=g) for n in prefix_lens]
        total_t = sum(extend_lens)
        q = torch.randn(total_t, h, dk, generator=g)
        k = torch.randn(total_t, h, dk, generator=g)
        v = torch.randn(total_t, h, self.V, generator=g)
        scale = 1.0 / math.sqrt(dk)
        req_to_token, num_pages = _build_layout(prefix_lens, page_size, c, seed=7)
        stride = page_size * c
        vals = [torch.cat([latent[i], rope[i]], -1) for i in range(len(prefix_lens))]
        block_ids = torch.cat(
            [req_to_token[i, :n:stride] // stride for i, n in enumerate(prefix_lens)]
        ).to(torch.int32)

        def project(x):
            out = x.reshape(-1, self.RANK) @ kv_b.T
            return out.view(*x.shape[:-1], kv_b.shape[0])

        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=h,
            tp_k_head_num=h,
            v_head_dim=self.V,
            scaling=scale,
            kv_b_proj=lambda x: (project(x),),
        )
        local = threading.local()
        collectives = []

        def rank_fn(r, group):
            local.group = group
            pool = _owned_pool(
                prefix_lens, req_to_token, num_pages, page_size, c, r, vals
            )
            backend = object.__new__(backend_mod.AscendAttnBackend)
            backend.dcp_size = c
            backend.page_size = page_size
            backend.dcp_prefix_chunk_tokens = chunk_tokens
            backend.dcp_merge_fp32 = False
            backend.dcp_lse_scale = 1.0
            backend.qk_nope_head_dim = self.NOPE
            backend.fia_mask = torch.ones(1, dtype=torch.bool)
            backend.forward_metadata = SimpleNamespace(
                extend_seq_lens_cpu_int=torch.tensor(extend_lens, dtype=torch.int32),
                prefix_lens=torch.tensor(prefix_lens),
                flatten_prefix_block_tables=block_ids,
            )
            buffers = pool.unsqueeze(2).split([self.RANK, self.ROPE], dim=-1)
            backend.token_to_kv_pool = SimpleNamespace(
                get_key_buffer=lambda _: buffers[0].contiguous(),
                get_value_buffer=lambda _: buffers[1].contiguous(),
            )
            out = backend._forward_extend_mla_prefix_dcp(q, k, v, layer)
            collectives.append((r, group._call))
            return out

        fake_ops = SimpleNamespace(
            npu=SimpleNamespace(npu_fused_infer_attention_score=_fake_fia_bsnd)
        )
        with patch.object(
            dcp_ops, "_attention_update_op", attention_update_reference
        ), patch.object(backend_mod, "is_fia_nz", return_value=False), patch.object(
            backend_mod,
            "get_parallel",
            side_effect=lambda: SimpleNamespace(dcp_group=local.group),
        ), patch.object(
            backend_mod, "torch", SimpleNamespace(**{**vars(torch), "ops": fake_ops})
        ):
            per_rank = _run_ranks(c, rank_fn)

        expect_calls = sum(math.ceil(n / chunk_tokens) for n in prefix_lens)
        self.assertEqual(sorted(collectives), [(r, expect_calls) for r in range(c)])
        offset = 0
        refs = []
        for i, n in enumerate(prefix_lens):
            t = extend_lens[i]
            kv = project(latent[i]).view(n, h, self.NOPE + self.V)
            k_nope, v_pre = kv.split([self.NOPE, self.V], dim=-1)
            k_pre = torch.cat([k_nope, rope[i][:, None].expand(-1, h, -1)], -1)
            out, _ = _fake_fia_bsnd(
                q[None, offset : offset + t],
                torch.cat([k_pre, k[offset : offset + t]])[None],
                torch.cat([v_pre, v[offset : offset + t]])[None],
                num_heads=h,
                num_key_value_heads=h,
                softmax_lse_flag=True,
                input_layout="BSND",
                sparse_mode=3,
                atten_mask=True,
                scale=scale,
            )
            refs.append(out[0])
            offset += t
        ref = torch.cat(refs).reshape(total_t, h * self.V)
        for r in range(c):
            self.assertEqual(per_rank[r].shape, (total_t, h * self.V))
            self.assertLess((per_rank[r] - ref).abs().max().item(), 1e-5)


def _fake_fia_v1_tnd_paged(
    query, key, value, *, query_rope, key_rope, workspace, out, **kw
):
    """FIA v1 .out stand-in for the DCP verify history call: TND query, paged
    [blocks, 1, P, D] MLA cache, no mask; FIA's +inf LSE / NaN out when a
    request has no local KV."""
    assert kw["input_layout"] == "TND" and kw["softmax_lse_flag"]
    assert kw["sparse_mode"] == 0 and kw["atten_mask"] is None
    assert kw["num_heads"] == query.shape[1] and workspace == "ws"
    page_size = kw["block_size"]
    q_ends = kw["actual_seq_lengths"]
    kv_lens = kw["actual_seq_lengths_kv"]
    block_table = kw["block_table"]
    assert len(q_ends) == len(kv_lens) == block_table.shape[0]
    out_t, lse_t = out
    start = 0
    for i, end in enumerate(q_ends):
        n = int(kv_lens[i])
        if n == 0:
            out_t[start:end] = float("nan")
            lse_t[start:end] = math.inf
        else:
            pages = block_table[i, : (n + page_size - 1) // page_size].long()
            kv = key[pages].reshape(-1, key.shape[-1])[:n].float()
            kr = key_rope[pages].reshape(-1, key_rope.shape[-1])[:n].float()
            lse, o = _mla_attn_with_lse(
                query[start:end].float(),
                query_rope[start:end].float(),
                kv,
                kr,
                kw["scale"],
                False,
            )
            out_t[start:end] = o
            lse_t[start:end] = lse[..., None]
        start = end


def _fake_fias_v2_bnsd(query, key, value, *, query_rope, key_rope, **kw):
    """FIAS v2 stand-in for the DCP verify current call: BNSD, non-paged K/V
    of the window, causal; returns (out [B, N, w, D], lse [B, N, w, 1])."""
    assert kw["input_layout"] == "BNSD" and kw["return_softmax_lse"]
    assert kw["sparse_mode"] == 3 and kw["atten_mask"] is not None
    assert "block_table" not in kw
    b, n, w, d = query.shape
    assert kw["num_query_heads"] == n
    assert kw["actual_seq_qlen"] == kw["actual_seq_kvlen"] == [w] * b
    outs, lses = [], []
    for i in range(b):
        lse, o = _mla_attn_with_lse(
            query[i].transpose(0, 1),
            query_rope[i].transpose(0, 1),
            key[i, 0],
            key_rope[i, 0],
            kw["softmax_scale"],
            True,
        )
        outs.append(o.transpose(0, 1))
        lses.append(lse.transpose(0, 1)[..., None])
    return torch.stack(outs), torch.stack(lses)


class TestAscendBackendVerifySplit(CustomTestCase):
    """AscendAttnBackend.forward_mtp -> _forward_verify_mla_dcp on c simulated
    ranks (fake FIA v1 / FIAS v2, reference npu_attention_update) == full
    causal attention over prefix + window, eager (with padding tokens) and
    graph mode."""

    H, D_C, D_R, PAGE, W = 3, 6, 3, 2, 3

    def _run(
        self,
        c,
        prefix_lens,
        graph_mode,
        comm_backend,
        merge_impl,
        pad_heads,
        fused_merge=False,
    ):
        backend_mod = _import_ascend_backend()
        g = torch.Generator().manual_seed(11)
        w, page_size, heads = self.W, self.PAGE, self.H * c
        bsz = len(prefix_lens)
        seq_lens = [n + w for n in prefix_lens]
        kv = [torch.randn(n, self.D_C, generator=g) for n in seq_lens]
        kr = [torch.randn(n, self.D_R, generator=g) for n in seq_lens]
        num_pad = 0 if graph_mode else 2
        total = bsz * w
        q = torch.randn(total + num_pad, heads, self.D_C, generator=g)
        q_rope = torch.randn(total + num_pad, heads, self.D_R, generator=g)
        k = torch.cat([x[-w:] for x in kv] + [torch.zeros(num_pad, self.D_C)])
        k_rope = torch.cat([x[-w:] for x in kr] + [torch.zeros(num_pad, self.D_R)])
        scale = 1.0 / math.sqrt(self.D_C + self.D_R)
        req_to_token, num_pages = _build_layout(seq_lens, page_size, c, seed=11)
        out_cache_loc = torch.cat(
            [req_to_token[i, p : p + w] for i, p in enumerate(prefix_lens)]
            + [torch.zeros(num_pad, dtype=torch.int32)]
        ).long()
        block_tables = dcp_block_tables(req_to_token, max(seq_lens), page_size, c)
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=heads,
            tp_k_head_num=1,
            v_head_dim=self.D_C,
            scaling=scale,
        )
        local = threading.local()

        def rank_fn(r, group):
            local.group = group
            c_kv = torch.zeros(num_pages * page_size, 1, self.D_C)
            c_rope = torch.zeros(num_pages * page_size, 1, self.D_R)
            # Prefix tokens were cached earlier (owner-filtered).
            for i, p in enumerate(prefix_lens):
                v = req_to_token[i, :p].long()
                mine = v % c == r
                c_kv[(v // c)[mine], 0] = kv[i][:p][mine]
                c_rope[(v // c)[mine], 0] = kr[i][:p][mine]

            def set_kv_buffer(_layer, loc, cache_k, cache_v):
                phys = dcp_physical_write_loc(loc, c, r)
                mine = (loc >= 0) & (loc % c == r)
                c_kv[phys[mine]] = cache_k[mine]
                c_rope[phys[mine]] = cache_v[mine]

            backend = object.__new__(backend_mod.AscendAttnBackend)
            backend.use_mla = True
            backend.dcp_size, backend.dcp_rank = c, r
            backend.tp_q_head_num = self.H
            backend.dcp_pad_heads = pad_heads
            backend.kv_lora_rank, backend.qk_rope_head_dim = self.D_C, self.D_R
            backend.page_size = page_size
            backend.speculative_num_draft_tokens = w
            backend.graph_mode = graph_mode
            backend.mtp_mask = torch.ones(1, dtype=torch.bool)
            backend.dcp_comm_backend = comm_backend
            backend.dcp_merge_impl = merge_impl
            backend.dcp_verify_fused_merge = fused_merge
            backend.dcp_merge_fp32 = False
            backend.dcp_lse_scale = 1.0
            hist = dcp_verify_history_local_lens(seq_lens, w, c, r)
            if graph_mode:
                # Captured bs = 5 (one padding request), lists rebound at replay.
                backend.forward_metadata = SimpleNamespace(
                    dcp_local_seq_lens=hist,
                    block_tables=block_tables,
                )
            else:
                backend.forward_metadata = SimpleNamespace(
                    dcp_local_seq_lens=hist + [0],
                    block_tables=torch.cat(
                        [block_tables, torch.zeros_like(block_tables[:1])]
                    ),
                )
            backend.token_to_kv_pool = SimpleNamespace(
                set_kv_buffer=set_kv_buffer,
                get_kv_buffer=lambda _: (c_kv, c_rope),
            )
            forward_batch = SimpleNamespace(
                forward_mode=SimpleNamespace(
                    is_target_verify=lambda: True, is_draft_extend_v2=lambda: False
                ),
                out_cache_loc=out_cache_loc,
                num_token_non_padded_cpu=None if graph_mode else total,
            )
            return backend.forward_mtp(
                q.reshape(q.shape[0], -1),
                k,
                None,
                layer,
                forward_batch,
                True,
                q_rope=q_rope.reshape(q.shape[0], -1),
                k_rope=k_rope,
            )

        fake_npu = MagicMock()
        fake_npu._npu_fused_infer_attention_score_get_max_workspace.return_value = "ws"
        fake_npu.npu_fused_infer_attention_score.out.side_effect = (
            _fake_fia_v1_tnd_paged
        )
        fake_npu.npu_fused_infer_attention_score_v2.side_effect = _fake_fias_v2_bnsd
        with patch.object(
            dcp_ops, "_attention_update_op", attention_update_reference
        ), patch.object(backend_mod, "is_fia_nz", return_value=False), patch.object(
            backend_mod, "torch_npu", fake_npu
        ), patch.object(
            backend_mod,
            "get_parallel",
            side_effect=lambda: SimpleNamespace(dcp_group=local.group),
        ):
            per_rank = _run_ranks(c, rank_fn)

        hist_heads = {
            call.kwargs["num_heads"]
            for call in fake_npu.npu_fused_infer_attention_score.out.call_args_list
        }
        cur_heads = {
            call.kwargs["num_query_heads"]
            for call in fake_npu.npu_fused_infer_attention_score_v2.call_args_list
        }
        if pad_heads:
            self.assertEqual(hist_heads, {1 << (heads - 1).bit_length()})
            self.assertEqual(cur_heads, {4})
        else:
            self.assertEqual(hist_heads, {heads})
            self.assertEqual(cur_heads, {self.H})
        for r in range(c):
            got = per_rank[r]
            self.assertEqual(got.shape, (total + num_pad, self.H * self.D_C))
            if num_pad:
                self.assertTrue(torch.all(got[total:] == 0))
            got = got[:total].view(bsz, w, self.H, self.D_C)
            for i in range(bsz):
                tok = slice(i * w, (i + 1) * w)
                _, ref = _mla_attn_with_lse(
                    q[tok], q_rope[tok], kv[i], kr[i], scale, True
                )
                exp = ref[:, r * self.H : (r + 1) * self.H]
                err = (got[i] - exp).abs().max().item()
                self.assertLess(err, 1e-5, (graph_mode, merge_impl, fused_merge, i, r))

    def test_eager(self):
        for merge_impl, comm in (("npu", "a2a"), ("vllm", "a2a"), ("torch", "ag_rs")):
            self._run(2, [0, 1, 7, 12], False, comm, merge_impl, pad_heads=True)
        self._run(3, [2, 0, 9], False, "a2a", "npu", pad_heads=False)

    def test_graph_mode(self):
        self._run(2, [0, 1, 7, 12], True, "a2a", "npu", pad_heads=True)

    def test_fused_merge_matches_two_stage(self):
        """SGLANG_NPU_DCP_VERIFY_FUSED_MERGE: the current window folded into the
        cross-rank merge as the (dcp + 1)-th shard gives the same attention."""
        for merge_impl in ("npu", "torch"):
            self._run(
                2,
                [0, 1, 7, 12],
                False,
                "a2a",
                merge_impl,
                pad_heads=True,
                fused_merge=True,
            )
        self._run(3, [2, 0, 9], True, "a2a", "npu", pad_heads=False, fused_merge=True)


class TestEnvDefaults(CustomTestCase):
    def test_defaults(self):
        self.assertEqual(envs.SGLANG_NPU_DCP_MERGE_IMPL.get(), "triton")
        self.assertFalse(envs.SGLANG_NPU_DCP_PAD_HEADS.get())
        self.assertFalse(envs.SGLANG_NPU_DCP_MERGE_FP32.get())
        self.assertEqual(envs.SGLANG_NPU_DCP_PREFIX_CHUNK_TOKENS.get(), 65536)

    def test_fusion_switches_default_on(self):
        """This branch carries the fused kernels, so every fused path is the
        default here; the unfused behaviour is the branch without them."""
        self.assertEqual(envs.SGLANG_NPU_DCP_MERGE_IMPL.get(), "triton")
        self.assertTrue(envs.SGLANG_NPU_DCP_VERIFY_FUSED_MERGE.get())
        self.assertTrue(envs.SGLANG_NPU_DCP_KV_STORE_TRITON.get())
        self.assertTrue(envs.SGLANG_NPU_FUSED_SPLIT_QK_NORM_TRITON.get())
        self.assertIn("triton", dcp_ops.DCP_MERGE_IMPLS)
        self.assertEqual(dcp_ops.DCP_SPLIT_MERGE_IMPLS, ("npu", "torch", "triton"))

    def test_verify_fused_merge_requires_a_splittable_merge(self):
        for impl in dcp_ops.DCP_SPLIT_MERGE_IMPLS:
            with envs.SGLANG_NPU_DCP_VERIFY_FUSED_MERGE.override(
                True
            ), envs.SGLANG_NPU_DCP_MERGE_IMPL.override(impl):
                _, backend, _ = _real_ascend_backend(
                    dcp_size=2, allocator_page_size=8, page=4
                )
                self.assertTrue(backend.dcp_verify_fused_merge)
        # vllm packs a different layout, ag_rs has no per-shard exchange.
        for impl, comm in (("vllm", "a2a"), ("npu", "ag_rs")):
            with envs.SGLANG_NPU_DCP_VERIFY_FUSED_MERGE.override(
                True
            ), envs.SGLANG_NPU_DCP_MERGE_IMPL.override(impl):
                with self.assertRaises(ValueError):
                    _real_ascend_backend(
                        dcp_size=2,
                        allocator_page_size=8,
                        page=4,
                        comm_backend=comm,
                    )

    def test_verify_fused_merge_on_by_default(self):
        _, backend, _ = _real_ascend_backend(dcp_size=2, allocator_page_size=8, page=4)
        self.assertTrue(backend.dcp_verify_fused_merge)

    def test_an_unsplittable_merge_downgrades_when_it_was_not_asked_for(self):
        """On by default, so a config that cannot split its merge must fall
        back with a warning rather than refuse to start -- only an explicit
        request is an error (a silent downgrade would spoil an A/B)."""
        with envs.SGLANG_NPU_DCP_MERGE_IMPL.override("vllm"):
            with self.assertLogs(level="WARNING") as logs:
                _, backend, _ = _real_ascend_backend(
                    dcp_size=2, allocator_page_size=8, page=4
                )
            self.assertFalse(backend.dcp_verify_fused_merge)
            self.assertTrue(
                any("fused target-verify merge" in m for m in logs.output), logs.output
            )

    def test_backend_reads_merge_settings_once(self):
        _, backend, _ = _real_ascend_backend(dcp_size=2, allocator_page_size=8, page=4)
        self.assertEqual(
            (
                backend.dcp_merge_impl,
                backend.dcp_comm_backend,
                backend.dcp_merge_fp32,
                backend.dcp_lse_scale,
                backend.dcp_pad_heads,
            ),
            ("triton", "a2a", False, 1.0, False),
        )
        with envs.SGLANG_NPU_DCP_LSE_BASE_E.override(False):
            _, base2, _ = _real_ascend_backend(
                dcp_size=2, allocator_page_size=8, page=4
            )
        lse = torch.tensor([1.0, -math.inf])
        self.assertTrue(torch.equal(backend._dcp_natural_lse(lse), lse))
        self.assertTrue(torch.equal(base2._dcp_natural_lse(lse), lse * math.log(2.0)))
        with envs.SGLANG_NPU_DCP_MERGE_IMPL.override("bogus"):
            with self.assertRaisesRegex(ValueError, "SGLANG_NPU_DCP_MERGE_IMPL"):
                _real_ascend_backend(dcp_size=2, allocator_page_size=8, page=4)

    def test_decode_merges_in_backend(self):
        backend_mod, backend, _ = _real_ascend_backend(
            dcp_size=2, allocator_page_size=8, page=4
        )
        heads, d_c, d_r, bs = 4, 3, 2, 5
        layer = SimpleNamespace(tp_q_head_num=heads)
        backend.kv_lora_rank, backend.qk_rope_head_dim = d_c, d_r
        backend.dcp_attn_impl = "fia"
        backend.dcp_lse_scale = math.log(2.0)
        out = torch.randn(bs, heads, d_c)
        lse = torch.randn(bs, heads)
        seen = {}

        def fake_merge(o, l, return_lse=False):
            seen.update(out=o, lse=l, return_lse=return_lse)
            return o[:, : heads // 2]

        with patch.object(
            backend, "_dcp_decode_fia", return_value=(out, lse)
        ), patch.object(backend, "_dcp_merge", side_effect=fake_merge):
            got = backend._forward_decode_mla_dcp(
                torch.randn(bs, heads * d_c), torch.randn(bs, heads * d_r), layer
            )
        self.assertTrue(torch.equal(seen["lse"], lse * math.log(2.0)))
        self.assertFalse(seen["return_lse"])
        self.assertEqual(got.shape, (bs, heads // 2 * d_c))


class TestMlaNpuVerifyDispatch(CustomTestCase):
    """forward_mla_core_npu: DCP decode and target verify call the full-head
    DCP layer and take the backend's merged output."""

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
    def _batch(mode):
        return SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_decode=lambda: mode == "decode",
                is_target_verify=lambda: mode == "verify",
            )
        )

    def test_gating(self):
        mla_npu = self._module()
        for enabled in (False, True):
            with patch.object(
                mla_npu,
                "get_parallel",
                return_value=SimpleNamespace(dcp_enabled=enabled),
            ):
                for mode in ("decode", "verify", "extend"):
                    self.assertEqual(
                        mla_npu._is_npu_dcp_mla_full_heads(self._batch(mode)),
                        enabled and mode in ("decode", "verify"),
                    )

    def test_core_dispatch(self):
        mla_npu = self._module()
        heads, d = 2, 3
        for mode in ("decode", "verify"):
            m = SimpleNamespace(
                num_local_heads=heads,
                kv_lora_rank=d,
                v_head_dim=d,
                w_vc=torch.zeros(heads, d, d),
                o_proj=lambda x: (x, None),
                attn_mqa=MagicMock(return_value=torch.zeros(5, heads * d)),
                attn_mqa_for_dcp_decode=MagicMock(
                    return_value=torch.zeros(5, heads * d)
                ),
            )
            parallel = SimpleNamespace(dcp_enabled=True)
            with patch.object(mla_npu, "get_parallel", return_value=parallel):
                mla_npu.forward_mla_core_npu(
                    m,
                    "q_pe",
                    "k_pe",
                    "q_nope",
                    "k_nope",
                    self._batch(mode),
                    None,
                    None,
                    None,
                )
            m.attn_mqa_for_dcp_decode.assert_called_once()
            args, kwargs = m.attn_mqa_for_dcp_decode.call_args
            self.assertEqual(args[:3], ("q_nope", "k_nope", "k_nope"))
            self.assertEqual(kwargs, dict(q_rope="q_pe", k_rope="k_pe"))
            m.attn_mqa.assert_not_called()


class TestKimiK3NpuDcpConfig(CustomTestCase):
    @staticmethod
    def _server_args(**kwargs):
        fields = dict(
            dcp_size=8,
            dcp_comm_backend="ag_rs",
            dcp_replicate_q_proj=None,
            speculative_algorithm=None,
            speculative_eagle_topk=None,
            enable_hierarchical_cache=False,
            disaggregation_mode="null",
            attention_backend=None,
            prefill_attention_backend=None,
            decode_attention_backend=None,
        )
        fields.update(kwargs)
        return SimpleNamespace(**fields)

    def _resolve(self, hf_config=None, **kwargs):
        with patch.object(
            kimi_k3_overrides,
            "get_platform",
            return_value=SimpleNamespace(is_npu=True, is_sm100=False),
        ):
            return kimi_k3_overrides._kimi_k3_overrides(
                self._server_args(**kwargs), hf_config
            )

    def test_default_replicates_q_with_a2a(self):
        self.assertEqual(
            self._resolve(),
            {"dcp_replicate_q_proj": True, "dcp_comm_backend": "a2a"},
        )
        self.assertEqual(
            self._resolve(attention_backend="ascend"),
            {"dcp_replicate_q_proj": True, "dcp_comm_backend": "a2a"},
        )

    def test_no_replicate_keeps_comm_backend(self):
        self.assertEqual(self._resolve(dcp_replicate_q_proj=False), {})
        self.assertEqual(
            self._resolve(dcp_replicate_q_proj=False, dcp_comm_backend="a2a"), {}
        )

    def test_explicit_replicate(self):
        self.assertEqual(
            self._resolve(dcp_replicate_q_proj=True), {"dcp_comm_backend": "a2a"}
        )

    def test_dspark_allowed_with_static_ragged_verify(self):
        expect = {"dcp_replicate_q_proj": True, "dcp_comm_backend": "a2a"}
        self.assertEqual(self._resolve(speculative_algorithm="DSPARK"), expect)
        self.assertEqual(
            self._resolve(speculative_algorithm="DSPARK", speculative_eagle_topk=1),
            expect,
        )
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
            self.assertEqual(self._resolve(speculative_algorithm="DSPARK"), expect)

    def test_dspark_rejections(self):
        for mode in ("compact", "cap-accept"):
            with envs.SGLANG_RAGGED_VERIFY_MODE.override(mode):
                with self.assertRaisesRegex(ValueError, "RAGGED_VERIFY_MODE"):
                    self._resolve(speculative_algorithm="DSPARK")
        with self.assertRaisesRegex(ValueError, "speculative_eagle_topk"):
            self._resolve(speculative_algorithm="DSPARK", speculative_eagle_topk=4)
        for algo in ("EAGLE", "EAGLE3", "NGRAM", "DFLASH", "STANDALONE"):
            with self.assertRaisesRegex(ValueError, "only speculative_algorithm"):
                self._resolve(speculative_algorithm=algo)
        # Other rejections still apply with DSPARK on.
        with self.assertRaises(ValueError):
            self._resolve(
                speculative_algorithm="DSPARK", enable_hierarchical_cache=True
            )

    def test_rejections(self):
        rejected = [
            dict(speculative_algorithm="EAGLE"),
            dict(enable_hierarchical_cache=True),
            dict(disaggregation_mode="decode"),
            dict(decode_attention_backend="cutedsl_mla"),
        ]
        for kwargs in rejected:
            with self.assertRaises(ValueError, msg=str(kwargs)):
                self._resolve(**kwargs)
        with envs.SGLANG_NPU_USE_MLAPO.override(True):
            with self.assertRaises(ValueError):
                self._resolve()

    def test_dsa_checkpoint_rejected(self):
        dense = SimpleNamespace(
            architectures=["KimiK3ForConditionalGeneration"],
            text_config=SimpleNamespace(index_topk=None),
        )
        self.assertEqual(
            self._resolve(hf_config=dense),
            {"dcp_replicate_q_proj": True, "dcp_comm_backend": "a2a"},
        )
        dsa = SimpleNamespace(
            architectures=["KimiK3ForConditionalGeneration"],
            text_config=SimpleNamespace(index_topk=2048),
        )
        with self.assertRaisesRegex(ValueError, "DSA"):
            self._resolve(hf_config=dsa)

    def test_dcp_disabled_is_untouched(self):
        self.assertEqual(self._resolve(dcp_size=1), {})


class TestDsparkWithoutDcp(CustomTestCase):
    """DSPARK with DCP off (dcp_size == 1) keeps every original non-DCP path:
    Ascend target / draft backends, K3 overrides, the MLA NPU module."""

    PAGE, W, HEADS, D_C, D_R = 4, 3, 8, 6, 2

    def _backend(self, is_draft_worker, fias_v2=False):
        backend_mod = _import_ascend_backend()
        cls = backend_mod.AscendAttnBackend
        req_to_token = torch.arange(2 * 64, dtype=torch.int32).view(2, 64) + 8
        model_runner = SimpleNamespace(
            device="cpu",
            page_size=self.PAGE,
            model_config=SimpleNamespace(
                dtype=torch.bfloat16,
                attention_arch=backend_mod.AttentionArch.MLA,
                kv_lora_rank=self.D_C,
                qk_rope_head_dim=self.D_R,
                qk_nope_head_dim=self.D_C,
                hf_config=SimpleNamespace(
                    architectures=["KimiK3ForConditionalGeneration"]
                ),
                context_len=64,
                num_attention_heads=self.HEADS,
            ),
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            token_to_kv_pool=object(),
            spec_algorithm=SimpleNamespace(
                is_dspark=lambda: True,
                get_num_tokens_per_req_for_target_verify=lambda n, is_draft_worker: n,
            ),
            is_draft_worker=is_draft_worker,
            is_hybrid_swa=False,
            server_args=None,
            ps=SimpleNamespace(attn_cp_size=1),
        )
        parallel = SimpleNamespace(attn_tp_size=1, attn_dcp_size=1, attn_dcp_rank=0)
        real_tensor = torch.tensor

        def cpu_tensor(*args, **kwargs):
            kwargs.pop("device", None)
            return real_tensor(*args, **kwargs)

        with patch.object(torch, "tensor", cpu_tensor), patch.object(
            backend_mod, "get_parallel", return_value=parallel
        ), patch.object(
            backend_mod,
            "get_spec",
            return_value=SimpleNamespace(speculative_num_draft_tokens=self.W),
        ), patch.object(
            backend_mod,
            "get_flags",
            return_value=SimpleNamespace(
                capture=SimpleNamespace(enable_torch_compile=False)
            ),
        ), patch.object(
            backend_mod, "AscendAttnMaskBuilder", MagicMock()
        ), patch.object(
            backend_mod, "AscendTorchNativeAttnBackend", MagicMock()
        ), patch.object(
            backend_mod, "DllmConfig", MagicMock(from_server_args=lambda _: None)
        ), patch.object(
            backend_mod, "is_fia_nz", return_value=False
        ), envs.SGLANG_NPU_USE_FIAS_V2_BSND.override(
            fias_v2
        ):
            backend = cls(model_runner)
        return backend_mod, backend, model_runner

    def test_target_backend_keeps_plain_layout_and_verify(self):
        for fias_v2 in (False, True):
            backend_mod, backend, mr = self._backend(False, fias_v2)
            self.assertEqual((backend.dcp_size, backend.dcp_rank), (1, 0))
            self.assertFalse(backend.dcp_replicated_draft)
            self.assertFalse(backend.dcp_pad_heads)
            self.assertEqual(backend.page_size, self.PAGE)
            self.assertEqual(backend.use_fias_v2_bsnd, fias_v2)

            seq_lens = [5, 13]  # DSPARK: CPU lens already count the window
            fb = SimpleNamespace(
                forward_mode=ForwardMode.TARGET_VERIFY,
                batch_size=2,
                seq_lens=torch.tensor(seq_lens),
                seq_lens_cpu=torch.tensor(seq_lens),
                spec_info=SimpleNamespace(draft_token_num=self.W),
                spec_algorithm=SimpleNamespace(is_dspark=lambda: True),
                req_pool_indices=torch.tensor([1, 0]),
                extend_seq_lens=None,
                out_cache_loc=None,
                num_token_non_padded_cpu=2 * self.W,
            )
            boom = MagicMock(side_effect=AssertionError("DCP path used"))
            with patch.object(backend_mod, "dcp_block_tables", boom), patch.object(
                backend_mod, "dcp_verify_history_local_lens", boom
            ), patch.object(backend_mod, "dcp_local_seq_lens", boom):
                backend.init_forward_metadata(fb)
            md = backend.forward_metadata
            # Without DCP the backend follows the env default
            # (SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS, default True), so the
            # block table is truncated to the max sequence length before striding.
            self.assertTrue(backend.needs_cpu_seq_lens)
            seq_max = max(seq_lens) + self.W
            expect = (
                mr.req_to_token_pool.req_to_token[[1, 0], :seq_max][:, :: self.PAGE]
                // self.PAGE
            )
            self.assertTrue(torch.equal(md.block_tables, expect))
            self.assertEqual(md.seq_lens_cpu_int.tolist(), seq_lens)
            self.assertIsNone(md.dcp_local_seq_lens)

            num_pages = 64 // self.PAGE + 4
            backend.token_to_kv_pool = SimpleNamespace(
                get_kv_buffer=lambda _: (
                    torch.zeros(num_pages * self.PAGE, 1, self.D_C),
                    torch.zeros(num_pages * self.PAGE, 1, self.D_R),
                )
            )
            fake_npu = MagicMock()
            fake_npu.npu_fused_infer_attention_score_v2.side_effect = (
                lambda q, *a, **kw: (torch.zeros_like(q), None)
            )
            layer = SimpleNamespace(
                layer_id=0,
                tp_q_head_num=self.HEADS,
                tp_k_head_num=1,
                tp_v_head_num=1,
                v_head_dim=self.D_C,
                scaling=1.0,
            )
            T = 2 * self.W
            with patch.object(backend_mod, "torch_npu", fake_npu), patch.object(
                backend_mod, "is_fia_nz", return_value=False
            ), patch.object(
                backend_mod.AscendAttnBackend, "_forward_verify_mla_dcp", boom
            ):
                out = backend.forward_mtp(
                    torch.zeros(T, self.HEADS * self.D_C),
                    torch.zeros(T, self.D_C),
                    None,
                    layer,
                    fb,
                    False,
                    q_rope=torch.zeros(T, self.HEADS * self.D_R),
                    k_rope=torch.zeros(T, self.D_R),
                )
            self.assertEqual(out.shape, (T, self.HEADS * self.D_C))
            if fias_v2:
                call = fake_npu.npu_fused_infer_attention_score_v2.call_args
                self.assertEqual(call.kwargs["actual_seq_kvlen"], seq_lens)
                self.assertEqual(call.kwargs["num_query_heads"], self.HEADS)
            else:
                call = fake_npu.npu_fused_infer_attention_score.out.call_args
                self.assertEqual(call.kwargs["actual_seq_lengths_kv"], seq_lens)
                self.assertEqual(call.kwargs["num_heads"], self.HEADS)
            self.assertEqual(call.kwargs["block_size"], self.PAGE)
            self.assertTrue(torch.equal(call.kwargs["block_table"], expect))

    def test_draft_backend_is_plain_dcp1(self):
        _, backend, _ = self._backend(True)
        self.assertEqual((backend.dcp_size, backend.dcp_rank), (1, 0))
        self.assertEqual(backend.page_size, self.PAGE)
        self.assertFalse(backend.dcp_replicated_draft)
        self.assertFalse(backend.dcp_pad_heads)

    def test_k3_overrides_untouched(self):
        boom = MagicMock(side_effect=AssertionError("NPU DCP resolver called"))
        with patch.object(kimi_k3_overrides, "_resolve_kimi_k3_npu_dcp", boom):
            for topk in (None, 1):
                self.assertEqual(
                    TestKimiK3NpuDcpConfig()._resolve(
                        dcp_size=1,
                        speculative_algorithm="DSPARK",
                        speculative_eagle_topk=topk,
                    ),
                    {},
                )
            # A ragged mode the DCP resolver rejects is not rejected at dcp=1.
            with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"):
                self.assertEqual(
                    TestKimiK3NpuDcpConfig()._resolve(
                        dcp_size=1, speculative_algorithm="DSPARK"
                    ),
                    {},
                )
        boom.assert_not_called()

    def test_mla_npu_target_verify_uses_plain_attn_mqa(self):
        mla_npu = TestMlaNpuVerifyDispatch()._module()
        heads, d = 2, 3
        for mode in ("verify", "decode"):
            m = SimpleNamespace(
                num_local_heads=heads,
                kv_lora_rank=d,
                v_head_dim=d,
                w_vc=torch.zeros(heads, d, d),
                o_proj=lambda x: (x, None),
                attn_mqa=MagicMock(return_value=torch.zeros(5, heads * d)),
                attn_mqa_for_dcp_decode=MagicMock(
                    side_effect=AssertionError("DCP attention used")
                ),
            )
            fb = SimpleNamespace(
                forward_mode=SimpleNamespace(
                    is_decode=lambda: mode == "decode",
                    is_target_verify=lambda: mode == "verify",
                )
            )
            parallel = SimpleNamespace(dcp_enabled=False, attn_dcp_size=1)
            with patch.object(mla_npu, "get_parallel", return_value=parallel):
                self.assertFalse(mla_npu._is_npu_dcp_mla_full_heads(fb))
                mla_npu.forward_mla_core_npu(
                    m, "q_pe", "k_pe", "q_nope", "k_nope", fb, None, None, None
                )
            m.attn_mqa.assert_called_once_with(
                "q_nope", "k_nope", "k_nope", fb, q_rope="q_pe", k_rope="k_pe"
            )
            m.attn_mqa_for_dcp_decode.assert_not_called()


class TestMergeShardsShapeContract(CustomTestCase):
    """Every merge implementation returns the documented [B*h, D] rows (and
    the [B*h] LSE), with and without the extra shard.

    Nothing pinned this before, and the same contract going unpinned in the
    op-bench reference cost a device round trip: its npu leg returned the
    flattened rows while its torch mirror kept [B, h, D], so stacking the two
    worked on a host and failed on a card with "aclnnStack ... dimnum of
    tensor 1 is [3], should be equal to tensor 0 [2]". Here the three legs
    agree by construction; this is what keeps them agreeing.
    """

    N, B, H, D = 4, 3, 2, 8

    def test_every_impl_returns_the_same_rows(self):
        torch.manual_seed(0)
        outs = torch.randn(self.N, self.B, self.H, self.D)
        lses = torch.randn(self.N, self.B, self.H)
        extra = dict(
            extra_out=torch.randn(self.B, self.H, self.D),
            extra_lse=torch.randn(self.B, self.H),
        )
        rows = self.B * self.H
        for label, kwargs in (("plain", {}), ("with the extra shard", extra)):
            merged = {}
            with patch.object(
                dcp_ops, "_attention_update_op", attention_update_reference
            ):
                for impl in ("npu", "torch"):
                    out, lse = dcp_ops.dcp_merge_shards(
                        outs, lses, impl, return_lse=True, **kwargs
                    )
                    self.assertEqual(tuple(out.shape), (rows, self.D), (impl, label))
                    self.assertEqual(tuple(lse.shape), (rows,), (impl, label))
                    merged[impl] = out
            err = (merged["npu"] - merged["torch"]).abs().max().item()
            self.assertLess(err, 1e-5, (label, err))


if __name__ == "__main__":
    unittest.main()
