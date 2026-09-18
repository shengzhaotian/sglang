"""CPU-only unit tests for --mamba-state-decay-base/-floor: age-based
geometric thinning of cached Mamba states, applied at the same point as the
--mamba-max-states-per-path cap."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from array import array
from collections import defaultdict

import torch
from test_mamba_path_state_cap import _FakeTreeCore, _FakeUnifiedCache

from sglang.srt.arg_groups.mamba_hook import (
    validate_mamba_extra_buffer,
    validate_mamba_no_buffer,
)
from sglang.srt.arg_groups.overrides import mamba_cache_chunk_size, resolved_view
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.mamba_component import (
    MambaComponent,
    compute_mamba_state_decay_evictions,
)
from sglang.srt.mem_cache.unified_cache.components.tree_component import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedTreeNode
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs


def _key(length: int) -> RadixKey:
    return RadixKey(array("q", [1] * length))


class TestComputeMambaStateDecayEvictions(unittest.TestCase):
    """The pure decision, independent of the tree: given holder depths, the
    tail depth, and (g0, gmax), which depths does the age band rule evict."""

    def test_32_turns_of_4096_tokens_with_floor(self):
        g0, gmax = 4096, 32768
        depths = [t * g0 for t in range(1, 33)]  # 4096 .. 131072
        tail = depths[-1]

        evicted = compute_mamba_state_decay_evictions(depths, tail, g0, gmax)

        self.assertEqual(
            sorted(set(depths) - evicted),
            [32768, 65536, 98304, 114688, 122880, 126976, 131072],
        )

    def test_32_turns_of_4096_tokens_without_floor(self):
        g0 = 4096
        depths = [t * g0 for t in range(1, 33)]
        tail = depths[-1]

        evicted = compute_mamba_state_decay_evictions(depths, tail, g0, 0)

        self.assertEqual(
            sorted(set(depths) - evicted),
            [65536, 98304, 114688, 122880, 126976, 131072],
        )

    def test_no_op_below_zero_or_at_g0(self):
        # decay_base <= 0 disables the rule entirely; ages younger than g0
        # (including the tail's own age-0 entry) are never eviction candidates.
        self.assertEqual(
            compute_mamba_state_decay_evictions([100, 200], 300, 0, 0), set()
        )
        self.assertEqual(
            compute_mamba_state_decay_evictions([4096, 8192], 8192, 4096, 0), set()
        )

    def test_unaligned_turn_ends_keep_only_deepest_per_slot(self):
        """Irregular turn lengths can land two holders in the same
        decay_base-wide slot; only the deeper (younger) one is ever a
        survival candidate for that slot -- the shallower one is evicted
        outright even when the slot's own alignment test would have passed
        it individually, and the slot is dropped entirely when even the
        deepest holder fails its band's alignment test."""
        g0 = 1000
        depths = [4321, 9110, 12800, 17000, 17800, 18500, 19100, 19800, 20500, 21200]
        tail = 21700

        evicted = compute_mamba_state_decay_evictions(depths, tail, g0, 0)

        # slot 17 = {17000, 17800}: the deepest (17800) still fails its own
        # band alignment, so neither of the pair survives.
        self.assertIn(17000, evicted)
        self.assertIn(17800, evicted)
        # slot 19 = {19100, 19800}: the deepest (19800) passes, the
        # shallower sibling (19100) does not get to ride along.
        self.assertIn(19100, evicted)
        self.assertNotIn(19800, evicted)
        self.assertEqual(sorted(set(depths) - evicted), [18500, 19800, 20500, 21200])


def _build_chain(*, cap, decay_base=0, decay_floor=0, key_len, length):
    cache = _FakeUnifiedCache()
    core = _FakeTreeCore()
    component = object.__new__(MambaComponent)
    component.cache = cache
    component.tree_core = core
    component.mamba_max_states_per_path = cap
    component.mamba_state_decay_base = decay_base
    component.mamba_state_decay_floor = decay_floor

    nodes = []
    parent = core.root_node
    for index in range(length):
        node = UnifiedTreeNode(core.tree_components)
        node.parent = parent
        node.key = _key(key_len)
        node.component_data[ComponentType.FULL].value = torch.tensor([100 + index])
        node.component_data[ComponentType.MAMBA].value = torch.tensor([index])
        parent.children[index] = node
        core.component_evictable_size_[ComponentType.MAMBA] += 1
        core.lru_lists[ComponentType.MAMBA].insert_mru(node)
        nodes.append(node)
        parent = node
    return component, nodes, core, cache


def _has_mamba(node) -> bool:
    return node.component_data[ComponentType.MAMBA].value is not None


class TestMambaComponentStateDecay(unittest.TestCase):
    def test_decay_alone_matches_the_32_turn_survivor_list(self):
        component, nodes, core, cache = _build_chain(
            cap=-1, decay_base=4096, decay_floor=32768, key_len=4096, length=32
        )

        device_frees = defaultdict(list)
        host_frees = defaultdict(list)
        component._evict_excess_path_states(nodes[-1], device_frees, host_frees)

        survivor_depths = {32768, 65536, 98304, 114688, 122880, 126976, 131072}
        survivor_indices = {depth // 4096 - 1 for depth in survivor_depths}
        for i, node in enumerate(nodes):
            self.assertEqual(
                _has_mamba(node),
                i in survivor_indices,
                f"node {i} (depth {(i+1)*4096})",
            )
            # Only the Mamba state is ever touched; KV always stays.
            self.assertIsNotNone(node.component_data[ComponentType.FULL].value)

    def test_locked_node_survives_decay_despite_being_marked(self):
        component, nodes, core, cache = _build_chain(
            cap=-1, decay_base=4096, decay_floor=0, key_len=4096, length=3
        )
        # node[0] at depth 4096, age 8192 (T=12288): ratio=2, band=1, m=2,
        # slot=1, 1%2 != 0 -> the rule wants it gone, but it is locked.
        nodes[0].component_data[ComponentType.MAMBA].lock_ref = 1

        device_frees = defaultdict(list)
        host_frees = defaultdict(list)
        component._evict_excess_path_states(nodes[-1], device_frees, host_frees)

        self.assertTrue(_has_mamba(nodes[0]))
        self.assertTrue(_has_mamba(nodes[1]))
        self.assertTrue(_has_mamba(nodes[2]))

    def test_fork_node_survives_decay_despite_being_marked(self):
        component, nodes, core, cache = _build_chain(
            cap=-1, decay_base=4096, decay_floor=0, key_len=4096, length=3
        )
        fork_child = UnifiedTreeNode(core.tree_components)
        fork_child.parent = nodes[0]
        nodes[0].children["fork"] = fork_child

        device_frees = defaultdict(list)
        host_frees = defaultdict(list)
        component._evict_excess_path_states(nodes[-1], device_frees, host_frees)

        self.assertTrue(_has_mamba(nodes[0]))

    def test_tail_is_never_evicted_by_decay(self):
        component, nodes, core, cache = _build_chain(
            cap=-1, decay_base=4096, decay_floor=0, key_len=4096, length=1
        )

        device_frees = defaultdict(list)
        host_frees = defaultdict(list)
        component._evict_excess_path_states(nodes[-1], device_frees, host_frees)

        self.assertTrue(_has_mamba(nodes[0]))

    def test_cap_applies_to_decay_survivors(self):
        # depths 4096..20480 (5 nodes), tail=20480. Decay (g0=4096, no floor)
        # alone keeps {8192, 16384, 20480} (nodes 1, 3, 4); the cap=2 then
        # evicts the shallowest of those three (node 1), leaving {3, 4}.
        component, nodes, core, cache = _build_chain(
            cap=2, decay_base=4096, decay_floor=0, key_len=4096, length=5
        )

        device_frees = defaultdict(list)
        host_frees = defaultdict(list)
        component._evict_excess_path_states(nodes[-1], device_frees, host_frees)

        self.assertFalse(_has_mamba(nodes[0]))  # decay
        self.assertFalse(_has_mamba(nodes[1]))  # cap, on top of decay
        self.assertFalse(_has_mamba(nodes[2]))  # decay
        self.assertTrue(_has_mamba(nodes[3]))
        self.assertTrue(_has_mamba(nodes[4]))  # tail
        for node in nodes:
            self.assertIsNotNone(node.component_data[ComponentType.FULL].value)

    def test_decay_disabled_by_default_leaves_cap_only_behavior_untouched(self):
        component, nodes, core, cache = _build_chain(
            cap=2, decay_base=0, decay_floor=0, key_len=4096, length=3
        )

        device_frees = defaultdict(list)
        host_frees = defaultdict(list)
        component._evict_excess_path_states(nodes[-1], device_frees, host_frees)

        self.assertFalse(_has_mamba(nodes[0]))
        self.assertTrue(_has_mamba(nodes[1]))
        self.assertTrue(_has_mamba(nodes[2]))


def _extra_buffer_server_args(**overrides):
    fields = dict(
        model_path="dummy",
        mamba_radix_cache_strategy="extra_buffer",
        page_size=64,
        chunked_prefill_size=24576,
    )
    fields.update(overrides)
    server_args = ServerArgs(**fields)
    # A pre-seeded `_mamba_cache_chunk_size` (as the neighbouring path-cap /
    # checkpoint-at tests do) so the validator never loads a real HF config
    # for the dummy model.
    server_args._mamba_cache_chunk_size = 64
    return server_args


def _validate_extra_buffer(server_args):
    validate_mamba_extra_buffer(
        resolved_view(server_args),
        "KimiK3ForConditionalGeneration",
        mamba_cache_chunk_size_of=lambda: mamba_cache_chunk_size(server_args),
    )


class TestMambaStateDecayServerArgs(unittest.TestCase):
    def test_server_arg_defaults_to_disabled(self):
        args = ServerArgs(model_path="dummy")
        self.assertEqual(args.mamba_state_decay_base, 0)
        self.assertEqual(args.mamba_state_decay_floor, 0)

    @override_platform(is_cuda=True)
    def test_negative_values_are_rejected(self):
        for field in ("mamba_state_decay_base", "mamba_state_decay_floor"):
            with self.subTest(field=field), self.assertRaisesRegex(
                AssertionError, "non-negative"
            ):
                _validate_extra_buffer(_extra_buffer_server_args(**{field: -1}))

    @override_platform(is_cuda=True)
    def test_base_off_the_grid_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "must be a positive multiple"):
            _validate_extra_buffer(
                _extra_buffer_server_args(mamba_state_decay_base=100)
            )

    @override_platform(is_cuda=True)
    def test_floor_not_a_power_of_two_multiple_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "times a power of two"):
            _validate_extra_buffer(
                _extra_buffer_server_args(
                    mamba_state_decay_base=4096, mamba_state_decay_floor=12288
                )
            )

    @override_platform(is_cuda=True)
    def test_grid_aligned_and_power_of_two_floor_are_accepted(self):
        _validate_extra_buffer(
            _extra_buffer_server_args(
                mamba_state_decay_base=4096, mamba_state_decay_floor=32768
            )
        )

    @override_platform(is_cuda=True)
    def test_base_grid_check_is_deferred_when_page_size_unresolved(self):
        def _must_not_be_read():
            raise AssertionError("the chunk size was read before page_size resolved")

        server_args = _extra_buffer_server_args(
            page_size=None, mamba_state_decay_base=4096
        )
        with self.assertLogs(level="WARNING") as logs:
            validate_mamba_extra_buffer(
                resolved_view(server_args),
                "KimiK3ForConditionalGeneration",
                mamba_cache_chunk_size_of=_must_not_be_read,
            )
        self.assertTrue(any("page-size" in m for m in logs.output))

    def test_no_buffer_warns_and_ignores(self):
        from types import SimpleNamespace

        view = SimpleNamespace(
            page_size=1,
            disable_overlap_schedule=True,
            attention_backend="triton",
            mamba_state_decay_base=4096,
            mamba_state_decay_floor=32768,
        )
        with self.assertLogs(level="WARNING") as logs:
            validate_mamba_no_buffer(view, "KimiK3ForConditionalGeneration")
        self.assertTrue(any("ignored" in m for m in logs.output))


if __name__ == "__main__":
    unittest.main()
