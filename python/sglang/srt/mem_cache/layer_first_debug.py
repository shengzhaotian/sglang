# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Debug instrumentation for the ENABLE_LAYER_FIRST (host_layout_mode=1)
kv_exchange path.

Gate with SGLANG_DEBUG_LAYER_FIRST=1. Output goes to a plain file via
os.write (no stderr, no logging module), so a stalled PTY/pipe consumer can
never block the scheduler thread. Each rank writes the same path with
O_APPEND, so lines interleave but never corrupt.
"""

from __future__ import annotations

import os
import time

import torch

_ENV = "SGLANG_DEBUG_LAYER_FIRST"
_DEBUG = os.environ.get(_ENV, "0") == "1"
_LOG_PATH = os.environ.get(
    "SGLANG_DEBUG_LAYER_FIRST_LOG_PATH", "/tmp/layer_first_debug.log"
)
_fd = None


def enabled() -> bool:
    return _DEBUG


def _write(msg: str) -> None:
    global _fd
    if _fd is None:
        _fd = os.open(_LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.write(_fd, f"{time.time():.3f} {msg}\n".encode("utf-8"))


def log_transfer_launch(
    tag: str, direction, layer_start: int, layer_num: int, num_pages: int
) -> None:
    """Called once per offload.kv_exchange_copy launch (mla.py).

    Verifies the "layer group = 1 -> per-layer launches" hypothesis: if every
    launch shows layer_num=1 while num_pages is large, the layer-outer loop of
    host_layout_mode=1 degenerates to a single layer and the peek-ahead scan
    cost is paid again on every launch.
    """
    if not _DEBUG:
        return
    _write(
        f"[launch] tag={tag} dir={direction} "
        f"layer_start={layer_start} layer_num={layer_num} num_pages={num_pages}"
    )


def log_merge_stats(
    tag: str,
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
    page_size: int,
    sort_ms: float = None,
) -> None:
    """Replicate the AIV kernel's LF_LoadPageAndMerge run detection on the
    exact index arrays the kernel reads (post-argsort), so the reported merge
    numbers are the same the kernel will achieve.

    Kernel rule (acc_offload_kv_exchange.h): a run extends while BOTH the host
    page id and the device page id advance by exactly +1.  If the device
    sequence is fragmented (paged allocator LIFO reuse), merge_count stays 1
    and the peek-ahead scan is pure overhead.
    """
    if not _DEBUG:
        return
    if host_indices is None or device_indices is None:
        return
    if host_indices.numel() == 0:
        return

    # Same page representatives as the kernel: token index at logical slot
    # page*page_size, divided by page_size -> physical page id.
    hp = host_indices[::page_size] // page_size
    dp = device_indices[::page_size] // page_size
    n = int(hp.numel())
    if n <= 1:
        _write(f"[merge] tag={tag} pages={n} (nothing to merge)")
        return

    host_contig = hp[1:] == hp[:-1] + 1
    dev_contig = dp[1:] == dp[:-1] + 1
    both = host_contig & dev_contig
    host_ratio = float(host_contig.float().mean().item())
    dev_ratio = float(dev_contig.float().mean().item())

    # Run lengths exactly as the kernel's mergeCount_ accumulates.
    flags = both.cpu().tolist()
    runs = []
    cur = 1
    for f in flags:
        if f:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    max_run = max(runs)
    n_runs = len(runs)
    merged_pages = sum(r for r in runs if r > 1)

    sort_str = f" sort_ms={sort_ms:.3f}" if sort_ms is not None else ""
    _write(
        f"[merge] tag={tag} pages={n} page_size={page_size} "
        f"host_contig={host_ratio:.2f} dev_contig={dev_ratio:.2f} "
        f"runs={n_runs} max_run={max_run} "
        f"merged_page_ratio={merged_pages / n:.2f}{sort_str}"
    )
