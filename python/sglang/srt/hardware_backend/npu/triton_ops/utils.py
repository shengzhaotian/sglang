"""Shared helpers for the in-tree Triton-Ascend kernels.

Every kernel here follows the Ascend idiom used by the vendor kernel library
(``triton_ascend_kernels``): a *fixed* grid of one program per vector core,
with each program walking a contiguous chunk of rows, rather than one program
per row. Ascend dispatch is per-core, so a grid far wider than the core count
only adds launch overhead.
"""

import functools

import torch

# Grid width when the vector core count cannot be read (CPU / interpreter).
_FALLBACK_GRID = 48


@functools.lru_cache(maxsize=1)
def npu_vector_cores() -> int:
    """Vector cores of the current device, or ``_FALLBACK_GRID`` off-device."""
    try:
        import triton.runtime.driver as driver

        device = torch.npu.current_device()
        cores = driver.active.utils.get_device_properties(device)["num_vectorcore"]
        return int(cores) if cores else _FALLBACK_GRID
    except Exception:
        return _FALLBACK_GRID


def row_grid(num_rows: int) -> int:
    """One program per vector core, never more programs than rows."""
    return max(1, min(int(num_rows), npu_vector_cores()))
