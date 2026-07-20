"""Small helpers for parallelizing the per-frame analysis drivers.

The per-frame work in Steps 1-3 is independent (Step 2's seed propagation is the
only cross-frame coupling, and is preserved by processing *contiguous* chunks
serially inside each worker). These helpers pick a worker count and split the
frame range into contiguous chunks; the drivers fall back to a plain serial loop
when only one worker is requested.
"""
from __future__ import annotations

import os
from typing import List, Tuple


def resolve_workers(num_workers) -> int:
    """Map a user setting to a concrete worker count.

    ``None``/1 → serial (1); ``0`` or negative → auto (CPU count − 1, min 1);
    otherwise the requested count (min 1).
    """
    if num_workers is None:
        return 1
    try:
        n = int(num_workers)
    except (TypeError, ValueError):
        return 1
    if n == 1:
        return 1
    if n <= 0:
        return max(1, (os.cpu_count() or 1) - 1)
    return max(1, n)


def chunk_ranges(n: int, k: int) -> List[Tuple[int, int]]:
    """Split ``range(n)`` into at most ``k`` contiguous ``(start, stop)`` ranges."""
    if n <= 0:
        return []
    k = max(1, min(int(k), n))
    bounds = [round(i * n / k) for i in range(k + 1)]
    return [(bounds[i], bounds[i + 1]) for i in range(k) if bounds[i + 1] > bounds[i]]
