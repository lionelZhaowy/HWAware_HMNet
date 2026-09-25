"""Exact closed-window offsets for DAT streams, including timestamp regressions.

A single bounded-memory sequential pass finds the first/last physical event in
all label windows. The resulting enclosing slices can contain unrelated events;
readers must filter by timestamp and stably sort, never binary-search raw DAT.
"""
import numpy as np

GEN1_WINDOW_INDEX = 'full_scan_closed_window_stable_sort_v2'


def exact_window_ranges(records, targets, chunk_size=1 << 20):
    targets = np.asarray(targets, dtype=np.int64)
    if targets.ndim != 1 or np.any(targets[1:] <= targets[:-1]):
        raise ValueError('GEN1 label targets must be strictly increasing')
    if chunk_size < 1:
        raise ValueError('Positive scan chunk size required')
    starts = np.full(len(targets), len(records), dtype=np.int64)
    ends = np.zeros(len(targets), dtype=np.int64)
    counts = np.zeros(len(targets), dtype=np.int64)
    inversions = 0
    largest_regression = 0
    previous = None
    for offset in range(0, len(records), chunk_size):
        block = records[offset:offset + chunk_size]
        t = block['t'].astype(np.int64)
        packed = block['_']
        x = packed & 16383
        y = (packed & 268419072) >> 14
        if np.any(x >= 304) or np.any(y >= 240):
            raise ValueError(f'Invalid GEN1 coordinates in physical events {offset}:{offset + len(t)}')
        delta = np.diff(t, prepend=previous) if previous is not None else np.diff(t)
        negative = delta[delta < 0]
        inversions += len(negative)
        if len(negative):
            largest_regression = max(largest_regression, int(-negative.min()))
        previous = int(t[-1])
        # Only label windows intersecting this block can contain any of its events.
        lo = np.searchsorted(targets, t.min(), side='left')
        hi = np.searchsorted(targets, int(t.max()) + 50000, side='right')
        for i in range(lo, hi):
            selected = np.flatnonzero((t >= targets[i] - 50000) & (t <= targets[i]))
            if len(selected):
                starts[i] = min(starts[i], offset + int(selected[0]))
                ends[i] = max(ends[i], offset + int(selected[-1]) + 1)
                counts[i] += len(selected)
    starts[counts == 0] = 0
    return starts, ends, counts, dict(events=len(records), timestamp_regressions=inversions,
                                     largest_regression_us=largest_regression)
