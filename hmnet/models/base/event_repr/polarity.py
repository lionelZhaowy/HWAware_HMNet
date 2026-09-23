"""Unclipped polarity counts on the raw event grid; no temporal subdivision."""
import numpy as np

REPRESENTATIONS = ("rvt_histogram", "polarity_binary", "polarity_count")
CACHE_SCHEMA = "dsec_polarity_counts_v1"


def input_spec(representation):
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown event representation: {representation}")
    polarity = representation != "rvt_histogram"
    return dict(representation=representation, channels=2 if polarity else 20,
                window_us=50000, interval="closed", polarity_order=[0, 1],
                bins=1 if polarity else 10, count_cutoff=None if polarity else 10,
                fastmode=False if polarity else True,
                storage_dtype="int32" if polarity else "uint8", scaling="none",
                binary=representation == "polarity_binary",
                initialization="canonical20_then_stem_v1" if polarity else "parent20")


def validate_input_spec(saved, representation):
    # Missing metadata is accepted only for legacy 20-channel checkpoints.
    expected = input_spec(representation)
    if saved is None and representation == "rvt_histogram":
        return
    if saved != expected:
        raise ValueError(f"Event input contract differs: saved={saved}, expected={expected}")


def polarity_counts(events, height=440, width=640):
    """events: int64 [N,4]=(t_us,x,y,p01), already windowed and cropped.

    int64 bincount prevents RVT's uint8 wraparound; serialize checked int32.
    Both polarity planes contain nonnegative counts (no signed cancellation).
    """
    events = np.asarray(events)
    if events.dtype != np.int64 or events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("Expected int64 [N,4] events")
    if height <= 0 or width <= 0:
        raise ValueError("Invalid event grid")
    t, x, y, p = events.T
    if (np.any(t[1:] < t[:-1]) or np.any((x < 0) | (x >= width))
            or np.any((y < 0) | (y >= height)) or np.any((p < 0) | (p > 1))):
        raise ValueError("Unsorted events or invalid coordinates/polarities")
    counts = np.bincount(p * height * width + y * width + x,
                         minlength=2 * height * width).reshape(2, height, width)
    if counts.max() > min(np.iinfo(np.int32).max, 2**24):
        raise OverflowError("Counts must be exactly representable as int32 and float32")
    return counts.astype(np.int32)


def validate_counts(counts, height=440, width=640):
    if (counts.dtype != np.int32 or counts.shape != (2, height, width)
            or np.any(counts < 0) or counts.max() > 2**24):
        raise ValueError("Invalid polarity count cache")
