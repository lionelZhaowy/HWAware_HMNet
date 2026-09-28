"""Same deterministic RGB-delay draw for synchronous and asynchronous ablations."""
import hashlib

DELAY_POLICY = "fixed_bernoulli_per_epoch_lane_sequence_one_rgb_frame"


def rgb_delay(seed, epoch, slot, sequence, probability):
    if not 0 <= probability <= 1:
        raise ValueError("RGB delay probability must be in [0, 1]")
    # Keep the historical Async B hash/token exactly; do not draw per frame.
    token = f"{seed}/{epoch}/{slot}/{sequence}".encode()
    draw = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / 2**64
    return int(draw < probability)
