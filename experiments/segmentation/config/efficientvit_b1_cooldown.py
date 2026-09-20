"""Explicit low-LR continuation; keep optimizer/data state in a separate run."""

from pathlib import Path
from hmnet.utils.config import load_config

_baseline = load_config(str(Path(__file__).with_name("efficientvit_b1.py")), "b1_cooldown_base")


class TrainSettings(_baseline.TrainSettings):
    # Exactly 20 additional epoch-equivalents; the parent epoch cursor is retained.
    epochs = 20
    updates = None
    learning_rate = 2e-5
    warmup_epochs = 0
    min_learning_rate = 2e-6
    start_new_stage = True
    output = str(Path(_baseline.TrainSettings.output) / "cooldown_20ep")


class TestSettings(_baseline.TestSettings):
    output = TrainSettings.output
