"""Alias of the current main B1 config; not an old-architecture selector."""

from pathlib import Path
from hmnet.utils.config import load_config

_config = load_config(str(Path(__file__).with_name("efficientvit_b1.py")), "b1_cross_config")
TrainSettings = _config.TrainSettings
TestSettings = _config.TestSettings
