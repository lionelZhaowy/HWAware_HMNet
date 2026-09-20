"""Alternate name retained for running experiment-C jobs; the main config is B1."""

from pathlib import Path
from hmnet.utils.config import load_config

_config = load_config(str(Path(__file__).with_name("efficientvit_b1.py")), "b1_cross_config")
TrainSettings = _config.TrainSettings
TestSettings = _config.TestSettings
