"""Experiment C: normalized LiteMLA cross-modal interaction between B1 stages."""

from pathlib import Path
from hmnet.utils.config import load_config

_base = load_config(str(Path(__file__).with_name("efficientvit_b1.py")), "b1_cross_base")


class TrainSettings(_base.TrainSettings):
    # All data, optimizer and schedule settings are inherited from the baseline.
    modality = "rgbdvs"
    fusion_mode = "cross_stage"
    output = str(Path(_base.TrainSettings.output).parent / "efficientvit_b1_cross")


class TestSettings(TrainSettings):
    frame_evaluation = True
    batch_size = TrainSettings.eval_batch_size

    def get_model(self, devices=None, mode="single_process"):
        if mode != "single_process":
            raise ValueError("Frame B1 uses a single device")
        return super().get_model()
