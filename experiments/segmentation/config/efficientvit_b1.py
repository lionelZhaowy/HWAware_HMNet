"""DSEC RGB + DVS: independent B1 branches, no recurrent state."""

from hmnet.models.efficientvit_tasks import build_frame_task, ROOT
from hmnet.dataset.dsec_frames import DSECFrames


class TrainSettings:
    frame_training = True
    task = "segmentation"
    # Normal training uses epochs; explicit updates take priority.
    epochs = 100
    updates = None
    batch_size = 32
    accumulation = 8
    pretrained = str(ROOT / "pretrained/efficientvit_b1_r224.pth")
    learning_rate = 2e-4
    weight_decay = 0.01
    workers = 2
    output = str(ROOT / "logs/segmentation/efficientvit_b1")
    cache = "/home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/dsec_b1"
    resume = ""
    overfit = 0

    def get_model(self):
        return build_frame_task(
            self.task,
            self.pretrained,
        )

    def get_dataset(self):
        return DSECFrames(self.cache, "train", augment=not self.overfit, limit=self.overfit or None)

    def get_validation_dataset(self):
        return DSECFrames(self.cache, "dev")


class TestSettings(TrainSettings):
    frame_evaluation = True
    batch_size = 2  # Evaluation does not need to use the training batch size.

    def get_model(self, devices=None, mode="single_process"):
        if mode != "single_process":
            raise ValueError("Frame B1 uses a single device; no memory-stage scheduling")
        return super().get_model()
