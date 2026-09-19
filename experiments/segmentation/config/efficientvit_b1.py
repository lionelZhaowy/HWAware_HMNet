"""DSEC frame segmentation: configurable RGB/DVS modality, no recurrent state."""

from hmnet.models.efficientvit_tasks import build_frame_task, ROOT
from hmnet.dataset.dsec_frames import DSECFrames


class TrainSettings:
    frame_training = True
    task = "segmentation"
    modality = "dvs"
    # Normal training uses epochs; explicit updates take priority.
    epochs = 300
    updates = None
    batch_size = 32
    accumulation = 1
    pretrained = str(ROOT / "pretrained/efficientvit_b1_r224.pth")
    learning_rate = 2e-4
    lr_schedule = "warmup_cosine"
    warmup_epochs = 5
    warmup_start_factor = 0.1
    min_learning_rate = 2e-6
    eval_every_epochs = 1
    eval_batch_size = 32
    weight_decay = 0.01
    workers = 16
    # One prefetched batch per worker limits memory across three experiments.
    prefetch_factor = 1
    output = str(ROOT / "logs/segmentation/efficientvit_b1")
    cache = (
        "/home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/dsec_b1"
    )
    resume = ""
    overfit = 0

    def get_model(self):
        return build_frame_task(
            self.task,
            self.pretrained,
            modality=self.modality,
        )

    def get_dataset(self):
        return DSECFrames(
            self.cache, "train", augment=not self.overfit, limit=self.overfit or None
        )

    def get_validation_dataset(self):
        return DSECFrames(self.cache, "dev")


class TestSettings(TrainSettings):
    frame_evaluation = True
    batch_size = TrainSettings.eval_batch_size

    def get_model(self, devices=None, mode="single_process"):
        if mode != "single_process":
            raise ValueError(
                "Frame B1 uses a single device; no memory-stage scheduling"
            )
        return super().get_model()
