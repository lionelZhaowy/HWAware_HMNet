"""DSEC frame segmentation: configurable RGB/DVS modality, DVS LiteMLA M=2 state."""

from hmnet.models.efficientvit_tasks import build_frame_task, ROOT
from hmnet.dataset.dsec_frames import DSECFrames


class TrainSettings:
    frame_training = True
    task = "segmentation"
    modality = "rgbdvs"
    fusion_mode = "add"
    precision = "bf16"
    temporal_window = 2
    event_representation = "polarity_binary"
    event_channels = 2
    # Normal training uses epochs; explicit updates take priority.
    epochs = 150
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
    workers = 8
    # One prefetched batch per worker limits memory across three experiments.
    prefetch_factor = 1
    output = str(ROOT / "logs/segmentation/efficientvit_b1_add_v12_T_dot1_binary_bf16")
    cache = (
        "/home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/dsec_b1_polarity50ms"
    )
    resume = ""
    overfit = 0

    def get_model(self):
        return build_frame_task(
            self.task,
            self.pretrained,
            modality=self.modality,
            fusion_mode=self.fusion_mode,
            temporal_window=self.temporal_window,
            event_channels=self.event_channels,
        )

    def get_dataset(self):
        return DSECFrames(
            self.cache, "train", augment=not self.overfit, limit=self.overfit or None,
            representation=self.event_representation
        )

    def get_validation_dataset(self):
        return DSECFrames(self.cache, "dev", representation=self.event_representation)


class TestSettings(TrainSettings):
    frame_evaluation = True
    batch_size = TrainSettings.eval_batch_size

    def get_model(self, devices=None, mode="single_process"):
        if mode != "single_process":
            raise ValueError(
                "Frame B1 uses a single device; no memory-stage scheduling"
            )
        return super().get_model()
