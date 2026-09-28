"""DSEC frame segmentation: configurable RGB/DVS modality, DVS LiteMLA M=2 state."""

from hmnet.models.efficientvit_tasks import build_frame_task, ROOT
from hmnet.dataset.dsec_frames import DSECFrames
from hmnet.dataset.dsec_rgb_delay import DSECFramesRGBDelay


class TrainSettings:
    frame_training = True
    task = "segmentation"
    modality = "rgbdvs"
    fusion_mode = "add"
    precision = "bf16"
    temporal_window = 2
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
    output = str(ROOT / "logs/segmentation/efficientvit_b1_add_v12_T_delay1_bf16")
    cache = (
        "/home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/dsec_b1"
    )
    rgb_delay_probability = 0.5
    rgb_catalog_root = "/home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/async_v1/dsec_async_B"
    resume = ""
    overfit = 0

    def get_model(self):
        return build_frame_task(
            self.task,
            self.pretrained,
            modality=self.modality,
            fusion_mode=self.fusion_mode,
            temporal_window=self.temporal_window,
        )

    def get_dataset(self):
        return DSECFramesRGBDelay(
            self.cache, self.rgb_catalog_root, "train", augment=not self.overfit,
            limit=self.overfit or None, delay_probability=self.rgb_delay_probability
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
