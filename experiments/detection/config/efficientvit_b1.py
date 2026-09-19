"""GEN1 DVS-only B1, four-scale Pyramid and YOLOX strides 8/16/32."""

from hmnet.models.efficientvit_tasks import build_frame_task, ROOT
from hmnet.dataset.frame_datasets import event_frames


class TrainSettings:
    frame_training = True
    data_root = "/data/lab_dataset/RGB_DVS_Fusion/GEN1/preprocessed/hmnet"
    task = "detection"
    # Normal training uses epochs; explicit updates take priority.
    epochs = 100
    updates = None
    batch_size = 2
    accumulation = 8
    pretrained = str(ROOT / "pretrained/efficientvit_b1_r224.pth")
    learning_rate = 2e-4
    weight_decay = 0.01
    workers = 2
    output = str(ROOT / "logs/detection/efficientvit_b1")
    resume = ""

    def get_model(self):
        return build_frame_task(
            self.task,
            self.pretrained,
        )

    def get_dataset(self):
        return event_frames("gen1", self.data_root)


class TestSettings:
    frame_evaluation = True
    output = TrainSettings.output
    checkpoint = "checkpoint.pth"
    batch_size = 1
    to_device_in_model = True

    def get_model(self, devices=None, mode="single_process"):
        if mode != "single_process":
            raise ValueError("Frame B1 has no memory-stage scheduling")
        model = build_frame_task("detection")
        if devices:
            model.devices = devices[:1]
        return model

    def get_dataset(
        self,
        fpath_evt,
        fpath_lbl,
        fpath_meta,
        fpath_gt_duration,
        base_path,
        fast_mode=False,
        delta_t=None,
    ):
        from hmnet.dataset.frame_datasets import GEN1Frames

        return GEN1Frames(
            fpath_evt_lst=[fpath_evt],
            fpath_lbl_lst=[fpath_lbl],
            base_path=base_path,
            fpath_meta=fpath_meta,
            fpath_gt_duration=fpath_gt_duration,
            train_duration=50000,
            sampling="regular",
            start_index_aug_method="none",
            output_type="long",
            event_repr=dict(type="RVTHistogram"),
        )
