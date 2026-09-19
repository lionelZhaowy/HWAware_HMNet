"""Eventscape/MVSEC DVS-only B1; separately instantiated depth models."""

from hmnet.models.efficientvit_tasks import build_frame_task, ROOT
from hmnet.dataset.frame_datasets import event_frames


class TrainSettings:
    frame_training = True
    data_root = "/data/lab_dataset/RGB_DVS_Fusion/Eventscape/preprocessed/hmnet"
    task = "depth"
    # Normal training uses epochs; explicit updates take priority.
    epochs = 100
    updates = None
    batch_size = 2
    accumulation = 8
    pretrained = str(ROOT / "pretrained/efficientvit_b1_r224.pth")
    learning_rate = 2e-4
    weight_decay = 0.01
    workers = 2
    output = str(ROOT / "logs/depth/efficientvit_b1/eventscape")
    resume = ""
    dataset = "eventscape"

    def get_model(self):
        return build_frame_task(
            self.task,
            self.pretrained,
            mvsec=self.dataset == "mvsec",
        )

    def get_dataset(self):
        root = self.data_root
        return event_frames(self.dataset, root, "day2" if self.dataset == "mvsec" else "train")


class FTSettings(TrainSettings):
    data_root = "/data/lab_dataset/RGB_DVS_Fusion/MVSEC/preprocessed/hmnet"
    dataset = "mvsec"
    output = str(ROOT / "logs/depth/efficientvit_b1/mvsec")


class TestEventscape:
    frame_evaluation = True
    output = TrainSettings.output
    checkpoint = "checkpoint.pth"
    batch_size = 1
    to_device_in_model = True
    dataset = "eventscape"

    def get_model(self, devices=None, mode="single_process"):
        if mode != "single_process":
            raise ValueError("Frame B1 has no memory-stage scheduling")
        model = build_frame_task("depth", mvsec=self.dataset == "mvsec")
        if devices:
            model.devices = devices[:1]
        return model

    def get_dataset(
        self,
        fpath_evt,
        fpath_rgb,
        fpath_lbl,
        fpath_meta,
        fpath_video_duration,
        base_path,
        fast_mode=False,
        delta_t=None,
    ):
        from hmnet.dataset.frame_datasets import EventscapeFrames

        return EventscapeFrames(
            fpath_evt_lst=[fpath_evt],
            fpath_image_lst=[fpath_rgb],
            fpath_label_lst=[fpath_lbl],
            base_path=base_path,
            fpath_meta=fpath_meta,
            fpath_video_duration=fpath_video_duration,
            train_duration=50000,
            sampling="regular",
            start_index_aug_method="none",
            output_type="long",
            skip_image_loading=True,
            event_repr=dict(type="RVTHistogram"),
        )


class TestMVSEC(TestEventscape):
    data_root = FTSettings.data_root
    dataset = "mvsec"
    output = FTSettings.output

    def get_dataset(
        self,
        fpath_data,
        fpath_gt,
        fpath_meta,
        image_mean_std="eventscape",
        fast_mode=False,
        delta_t=None,
        debug=False,
    ):
        from hmnet.dataset.frame_datasets import MVSECFrames

        return MVSECFrames(
            fpath_data=fpath_data,
            fpath_gt=fpath_gt,
            fpath_meta=fpath_meta,
            image_mean_std=image_mean_std,
            train_duration=50000,
            sampling="regular",
            start_index_aug_method="none",
            output_type="long",
            skip_image_loading=True,
            event_repr=dict(type="RVTHistogram"),
        )
