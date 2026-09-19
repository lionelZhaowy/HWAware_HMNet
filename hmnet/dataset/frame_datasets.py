"""Label-anchored EventFrame readers with exact inclusive microsecond windows.

The existing readers retain metadata, label conversion, box filtering and frame
contracts. Only sampling/window selection changes; Histogram counting is RVT's.
"""

from pathlib import Path
import numpy as np
from hmnet.dataset import gen1, eventscape, mvsec


def event_range(metadata, start, end):
    indices, counts = metadata
    a = max(0, int(start) // 1000)
    b = min(len(indices), int(end) // 1000 + 1)
    if a >= b:
        return 0, 0
    return int(indices[a]), int(indices[b - 1] + counts[b - 1])


def timestamps(path):
    labels = np.load(path, mmap_mode="r")
    return np.unique(labels["t" if "t" in labels.dtype.names else "ts"]).astype(np.int64)


class GEN1Frames(gen1.EventFrame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sampling_timings = [
            (i, int(t))
            for i, p in enumerate(self.list_fpath_lbl)
            for t in timestamps(self._get_path(p))
        ]
        self.total_seq = len(self.sampling_timings)

    def _choose_file_and_time(self, index, duration):
        i, end = self.sampling_timings[index]
        start = end - duration
        ev = self._get_path(self.list_fpath_evt[i])
        label = self._get_path(self.list_fpath_lbl[i])
        return ev, label, start, event_range(self.ev_meta[Path(ev).name], start, end), 1

    def _load(self, ev, label, start, duration, ev_range, gt_duration):
        events = np.array(np.load(ev, mmap_mode="r")[slice(*ev_range)])
        # Prophesee raw polarity can be uint8; signed conversion must precede -1.
        dtype = [
            (name, np.int16 if name == "p" else events.dtype[name]) for name in events.dtype.names
        ]
        events = events.astype(dtype)
        events["p"] = events["p"] * 2 - 1
        labels = self._load_label_delta_t(label, start, duration + 1)
        events = events[(events["t"] >= start) & (events["t"] <= start + duration)]
        labels = labels[labels["t"] == start + duration]
        return events, labels

    def _label_padding(self, labels, train_duration, gt_duration):
        # A timestamp exists in the annotation schedule even if all its boxes are
        # filtered out. Preserve the empty background target instead of dummy GT.
        return labels


class EventscapeFrames(eventscape.EventFrame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sampling_timings = [
            (i, int(t))
            for i, p in enumerate(self.list_fpath_label)
            for t in timestamps(self._get_path(p))
        ]
        self.total_seq = len(self.sampling_timings)

    def _choose_file_and_time(self, index, duration):
        i, end = self.sampling_timings[index]
        start = end - duration
        ev = self._get_path(self.list_fpath_evt[i])
        return (
            ev,
            self._get_path(self.list_fpath_image[i]),
            self._get_path(self.list_fpath_label[i]),
            start,
            event_range(self.ev_meta[Path(ev).name], start, end),
        )

    def _load(self, ev, image, label, start, duration, ev_range):
        values = list(super()._load(ev, image, label, start, duration + 1, ev_range))
        events = values[0]
        values[0] = events[(events["t"] >= start) & (events["t"] <= start + duration)]
        return tuple(values)


class MVSECFrames(mvsec.EventFrame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label_t = np.rint(self.label_t).astype(np.int64)
        self.sampling_timings = np.unique(self.label_t).tolist()
        self.total_seq = len(self.sampling_timings)

    def _choose_time(self, index, duration):
        end = self.sampling_timings[index]
        start = end - duration
        return start, event_range(self.ev_meta.T, start, end)

    def _load(self, start, duration, ev_range):
        values = list(super()._load(start, duration + 1, ev_range))
        events = values[0]
        events[:, 2] = np.rint(events[:, 2])
        values[0] = events[(events[:, 2] >= start) & (events[:, 2] <= start + duration)]
        return tuple(values)


def event_frames(kind, root, split="train"):
    root = Path(root)
    # Parent readers initialize their indexes in regular mode. The subclasses
    # replace them with exact label timestamps, without random temporal scaling.
    common = dict(
        train_duration=50000,
        sampling="regular",
        start_index_aug_method="none",
        start_index_aug_ratio=0.0,
        random_time_scaling=False,
        output_type="long",
        event_repr=dict(type="RVTHistogram"),
    )
    if kind == "gen1":
        ls = root / "list" / split
        return GEN1Frames(
            fpath_evt_lst=str(ls / "events.txt"),
            fpath_lbl_lst=str(ls / "labels.txt"),
            base_path=str(root),
            fpath_meta=str(ls / "meta.pkl"),
            fpath_gt_duration=str(ls / "gt_interval.csv"),
            **common
        )
    if kind == "eventscape":
        ls = root / "list" / split
        return EventscapeFrames(
            fpath_evt_lst=str(ls / "events.txt"),
            fpath_image_lst=str(ls / "images.txt"),
            fpath_label_lst=str(ls / "labels.txt"),
            base_path=str(root),
            fpath_meta=str(ls / "meta.pkl"),
            fpath_video_duration=str(ls / "video_duration.csv"),
            skip_image_loading=True,
            **common
        )
    if kind == "mvsec":
        name = "outdoor_" + split
        return MVSECFrames(
            fpath_data=str(root / (name + "_data.hdf5")),
            fpath_gt=str(root / (name + "_gt.hdf5")),
            fpath_meta=str(root / (name + "_meta.npy")),
            skip_image_loading=True,
            **common
        )
    raise ValueError(kind)
