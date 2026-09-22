"""Registered DSEC cache using the existing dense-frame training contract."""

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class DSECFrames(Dataset):
    def __init__(self, root, split="train", augment=False, limit=None):
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        if (
            manifest["window_us"],
            manifest["bins"],
            manifest["count_cutoff"],
            manifest["fastmode"],
        ) != (50000, 10, 10, True):
            raise ValueError(
                "Cache representation does not match first-version RVT settings"
            )
        self.samples = [s for s in manifest["samples"] if s["split"] == split]
        if limit is not None:
            self.samples = self.samples[:limit]
        if not self.samples:
            raise ValueError(f"Empty DSEC split: {split}")
        self.augment = augment
        self.augmentation_epoch = None
        self.augmentation_seed = 42

    def set_epoch(self, epoch, seed=42):
        # Per-sample augmentation is identical across modalities and across resume.
        self.augmentation_epoch, self.augmentation_seed = epoch, seed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        temporal_key = index if isinstance(index, tuple) else None
        if temporal_key is not None:
            index, stream_slot, reset, flip = temporal_key
        sample = self.samples[index]
        with np.load(self.root / sample["file"]) as f:
            hist = torch.from_numpy(f["histogram"].copy()).float()
            rgb = torch.from_numpy(f["rgb"].copy()).permute(2, 0, 1).float() / 255
            label = torch.from_numpy(f["label"].copy()).long()
        # Exactly the same spatial augmentation for events, registered RGB and GT.
        generator = None
        if self.augmentation_epoch is not None:
            generator = torch.Generator().manual_seed(
                self.augmentation_seed + self.augmentation_epoch * len(self) + index
            )
        do_flip = (bool(flip) if temporal_key is not None else
                   self.augment and bool(torch.rand((), generator=generator) < 0.5))
        if do_flip:
            hist, rgb, label = (x.flip(-1) for x in (hist, rgb, label))
        rgb = (
            rgb - rgb.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        ) / rgb.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        meta = dict(
            height=440,
            width=640,
            filename=sample["file"],
            label_path=sample["label_path"],
            curr_time_org=sample["target_us"],
            rgb_time=sample["rgb_us"],
        )
        if temporal_key is not None:
            meta.update(sequence=sample["sequence"], stream_slot=stream_slot,
                        reset=reset, flipped=do_flip, sample_index=index)
        return dict(events=hist, images=rgb), dict(labels=label), dict(image_meta=meta)
