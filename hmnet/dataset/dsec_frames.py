"""Registered DSEC cache using the existing dense-frame training contract."""

import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from hmnet.models.base.event_repr.polarity import CACHE_SCHEMA, input_spec, validate_counts


class DSECFrames(Dataset):
    def __init__(self, root, split="train", augment=False, limit=None, representation="rvt_histogram"):
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.representation = representation
        self.input_spec = input_spec(representation)
        self.asset_root = self.root
        self.data_contract = None
        if representation != "rvt_histogram":
            expected = dict(schema=CACHE_SCHEMA, complete=True, window_us=50000, interval="closed",
                            channels=2, polarity_order=[0,1], dtype="int32", scaling="none", count_cutoff=None)
            if any(manifest.get(k) != v for k,v in expected.items()):
                raise ValueError("Invalid polarity cache schema")
            self.asset_root = Path(manifest["parent_cache"])
            parent_bytes = (self.asset_root / "manifest.json").read_bytes()
            if hashlib.sha256(parent_bytes).hexdigest() != manifest["parent_manifest_sha256"]:
                raise ValueError("Parent RGB/GT manifest changed")
            parent_samples = json.loads(parent_bytes)["samples"]
            stripped = [{k:v for k,v in s.items() if k not in
                         ("count_file", "count_sha256", "asset_sha256")} for s in manifest["samples"]]
            if manifest.get("diagnostic_subset"):
                parent_by_file = {s["file"]:s for s in parent_samples}
                if (len({s["file"] for s in stripped}) != len(stripped) or
                        any(parent_by_file.get(s["file"]) != s for s in stripped)):
                    raise ValueError("Diagnostic cache contains invalid parent samples")
            elif stripped != parent_samples:
                raise ValueError("Polarity cache sample membership differs from parent")
            self.data_contract = dict(schema=CACHE_SCHEMA,
                manifest_sha256=hashlib.sha256((self.root / "manifest.json").read_bytes()).hexdigest(),
                parent_manifest_sha256=manifest["parent_manifest_sha256"],
                diagnostic_subset=manifest.get("diagnostic_subset", False))
        elif (
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
        if self.representation != "rvt_histogram":
            with np.load(self.root / sample["count_file"]) as f:
                counts = f["counts"]
            validate_counts(counts)
            hist = torch.from_numpy((counts > 0) if self.representation == "polarity_binary" else counts).float()
        with np.load(self.asset_root / sample["file"]) as f:
            if self.representation == "rvt_histogram":
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
