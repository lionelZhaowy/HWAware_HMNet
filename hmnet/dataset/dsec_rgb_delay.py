"""Native 20Hz DSEC events/GT, with the historical Async B RGB-only augmentation.

Read the already registered RGB catalog; never rebuild events, shift labels,
replace the native closed-window histograms or fall back to a future RGB.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from hmnet.dataset.dsec_frames import DSECFrames
from hmnet.utils.rgb_delay import rgb_delay, DELAY_POLICY


class DSECFramesRGBDelay(DSECFrames):
    def __init__(self, root, rgb_catalog_root, split="train", augment=False,
                 limit=None, delay_probability=0.5):
        super().__init__(root, split, augment, limit)
        if split != "train":
            raise ValueError("RGB delay augmentation is train-only; use DSECFrames for dev")
        if not 0 <= delay_probability <= 1:
            raise ValueError("RGB delay probability must be in [0, 1]")
        self.delay_probability = delay_probability
        self.catalog_root = Path(rgb_catalog_root)
        raw = (self.catalog_root / "manifest.json").read_bytes()
        manifest = json.loads(raw)
        base_sha = hashlib.sha256((self.root / "manifest.json").read_bytes()).hexdigest()
        if manifest.get("base_manifest_sha256") != base_sha:
            raise ValueError("RGB catalog was prepared from a different native manifest")
        self.catalog_signature = hashlib.sha256(raw).hexdigest()
        self.catalog = manifest["rgb"]
        self.lookup = {(s["sequence"], s["target_us"]): s for s in manifest["samples"] if s["has_gt"]}
        for s in self.samples:
            row = self.lookup[(s["sequence"], s["target_us"])]
            if (row["split"] != split or row["label_path"] != s["label_path"] or
                    self.catalog[s["sequence"]][row["rgb_index"]]["time_us"] != s["rgb_us"]):
                raise ValueError("RGB/GT membership or timestamps differ from native cache")

    def rgb_delay_contract(self):
        return dict(probability=self.delay_probability, policy=DELAY_POLICY,
                    catalog_manifest_sha256=self.catalog_signature,
                    cold_start="zero_rgb_features", event_clock_us=50000,
                    event_source="unchanged_native_cache")

    def __getitem__(self, key):
        if not isinstance(key, tuple) or self.augmentation_epoch is None:
            raise ValueError("Delay training requires epoch-aware sequence lane keys")
        index, slot, _, _ = key
        data, target, info = super().__getitem__(key)
        sample = self.samples[index]
        delay = rgb_delay(self.augmentation_seed, self.augmentation_epoch, slot,
                          sample["sequence"], self.delay_probability)
        row = self.lookup[(sample["sequence"], sample["target_us"])]
        rgb_index = row["rgb_index"] - delay
        meta = info["image_meta"]
        if delay:
            if rgb_index >= 0:
                entry = self.catalog[sample["sequence"]][rgb_index]
                rgb = torch.from_numpy(np.load(self.catalog_root / entry["file"])).permute(2,0,1).float()/255
                if meta["flipped"]: rgb = rgb.flip(-1)
                data["images"] = (rgb-rgb.new_tensor([.485,.456,.406])[:,None,None])/rgb.new_tensor([.229,.224,.225])[:,None,None]
                rgb_time = entry["time_us"]
            else:
                # Metadata makes the model bypass this placeholder entirely.
                data["images"] = torch.zeros_like(data["images"])
                rgb_time = -1
        else:
            rgb_time = sample["rgb_us"]
        if rgb_index >= 0 and rgb_time > sample["target_us"]:
            raise ValueError("Future RGB is forbidden")
        meta.update(rgb_valid=rgb_index >= 0, rgb_time=rgb_time,
                    rgb_age_us=sample["target_us"]-rgb_time if rgb_index >= 0 else -1,
                    rgb_delay_frames=delay)
        return data, target, info
