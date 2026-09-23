"""Shared fixed-window DSEC data: two-step GT clips or every-step causal stream."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DSECAsync(Dataset):
    async_data = True
    def __init__(self, root, split="train", augment=False, clips=False,
                 window_us=50000, bins=10, pseudo_root=None, pseudo_weight=0.2,
                 pseudo_ramp_epochs=5, delay_probability=0.5, rgb_delay_frames=0, rgb_keep_every=1, limit=None, diagnostic_pseudo=False):
        self.root = Path(root)
        path = self.root/"manifest.json"
        self.manifest = json.loads(path.read_text())
        self.signature = hashlib.sha256(path.read_bytes()).hexdigest()
        if (self.manifest.get("format"),self.manifest["window_us"],self.manifest["bins"],
            self.manifest["boundary"]) != ("dsec_async_v1",window_us,bins,"(start,end]"):
            raise ValueError("Async cache representation/interval mismatch")
        if self.manifest.get("count_cutoff") != 10 or self.manifest.get("fastmode") is not True:
            raise ValueError("RVT count/overflow semantics mismatch")
        self.rows = self.manifest["samples"]
        self.samples = [dict(s,row_index=i) for i,s in enumerate(self.rows)
                        if s["split"]==split and (not clips or s["has_gt"])]
        if limit is not None:self.samples = self.samples[:limit]
        if not self.samples:raise ValueError(f"Empty async split {split}")
        self.split,self.clips,self.augment = split,clips,augment
        self.epoch,self.seed = 0,42
        self.delay_probability,self.rgb_delay_frames = delay_probability,rgb_delay_frames
        self.rgb_keep_every = rgb_keep_every
        if rgb_delay_frames < 0 or rgb_keep_every not in (1,2):raise ValueError("Invalid RGB schedule")
        self.pseudo_root = Path(pseudo_root) if pseudo_root else None
        self.pseudo_weight,self.pseudo_ramp_epochs = pseudo_weight,pseudo_ramp_epochs
        self.pseudo_signature = None
        self.pseudo_files = {}
        if self.pseudo_root:
            if split != "train" or not clips:raise ValueError("Pseudo supervision is train-clips only")
            manifest_path = self.pseudo_root/"manifest.json"
            pseudo = json.loads(manifest_path.read_text())
            if pseudo.get("diagnostic_only") and not diagnostic_pseudo:
                raise ValueError("Synthetic diagnostic pseudo labels cannot enter formal training")
            if pseudo.get("format") != "dsec_async_pseudo_v1" or not pseudo.get("audit_passed"):
                raise ValueError("Pseudo labels require a passing independent audit")
            if self.manifest["grid_sha256"] != pseudo["grid_sha256"]:
                raise ValueError("Pseudo time grid differs")
            if pseudo.get("split") != "train":raise ValueError("Pseudo labels must be train-only")
            self.pseudo_signature = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            self.pseudo_files = pseudo["files"]

    def __len__(self):return len(self.samples)

    def set_epoch(self,epoch,seed=42):self.epoch,self.seed = epoch,seed

    def contract(self):
        return dict(format="dsec_async_v1",window_us=self.manifest["window_us"],bins=self.manifest["bins"],
            channels=2*self.manifest["bins"],boundary=self.manifest["boundary"],
            grid_sha256=self.manifest["grid_sha256"],manifest_sha256=self.signature,
            tbptt=2,rgb_delay_probability=self.delay_probability,
            delay_policy="fixed_bernoulli_per_epoch_lane_sequence_one_rgb_frame",
            rgb_latency_assumption="zero_transport_delay",cold_start="zero_rgb_features",
            phase=2 if self.pseudo_root else 1,pseudo_sha256=self.pseudo_signature,
            pseudo_weight=self.pseudo_weight if self.pseudo_root else 0.,
            pseudo_ramp_epochs=self.pseudo_ramp_epochs if self.pseudo_root else 0)

    def _read(self,row,delay,flip,reset=False):
        with np.load(self.root/row["file"]) as data:
            hist = torch.from_numpy(data["histogram"].copy()).float()
        rgb_index = row["rgb_index"]-delay
        if not self.clips and rgb_index >= 0:rgb_index = rgb_index//self.rgb_keep_every*self.rgb_keep_every
        catalog = self.manifest["rgb"][row["sequence"]]
        valid = rgb_index >= 0
        if valid:
            image = catalog[rgb_index]
            rgb = torch.from_numpy(np.load(self.root/image["file"])).permute(2,0,1).float()/255
            rgb = (rgb-rgb.new_tensor([.485,.456,.406])[:,None,None])/rgb.new_tensor([.229,.224,.225])[:,None,None]
            rgb_time = image["time_us"]
        else:
            rgb = torch.zeros(3,440,640);rgb_time = -1
        if valid and rgb_time > row["target_us"]:raise ValueError("Future RGB in cache")
        label = torch.full((440,640),255,dtype=torch.long)
        if row["has_gt"]:
            label = torch.from_numpy(cv2.imread(row["label_path"],cv2.IMREAD_UNCHANGED).astype(np.int64))
        elif self.pseudo_root:
            key = f'{row["sequence"]}/{row["target_us"]}'
            if key not in self.pseudo_files:raise ValueError(f"Missing pseudo step {key}")
            label = torch.from_numpy(np.load(self.pseudo_root/self.pseudo_files[key]).astype(np.int64))
        if flip:hist,rgb,label = (x.flip(-1) for x in (hist,rgb,label))
        meta = dict(height=440,width=640,sequence=row["sequence"],curr_time_org=row["target_us"],
            rgb_time=rgb_time,rgb_id=f'{row["sequence"]}/{rgb_index}' if valid else None,
            rgb_age_us=row["target_us"]-rgb_time if valid else -1,rgb_valid=valid,
            has_gt=row["has_gt"],filename=row["file"],flipped=flip,reset=reset or row["reset"])
        return hist,rgb,label,meta

    def __getitem__(self,key):
        index,slot,reset,flip = key if isinstance(key,tuple) else (key,0,False,False)
        sample = self.samples[index]
        if self.clips:
            # Fixed per lane/scene avoids toggling backwards to an older RGB cache.
            token = f'{self.seed}/{self.epoch}/{slot}/{sample["sequence"]}'.encode()
            draw = int.from_bytes(hashlib.sha256(token).digest()[:8],"big")/2**64
            delay = int(draw < self.delay_probability)
            indices = [sample["row_index"]-1,sample["row_index"]]
            if indices[0] < 0 or self.rows[indices[0]]["has_gt"]:
                raise ValueError("Every GT needs exactly one preceding unlabelled step")
            steps = [self._read(self.rows[i],delay,flip,reset if t==0 else False) for t,i in enumerate(indices)]
            meta = dict(steps[-1][3],steps=[s[3] for s in steps],stream_slot=slot,
                        sample_index=index,reset=reset or steps[0][3]["reset"],
                        pseudo_weight=(self.pseudo_weight*min(1.,(self.epoch+1)/max(1,self.pseudo_ramp_epochs))
                                       if self.pseudo_root else 0.))
            return dict(events=torch.stack([s[0] for s in steps]),images=torch.stack([s[1] for s in steps])),dict(labels=torch.stack([s[2] for s in steps])),dict(image_meta=meta)
        hist,rgb,label,meta = self._read(sample,self.rgb_delay_frames,False,reset)
        meta.update(stream_slot=slot,sample_index=index)
        return dict(events=hist,images=rgb),dict(labels=label),dict(image_meta=meta)
