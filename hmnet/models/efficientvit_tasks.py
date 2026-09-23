"""Build independent frame models with the existing HMNet necks and task heads."""

import copy
from pathlib import Path
import torch
from hmnet.utils.config import load_config
from hmnet.models.segmentation import HMSeg
from hmnet.models.detection import HMDet
from hmnet.models.depth import HMDepth

ROOT = Path(__file__).resolve().parents[2]


def build_frame_task(task, pretrained=None, mvsec=False, modality=None, fusion_mode="add", temporal_window=0, event_channels=20):
    if task not in ("segmentation", "detection", "depth"):
        raise ValueError(task)
    name = "hmnet_B3_yolox.py" if task == "detection" else "hmnet_B3.py"
    base = load_config(
        str(ROOT / "experiments" / task / "config" / name), f"frame_reference_{task}"
    )
    backbone = dict(
        type="EfficientViTB1",
        fusion=task == "segmentation",
        pretrained=pretrained,
        modality=modality,
        fusion_mode=fusion_mode,
        temporal_window=temporal_window,
        event_channels=20,  # Canonical initialization keeps all common B/C parameters identical.
    )
    neck = copy.deepcopy(base.neck)
    # Fuse all four scales. YOLOX consumes /8,/16,/32; dense prediction uses /4.
    neck.update(
        in_channels=[256] * 4,
        out_indices=[1, 2, 3] if task == "detection" else [0, 1, 2, 3],
    )
    head = copy.deepcopy(base.head)
    if task == "detection":
        head["strides"] = [8, 16, 32]
    if task == "depth" and mvsec:
        head.update(min_depth=1.978, max_depth=80)
    cls = {"segmentation": HMSeg, "detection": HMDet, "depth": HMDepth}[task]
    positional = [backbone, neck, head]
    if task == "segmentation":
        positional.append(copy.deepcopy(base.aux_head))
    model = cls(*positional, devices=[torch.device("cuda:0")])
    model.init_weights()
    if model.backbone.use_events and event_channels != 20:
        old = model.backbone.event_encoder.input_stem.op_list[0].conv
        # Adapt after ALL generic/pretrained initialization. A narrower stem
        # must not shift the RNG sequence used to initialize Neck and Head.
        with torch.random.fork_rng(devices=[]):
            new = torch.nn.Conv2d(event_channels, old.out_channels, old.kernel_size,
                                 old.stride, old.padding, bias=old.bias is not None)
        with torch.no_grad():
            new.weight.copy_(old.weight.mean(1,keepdim=True).repeat(1,event_channels,1,1)*(20/event_channels))
            if old.bias is not None:new.bias.copy_(old.bias)
        model.backbone.event_encoder.input_stem.op_list[0].conv = new
        model.backbone.event_channels = event_channels
    return model
