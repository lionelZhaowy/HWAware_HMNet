"""Build independent frame models with the existing HMNet necks and task heads."""

import copy
from pathlib import Path
import torch
from hmnet.utils.config import load_config
from hmnet.models.segmentation import HMSeg
from hmnet.models.detection import HMDet
from hmnet.models.depth import HMDepth

ROOT = Path(__file__).resolve().parents[2]


def build_frame_task(task, pretrained=None, mvsec=False, modality=None):
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
    return model
