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
    if event_channels != 20:
        if event_channels != 2 or not model.backbone.use_events:
            raise ValueError("Input ablations support two polarity channels")
        # Construct the full canonical model first: changing Conv2d size earlier
        # shifts RNG consumption and changes otherwise unrelated neck/head weights.
        block = model.backbone.event_encoder.input_stem.op_list[0]
        old = block.conv
        with torch.random.fork_rng(devices=[]):
            replacement = torch.nn.Conv2d(2, old.out_channels, old.kernel_size,
                old.stride, old.padding, old.dilation, old.groups,
                old.bias is not None, old.padding_mode)
        with torch.no_grad():
            if pretrained is not None:
                checkpoint = torch.load(pretrained, map_location="cpu", weights_only=True)
                weights = checkpoint.get("state_dict", checkpoint)
                weights = {k.removeprefix("module."):v for k,v in weights.items()}
                source = weights["backbone.input_stem.op_list.0.conv.weight"]
                replacement.weight.copy_(source.mean(1, keepdim=True).repeat(1,2,1,1) * (3/2))
            else:
                # Defined deterministic diagnostic initialization with preserved
                # channel-summed response; no effect on the external RNG stream.
                replacement.weight.copy_(old.weight.sum(1, keepdim=True).repeat(1,2,1,1) / 2)
            if old.bias is not None:
                replacement.bias.copy_(old.bias)
        block.conv = replacement
        model.backbone.event_channels = 2
    return model
