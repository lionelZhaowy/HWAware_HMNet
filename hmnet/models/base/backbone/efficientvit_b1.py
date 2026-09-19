"""Stateless official EfficientViT-B1 with optional, independent RGB encoding."""

from pathlib import Path
import torch
from torch import nn
from .vendor.efficientvit.models.efficientvit.backbone import efficientvit_backbone_b1


class EfficientViTB1(nn.Module):
    input_format = "histogram"
    out_channels = (256, 256, 256, 256)
    out_strides = (4, 8, 16, 32)

    def __init__(self, fusion=False, event_channels=20, pretrained=None):
        super().__init__()
        self.fusion = fusion
        self.event_channels = event_channels
        self.pretrained = pretrained
        self.event_encoder = efficientvit_backbone_b1(in_channels=event_channels)
        if fusion:
            self.rgb_encoder = efficientvit_backbone_b1(in_channels=3)

        def projections():
            return nn.ModuleList(
                [
                    nn.Sequential(nn.Conv2d(c, 256, 1, bias=False), nn.BatchNorm2d(256))
                    for c in (32, 64, 128, 256)
                ]
            )

        self.event_proj = projections()
        if fusion:
            self.rgb_proj = projections()
        self.relu = nn.ReLU()

    def init_weights(self):
        # Called by the task wrapper AFTER its generic initialization, so pretrained
        # BN/linear parameters cannot subsequently be silently overwritten.
        if self.pretrained is not None:
            self.load_pretrained(self.pretrained)

    def load_pretrained(self, path):
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        unexpected = [k for k in state if not k.startswith(("backbone.", "head."))]
        if unexpected:
            raise ValueError(f"Unexpected classifier checkpoint keys: {unexpected[:5]}")
        backbone = {
            k.removeprefix("backbone."): v for k, v in state.items() if k.startswith("backbone.")
        }
        if self.fusion:
            self.rgb_encoder.load_state_dict(backbone, strict=True)
        event = dict(backbone)
        stem = "input_stem.op_list.0.conv.weight"
        # Repeating the RGB mean with 3/C scaling preserves the response when all
        # input channels contain the same signal; RGB and DVS never share Parameters.
        event[stem] = backbone[stem].mean(1, keepdim=True).repeat(1, self.event_channels, 1, 1) * (
            3 / self.event_channels
        )
        self.event_encoder.load_state_dict(event, strict=True)

    def forward(self, event_hist, rgb=None):
        if not torch.jit.is_tracing():
            if event_hist.ndim != 4 or event_hist.shape[1] != self.event_channels:
                raise ValueError("event_hist must be [B,20,H,W]")
            if self.fusion and (
                rgb is None or rgb.shape != (event_hist.shape[0], 3, *event_hist.shape[-2:])
            ):
                raise ValueError("Fusion requires registered RGB [B,3,H,W]")
            if not self.fusion and rgb is not None:
                raise ValueError("DVS-only backbone does not accept RGB")
        ev = self.event_encoder(event_hist)
        im = self.rgb_encoder(rgb) if self.fusion else None
        outputs = []
        for i, proj in enumerate(self.event_proj):
            feature = proj(ev[f"stage{i+1}"])
            if im is not None:
                feature = feature + self.rgb_proj[i](im[f"stage{i+1}"])
            outputs.append(self.relu(feature))
        return tuple(outputs)
