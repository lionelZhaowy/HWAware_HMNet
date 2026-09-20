"""Stateless official EfficientViT-B1 with optional, independent RGB encoding."""

from pathlib import Path
import torch
from torch import nn
from .vendor.efficientvit.models.efficientvit.backbone import efficientvit_backbone_b1
from .cross_modal_litemla import CrossModalLiteMLA


class EfficientViTB1(nn.Module):
    input_format = "histogram"
    out_channels = (256, 256, 256, 256)
    out_strides = (4, 8, 16, 32)

    def __init__(
        self, fusion=False, event_channels=20, pretrained=None, modality=None,
    ):
        super().__init__()
        self.modality = modality or ("rgbdvs" if fusion else "dvs")
        if self.modality not in ("rgb", "dvs", "rgbdvs"):
            raise ValueError(f"Unknown input modality: {self.modality}")
        self.fusion = self.modality == "rgbdvs"
        self.use_events = self.modality in ("dvs", "rgbdvs")
        self.use_rgb = self.modality in ("rgb", "rgbdvs")
        self.event_channels = event_channels
        self.pretrained = pretrained
        if self.use_events:
            self.event_encoder = efficientvit_backbone_b1(in_channels=event_channels)
        if self.use_rgb:
            self.rgb_encoder = efficientvit_backbone_b1(in_channels=3)

        def projections():
            return nn.ModuleList(
                [
                    nn.Sequential(nn.Conv2d(c, 256, 1, bias=False), nn.BatchNorm2d(256))
                    for c in (32, 64, 128, 256)
                ]
            )

        if self.fusion:
            self.interactions = nn.ModuleList(
                [CrossModalLiteMLA(c) for c in (32, 64, 128, 256)]
            )
            self.fused_proj = projections()
        else:
            if self.use_events:
                self.event_proj = projections()
            if self.use_rgb:
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
            k.removeprefix("backbone."): v
            for k, v in state.items()
            if k.startswith("backbone.")
        }
        if self.use_rgb:
            self.rgb_encoder.load_state_dict(backbone, strict=True)
        if self.use_events:
            event = dict(backbone)
            stem = "input_stem.op_list.0.conv.weight"
            # Repeating the RGB mean with 3/C scaling preserves the response when all
            # input channels contain the same signal; RGB and DVS never share Parameters.
            event[stem] = backbone[stem].mean(1, keepdim=True).repeat(
                1, self.event_channels, 1, 1
            ) * (3 / self.event_channels)
            self.event_encoder.load_state_dict(event, strict=True)

    def forward(self, event_hist=None, rgb=None):
        # Single-modality experiments instantiate only their active branch.
        # They do not replace an unused modality with zero input.
        if not torch.jit.is_tracing():
            if self.use_events and (
                event_hist is None
                or event_hist.ndim != 4
                or event_hist.shape[1] != self.event_channels
            ):
                raise ValueError("DVS input must be [B,20,H,W]")
            if self.use_rgb and (rgb is None or rgb.ndim != 4 or rgb.shape[1] != 3):
                raise ValueError("RGB input must be [B,3,H,W]")
            if self.fusion and (
                rgb.shape[0] != event_hist.shape[0]
                or rgb.shape[2:] != event_hist.shape[2:]
            ):
                raise ValueError("RGB and DVS must share batch and spatial dimensions")
        if self.fusion:
            return self._forward_cross_stage(event_hist, rgb)
        encoder = self.event_encoder if self.use_events else self.rgb_encoder
        projection = self.event_proj if self.use_events else self.rgb_proj
        features = encoder(event_hist if self.use_events else rgb)
        return tuple(self.relu(projection[i](features[f"stage{i+1}"])) for i in range(4))

    def _forward_cross_stage(self, event_hist, rgb):
        event = self.event_encoder.input_stem(event_hist)
        rgb = self.rgb_encoder.input_stem(rgb)
        outputs = []
        for event_stage, rgb_stage, interaction, projection in zip(
            self.event_encoder.stages, self.rgb_encoder.stages,
            self.interactions, self.fused_proj,
        ):
            event, rgb = event_stage(event), rgb_stage(rgb)
            # Only the two updated native-width streams enter the next stage.
            # The merged 256-channel feature is exclusively a pyramid output.
            rgb, event, fused = interaction(rgb, event)
            outputs.append(self.relu(projection(fused)))
        return tuple(outputs)
