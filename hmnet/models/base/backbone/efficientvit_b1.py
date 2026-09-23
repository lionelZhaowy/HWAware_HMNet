"""Official EfficientViT-B1 with independent RGB and optional explicit DVS memory."""

from pathlib import Path
import torch
from torch import nn
from .vendor.efficientvit.models.efficientvit.backbone import efficientvit_backbone_b1
from .cross_modal_litemla import CrossModalLiteMLA
from .adaptive_add import AdaptiveAdd
from .temporal_litemla import encoder_step, zero_memory


class EfficientViTB1(nn.Module):
    input_format = "histogram"
    out_channels = (256, 256, 256, 256)
    out_strides = (4, 8, 16, 32)

    def __init__(self, fusion=False, event_channels=20, pretrained=None, modality=None,
                 fusion_mode="add", temporal_window=0):
        super().__init__()
        self.modality = modality or ("rgbdvs" if fusion else "dvs")
        if self.modality not in ("rgb", "dvs", "rgbdvs"):
            raise ValueError(f"Unknown input modality: {self.modality}")
        self.fusion = self.modality == "rgbdvs"
        if fusion_mode not in ("add", "adaptive_add", "cross_stage_post_mbconv",
                               "cross_stage_post_mbconv_no_feedback"):
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")
        self.fusion_mode = fusion_mode if self.fusion else "none"
        self.cross_fusion = self.fusion and fusion_mode.startswith("cross_stage_")
        self.use_events = self.modality in ("dvs", "rgbdvs")
        self.use_rgb = self.modality in ("rgb", "rgbdvs")
        self.event_channels = event_channels
        self.pretrained = pretrained
        if temporal_window not in (0, 2):
            raise ValueError("Temporal experiment supports only M=2 or disabled")
        if temporal_window and (not self.use_events or fusion_mode not in
                ("add", "cross_stage_post_mbconv_no_feedback")):
            raise ValueError("Temporal DVS requires Add or no-feedback cross fusion")
        self.temporal_window = temporal_window
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

        if self.cross_fusion:
            self.interactions = nn.ModuleList([CrossModalLiteMLA(c) for c in (32, 64, 128, 256)])
            self.fused_proj = projections()
        else:
            if self.use_events:
                self.event_proj = projections()
            if self.use_rgb:
                self.rgb_proj = projections()
        if self.fusion_mode == "adaptive_add":
            self.gates = nn.ModuleList([AdaptiveAdd() for _ in range(4)])
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

    def zero_temporal_state(self, batch_size, device=None):
        if not self.temporal_window:
            raise ValueError("This backbone is stateless")
        return zero_memory(self.event_encoder, batch_size,
                           device or next(self.parameters()).device)

    def forward(self, event_hist=None, rgb=None, temporal_state=None):
        # Single-modality experiments instantiate only their active branch.
        # They do not replace an unused modality with zero input.
        if not torch.jit.is_tracing():
            if self.use_events and (
                event_hist is None
                or event_hist.ndim != 4
                or event_hist.shape[1] != self.event_channels
            ):
                raise ValueError(f"DVS input must be [B,{self.event_channels},H,W]")
            if self.use_rgb and (rgb is None or rgb.ndim != 4 or rgb.shape[1] != 3):
                raise ValueError("RGB input must be [B,3,H,W]")
            if self.fusion and (
                rgb.shape[0] != event_hist.shape[0]
                or rgb.shape[2:] != event_hist.shape[2:]
            ):
                raise ValueError("RGB and DVS must share batch and spatial dimensions")
        if self.temporal_window:
            if temporal_state is None:
                raise ValueError("Explicit temporal state required; initialize/reset per stream")
            ev, next_state = encoder_step(self.event_encoder, event_hist, temporal_state)
        else:
            if temporal_state is not None:
                raise ValueError("Stateless backbone cannot consume temporal memory")
            if self.cross_fusion:
                return self._forward_cross_stage(event_hist, rgb)
            ev = self.event_encoder(event_hist) if self.use_events else None
        im = self.rgb_encoder(rgb) if self.use_rgb else None
        outputs = []
        for i in range(4):
            if self.cross_fusion:
                # No cross-modal feedback: the RGB and DVS encoders remain independent.
                _, _, fused = self.interactions[i](im[f"stage{i+1}"], ev[f"stage{i+1}"])
                outputs.append(self.relu(self.fused_proj[i](fused)))
                continue
            feature = self.event_proj[i](ev[f"stage{i+1}"]) if ev is not None else None
            if im is not None:
                rgb_feature = self.rgb_proj[i](im[f"stage{i+1}"])
                if self.fusion_mode == "adaptive_add":
                    feature = self.gates[i](feature, rgb_feature)
                else:
                    feature = rgb_feature if feature is None else feature + rgb_feature
            outputs.append(self.relu(feature))
        return (tuple(outputs), next_state) if self.temporal_window else tuple(outputs)

    def _forward_cross_stage(self, event_hist, rgb):
        event = self.event_encoder.input_stem(event_hist)
        rgb = self.rgb_encoder.input_stem(rgb)
        outputs = []
        for event_stage, rgb_stage, interaction, projection in zip(
            self.event_encoder.stages, self.rgb_encoder.stages,
            self.interactions, self.fused_proj,
        ):
            event, rgb = event_stage(event), rgb_stage(rgb)
            rgb_next, event_next, fused = interaction(rgb, event)
            if self.fusion_mode == "cross_stage_post_mbconv":
                rgb, event = rgb_next, event_next
            # In no-feedback mode each next Stage consumes its ORIGINAL stream.
            # The unchanged interaction output still feeds projection and Neck.
            outputs.append(self.relu(projection(fused)))
        return tuple(outputs)
