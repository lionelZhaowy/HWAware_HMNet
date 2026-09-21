"""Normalized bidirectional LiteMLA interaction for native B1 stage features.

The ReLU kernel, multi-scale aggregation and padded-V normalization follow the
vendored official EfficientViT LiteMLA (vendor/efficientvit/models/nn/ops.py;
upstream license: vendor/efficientvit/LICENSE). Queries attend to the OTHER stream's
keys/values. Two residual streams continue through the backbone; a separate
merged feature goes to the pyramid. v2.1 follows EfficientViTBlock ordering:
attention projection + residual, then local MBConv + residual. The learned
attention residual scales are retained from the previous cross-modal experiment.
"""

from contextlib import contextmanager, nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .vendor.efficientvit.models.nn import MBConv


class CrossModalLiteMLA(nn.Module):
    def __init__(self, channels, dim=16, scales=(5,), residual_scale=0.1, eps=1e-15):
        super().__init__()
        if channels % dim:
            raise ValueError("Stage channels must be divisible by head dimension")
        self.channels = channels
        self.dim = dim
        self.heads = channels // dim
        self.eps = eps

        def local():
            # Match the official EfficientViTBlock's POST-attention local block:
            # expansion=4, HardSwish, bias in expansion/depthwise, final BN only.
            return MBConv(
                channels, channels, expand_ratio=4,
                use_bias=(True, True, False), norm=(None, None, "bn2d"),
                act_func=("hswish", "hswish", None),
            )

        self.rgb_post = local()
        self.event_post = local()
        self.rgb_qkv = nn.Conv2d(channels, 3 * channels, 1, bias=False)
        self.event_qkv = nn.Conv2d(channels, 3 * channels, 1, bias=False)

        def aggregations():
            # Each Q/K/V head is aggregated independently as in LiteMLA.
            return nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(3 * channels, 3 * channels, kernel,
                              padding=kernel // 2, groups=3 * channels, bias=False),
                    nn.Conv2d(3 * channels, 3 * channels, 1,
                              groups=3 * self.heads, bias=False),
                ) for kernel in scales
            ])

        self.rgb_aggreg = aggregations()
        self.event_aggreg = aggregations()
        attention_channels = channels * (1 + len(scales))

        def projection(in_channels):
            return nn.Sequential(nn.Conv2d(in_channels, channels, 1, bias=False),
                                 nn.BatchNorm2d(channels))

        self.rgb_update = projection(attention_channels)
        self.event_update = projection(attention_channels)
        self.rgb_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.event_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.merge = nn.Sequential(projection(2 * channels), nn.ReLU())

    def _qkv(self, x, projection, aggregations):
        base = projection(x)
        # Layout is [all Q channels, all K channels, all V channels]. Split
        # BEFORE concatenating scales, so modality/head/scale ordering agrees.
        pyramids = [base] + [op(base) for op in aggregations]
        triples = [level.chunk(3, dim=1) for level in pyramids]
        b, _, h, w = base.shape
        return tuple(torch.cat([level[i] for level in triples], dim=1)
                     .reshape(b, -1, self.dim, h * w) for i in range(3))

    @staticmethod
    def normalized_attention(q, k, v, eps=1e-15):
        """Channel-first tensors [B, heads*scales, d, N]; no NxN matrix.

        Pad OTHER-modality V with ones to compute numerator and denominator
        together: V K^T Q / (1 K^T Q + eps). FP32 inside AMP prevents overflow
        during the long spatial reduction; V stays signed, Q/K use ReLU.
        """
        dtype = q.dtype
        with torch.autocast(device_type=q.device.type, enabled=False):
            q, k, v = q.float().relu(), k.float().relu(), v.float()
            context = F.pad(v, (0, 0, 0, 1), value=1.0) @ k.transpose(-1, -2)
            numerator_and_mass = context @ q
            result = numerator_and_mass[:, :, :-1] / (
                numerator_and_mass[:, :, -1:] + eps
            )
        return result.to(dtype)

    @contextmanager
    def _recompute_batch_norm(self):
        # Checkpoint recomputation must use batch statistics but must NOT update
        # persistent BN statistics a second time. Swap in temporary buffers;
        # restoring references avoids in-place changes to autograd-saved tensors.
        saved = []
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                buffers = (module.running_mean, module.running_var, module.num_batches_tracked)
                saved.append((module, buffers))
                module.running_mean, module.running_var, module.num_batches_tracked = (
                    value.clone() for value in buffers
                )
        try:
            yield
        finally:
            for module, buffers in saved:
                module.running_mean, module.running_var, module.num_batches_tracked = buffers

    def forward(self, rgb, event):
        if self.training and torch.is_grad_enabled():
            # Keep batch=32 on 24 GiB GPUs without changing effective batch or BN
            # semantics. Non-reentrant checkpoint also supports autograd.grad.
            return checkpoint(
                self._forward, rgb, event, use_reentrant=False,
                context_fn=lambda: (nullcontext(), self._recompute_batch_norm()),
            )
        return self._forward(rgb, event)

    def _forward(self, rgb, event):
        # Residual paths keep each stream's spatial information and identity;
        # only the cross-attention context is borrowed from the other stream.
        qr, kr, vr = self._qkv(rgb, self.rgb_qkv, self.rgb_aggreg)
        qe, ke, ve = self._qkv(event, self.event_qkv, self.event_aggreg)
        ar = self.normalized_attention(qr, ke, ve, self.eps).reshape(
            rgb.shape[0], -1, rgb.shape[2], rgb.shape[3])
        ae = self.normalized_attention(qe, kr, vr, self.eps).reshape(
            event.shape[0], -1, event.shape[2], event.shape[3])
        # Project attention alone (2*C -> C for the two default scales), then
        # retain the receiving stream's identity. Both directions use the same
        # pre-update inputs, so neither branch sees an already-updated partner.
        rgb_context = rgb + self.rgb_scale * self.rgb_update(ar)
        event_context = event + self.event_scale * self.event_update(ae)
        # Local processing now sees the exchanged information. Feed these two
        # distinct outputs to the next stage AND to the current pyramid output.
        rgb_next = rgb_context + self.rgb_post(rgb_context)
        event_next = event_context + self.event_post(event_context)
        output = self.merge(torch.cat((rgb_next, event_next), dim=1))
        return rgb_next, event_next, output
