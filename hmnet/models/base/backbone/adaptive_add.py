"""Spatially adaptive Add, initially identical to the original projected Add."""
import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveAdd(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 != 1:
            raise ValueError("The spatial kernel must be positive and odd")
        self.padding = kernel_size // 2
        # Functional Conv2d with zero Parameters avoids consuming RNG and avoids
        # the task wrapper's generic Conv initialization overwriting zero logits.
        # Thus adding this module preserves ALL common Add-model initialization.
        self.weight = nn.Parameter(torch.zeros(1, 4, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, event, rgb):
        # Inputs: [B,256,H,W] after separate Conv+BN projections.
        # Four spatial summaries -> [B,1,H,W]; broadcast across channels.
        stats = torch.cat((event.mean(1, keepdim=True), event.amax(1, keepdim=True),
                           rgb.mean(1, keepdim=True), rgb.amax(1, keepdim=True)), dim=1)
        logits = F.conv2d(stats, self.weight, self.bias, padding=self.padding)
        delta = 2 * F.hardsigmoid(logits) - 1
        # Coefficients lie in [0,2] and sum to 2. Zero logits yield EXACT Add.
        # ReLU remains outside, in the unchanged backbone output path.
        return event * (1 - delta) + rgb * (1 + delta)
