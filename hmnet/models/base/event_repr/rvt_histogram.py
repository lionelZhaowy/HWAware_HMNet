"""HMNet metadata adapter; histogram construction is the unmodified RVT code."""

import torch
from torch import nn
from .vendor.rvt_histogram import StackedHistogram


class RVTHistogram(nn.Module):
    def __init__(self, bins=10, count_cutoff=10, fastmode=True):
        super().__init__()
        self.bins, self.count_cutoff, self.fastmode = bins, count_cutoff, fastmode
        self.num_channels = 2 * bins

    def forward(self, events, image_meta):
        if events.ndim != 2 or events.shape[1] != 4 or events.dtype != torch.int64:
            raise ValueError(
                'RVT requires int64 [N,4] = (time_us,x,y,polarity_01); use output_type="long"'
            )
        t, x, y, p = events.unbind(1)
        h, w = int(image_meta["height"]), int(image_meta["width"])
        if len(t):
            if (t[1:] < t[:-1]).any():
                raise ValueError("Event timestamps must be sorted")
            if ((x < 0) | (x >= w) | (y < 0) | (y >= h) | (p < 0) | (p > 1)).any():
                raise ValueError("Event coordinate or polarity outside representation domain")
        # RVT orders channels polarity-major, then time bin: p0 bins, p1 bins.
        # Its first/last event normalization and uint8 fastmode overflow are retained.
        histogram = StackedHistogram(self.bins, h, w, self.count_cutoff, self.fastmode).construct(
            x, y, p, t
        )
        meta = dict(
            image_meta, img_shape=[w, h, self.num_channels], pad_shape=[w, h, self.num_channels]
        )
        return histogram.float(), meta
