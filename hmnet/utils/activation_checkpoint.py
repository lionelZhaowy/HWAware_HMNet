"""BN-safe non-reentrant checkpointing, adapted from local CrossModalLiteMLA."""
from contextlib import contextmanager, nullcontext
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@contextmanager
def temporary_bn_buffers(module):
    saved = []
    for layer in module.modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            buffers = (layer.running_mean, layer.running_var, layer.num_batches_tracked)
            saved.append((layer, buffers))
            layer.running_mean, layer.running_var, layer.num_batches_tracked = (
                value.clone() if value is not None else None for value in buffers)
    try:
        yield
    finally:
        for layer, buffers in saved:
            layer.running_mean, layer.running_var, layer.num_batches_tracked = buffers


def recompute(module, function, *args):
    if module.training and torch.is_grad_enabled():
        return checkpoint(function, *args, use_reentrant=False,
                          context_fn=lambda: (nullcontext(), temporary_bn_buffers(module)))
    return function(*args)
