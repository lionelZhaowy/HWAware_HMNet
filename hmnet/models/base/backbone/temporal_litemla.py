"""Pure M=2 DVS LiteMLA steps. No mutable module state or extra parameters.

Token-row notation: S=K.T@V, z=K.T@1. Official channel-first layout
stores [S.T; z.T] as [B, heads*scales, d+1, d]. Only the CURRENT summary
is returned; feeding a running sum back would incorrectly retain all history.
"""
import torch
from torch.nn import functional as F
from .vendor.efficientvit.models.nn.ops import EfficientViTBlock, LiteMLA


def memory_modules(encoder):
    return [m for m in encoder.modules() if isinstance(m, LiteMLA)]


def zero_memory(encoder, batch_size, device):
    return tuple(torch.zeros(batch_size,
        m.qkv.conv.out_channels // (3*m.dim) * (1+len(m.aggreg)),
        m.dim+1, m.dim, device=device, dtype=torch.float32)
        for m in memory_modules(encoder))


def attention_step(module, x, previous):
    base = module.qkv(x)
    packed = torch.cat([base] + [op(base) for op in module.aggreg], dim=1)
    b, _, h, w = packed.shape
    # Disabling autocast alone does NOT upcast BF16 operands. Both MatMuls,
    # persistent summaries and the normalization execute explicitly in FP32.
    with torch.autocast(device_type=x.device.type, enabled=False):
        q, k, v = packed.float().reshape(b, -1, 3*module.dim, h*w).split(module.dim, dim=2)
        q, k = q.relu(), k.relu()
        current = F.pad(v, (0, 0, 0, 1), value=1.) @ k.transpose(-1, -2)
        if not torch.jit.is_tracing():
            if previous.dtype != torch.float32 or previous.shape != current.shape:
                raise ValueError("Temporal memory must be FP32 [B,heads*scales,d+1,d]")
        read = (current + previous) @ q
        out = read[:, :, :-1] / (read[:, :, -1:] + module.eps)
    out = out.reshape(b, -1, h, w).to(packed.dtype)
    return module.proj(out), current


def encoder_step(encoder, events, previous):
    if len(previous) != len(memory_modules(encoder)):
        raise ValueError("Expected one memory tensor per DVS LiteMLA")
    x = encoder.input_stem(events)
    outputs, next_state = {}, []
    for stage_id, stage in enumerate(encoder.stages, 1):
        for block in stage.op_list:
            if isinstance(block, EfficientViTBlock):
                residual = block.context_module
                source = x if residual.pre_norm is None else residual.pre_norm(x)
                context, state = attention_step(residual.main, source, previous[len(next_state)])
                x = context if residual.shortcut is None else context + residual.shortcut(x)
                if residual.post_act is not None:
                    x = residual.post_act(x)
                x = block.local_module(x)
                next_state.append(state)
            else:
                x = block(x)
        outputs[f"stage{stage_id}"] = x
    return outputs, tuple(next_state)
