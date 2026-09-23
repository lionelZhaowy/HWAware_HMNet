"""Two-step segmentation learning and causal inference with explicit RGB cache.

Only the DVS summaries carry gradients between steps. RGB cache is local to a
training clip, so no features computed before an optimizer update are reused.
"""
import torch
import torch.nn.functional as F
from hmnet.utils.activation_checkpoint import recompute


def update_rgb(backbone, images, valid, ids, cache=None, previous_ids=None):
    """Encode only changed/available images. IDs must include sequence identity."""
    b, _, h, w = images.shape
    changed = [i for i in range(b) if valid[i] and
               (previous_ids is None or ids[i] != previous_ids[i])]
    if cache is None:
        # ceil downsampling matches EfficientViT for arbitrary image sizes.
        dtype = torch.get_autocast_dtype(images.device.type) if torch.is_autocast_enabled(images.device.type) else images.dtype
        cache = tuple(torch.zeros(b,256,(h+s-1)//s,(w+s-1)//s,device=images.device,dtype=dtype)
                      for s in (4,8,16,32))
    if changed:
        index = torch.tensor(changed, device=images.device)
        encoded = backbone.encode_rgb(images.index_select(0,index))
        cache = tuple(old.to(new.dtype).index_copy(0,index,new) for old,new in zip(cache,encoded))
    return cache, list(ids)


def masked_ce(logits, target):
    """Accepted-pixel mean (unlike UniMatch's all-nonignore denominator)."""
    valid = target != 255
    loss = F.cross_entropy(logits.float(),target,ignore_index=255,reduction="sum")
    return loss / valid.sum().clamp_min(1)


def segmentation_loss(model, features, metas, target):
    # Checkpoint includes full-resolution CE to avoid retaining two large heads.
    def loss(*values):
        pred = model.seg_head(model.neck(list(values)),metas)
        aux = model.aux_head(values[0],metas)
        return masked_ce(pred,target), masked_ce(aux,target)*0.4
    main, aux = recompute(model,loss,*features)
    return main+aux, main, aux


def clip_loss(model, events, images, metas, targets, temporal_state):
    device = next(model.parameters()).device
    events = events.to(device)
    images = torch.stack(images).to(device)  # [B,T,3,H,W]
    if events.ndim != 5 or events.shape[1] != 2:
        raise ValueError("Async learning expects [B,2,C,H,W] clips")
    cache, ids = None, None
    logs = {}; total = events.sum()*0.; state = temporal_state
    for t in range(2):
        step_meta = [m["steps"][t] for m in metas]
        current_ids = [m["rgb_id"] for m in step_meta]
        cache, ids = update_rgb(model.backbone,images[:,t],
            [m["rgb_valid"] for m in step_meta],current_ids,cache,ids)
        features,state = model.backbone.async_step(events[:,t],cache,state)
        # First-step head is omitted in GT-only training: a head without loss
        # would merely perturb BN. Inference always decodes every event step.
        weight = float(metas[0].get("pseudo_weight",0.)) if t == 0 else 1.
        label = torch.stack([target[t] for target in targets]).to(device)
        if weight > 0 and (label != 255).any():
            value,main,aux = segmentation_loss(model,features,step_meta,label)
            total = total + weight*value
            prefix = "pseudo" if t == 0 else "gt"
            logs[prefix+"_main"] = float(main.detach())
            logs[prefix+"_aux"] = float(aux.detach())
        elif t == 0 and weight > 0:
            logs.update(pseudo_main=0.,pseudo_aux=0.)
    valid_endpoints = sum(bool((target[1] != 255).any()) for target in targets)
    # Cold-start event-only clips still participate safely in DDP-style graphs.
    total = total + sum(param.reshape(-1)[0]*0 for param in model.parameters())
    logs["pseudo_weight"] = float(metas[0].get("pseudo_weight",0.))
    logs["rgb_age_ms"] = sum(m["steps"][1]["rgb_age_us"] for m in metas)/len(metas)/1000
    return dict(loss=total,log_vars=logs,num_samples=len(metas),
                skip_step=valid_endpoints == 0,temporal_state=state)


class AsyncPredictor:
    """One chronological stream; reuse across events, reset on scene changes.

    `rgb` is normalized [1,3,H,W]; None means no image arrival. Neither current
    nor past cached RGB requires re-encoding while inference weights are fixed.
    """
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.state = None
        self.cache = None
        self.ids = None
        self.last_time = None
        self.rgb_time = None
        self.sequence = None

    @torch.no_grad()
    def step(self, events, meta, rgb=None):
        if self.model.training:
            raise ValueError("AsyncPredictor requires eval(); training uses clip_loss")
        time = int(meta["curr_time_org"])
        reset = (meta.get("reset",False) or self.sequence != meta["sequence"] or
                 (self.last_time is not None and time-self.last_time > 37500))
        if reset:
            self.reset()
        if self.last_time is not None and time <= self.last_time:
            raise ValueError("Stream time must increase")
        device = next(self.model.parameters()).device
        events = events.to(device)
        if events.shape[0] != 1:
            raise ValueError("One AsyncPredictor per stream")
        if self.state is None:
            self.state = self.model.backbone.zero_temporal_state(1)
        if rgb is not None:
            source_time = int(meta["rgb_time"])
            if source_time > time or (self.rgb_time is not None and source_time < self.rgb_time):
                raise ValueError("RGB must be causal and must not move backwards")
            if self.ids != [meta["rgb_id"]]:
                self.cache = self.model.backbone.encode_rgb(rgb.to(device))
                self.ids = [meta["rgb_id"]]
                self.rgb_time = source_time
        if self.cache is None:
            h,w = events.shape[-2:]
            self.cache = tuple(events.new_zeros(1,256,(h+s-1)//s,(w+s-1)//s) for s in (4,8,16,32))
        features,self.state = self.model.backbone.async_step(events,self.cache,self.state)
        logits = self.model.seg_head(self.model.neck(list(features)),[meta])
        self.sequence,self.last_time = meta["sequence"],time
        return logits
