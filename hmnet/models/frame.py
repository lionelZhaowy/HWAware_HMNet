"""Thin task-wrapper paths for stateless dense backbones; legacy HMNet is untouched."""

import torch


def is_frame_model(model):
    return getattr(model.backbone, "input_format", None) == "histogram"


def frame_features(model, events, images=None):
    device = next(model.parameters()).device
    events = torch.stack(events) if isinstance(events, (list, tuple)) else events
    rgb = None
    if getattr(model.backbone, "use_rgb", model.backbone.fusion):
        if images is None or any(x is None for x in images):
            raise ValueError("RGB fusion sample has no synchronized image")
        rgb = torch.stack(images) if isinstance(images, (list, tuple)) else images
        rgb = rgb.to(device)
    events = events.to(device) if getattr(model.backbone, "use_events", True) else None
    return model.backbone(events, rgb)


def frame_loss(model, events, images, metas, targets, kind, boxes=None, ignore=None):
    if kind == "segmentation":
        keep = [i for i, t in enumerate(targets) if t is not None and (t != 255).any()]
    elif kind == "depth":
        lo, hi = model.reg_head.min_depth, model.reg_head.max_depth
        keep = [
            i
            for i, t in enumerate(targets)
            if t is not None
            and (
                torch.isfinite(t)
                & ((t > 0) if model.reg_head.clip_gt else ((t >= lo) & (t <= hi)))
            ).any()
        ]
    else:
        # An empty annotated box list is a supervised background sample, not missing GT.
        keep = [i for i, t in enumerate(targets) if t is not None]
    if not keep:
        zero = sum(p.sum() * 0 for p in model.parameters())
        return dict(loss=zero, log_vars={}, num_samples=0, skip_step=True)
    ev = events[keep] if torch.is_tensor(events) else [events[i] for i in keep]
    im = [images[i] for i in keep] if images is not None else None
    features = list(frame_features(model, ev, im))
    selected = [metas[i] for i in keep]
    device = features[0].device
    if kind == "detection":
        loss, logs = model._forward_head(
            features,
            [boxes[i].to(device) for i in keep],
            [targets[i].to(device) for i in keep],
            [
                (
                    ignore[i].to(device)
                    if ignore is not None and ignore[i] is not None
                    else torch.zeros_like(targets[i], dtype=torch.bool, device=device)
                )
                for i in keep
            ],
        )
    else:
        gt = torch.stack([targets[i] for i in keep]).to(device)
        if kind == "depth":
            gt = torch.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0)
        loss, logs = model._forward_head(
            features, selected, gt, torch.arange(len(keep), device=device)
        )
    return dict(loss=loss, log_vars=logs, num_samples=len(keep), skip_step=False)


def frame_inference(model, events, images, metas, kind):
    # Existing test loaders may deliver a sequence of dense batches. Each is an
    # independent window: there is deliberately no reset, warmup or feature cache.
    if metas and isinstance(metas[0], (list, tuple)):
        results, metadata = [], []
        for i, batch_meta in enumerate(metas):
            output, selected = frame_inference(
                model,
                events[i],
                images[i] if images is not None else None,
                batch_meta,
                kind,
            )
            results.extend(output)
            metadata.extend(selected)
        return (results if kind == "detection" else torch.stack(results)), metadata
    features = frame_features(model, events, images)
    pyramid = model.neck(list(features))
    if kind == "detection":
        detections = model.bbox_head.postprocess(model.bbox_head.inference(pyramid))
        pred = [
            (
                dict(bboxes=d[:, :4], labels=d[:, 6], scores=d[:, 4] * d[:, 5])
                if d is not None
                else dict(
                    bboxes=torch.empty(0, 4),
                    labels=torch.empty(0),
                    scores=torch.empty(0),
                )
            )
            for d in detections
        ]
    else:
        head = model.seg_head if kind == "segmentation" else model.reg_head
        pred = head.inference(pyramid, metas).cpu()
    return pred, metas
