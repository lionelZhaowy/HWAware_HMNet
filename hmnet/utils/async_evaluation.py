"""Decode every event step; report GT-only metrics separately from output rate."""
import time
import torch
from torch.utils.data import DataLoader
from hmnet.dataset.custom_collate_fn import collate_keep_dict
from hmnet.models.async_frame import AsyncPredictor


def confusion_metrics(confusion):
    union = confusion.sum(0)+confusion.sum(1)-confusion.diag()
    iou = confusion.diag().double()/union.clamp_min(1)
    return dict(miou=float(iou[union>0].mean()) if (union>0).any() else None,
                class_iou=[float(iou[i]) if union[i]>0 else None for i in range(11)],
                confusion=confusion.tolist())


@torch.no_grad()
def evaluate_async(model,dataset,workers=0,runtime=None,on_prediction=None):
    if dataset.clips:raise ValueError("Evaluation needs all event steps, not training clips")
    if runtime is not None and runtime.distributed:raise ValueError("Async eval is single stream per process")
    model.eval();predictor = AsyncPredictor(model)
    totals = {key:torch.zeros(11,11,dtype=torch.int64) for key in ("all","rgb_age_0_25ms","rgb_age_25_50ms","rgb_age_50ms_plus","no_rgb")}
    output_count,gt_count,rgb_updates = 0,0,0
    elapsed = 0.
    latencies = []
    loader = DataLoader(dataset,batch_size=1,num_workers=workers,collate_fn=collate_keep_dict,
                        **(dict(prefetch_factor=1) if workers else {}))
    for data,targets,metadata in loader:
        meta = metadata[0]["image_meta"]
        changed = predictor.ids != [meta["rgb_id"]] or meta["reset"] or predictor.sequence != meta["sequence"]
        rgb = data[0]["images"][None] if changed and meta["rgb_valid"] else None
        device = next(model.parameters()).device
        if device.type=="cuda":torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.autocast(device.type,enabled=device.type=="cuda",dtype=torch.bfloat16):
            logits = predictor.step(data[0]["events"][None],meta,rgb)
        if device.type=="cuda":torch.cuda.synchronize(device)
        duration = time.perf_counter()-start
        elapsed += duration
        latencies.append(duration*1000)
        meta = dict(meta,model_latency_ms=duration*1000)
        output_count += 1;rgb_updates += int(rgb is not None)
        if on_prediction is not None:on_prediction(logits,targets[0]["labels"],meta)
        if not meta["has_gt"]:continue
        gt_count += 1
        gt = targets[0]["labels"];pred = logits.argmax(1)[0].cpu()
        valid = (gt>=0)&(gt<11)
        confusion = torch.bincount((gt[valid]*11+pred[valid]).flatten(),minlength=121).reshape(11,11)
        totals["all"] += confusion
        age = meta["rgb_age_us"]
        group = ("no_rgb" if age<0 else "rgb_age_0_25ms" if age<25000 else
                 "rgb_age_25_50ms" if age<50000 else "rgb_age_50ms_plus")
        totals[group] += confusion
    return dict(**confusion_metrics(totals["all"]),age_groups={k:confusion_metrics(v) for k,v in totals.items() if k!="all"},
                event_outputs=output_count,gt_outputs=gt_count,rgb_encoder_calls=rgb_updates,
                latency_ms=dict(first=latencies[0],median=float(torch.tensor(latencies).median()),
                                p95=float(torch.quantile(torch.tensor(latencies),.95))),
                model_seconds=elapsed,model_outputs_per_second=output_count/max(elapsed,1e-9),
                latency_scope="host input transfer and model, excludes disk/data loading",
                inference_precision="bf16" if next(model.parameters()).is_cuda else "fp32")
