#!/usr/bin/env python3
"""Evaluate a frozen B1 checkpoint and render one DSEC sequence (no training)."""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from hmnet.dataset.dsec_frames import DSECFrames
from hmnet.dataset.custom_collate_fn import collate_keep_dict
from hmnet.utils.config import load_config
from hmnet.utils.frame_train import unpack

# DSEC-Semantic's official 11-class order; black denotes ignored/missing labels.
NAMES = [
    "background",
    "building",
    "fence",
    "person",
    "pole",
    "road",
    "sidewalk",
    "vegetation",
    "car",
    "wall",
    "traffic sign",
]
COLORS = np.array(
    [
        [25, 28, 35],
        [70, 70, 70],
        [190, 153, 153],
        [220, 20, 60],
        [153, 153, 153],
        [128, 64, 128],
        [244, 35, 232],
        [107, 142, 35],
        [0, 0, 142],
        [102, 102, 156],
        [220, 220, 0],
    ],
    dtype=np.uint8,
)


def confusion(gt, pred):
    valid = (gt >= 0) & (gt < 11)
    # Column 11 penalizes missing HMNet predictions (including sequence warmup).
    pred = np.where((pred >= 0) & (pred < 11), pred, 11)
    return np.bincount(
        gt[valid].astype(np.int64) * 12 + pred[valid], minlength=132
    ).reshape(11, 12)


def metrics(c):
    tp = np.diag(c[:, :11])
    union = c.sum(1) + c[:, :11].sum(0) - tp
    iou = np.divide(tp, union, out=np.zeros(11, dtype=float), where=union > 0)
    return dict(
        miou=float(iou[union > 0].mean()),
        pixel_accuracy=float(tp.sum() / c.sum()),
        class_iou=[float(v) if u else None for v, u in zip(iou, union)],
        gt_pixel_fraction=(c.sum(1) / c.sum()).tolist(),
        confusion=c.tolist(),
    )


def color(labels):
    result = np.zeros((*labels.shape, 3), np.uint8)
    valid = (labels >= 0) & (labels < 11)
    result[valid] = COLORS[labels[valid]]
    return result


def panel(rgb, hist, gt, pred, baseline, title, model_label="EfficientViT-B1 RGB+DVS"):
    # Histogram channel order is [negative bins, positive bins].
    events = np.zeros_like(rgb)
    events[..., 0] = np.minimum(hist[10:].sum(0) * 32, 255).astype(np.uint8)
    events[..., 2] = np.minimum(hist[:10].sum(0) * 32, 255).astype(np.uint8)
    valid = (gt >= 0) & (gt < 11)
    error = np.zeros_like(rgb)
    error[valid & (gt == pred)] = (35, 95, 55)
    error[valid & (gt != pred)] = (240, 55, 55)
    views = [
        (rgb, "Registered RGB"),
        (events, "DVS: red positive / blue negative"),
        (color(gt), "Ground truth"),
        (color(pred), model_label),
        (color(baseline), "HMNet-B3 events (official weights)"),
        (error, "B1 error: red wrong / green correct"),
    ]
    canvas = np.full((720, 1440, 3), 20, np.uint8)
    cv2.putText(
        canvas,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    for k, (im, label) in enumerate(views):
        x = (k % 3) * 480
        y = 45 + (k // 3) * 310
        cv2.putText(
            canvas,
            label,
            (x + 10, y + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        canvas[y + 28 : y + 303, x + 40 : x + 440] = cv2.resize(
            im, (400, 275), interpolation=cv2.INTER_NEAREST
        )
    for i, name in enumerate(NAMES):
        x = (i % 6) * 240
        y = 676 + (i // 6) * 23
        cv2.rectangle(
            canvas,
            (x + 10, y - 10),
            (x + 24, y + 3),
            tuple(int(c) for c in COLORS[i]),
            -1,
        )
        cv2.putText(
            canvas,
            name,
            (x + 30, y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return canvas


@torch.inference_mode()
def main(args):
    torch.set_num_threads(4)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config, "b1_assessment").TestSettings()
    model = cfg.get_model().cuda().eval()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    checkpoint_step = ckpt["step"]
    del ckpt
    ds = DSECFrames(args.cache, args.split)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size or cfg.batch_size,
        num_workers=cfg.workers,
        collate_fn=collate_keep_dict,
        pin_memory=True,
        **({"prefetch_factor": getattr(cfg, "prefetch_factor", 1)} if cfg.workers > 0 else {}),
    )
    seqs = sorted({s["sequence"] for s in ds.samples})
    counts = {s: sum(r["sequence"] == s for r in ds.samples) for s in seqs}
    predicted = {
        s: np.lib.format.open_memmap(
            out / (s + "_predictions.npy"),
            mode="w+",
            dtype=np.uint8,
            shape=(counts[s], 440, 640),
        )
        for s in seqs
    }
    baseline = (
        {
            s: np.load(Path(args.hmnet_dir) / (s + "_predictions.npy"), mmap_mode="r")
            for s in seqs
        }
        if args.hmnet_dir
        else {}
    )
    cs = {s: np.zeros((11, 12), np.int64) for s in seqs}
    hs = {s: np.zeros((11, 12), np.int64) for s in seqs}
    common_c = np.zeros((11, 12), np.int64)
    common_h = common_c.copy()
    positions = {s: 0 for s in seqs}
    records = []
    times = []
    cursor = 0
    demo = [s for s in ds.samples if s["sequence"] == args.demo_sequence]
    fps = (
        1e6 / np.median(np.diff([s["target_us"] for s in demo]))
        if len(demo) > 1
        else 20
    )
    video = None
    if demo:
        ffmpeg = shutil.which("ffmpeg") or str(Path(sys.executable).parent / "ffmpeg")
        video = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "1440x720",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "4",
                "-preset",
                "fast",
                "-crf",
                "21",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out / "sequence_demo.mp4"),
            ],
            stdin=subprocess.PIPE,
        )
    started = time.perf_counter()
    try:
        for batch in loader:
            ev, ims, metas, targets = unpack(batch, "segmentation")
            torch.cuda.synchronize()
            t = time.perf_counter()
            logits, _ = model.inference(ev, ims, metas)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
            preds = logits.argmax(1).numpy()
            for pred, gt_tensor in zip(preds, targets):
                sample = ds.samples[cursor]
                cursor += 1
                seq = sample["sequence"]
                idx = positions[seq]
                positions[seq] += 1
                gt = gt_tensor.numpy()
                c = confusion(gt, pred)
                cs[seq] += c
                predicted[seq][idx] = pred
                # Original semantic-file index ties both models to the exact same GT.
                original_index = int(Path(sample["file"]).stem)
                base = (
                    baseline[seq][original_index, 0]
                    if baseline
                    else np.full_like(gt, -1)
                )
                if baseline:
                    hs[seq] += confusion(gt, base)
                    if (base >= 0).any():
                        common_c += c
                        common_h += confusion(gt, base)
                record = dict(
                    sequence=seq,
                    file=sample["file"],
                    original_index=original_index,
                    target_us=sample["target_us"],
                    frame_miou=metrics(c)["miou"],
                )
                records.append(record)
                if seq == args.demo_sequence:
                    with np.load(Path(args.cache) / sample["file"]) as data:
                        frame = panel(
                            data["rgb"],
                            data["histogram"].astype(np.int32),
                            gt,
                            pred,
                            base,
                            f"{seq} | frame {original_index} | t={(sample['target_us']-demo[0]['target_us'])/1e6:.2f}s | checkpoint {checkpoint_step}",
                            model_label=f"EfficientViT-B1 {cfg.modality.upper()}",
                        )
                    video.stdin.write(frame.tobytes())
                    if idx in {
                        0,
                        len(demo) // 4,
                        len(demo) // 2,
                        3 * len(demo) // 4,
                        len(demo) - 1,
                    }:
                        cv2.imwrite(str(out / f"demo_{idx:06d}.jpg"), frame[:, :, ::-1])
            if cursor % 100 < args.batch_size:
                print(f"{cursor}/{len(ds)}", flush=True)
    finally:
        if video:
            video.stdin.close()
            if video.wait() != 0:
                raise RuntimeError("Video encoding failed")
    for p in predicted.values():
        p.flush()
    result = dict(
        checkpoint=str(Path(args.checkpoint).resolve()),
        step=checkpoint_step,
        split=args.split,
        frames=len(ds),
        cache=str(Path(args.cache).resolve()),
        classes=NAMES,
        precision="FP32",
        modality=cfg.modality,
        fusion_mode="cross_stage_post_mbconv_muladd" if cfg.modality == "rgbdvs" else "none",
        parameters=sum(p.numel() for p in model.parameters()),
        evaluated=metrics(sum(cs.values())),
        sequences={s: dict(frames=counts[s], **metrics(cs[s])) for s in seqs},
        seconds=time.perf_counter() - started,
        inference_ms_per_frame_excluding_first_batch=sum(times[1:])
        * 1000
        / max(1, len(ds) - args.batch_size),
        timing_note="Includes input transfer and CPU logits return; excludes cache IO, rendering; batch inference, not Dremi latency.",
        demo_sequence=args.demo_sequence,
        demo_fps=float(fps),
    )
    if baseline:
        result.update(
            hmnet_same_frames=metrics(sum(hs.values())),
            hmnet_sequences={s: metrics(hs[s]) for s in seqs},
            common_predicted_frames=dict(b1=metrics(common_c), hmnet=metrics(common_h)),
        )
    (out / "assessment.json").write_text(json.dumps(result, indent=2))
    (out / "frame_metrics.json").write_text(json.dumps(records, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("sequences", "hmnet_sequences")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", default="experiments/segmentation/config/efficientvit_b1.py"
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--split", choices=("train", "dev", "test"), default="test")
    p.add_argument("--output", required=True)
    p.add_argument(
        "--hmnet-dir",
        help="Optional HMNet arrays indexed by original semantic-frame index",
    )
    p.add_argument("--demo-sequence", default="zurich_city_13_a")
    p.add_argument("--batch-size", type=int, default=None, help="Default: TestSettings.batch_size")
    main(p.parse_args())
