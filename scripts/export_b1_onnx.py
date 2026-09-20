#!/usr/bin/env python3
"""Export fixed-resolution stateless B1 task graphs, simplify and compare numerics."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnxsim import simplify
import torch
from hmnet.models.efficientvit_tasks import build_frame_task


class FrameGraph(torch.nn.Module):
    def __init__(self, model, task, height, width, backbone_only=False):
        super().__init__()
        self.model = model
        self.task = task
        self.metas = [dict(height=height, width=width)]
        self.backbone_only = backbone_only

    def forward(self, event_hist, rgb=None):
        features = self.model.backbone(event_hist, rgb)
        if self.backbone_only:
            return features
        features = self.model.neck(list(features))
        if self.task == "segmentation":
            return self.model.seg_head(features, self.metas)
        if self.task == "depth":
            return self.model.reg_head.inference(features, self.metas)
        # Dense decoded xywh, objectness, class probabilities. NMS remains outside
        # the graph; it has data-dependent output length and no trainable operators.
        return self.model.bbox_head.inference(features)


def export(args):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    task = "depth" if args.task.startswith("depth") else args.task
    h, w = {
        "segmentation": (440, 640),
        "detection": (240, 304),
        "depth_eventscape": (256, 512),
        "depth_mvsec": (260, 346),
    }[args.task]
    model = build_frame_task(
        task, mvsec=args.task == "depth_mvsec"
    ).eval()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    wrapper = FrameGraph(model, task, h, w, args.backbone_only).eval()
    # Counts use the actual input domain; RGB uses normalized intensities.
    event = torch.randint(0, 11, (1, 20, h, w)).float()
    inputs = (event, torch.randn(1, 3, h, w)) if task == "segmentation" else (event,)
    if args.sample:
        with np.load(args.sample) as f:
            e = torch.from_numpy(f["histogram"].copy()).float()[None]
            r = torch.from_numpy(f["rgb"].copy()).permute(2, 0, 1).float()[None] / 255
            r = (r - r.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]) / r.new_tensor(
                [0.229, 0.224, 0.225]
            )[None, :, None, None]
        inputs = (e, r) if task == "segmentation" else (e,)
    names = ["event_hist", "rgb"] if task == "segmentation" else ["event_hist"]
    outputs = (
        ["f4", "f8", "f16", "f32"]
        if args.backbone_only
        else [{"segmentation": "logits", "depth": "depth", "detection": "detections"}[task]]
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stem = args.task + ("_backbone" if args.backbone_only else "")
    path = out / (stem + ".onnx")
    simple = out / (stem + ".sim.onnx")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs,
            str(path),
            input_names=names,
            output_names=outputs,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    original = onnx.load(path)
    onnx.checker.check_model(original, full_check=True)
    simplified, ok = simplify(original, check_n=0)
    if not ok:
        raise RuntimeError("onnxsim validation failed")
    onnx.checker.check_model(simplified, full_check=True)
    onnx.save(simplified, simple)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    sessions = [
        ort.InferenceSession(str(p), options, providers=["CPUExecutionProvider"])
        for p in (path, simple)
    ]
    checks, failures = [], []
    cases = [("sample" if args.sample else "synthetic", inputs),
             ("all_zero", tuple(torch.zeros_like(x) for x in inputs))]
    if task == "segmentation":
        cases.append(("empty_event", (torch.zeros_like(inputs[0]), inputs[1])))
    for case, test_inputs in cases:
        with torch.no_grad():
            expected = wrapper(*test_inputs)
        expected = expected if isinstance(expected, tuple) else (expected,)
        feed = {name: x.numpy() for name, x in zip(names, test_inputs)}
        values = [s.run(None, feed) for s in sessions]
        for j, want in enumerate(expected):
            want = want.numpy()
            for label, got in zip(("original", "simplified"), [v[j] for v in values]):
                # FP32 backend differences may be amplified by normalized attention.
                # Record the actual error and segmentation decisions, not only pass/fail.
                atol = 1e-3 if task == "segmentation" else 1e-4
                finite = bool(np.isfinite(got).all() and np.isfinite(want).all())
                close = finite and bool(np.allclose(got, want, atol=atol, rtol=1e-4))
                check = dict(
                    input_case=case,
                    graph=label,
                    output=outputs[j],
                    max_abs_error=float(np.max(np.abs(got - want))),
                    mean_abs_error=float(np.mean(np.abs(got - want))),
                    atol=atol,
                    rtol=1e-4,
                    finite=finite,
                    passed=close,
                    mismatch_fraction=float(np.mean(
                        ~np.isfinite(got) | ~np.isfinite(want)
                        | (np.abs(got - want) > atol + 1e-4 * np.abs(want))
                    )),
                )
                if task == "segmentation" and not args.backbone_only:
                    check["argmax_agreement"] = float(np.mean(got.argmax(1) == want.argmax(1)))
                    check["passed"] &= check["argmax_agreement"] >= 0.9999
                checks.append(check)
                if not check["passed"]:
                    failures.append(f"{case}/{outputs[j]}/{label}: PyTorch mismatch")
            # Keep the stricter original-vs-simplified check, but finish all
            # cases and write diagnostics BEFORE reporting a failed export.
            simplification_ok = bool(
                np.isfinite(values[0][j]).all() and np.isfinite(values[1][j]).all()
                and np.allclose(values[0][j], values[1][j], atol=1e-5, rtol=1e-4)
            )
            if not simplification_ok:
                failures.append(f"{case}/{outputs[j]}: simplification mismatch")
    report = dict(
        passed=not failures,
        failures=failures,
        sample=args.sample,
        task=args.task,
        fusion_mode="cross_stage_post_mbconv_muladd" if task == "segmentation" else "none",
        checkpoint=args.checkpoint,
        checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        shape=[h, w],
        inputs=names,
        outputs=outputs,
        checks=checks,
        nodes={
            label: dict(Counter(n.op_type for n in g.graph.node))
            for label, g in [("original", original), ("simplified", simplified)]
        },
    )
    (out / (stem + ".report.json")).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if failures:
        raise RuntimeError(
            f"ONNX numerical validation failed; graphs are diagnostic only. "
            f"See {out / (stem + '.report.json')}: " + "; ".join(failures)
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task",
        choices=["segmentation", "detection", "depth_eventscape", "depth_mvsec"],
        required=True,
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="logs/onnx/efficientvit_b1")
    p.add_argument("--sample")
    p.add_argument("--backbone-only", action="store_true")
    export(p.parse_args())
