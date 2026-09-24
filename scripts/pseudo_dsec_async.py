#!/usr/bin/env python3
"""Audit frozen semantic teachers, then generate ONE shared train pseudo set.
Supports original B+C, synchronous binary, or binary+C confidence agreement.

Algorithm references: FAOD past-RGB shifts; UniMatch confidence/ignore masks.
No boxes, track interpolation, future frames or temporal pixel copying.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from hmnet.dataset.dsec_async import DSECAsync
from hmnet.models.async_frame import AsyncPredictor
from hmnet.utils.async_checkpoint import load_async_model,file_sha256
from hmnet.utils.binary_teacher import load_binary_teacher,BinaryTeacherInputs,BinaryTeacherPredictor


def confidence_mask(logits, confidence):
    """All teachers must agree and individually exceed the confidence threshold."""
    probability, label = logits[0].float().softmax(1).max(1)
    accepted = probability >= confidence
    for prediction in logits[1:]:
        probability, other = prediction.float().softmax(1).max(1)
        accepted &= (other == label) & (probability >= confidence)
    return label.masked_fill(~accepted, 255), accepted


def agreement_mask(logits_b, logits_c, confidence):
    # Preserve the original B/C utility API.
    return confidence_mask([logits_b, logits_c], confidence)



def validate_audit(audit, provenance, protocol, confidence, grid_sha256):
    if (not audit["passed"] or audit["teachers"] != provenance
            or audit["confidence"] != confidence or audit.get("teacher_protocol") != protocol
            or audit["grid_sha256"] != grid_sha256):
        raise ValueError("Failed/stale audit or teacher/confidence/protocol/grid mismatch; rerun audit")


def boundary_mask(label):
    valid=label!=255;edge=torch.zeros_like(valid)
    horizontal=(label[:,1:]!=label[:,:-1])&valid[:,1:]&valid[:,:-1]
    vertical=(label[1:]!=label[:-1])&valid[1:]&valid[:-1]
    edge[:,1:] |= horizontal;edge[:,:-1] |= horizontal
    edge[1:] |= vertical;edge[:-1] |= vertical
    return edge


def main(args):
    torch.set_num_threads(2)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/"manifest.json").exists() or (out/"audit.json").exists():
        raise FileExistsError("Choose a new output directory; pseudo provenance is immutable")
    names={"bc":("B","C"),"binary":("binary",),"binary_c":("binary","C")}[args.teacher_mode]
    provenance={v:dict(checkpoint=str(Path(getattr(args,v.lower()+"_checkpoint")).resolve()),
                      sha256=file_sha256(getattr(args,v.lower()+"_checkpoint"))) for v in names}
    protocol=dict(teacher_mode=args.teacher_mode, inference_precision="bf16" if torch.device(args.device).type=="cuda" else "fp32",
                  binary_temporal_policy="two_interleaved_50ms_streams_v1" if "binary" in names else None)
    split="dev" if args.mode=="audit" else "train"
    data=[];predictors=[];binary_inputs=[]
    for v in names:
        if v == "binary":
            model,contract=load_binary_teacher(args.binary_checkpoint,args.device)
            ds=DSECAsync(args.b_data,split,window_us=50000,bins=10,
                         rgb_delay_frames=1 if args.mode=="audit" else 0)
            inputs=BinaryTeacherInputs(args.binary_data,ds.manifest,contract)
            protocol["binary_manifest_sha256"]=inputs.signature
            predictor=BinaryTeacherPredictor(model)
        else:
            model,contract=load_async_model(getattr(args,v.lower()+"_checkpoint"),args.device,v,phase=1)
            ds=DSECAsync(getattr(args,v.lower()+"_data"),split,window_us=50000 if v=="B" else 25000,
                bins=10 if v=="B" else 5,rgb_delay_frames=1 if args.mode=="audit" else 0)
            if ds.manifest["grid_sha256"]!=contract["asynchronous"]["grid_sha256"]:
                raise ValueError("Teacher/data time grid differs")
            predictor=AsyncPredictor(model);inputs=None
        data.append(ds);predictors.append(predictor);binary_inputs.append(inputs)
    if any(ds.manifest["grid_sha256"]!=data[0].manifest["grid_sha256"] or len(ds)!=len(data[0]) for ds in data):
        raise ValueError("Teachers must have identical event times and GT membership")
    protocol["dataset_manifest_sha256"]={v:ds.signature for v,ds in zip(names,data)}
    if args.mode=="generate":
        if not args.audit:raise ValueError("Generation requires --audit from held-out dev labels")
        audit=json.loads(Path(args.audit).read_text())
        validate_audit(audit,provenance,protocol,args.confidence,data[0].manifest["grid_sha256"])
    total=np.zeros(11,dtype=np.int64);accepted=np.zeros(11,dtype=np.int64);correct=np.zeros(11,dtype=np.int64)
    boundary_total=boundary_accepted=boundary_correct=0
    files={};pseudo_pixels=0;all_pixels=0
    with torch.inference_mode():
        for i in range(len(data[0])):
            samples=[ds[i] for ds in data];metas=[x[2]["image_meta"] for x in samples]
            if any((m["sequence"],m["curr_time_org"],m["has_gt"]) !=
                   (metas[0]["sequence"],metas[0]["curr_time_org"],metas[0]["has_gt"]) for m in metas):
                raise ValueError("Misaligned teachers")
            logits=[]
            for predictor,sample,meta,inputs in zip(predictors,samples,metas,binary_inputs):
                changed=predictor.ids!=[meta["rgb_id"]] or meta["reset"] or predictor.sequence!=meta["sequence"]
                rgb=sample[0]["images"][None] if meta["rgb_valid"] and changed else None
                events=inputs.read(meta) if inputs is not None else sample[0]["events"][None]
                with torch.autocast(torch.device(args.device).type,enabled=torch.device(args.device).type=="cuda",dtype=torch.bfloat16):
                    logits.append(predictor.step(events,meta,rgb))
            label,mask=confidence_mask(logits,args.confidence)
            label,mask=label[0].cpu(),mask[0].cpu();meta=metas[0]
            if args.mode=="audit" and meta["has_gt"]:
                gt=samples[0][1]["labels"];valid=gt!=255;keep=mask&valid;ok=keep&(label==gt)
                total+=torch.bincount(gt[valid],minlength=11).numpy()
                accepted+=torch.bincount(gt[keep],minlength=11).numpy()
                correct+=torch.bincount(gt[ok],minlength=11).numpy()
                edges=boundary_mask(gt);boundary_total+=int(edges.sum())
                boundary_accepted+=int((edges&keep).sum());boundary_correct+=int((edges&ok).sum())
            elif args.mode=="generate" and not meta["has_gt"]:
                key=f'{meta["sequence"]}/{meta["curr_time_org"]}'
                relative=key+".npy";path=out/relative;path.parent.mkdir(parents=True,exist_ok=True)
                np.save(path,label.byte().numpy());files[key]=relative
                pseudo_pixels+=int(mask.sum());all_pixels+=mask.numel()
            if (i+1)%100==0:print(json.dumps(dict(mode=args.mode,steps=i+1,total=len(data[0]))),flush=True)
    if args.mode=="audit":
        precision=float(correct.sum()/max(1,accepted.sum()));coverage=float(accepted.sum()/max(1,total.sum()))
        class_precision=[float(c/a) if a else None for c,a in zip(correct,accepted)]
        supported=accepted>=args.min_class_pixels
        passed=(accepted.sum()>0 and precision>=args.min_precision and coverage>=args.min_coverage and
                all(class_precision[i]>=args.min_class_precision for i in range(11) if supported[i]))
        result=dict(format="dsec_async_audit_v1",passed=bool(passed),teachers=provenance,teacher_protocol=protocol,confidence=args.confidence,
            calibration_split="dev (not independent final test)",rgb_delay_frames=1,
            precision=precision,coverage=coverage,class_precision=class_precision,
            accepted_by_gt_class=accepted.tolist(),correct_by_gt_class=correct.tolist(),gt_by_class=total.tolist(),
            boundary_precision=boundary_correct/max(1,boundary_accepted),boundary_coverage=boundary_accepted/max(1,boundary_total),
            thresholds=dict(min_precision=args.min_precision,min_coverage=args.min_coverage,
                min_class_precision=args.min_class_precision,min_class_pixels=args.min_class_pixels),
            grid_sha256=data[0].manifest["grid_sha256"],
            limitation="Held-out GT time proxy; does not establish unlabelled 25ms accuracy")
        (out/"audit.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
        if not passed:raise SystemExit("Pseudo quality gate failed; labels were not generated")
    else:
        if audit["grid_sha256"]!=data[0].manifest["grid_sha256"]:raise ValueError("Audit grid differs")
        result=dict(format="dsec_async_pseudo_v1",audit_passed=True,audit_sha256=file_sha256(args.audit),
            teachers=provenance,teacher_protocol=protocol,confidence=args.confidence,grid_sha256=data[0].manifest["grid_sha256"],
            split="train",files=files,coverage=pseudo_pixels/max(1,all_pixels),
            loss_normalization="accepted pixels, independent of GT CE; weight ramp")
        (out/"manifest.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(dict(files=len(files),coverage=result["coverage"])))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("audit","generate"))
    p.add_argument("--teacher-mode",choices=("bc","binary","binary_c"),default="bc")
    for v in ("b","c","binary"):
        p.add_argument(f"--{v}-checkpoint");p.add_argument(f"--{v}-data")
    p.add_argument("--output",required=True);p.add_argument("--device",default="cuda:0")
    p.add_argument("--confidence",type=float,default=.95)
    p.add_argument("--min-precision",type=float,default=.90)
    p.add_argument("--min-coverage",type=float,default=.10)
    p.add_argument("--min-class-precision",type=float,default=.70)
    p.add_argument("--min-class-pixels",type=int,default=1000)
    p.add_argument("--audit")
    args=p.parse_args()
    if any(not 0<=getattr(args,k)<=1 for k in ("confidence","min_precision","min_coverage","min_class_precision")):
        p.error("Confidence/quality thresholds must be in [0,1]")
    required={"bc":("b_checkpoint","c_checkpoint","b_data","c_data"),
              "binary":("binary_checkpoint","binary_data","b_data"),
              "binary_c":("binary_checkpoint","binary_data","b_data","c_checkpoint","c_data")}[args.teacher_mode]
    if args.min_class_pixels < 1:p.error("--min-class-pixels must be positive")
    missing=["--"+key.replace("_","-") for key in required if not getattr(args,key)]
    if missing:p.error("Selected teacher mode requires: "+", ".join(missing))
    main(args)
