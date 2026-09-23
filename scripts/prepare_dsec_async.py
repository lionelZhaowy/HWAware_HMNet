#!/usr/bin/env python3
"""Create B/C caches together, reusing DSEC alignment and frozen RVT histogram.

GT membership comes from the parent manifest. Each GT has one preceding event
step. Regular intervals use their actual midpoint; gaps start a fresh clip.
"""
import argparse
import hashlib
import json
import os
import copy
import multiprocessing
import zipfile
from pathlib import Path
import cv2
import h5py
import hdf5plugin
import numpy as np
import torch
import yaml
from scripts.prepare_dsec_b1 import ensure_assets, validate_rectify, read_window
from hmnet.dataset.vendor.dsec_remapping import compute_remapping
from hmnet.models.base.event_repr.rvt_histogram import RVTHistogram


def time_grid(samples):
    rows = []
    previous = None
    for sample in sorted(samples,key=lambda s:s["target_us"]):
        time = int(sample["target_us"])
        regular = previous is not None and 40000 <= time-previous <= 60000
        midpoint = (previous+time)//2 if regular else time-25000
        for target,has_gt in ((midpoint,False),(time,True)):
            rows.append(dict(sequence=sample["sequence"],target_us=target,
                split=sample["split"],has_gt=has_gt,label_path=sample["label_path"] if has_gt else None,
                reset=not regular and not has_gt))
        previous = time
    return rows


def prepare(args):
    torch.set_num_threads(2)
    source,root,assets = Path(args.source),Path(args.output_root),Path(args.assets)
    base_path = Path(args.base_cache)/"manifest.json"
    base = json.loads(base_path.read_text())
    variants = {"B":(50000,10),"C":(25000,5)}
    outputs = {v:root/f"dsec_async_{v}" for v in args.variants}
    for out in outputs.values():
        if (out/"manifest.json").exists():raise FileExistsError(f"Completed cache: {out}; choose another output root")
    groups = {}
    for sample in base["samples"]:
        if args.sequences and sample["sequence"] not in args.sequences:continue
        groups.setdefault(sample["sequence"],[]).append(sample)
    if not groups:raise ValueError("No selected sequences")
    root.mkdir(parents=True,exist_ok=True)
    if args.workers > 1 and len(groups) > 1:
        jobs=[]
        for seq in sorted(groups):
            child=copy.copy(args);child.sequences=[seq];child.workers=1;child.part_sequence=seq
            jobs.append(child)
        with multiprocessing.get_context("spawn").Pool(min(args.workers,len(jobs))) as pool:
            pool.map(prepare,jobs)
        for v,out in outputs.items():
            parts=[json.loads((out/"parts"/(seq+".json")).read_text()) for seq in sorted(groups)]
            merged=dict(parts[0])
            merged["samples"]=[row for part in parts for row in part["samples"]]
            for key in ("rgb","counts","assets"):
                merged[key]={k:value for part in parts for k,value in part[key].items()}
            grid=[(r["sequence"],r["target_us"],r["has_gt"],r["split"]) for r in merged["samples"]]
            merged["grid_sha256"]=hashlib.sha256(json.dumps(grid,separators=(",",":")).encode()).hexdigest()
            temp=out/"manifest.tmp.json";temp.write_text(json.dumps(merged,indent=2));temp.replace(out/"manifest.json")
        print(json.dumps(dict(completed=[str(out) for out in outputs.values()])),flush=True)
        return
    shared = root/"dsec_async_rgb"
    all_rows,catalog,counts,provenance = [],{}, {},{}
    reprs = {v:RVTHistogram(bins=variants[v][1]) for v in outputs}
    for seq,samples in sorted(groups.items()):
        samples = sorted(samples,key=lambda s:s["target_us"])
        if args.per_sequence:samples = samples[:args.per_sequence]
        rows = time_grid(samples)
        calibration,rectify = ensure_assets(seq,assets,False,base.get("source_split","train"))
        conf = yaml.safe_load(calibration.read_text())
        with h5py.File(rectify) as f:rect = f["rectify_map"][...]
        validate_rectify(conf,rect)
        mapping = compute_remapping(conf,{"rectify_map":rect})
        if mapping.shape != (480,640,2) or not np.isfinite(mapping).all():
            raise ValueError(f"{seq}: invalid RGB remapping")
        provenance[seq] = {q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in (calibration,rectify)}
        rgb_ts = np.loadtxt(source/seq/"images/timestamps.txt",dtype=np.int64)
        paths = sorted((source/seq/"images/left/rectified").glob("*.png"))
        if len(paths)!=len(rgb_ts) or (np.diff(rgb_ts)<=0).any():raise ValueError("Invalid RGB timeline")
        catalog[seq] = [dict(time_us=int(t),file=f"../dsec_async_rgb/{seq}/{i:06d}.npy") for i,t in enumerate(rgb_ts)]
        needed = set()
        for row in rows:
            i = int(np.searchsorted(rgb_ts,row["target_us"],side="right"))-1
            row["rgb_index"] = i
            needed.update(j for j in (i,i-1) if j>=0)
            row["file"] = f'{seq}/{row["target_us"]}.npz'
        directory = shared/seq;directory.mkdir(parents=True,exist_ok=True)
        for i in sorted(needed):
            destination = directory/f"{i:06d}.npy"
            complete=False
            if destination.exists():
                try:
                    value=np.load(destination,mmap_mode="r")
                    complete=value.shape==(440,640,3) and value.dtype==np.uint8
                    del value
                except (OSError,ValueError,EOFError):
                    complete=False
            if not complete:
                raw = cv2.imread(str(paths[i]))
                if raw is None:raise FileNotFoundError(paths[i])
                expected=tuple(conf["intrinsics"]["camRect1"]["resolution"][::-1])
                if raw.shape[:2]!=expected:raise ValueError(f"{seq}: RGB/calibration dimensions differ")
                rgb = cv2.remap(raw,mapping,None,cv2.INTER_CUBIC)[:440,:,::-1].copy()
                temporary=destination.with_suffix(".tmp.npy")
                np.save(temporary,rgb);temporary.replace(destination)
        boundary_events = {v:0 for v in outputs}
        bytes_written = {v:0 for v in outputs}
        with h5py.File(source/seq/"events/left/events.h5") as handle:
            offset = int(handle["t_offset"][()])
            for row in rows:
                for v,out in outputs.items():
                    duration,bins = variants[v]
                    events = read_window(handle,row["target_us"],duration)
                    lower = row["target_us"]-offset-duration
                    boundary_events[v] += int((events[:,0]==lower).sum())
                    events = events[(events[:,0]>lower)&(events[:,2]<440)]
                    destination = out/row["file"];destination.parent.mkdir(parents=True,exist_ok=True)
                    complete=False
                    if destination.exists():
                        # Resume this immutable representation after interrupted preparation.
                        try:
                            with np.load(destination) as cached:
                                value=cached["histogram"]
                                complete=value.shape==(2*bins,440,640) and value.dtype==np.uint8
                        except (OSError,ValueError,EOFError,zipfile.BadZipFile):
                            complete=False
                    if not complete:
                        hist,_ = reprs[v](torch.from_numpy(events),dict(height=440,width=640))
                        temporary=destination.with_suffix(".tmp.npz")
                        np.savez_compressed(temporary,histogram=hist.numpy().astype(np.uint8))
                        temporary.replace(destination)
                    bytes_written[v] += destination.stat().st_size
        all_rows.extend(rows)
        counts[seq] = dict(gt=len(samples),steps=len(rows),removed_left_boundary_events=boundary_events,
                           event_bytes=bytes_written,rgb_assets=len(needed))
        print(json.dumps(dict(sequence=seq,**counts[seq])),flush=True)
    grid = [(r["sequence"],r["target_us"],r["has_gt"],r["split"]) for r in all_rows]
    grid_sha = hashlib.sha256(json.dumps(grid,separators=(",",":")).encode()).hexdigest()
    for v,out in outputs.items():
        duration,bins = variants[v]
        manifest = dict(format="dsec_async_v1",variant=v,window_us=duration,bins=bins,count_cutoff=10,
            fastmode=True,boundary="(start,end]",grid="GT-preserving midpoint; reset on abnormal gaps",
            grid_sha256=grid_sha,rgb_latency_assumption="zero_transport_delay",crop=[0,0,440,640],
            base_manifest_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),assets=provenance,
            counts=counts,rgb=catalog,samples=all_rows)
        part=getattr(args,"part_sequence",None)
        target=out/"parts"/(part+".json") if part else out/"manifest.json"
        target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_suffix(".tmp.json");temp.write_text(json.dumps(manifest,indent=2));temp.replace(target)
    print(json.dumps(dict(outputs={v:str(p) for v,p in outputs.items()},grid_sha256=grid_sha)),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    base="/home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic"
    p.add_argument("--source",default=base+"/source")
    p.add_argument("--base-cache",default=base+"/preprocessed/dsec_b1")
    p.add_argument("--assets",default=base+"/preprocessed/dsec_b1_assets")
    p.add_argument("--output-root",default=base+"/preprocessed/async_v1")
    p.add_argument("--variants",nargs="+",choices=("B","C"),default=["B","C"])
    p.add_argument("--sequences",nargs="+")
    p.add_argument("--per-sequence",type=int,default=0)
    p.add_argument("--workers",type=int,default=4,help="independent raw sequences, 1 disables multiprocessing")
    args=p.parse_args()
    if args.workers<1:p.error("workers must be positive")
    if args.per_sequence<0:p.error("per-sequence must be nonnegative")
    prepare(args)
