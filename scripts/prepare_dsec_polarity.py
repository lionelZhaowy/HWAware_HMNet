#!/usr/bin/env python3
"""Build/audit a shared 50ms count cache using a frozen RVT manifest's members.

RGB/GT stay in the original cache, referenced read-only. Counts are rebuilt
from raw HDF5, never summed from clipped RVT bins. No network/download needed.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from scripts.prepare_dsec_b1 import read_window
from hmnet.models.base.event_repr.polarity import CACHE_SCHEMA, polarity_counts, validate_counts


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def run(args):
    out, parent, source = map(lambda p: Path(p).resolve(),
                              (args.output, args.parent_cache, args.source))
    if out == parent:
        raise ValueError("Output must be independent of the parent cache")
    parent_hash = sha(parent / "manifest.json")
    original = json.loads((parent / "manifest.json").read_text())
    if tuple(original[k] for k in ("window_us", "bins", "count_cutoff", "fastmode")) != (50000, 10, 10, True):
        raise ValueError("Expected the v1.2_T RVT parent manifest")
    selected = original["samples"]
    if args.limit:
        selected = [s for split in sorted({s["split"] for s in selected})
                    for s in [x for x in selected if x["split"] == split][:args.limit]]
    out.mkdir(parents=True, exist_ok=True)
    # Nonblocking single writer; the presence of this file alone is not a lock.
    with (out / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = out / "manifest.json"
        if args.audit:
            manifest = json.loads(manifest_path.read_text())
            expected = dict(schema=CACHE_SCHEMA, complete=True, window_us=50000, interval="closed",
                            channels=2, polarity_order=[0,1], dtype="int32", scaling="none", count_cutoff=None)
            if (any(manifest.get(k) != v for k,v in expected.items()) or
                    manifest["parent_manifest_sha256"] != sha(parent / "manifest.json") or
                    Path(manifest["parent_cache"]).resolve() != parent):
                raise ValueError("Parent/schema mismatch")
            entries = manifest["samples"]
            stripped = [{k:v for k,v in s.items() if k not in
                         ("count_file", "count_sha256", "asset_sha256")} for s in entries]
            if stripped != selected:
                raise ValueError("Sample membership/order differs from parent")
        else:
            if manifest_path.exists():
                raise FileExistsError(f"Completed cache exists: {out}; use --audit")
            entries = []
        histogram = Counter()
        handle, current = None, None
        try:
            for i, sample in enumerate(selected):
                seq = sample["sequence"]
                if seq != current:
                    if handle is not None: handle.close()
                    handle = h5py.File(source / seq / "events/left/events.h5", "r")
                    current = seq
                ev = read_window(handle, sample["target_us"], 50000)
                ev = ev[ev[:, 2] < 440]  # Exact parent crop, then validate remaining coordinates.
                counts = polarity_counts(ev)
                assert int(counts.sum(dtype=np.int64)) == len(ev)
                asset = parent / sample["file"]
                with np.load(asset) as f:
                    if f["rgb"].shape != (440,640,3) or f["label"].shape != (440,640):
                        raise ValueError(f"Invalid RGB/GT asset: {asset}")
                if args.audit:
                    entry = entries[i]; path = out / entry["count_file"]
                    with np.load(path) as f: stored = f["counts"]
                    validate_counts(stored)
                    if (not np.array_equal(counts, stored) or sha(path) != entry["count_sha256"]
                            or sha(asset) != entry["asset_sha256"]):
                        raise ValueError(f"Count/asset audit mismatch: {path}")
                else:
                    relative = "counts/" + sample["file"]
                    path = out / relative; path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(".tmp")
                    with tmp.open("wb") as f: np.savez_compressed(f, counts=counts)
                    tmp.replace(path)
                    entries.append(dict(sample, count_file=relative, count_sha256=sha(path),
                                        asset_sha256=sha(asset)))
                freq = np.bincount(counts.ravel())
                histogram.update({int(k):int(freq[k]) for k in np.flatnonzero(freq)})
                if (i+1) % 100 == 0 or i+1 == len(selected):
                    print(f"{'audit' if args.audit else 'build'} {i+1}/{len(selected)}", flush=True)
        finally:
            if handle is not None: handle.close()
        if sha(parent / "manifest.json") != parent_hash:
            raise ValueError("Parent manifest changed while processing; cache not finalized")
        total = sum(histogram.values()); nonzero = total - histogram[0]
        ordered = sorted((k,v) for k,v in histogram.items() if k)
        quantiles = {}
        for q in (.5,.9,.99,.999):
            cumulative = 0; value = 0
            for k,v in ordered:
                cumulative += v
                if cumulative >= nonzero*q: value=k;break
            quantiles[str(q)] = value
        report = dict(passed=True, samples=len(entries), splits=dict(Counter(s["split"] for s in entries)),
                      maximum=max(histogram, default=0), nonzero_fraction=nonzero/max(1,total),
                      nonzero_quantiles=quantiles, above255=sum(v for k,v in histogram.items() if k>255),
                      parent_manifest_sha256=sha(parent / "manifest.json"),
                      raw_event_reconstruction=True, rgb_gt="unchanged parent NPZ assets with SHA256",
                      diagnostic_subset=bool(args.limit))
        if not args.audit:
            manifest = dict(schema=CACHE_SCHEMA, complete=True, window_us=50000, interval="closed",
                            channels=2, polarity_order=[0,1], dtype="int32", scaling="none", count_cutoff=None,
                            parent_cache=str(parent), parent_manifest_sha256=sha(parent / "manifest.json"),
                            source=str(source), diagnostic_subset=bool(args.limit), samples=entries)
            atomic_json(manifest_path, manifest)
        atomic_json(out / ("audit.json" if args.audit else "build_report.json"), report)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    base = "/home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default=base+"/source")
    p.add_argument("--parent-cache", default=base+"/preprocessed/dsec_b1")
    p.add_argument("--output", default=base+"/preprocessed/dsec_b1_polarity50ms")
    p.add_argument("--audit", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Diagnostic samples per split, 0 means all")
    args = p.parse_args()
    if args.limit < 0: p.error("--limit must be nonnegative")
    run(args)
