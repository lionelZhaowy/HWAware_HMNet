#!/usr/bin/env python3
"""Build a bounded, independent RGB/DVS frame cache from official DSEC sources."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile
import time

import cv2
import h5py
import hdf5plugin  # Registers the compression used in DSEC events.h5.
import numpy as np
import torch
import yaml
from hmnet.dataset.vendor.dsec_remapping import compute_remapping, conf_to_K
from hmnet.models.base.event_repr.rvt_histogram import RVTHistogram


class HTTPRangeFile(io.RawIOBase):
    """Read small ZIP members without downloading multi-GB event archives."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos if whence == 1 else self.size) + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):
        size = min(size if size >= 0 else self.size, self.size - self.pos)
        if size <= 0:
            return b""
        # Fail closed if the server ignores Range: never fetch an entire archive.
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{self.pos+size-1}"}
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=40) as r:
                    if r.status != 206:
                        raise RuntimeError(f"Server did not honor HTTP Range: {self.url}")
                    data = r.read(size)
                if len(data) != size:
                    raise IOError("Incomplete HTTP range response")
                break
            except (OSError, IOError):
                if attempt == 2:
                    raise
                time.sleep(1)
        self.pos += len(data)
        return data


def ensure_assets(sequence, root, download=False):
    directory = root / sequence
    calibration = directory / "calibration/cam_to_cam.yaml"
    rectify = directory / "events/left/rectify_map.h5"
    base = f"https://download.ifi.uzh.ch/rpg/DSEC/train/{sequence}/{sequence}"
    if not calibration.exists() and download:
        with urllib.request.urlopen(base + "_calibration.zip", timeout=60) as r:
            archive = zipfile.ZipFile(io.BytesIO(r.read()))
        member = next(n for n in archive.namelist() if n.endswith("cam_to_cam.yaml"))
        calibration.parent.mkdir(parents=True, exist_ok=True)
        calibration.write_bytes(archive.read(member))
    if not rectify.exists() and download:
        with zipfile.ZipFile(HTTPRangeFile(base + "_events_left.zip")) as archive:
            member = next(n for n in archive.namelist() if n.endswith("rectify_map.h5"))
            rectify.parent.mkdir(parents=True, exist_ok=True)
            rectify.write_bytes(archive.read(member))
    if not calibration.exists() or not rectify.exists():
        raise FileNotFoundError(
            f"{sequence}: official calibration/rectify_map missing in {directory}; use --download-assets"
        )
    return calibration, rectify


def latest_past_image(timestamps, target, max_age=50000):
    i = int(np.searchsorted(timestamps, target, side="right")) - 1
    return i if i >= 0 and 0 <= target - int(timestamps[i]) <= max_age else None


def read_window(handle, target, duration=50000):
    # Raw event times are local; RGB/semantic times include events.h5:t_offset.
    end = int(target) - int(handle["t_offset"][()])
    start = end - duration
    if end < 0:
        return np.empty((0, 4), dtype=np.int64)
    index = handle["ms_to_idx"]
    lo = int(index[max(0, min(len(index) - 1, start // 1000))])
    hi_ms = end // 1000 + 1
    hi = int(index[hi_ms]) if 0 <= hi_ms < len(index) else len(handle["events/t"])
    ts = handle["events/t"][lo:hi].astype(np.int64)
    a, b = np.searchsorted(ts, start, side="left"), np.searchsorted(ts, end, side="right")
    lo, hi = lo + int(a), lo + int(b)
    return np.stack(
        [handle["events/" + key][lo:hi] for key in ("t", "x", "y", "p")], axis=1
    ).astype(np.int64)


def validate_rectify(calibration, mapping):
    """Reject a map copied from an incompatible camera/sequence calibration."""
    intr = calibration["intrinsics"]
    w, h = intr["cam0"]["resolution"]
    if mapping.shape != (h, w, 2):
        raise ValueError("rectify_map resolution does not match event calibration")
    y, x = np.mgrid[:h, :w]
    points = np.stack([x, y], axis=-1).astype(np.float32).reshape(-1, 1, 2)
    expected = cv2.undistortPointsIter(
        points,
        conf_to_K(intr["cam0"]["camera_matrix"]),
        np.array(intr["cam0"]["distortion_coeffs"]),
        np.array(calibration["extrinsics"]["R_rect0"]),
        conf_to_K(intr["camRect0"]["camera_matrix"]),
        (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 0.001),
    ).reshape(h, w, 2)
    if not np.isfinite(mapping).all() or not np.allclose(expected, mapping, rtol=0, atol=0.01):
        raise ValueError("rectify_map is incompatible with the supplied calibration")


def prepare(args):
    torch.set_num_threads(2)
    source = Path(args.source).resolve()
    out = Path(args.output) if args.output else source.parent / "preprocessed/dsec_b1"
    assets = Path(args.assets) if args.assets else out / "assets"
    if (out / "manifest.json").exists():
        raise FileExistsError(
            f"Completed cache already exists: {out}; reuse it or select a new --output"
        )
    if args.per_sequence < 0:
        raise ValueError("--per-sequence must be >= 0 (0 means all)")
    out.mkdir(parents=True, exist_ok=True)
    representation = RVTHistogram()
    entries, counts, provenance = [], {}, {}
    for labels in sorted((source / "train").glob("*/11classes")):
        seq = labels.parent.name
        if args.sequences and seq not in args.sequences:
            continue
        calibration, rectify = ensure_assets(seq, assets, args.download_assets)
        conf = yaml.safe_load(calibration.read_text())
        with h5py.File(rectify) as f:
            rectify_values = f["rectify_map"][...]
        validate_rectify(conf, rectify_values)
        mapping = compute_remapping(conf, {"rectify_map": rectify_values})
        if mapping.shape != (480, 640, 2) or not np.isfinite(mapping).all():
            raise ValueError(f"{seq}: incompatible event remapping shape/values")
        provenance[seq] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (calibration, rectify)
        }
        timestamps = np.loadtxt(labels.parent / f"{seq}_semantic_timestamps.txt", dtype=np.int64)
        paths = sorted(labels.glob("*.png"))
        if len(paths) != len(timestamps):
            raise ValueError(f"{seq}: label timestamp count mismatch")
        rgb_ts = np.loadtxt(source / seq / "images/timestamps.txt", dtype=np.int64)
        rgb_paths = sorted((source / seq / "images/left/rectified").glob("*.png"))
        if len(rgb_paths) != len(rgb_ts) or np.any(np.diff(rgb_ts) < 0):
            raise ValueError(f"{seq}: RGB timestamp count/order mismatch")
        selected = np.linspace(
            0,
            len(paths) - 1,
            min(args.per_sequence, len(paths)) if args.per_sequence else len(paths),
            dtype=int,
        )
        counts[seq] = {"selected": len(selected), "no_past_rgb": 0, "all_ignore": 0, "saved": 0}
        with h5py.File(source / seq / "events/left/events.h5") as handle:
            for idx in selected:
                target = int(timestamps[idx])
                ri = latest_past_image(rgb_ts, target, args.max_rgb_age)
                if ri is None:
                    counts[seq]["no_past_rgb"] += 1
                    continue
                label = cv2.imread(str(paths[idx]), cv2.IMREAD_UNCHANGED)
                if label.shape != (440, 640) or not np.isin(label, list(range(11)) + [255]).all():
                    raise ValueError(f"Unexpected label shape/classes: {paths[idx]}")
                if np.all(label == 255):
                    counts[seq]["all_ignore"] += 1
                    continue
                rgb = cv2.imread(str(rgb_paths[ri]))
                expected = tuple(conf["intrinsics"]["camRect1"]["resolution"][::-1])
                if rgb.shape[:2] != expected:
                    raise ValueError(
                        f"{seq}: RGB resolution {rgb.shape[:2]} != calibration {expected}"
                    )
                # Output grid is RAW event-left coordinates; sampling grid points
                # into the RECTIFIED RGB-left input. Crop only AFTER remapping.
                rgb = cv2.remap(rgb, mapping, None, cv2.INTER_CUBIC)[:440, :, ::-1].copy()
                events = read_window(handle, target)
                events = events[events[:, 2] < 440]
                hist, _ = representation(torch.from_numpy(events), dict(height=440, width=640))
                relative = f"{seq}/{idx:06d}.npz"
                destination = out / relative
                destination.parent.mkdir(exist_ok=True)
                np.savez_compressed(
                    destination, histogram=hist.numpy().astype(np.uint8), rgb=rgb, label=label
                )
                entry = dict(
                    file=relative,
                    sequence=seq,
                    target_us=target,
                    rgb_us=int(rgb_ts[ri]),
                    rgb_age_us=target - int(rgb_ts[ri]),
                    label_path=str(paths[idx]),
                    rgb_path=str(rgb_paths[ri]),
                    split="dev" if seq == "zurich_city_08_a" else "train",
                )
                entries.append(entry)
                counts[seq]["saved"] += 1
                if counts[seq]["saved"] == 1:
                    active = hist.sum(0).numpy() > 0
                    overlay = rgb.copy()
                    overlay[active] = (0.5 * overlay[active] + np.array([127, 0, 127])).astype(
                        np.uint8
                    )
                    palette = np.random.default_rng(42).integers(0, 255, (256, 3), dtype=np.uint8)
                    palette[255] = 0
                    panel = np.concatenate(
                        [rgb, overlay, ((rgb.astype(float) + palette[label]) / 2).astype(np.uint8)],
                        axis=1,
                    )
                    cv2.imwrite(str(out / f"{seq}_overlay.png"), panel[:, :, ::-1])
        print(seq, counts[seq], flush=True)
    manifest = dict(
        version=1,
        window_us=50000,
        bins=10,
        count_cutoff=10,
        fastmode=True,
        crop=[0, 0, 440, 640],
        holdout="zurich_city_08_a",
        max_rgb_age_us=args.max_rgb_age,
        assets=provenance,
        counts=counts,
        samples=entries,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        required=True,
        help="DSEC source directory containing train/ and per-sequence raw data",
    )
    p.add_argument("--output", help="Shared cache; default: source/../preprocessed/dsec_b1")
    p.add_argument("--assets")
    p.add_argument(
        "--per-sequence", type=int, default=0, help="0: all labels; N: at most N per sequence"
    )
    p.add_argument("--max-rgb-age", type=int, default=50000)
    p.add_argument("--sequences", nargs="+")
    p.add_argument("--download-assets", action="store_true")
    prepare(p.parse_args())
