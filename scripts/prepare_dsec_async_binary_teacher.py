#!/usr/bin/env python3
"""Build exact 50ms binary inputs on the existing asynchronous time grid.

Reuses read_window and the polarity experiment's unclipped count operator.
The teacher's CLOSED interval is retained; student RVT caches are untouched.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import multiprocessing
from pathlib import Path
import h5py
import numpy as np
from scripts.prepare_dsec_b1 import read_window
from hmnet.models.base.event_repr.polarity import polarity_counts


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sequence_job(job):
    source, output, sequence, rows, reference = job
    output = Path(output)
    files = {}
    checked = 0
    with h5py.File(Path(source)/sequence/'events/left/events.h5', 'r') as handle:
        for row in rows:
            events = read_window(handle, row['target_us'], 50000)
            binary = (polarity_counts(events[events[:, 2] < 440]) > 0).astype(np.uint8)
            key = f"{sequence}/{row['target_us']}"
            # All GT endpoints must match the already audited synchronous input.
            if row['has_gt']:
                with np.load(reference[key]) as values:
                    if not np.array_equal(binary, values['counts'] > 0):
                        raise ValueError(f'Binary teacher differs from trained input: {key}')
                checked += 1
            relative = key + '.npz'
            path = output/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, binary=binary)
            temporary.replace(path)
            files[key] = dict(file=relative, sha256=sha(path))
    print(json.dumps(dict(sequence=sequence, steps=len(rows), gt_verified=checked)), flush=True)
    return files, checked


def main(args):
    root = Path(args.output).resolve()
    grid_path = Path(args.grid)/'manifest.json'
    reference_path = Path(args.polarity_cache)/'manifest.json'
    grid = json.loads(grid_path.read_text())
    reference = json.loads(reference_path.read_text())
    if grid.get('format') != 'dsec_async_v1':
        raise ValueError('Expected existing full asynchronous time grid')
    if (reference.get('schema') != 'dsec_polarity_counts_v1'
            or not reference.get('complete') or reference.get('diagnostic_subset')
            or reference.get('interval') != 'closed' or reference.get('window_us') != 50000):
        raise ValueError('Expected complete synchronous 50ms count cache')
    reference_files = {f"{s['sequence']}/{s['target_us']}":
                       str(Path(args.polarity_cache)/s['count_file']) for s in reference['samples']}
    groups = {}
    for row in grid['samples']:
        groups.setdefault(row['sequence'], []).append(row)
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'manifest.json').exists():
            raise FileExistsError('Completed binary teacher cache exists; reuse it')
        grid_hash, ref_hash = sha(grid_path), sha(reference_path)
        jobs = [(args.source, str(root), seq, rows,
                 {f"{r['sequence']}/{r['target_us']}": reference_files[f"{r['sequence']}/{r['target_us']}"]
                  for r in rows if r['has_gt']}) for seq, rows in sorted(groups.items())]
        with multiprocessing.get_context('spawn').Pool(args.workers) as pool:
            results = pool.map(sequence_job, jobs)
        if (grid_hash, ref_hash) != (sha(grid_path), sha(reference_path)):
            raise ValueError('Source manifests changed')
        files = {key: value for part, _ in results for key, value in part.items()}
        manifest = dict(format='dsec_async_binary_teacher_v1', complete=True,
                        grid_sha256=grid['grid_sha256'], grid_manifest_sha256=grid_hash,
                        polarity_manifest_sha256=ref_hash, window_us=50000,
                        boundary='[start,end]', channels=2, dtype='uint8',
                        representation='polarity_binary', polarity_order=[0, 1],
                        temporal_policy='two_interleaved_50ms_streams_v1',
                        counts=dict(Counter(r['split'] for r in grid['samples'])),
                        gt_verified=sum(count for _, count in results), files=files)
        temporary = root/'manifest.tmp.json'
        temporary.write_text(json.dumps(manifest, indent=2)+'\n')
        temporary.replace(root/'manifest.json')
        print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}), flush=True)


if __name__ == '__main__':
    base = '/home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=base+'/source')
    parser.add_argument('--grid', default=base+'/preprocessed/async_v1/dsec_async_B')
    parser.add_argument('--polarity-cache', default=base+'/preprocessed/dsec_b1_polarity50ms')
    parser.add_argument('--output', default=base+'/preprocessed/async_v1/binary_teacher50ms')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    main(args)
