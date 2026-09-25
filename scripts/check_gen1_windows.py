#!/usr/bin/env python3
"""Independent full-scan oracle for unordered DAT, closed endpoints and stable ties."""
import importlib.util
import json
from pathlib import Path
import tempfile
import numpy as np
from hmnet.dataset.gen1_windows import exact_window_ranges, GEN1_WINDOW_INDEX
from hmnet.dataset.task_frames import TaskFrames, represent

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_task_frames', ROOT/'scripts/prepare_task_frames.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def main():
    rng = np.random.default_rng(42)
    cases = 0
    for n in (0, 1, 100, 1000):
        a = np.zeros(n, dtype=[('t','<u4'),('_','<i4')])
        a['t'] = rng.integers(0,200000,n)
        targets = np.array([0,50000,50001,100000,150000,250001])
        for chunk in (1,7,1<<20):
            lo,hi,counts,audit = exact_window_ranges(a,targets,chunk)
            for i,t in enumerate(targets):
                selected = np.flatnonzero((a['t'].astype('int64')>=t-50000)&(a['t']<=t))
                assert counts[i] == len(selected)
                assert (lo[i],hi[i]) == ((selected[0],selected[-1]+1) if len(selected) else (0,0))
                cases += 1
            assert audit['timestamp_regressions'] == int((np.diff(a['t'].astype('int64'))<0).sum())
    invalid = np.array([(0,304)], dtype=[('t','<u4'),('_','<i4')])
    try:
        exact_window_ranges(invalid,[0])
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid coordinate accepted')
    with tempfile.TemporaryDirectory(prefix='gen1_windows_') as tmp:
        root = Path(tmp)
        # Nonlocal regression, exact lower/upper endpoints, outside events and stable ties.
        events = np.array([(51001,1,2,0),(1000,2,3,1),(51000,3,4,0),(999,4,5,1),
                           (51000,5,6,1),(2000,6,7,0),(1000,7,8,0),(200000,8,9,1)],np.int64)
        a = np.zeros(len(events),dtype=[('t','<u4'),('_','<i4')])
        a['t'] = events[:,0]; a['_'] = events[:,1]+(events[:,2]<<14)+(events[:,3]<<28)
        with (root/'fixture_td.dat').open('wb') as f:
            f.write(b'% Height 240\n% Width 304\n'+bytes([0,8]));a.tofile(f)
        labels = np.zeros(3,dtype=[('ts','<u8'),('x','<f4'),('y','<f4'),('w','<f4'),('h','<f4'),('class_id','u1')])
        labels['ts'] = [51000,150000,250001];labels['x']=10;labels['y']=10;labels['w']=40;labels['h']=40
        np.save(root/'fixture_bbox.npy',labels)
        rows,sources,audit = prepare.gen1_sequence(root/'fixture_bbox.npy')
        (root/'COMPLETE.json').write_text(json.dumps(dict(diagnostic_subset=True)))
        for split in ('train','val','test'):
            prepare.save(root,split,'gen1',rows,sources,gen1_window_index=GEN1_WINDOW_INDEX,gen1_audit=[audit])
            for representation in ('rvt_histogram','polarity_binary'):
                d = TaskFrames(root,split,representation)
                for i,t in enumerate(labels['ts'].astype(np.int64)):
                    oracle=events[(events[:,0]>=t-50000)&(events[:,0]<=t)]
                    oracle=oracle[np.argsort(oracle[:,0],kind='stable')]
                    assert np.array_equal(d.raw(i),oracle)
                    assert np.array_equal(d[i][0].numpy(),represent(oracle,representation,240,304).numpy())
                d.samples[0]['event_count'] += 1
                try:d.raw(0)
                except ValueError:pass
                else:raise AssertionError('Incorrect count accepted')
        p=root/'train.json';manifest=json.loads(p.read_text());manifest.pop('gen1_window_index');p.write_text(json.dumps(manifest))
        try:TaskFrames(root,'train')
        except ValueError as e:assert 'Legacy GEN1' in str(e)
        else:raise AssertionError('Legacy unsafe index accepted')
    print(json.dumps(dict(passed=True,oracle_cases=cases,split_representation_pairs=6,
                         stable_ties=True,empty_windows=True,legacy_rejected=True,count_mismatch_rejected=True)))

if __name__=='__main__':main()
