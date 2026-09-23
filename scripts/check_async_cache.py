#!/usr/bin/env python3
"""Audit B/C metadata and all referenced files, plus representative tensor payloads."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
from hmnet.dataset.dsec_async import DSECAsync
from hmnet.utils.async_checkpoint import file_sha256

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--base-cache',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(args.root)
    a=json.loads((root/'dsec_async_B/manifest.json').read_text())
    b=json.loads((root/'dsec_async_C/manifest.json').read_text())
    assert a['samples']==b['samples'] and a['grid_sha256']==b['grid_sha256']
    base=json.loads((Path(args.base_cache)/'manifest.json').read_text())
    expected={(s['sequence'],s['target_us'],s['label_path'],s['split']) for s in base['samples']}
    actual={(s['sequence'],s['target_us'],s['label_path'],s['split']) for s in a['samples'] if s['has_gt']}
    assert expected==actual,'GT membership changed'
    previous={};gaps=[];checked=set();rgb_paths=set()
    for i,sample in enumerate(a['samples']):
        seq,time=sample['sequence'],sample['target_us']
        if seq in previous:
            delta=time-previous[seq];assert delta>0
            if not sample['reset']:assert 20000<=delta<=30000
            gaps.append(delta)
        previous[seq]=time
        if sample['has_gt']:
            assert i>0 and not a['samples'][i-1]['has_gt']
            assert a['samples'][i-1]['sequence']==seq
            assert Path(sample['label_path']).is_file()
        for variant in ('B','C'):
            path=root/f'dsec_async_{variant}'/sample['file'];assert path.is_file() and path.stat().st_size>0
        catalog=a['rgb'][seq];ri=sample['rgb_index']
        if ri>=0:
            assert catalog[ri]['time_us']<=time
            if ri+1<len(catalog):assert catalog[ri+1]['time_us']>time
        for j in (ri,ri-1):
            if j>=0:rgb_paths.add(root/'dsec_async_B'/catalog[j]['file'])
        # One cold and one labelled payload from every sequence, both representations.
        token=(seq,sample['has_gt'])
        if token not in checked:
            for variant,channels in (('B',20),('C',10)):
                with np.load(root/f'dsec_async_{variant}'/sample['file']) as f:
                    value=f['histogram'];assert value.shape==(channels,440,640)
                    assert value.dtype==np.uint8 and value.max()<=10
            checked.add(token)
    assert all(path.is_file() for path in rgb_paths)
    counts=Counter(s['split'] for s in a['samples']);gt=Counter(s['split'] for s in a['samples'] if s['has_gt'])
    result=dict(passed=True,event_steps=dict(counts),gt_frames=dict(gt),rgb_assets=len(rgb_paths),
        grid_sha256=a['grid_sha256'],b_manifest_sha256=file_sha256(root/'dsec_async_B/manifest.json'),
        c_manifest_sha256=file_sha256(root/'dsec_async_C/manifest.json'),
        representative_payloads_checked=len(checked)*2,all_event_and_rgb_paths_checked=True,
        step_delta_us_percentiles=np.percentile(gaps,[0,50,95,100]).tolist())
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
