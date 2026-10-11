#!/usr/bin/env python3
"""从原 train 的整条序列固定 70/18 开发划分；只读注释，不使用 test 类别统计。"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True);p.add_argument('--output',required=True);p.add_argument('--report',required=True)
    a=p.parse_args();root=Path(a.data_root);out=Path(a.output)
    if out.exists():raise FileExistsError('Refuse to replace a locked selection')
    raw={s:(root/f'{s}.json').read_bytes() for s in ('train','val','test')}
    train=json.loads(raw['train']);labels={};profiles={};source_groups={}
    for s in train['samples']:
        source_groups.setdefault(s['file'],set()).add(s['sequence'])
        v=profiles.setdefault(s['sequence'],dict(condition=s['condition'],frames=0,gt=np.zeros(6,dtype=int),area=np.zeros((6,3),dtype=int)))
        assert v['condition']==s['condition'];v['frames']+=1
        if s['label_file'] not in labels:
            by=collections.defaultdict(list)
            for g in json.loads(Path(s['label_file']).read_text())['annotations']:by[g['image_id']].append(g)
            labels[s['label_file']]=by
        for g in labels[s['label_file']][s['image_id']]:
            if g.get('ignore') or g.get('iscrowd'):continue
            k=g['category_id'];v['gt'][k]+=1;v['area'][k,0 if g['area']<1024 else 1 if g['area']<9216 else 2]+=1
    assert all(len(v)==1 for v in source_groups.values()),'Group related source files before selection'
    seqs=sorted(profiles);assert len(seqs)==88
    gt=np.array([profiles[s]['gt'] for s in seqs]);area=np.array([profiles[s]['area'] for s in seqs]);frames=np.array([profiles[s]['frames'] for s in seqs]);support=gt>0
    pools={c:np.array([i for i,s in enumerate(seqs) if profiles[s]['condition']==c]) for c in ('normal','challenge','motion_blur')}
    quotas=dict(normal=8,challenge=6,motion_blur=4);minimum=np.minimum(4,support.sum(0)//3)
    rng=np.random.default_rng(42);best=None;attempts=10000
    for _ in range(attempts):
        ids=np.sort(np.concatenate([rng.choice(pools[c],q,replace=False) for c,q in quotas.items()]))
        if np.any(support[ids].sum(0)<minimum) or np.any(support.sum(0)-support[ids].sum(0)<minimum):continue
        score=float(np.mean((gt[ids].sum(0)/gt.sum(0)-.2)**2)+np.mean((support[ids].sum(0)/support.sum(0)-.2)**2)
            +.25*np.mean((area[ids].sum(0)[area.sum(0)>0]/area.sum(0)[area.sum(0)>0]-.2)**2)
            +(frames[ids].sum()/frames.sum()-.2)**2)
        candidate=(score,tuple(ids.tolist()))
        if best is None or candidate<best:best=candidate
    if best is None:raise RuntimeError('No split meets support constraints')
    dev=[seqs[i] for i in best[1]];kept=[s for s in seqs if s not in dev]
    lines=['schema = "peod_sequence_development_v1"','selection_seed = 42','candidate_count = 10000',
        'train_sequences = '+json.dumps(kept),'dev_sequences = '+json.dumps(dev),'','[parent_sha256]']
    lines += [f'{s} = "{hashlib.sha256(raw[s]).hexdigest()}"' for s in raw]
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text('\n'.join(lines)+'\n')
    result={}
    for name,names in [('train',kept),('dev',dev)]:
        ids=[seqs.index(s) for s in names]
        result[name]=dict(frames=int(frames[ids].sum()),sequences=names,conditions=dict(collections.Counter(profiles[s]['condition'] for s in names)),
            classes={c:dict(gt=int(gt[ids,k].sum()),sequences=int(support[ids,k].sum()),area=area[ids,k].sum(0).tolist()) for k,c in enumerate(train['classes'])})
    report=dict(selection_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),score=best[0],splits=result,
        method='10000 seed42 candidates; fixed 8/6/4 condition quotas; closest train-only GT/sequence/area/frame proportions to 20%; no model outputs',
        limitations='Published DAT sequence grouping only; separate recording/session IDs unavailable. Shared physical scenes across different sequences cannot be ruled out.',
        test_statistics_used=False,cache_modified=False)
    dest=Path(a.report);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
