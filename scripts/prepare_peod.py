#!/usr/bin/env python3
"""Audit published PEOD and build shared, resumable exact 50 ms DAT indices.

No original data is changed. Each completed sequence is reusable after restart.
Full physical scans, not binary search on unverified timestamps, define windows.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
import os
import numpy as np
from PIL import Image
from hmnet.dataset.vendor.dat_events_tools import parse_header
from hmnet.dataset.task_frames import sha_file

SCHEMA = 'peod_closed_50ms_letterbox_v1'
CLASSES = ['car', 'person', 'bus', 'truck', '2-wheeler', '3-wheeler']
GEOMETRY = dict(source_hw=[720,1280], resized_hw=[171,304], input_hw=[240,304],
                scale=0.2375, pad_ltrb=[0,34,0,35], event_rounding='floor_after_scale',
                rgb_interpolation='PIL_bilinear', bbox='float_xyxy_clip_source_then_affine')

def stat_record(path):
    s=path.stat()
    return dict(path=str(path.resolve()),bytes=s.st_size,mtime_ns=s.st_mtime_ns)

def signature(rows):
    return hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()

def discover(root):
    rows=[]
    for event in sorted(root.glob('train/*.dat'))+sorted(root.glob('test/*/*.dat')):
        rel=event.relative_to(root); seq=event.stem
        labels=list((root/'annotations'/rel.parent).glob(f'*/{seq}.json')) if rel.parts[0]=='train' else [root/'annotations'/rel.with_suffix('.json')]
        if len(labels)!=1:raise ValueError(f'Ambiguous annotation: {event}')
        rows.append(dict(sequence=seq,file=event,rgb=root/'rgb'/rel.parent/seq,
            label=labels[0],timestamps=root/'timestamp'/rel.with_suffix('.csv'),
            official_split=rel.parts[0],condition=labels[0].parent.name))
    if not rows:raise ValueError('No PEOD DAT files')
    return rows

def metadata(row):
    labels=json.loads(row['label'].read_text())
    if [(c['id'],c['name']) for c in labels['categories']]!=list(enumerate(CLASSES)):
        raise ValueError('Unexpected PEOD class mapping')
    times=np.loadtxt(row['timestamps'],delimiter=',',dtype=np.int64,ndmin=2)[:,1]
    images=sorted(labels['images'],key=lambda x:x['file_name'])
    rgb=sorted(row['rgb'].glob('*.png'))
    if len(times)!=len(images) or [p.name for p in rgb]!=[im['file_name'] for im in images]:
        raise ValueError(f'RGB/timestamp/annotation mapping differs: {row["sequence"]}')
    if np.any(np.diff(times)<=0):raise ValueError('Nonmonotonic RGB timestamps')
    by_id=defaultdict(list)
    for a in labels['annotations']:
        if a['category_id'] not in range(6) or not np.isfinite(a['bbox']).all():raise ValueError('Invalid annotation')
        by_id[a['image_id']].append(a)
    for im in images:
        if (im['height'],im['width'])!=(720,1280):raise ValueError('RGB geometry differs')
    sample_rgb=[]
    for j in sorted(set([0,len(rgb)//2,len(rgb)-1])):
        with Image.open(rgb[j]) as im:
            a=np.asarray(im)
            if im.mode!='RGB' or a.shape!=(720,1280,3):raise ValueError('Not published RGB grid')
            sample_rgb.append(dict(file=rgb[j].name,mean=a.mean((0,1)).tolist(),different_rg_fraction=float(np.mean(a[:,:,0]!=a[:,:,1]))))
    sources=[stat_record(p) for p in [row['file'],row['label'],row['timestamps']]+rgb]
    samples=[dict(file=str(row['file']),sequence=row['sequence'],target_us=int(t),rgb_us=int(t),
                  rgb_file=str(p),label_file=str(row['label']),image_id=im['id'],condition=row['condition'],
                  bbox_count=len(by_id[im['id']])) for t,p,im in zip(times,rgb,images)]
    return times,sources,samples,sample_rgb,labels

def scan(path,targets,chunk_size=1<<20):
    with path.open('rb') as f:offset,kind,size,shape=parse_header(f)
    if size!=8 or shape!=[720,1280] or kind not in (0,12):raise ValueError('Unexpected DAT encoding')
    if (path.stat().st_size-offset)%8:raise ValueError('Truncated DAT')
    records=np.memmap(path,mode='r',offset=offset,dtype=[('t','<u4'),('_','<u4')])
    starts=np.full(len(targets),len(records),np.int64);ends=np.zeros(len(targets),np.int64);counts=ends.copy()
    # Buffered physical reads coalesce HDD I/O; mmap remains only for five oracle slices.
    stream=path.open('rb');stream.seek(offset)
    if hasattr(os,'posix_fadvise'):os.posix_fadvise(stream.fileno(),offset,0,os.POSIX_FADV_SEQUENTIAL)
    previous=None; inversions=0;regression=0;total=0;first=2**63-1;last=0
    for begin in range(0,len(records),chunk_size):
        a=np.fromfile(stream,dtype=records.dtype,count=min(chunk_size,len(records)-begin));t=a['t'].astype(np.int64);packed=a['_']
        if np.any((packed&16383)>=1280) or np.any(((packed>>14)&16383)>=720):raise ValueError('Out-of-grid event')
        d=np.diff(t,prepend=previous) if previous is not None else np.diff(t)
        neg=d[d<0];inversions+=len(neg)
        if len(neg):regression=max(regression,int(-neg.min()))
        if regression>2**31:raise ValueError('Timestamp wrap requires explicit unwrapping; refusing guessed epoch')
        previous=int(t[-1]);first=min(first,int(t.min()));last=max(last,int(t.max()));total+=len(t)
        lo=np.searchsorted(targets,t.min());hi=np.searchsorted(targets,int(t.max())+50000,side='right')
        for i in range(lo,hi):
            hit=np.flatnonzero((t>=targets[i]-50000)&(t<=targets[i]))
            if len(hit):starts[i]=min(starts[i],begin+int(hit[0]));ends[i]=max(ends[i],begin+int(hit[-1])+1);counts[i]+=len(hit)
    stream.close()
    starts[counts==0]=0
    # Independent slice/filter checks against counts obtained by full physical scan.
    oracle=[]
    for i in sorted(set([0,len(targets)//2,len(targets)-1,int(np.argmax(counts)),int(np.argmin(counts))])):
        a=records[starts[i]:ends[i]]; t=a['t'].astype(np.int64)
        selected=(t>=targets[i]-50000)&(t<=targets[i])
        if int(selected.sum())!=counts[i]:raise AssertionError('Full scan oracle differs')
        oracle.append(dict(index=i,target_us=int(targets[i]),events=int(counts[i])))
    return starts,ends,counts,dict(events=total,first_us=first,last_us=last,dat_type=int(kind),
        timestamp_regressions=inversions,largest_regression_us=regression,wraps=0,oracle=oracle)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='/data/lab_dataset/RGB_DVS_Fusion/PEOD_orig')
    p.add_argument('--output');p.add_argument('--limit-sequences',type=int,default=0);p.add_argument('--continue-errors',action='store_true',help='Audit other sequences, but never publish COMPLETE if any source fails');args=p.parse_args()
    root=Path(args.root).resolve();out=Path(args.output) if args.output else root/'preprocessed/hmnet_v12t_240x304_v1'
    out.mkdir(parents=True,exist_ok=True);(out/'sequences').mkdir(exist_ok=True)
    all_rows=discover(root);rows=all_rows[:args.limit_sequences or None]
    # Stable condition-stratified held-out sequences: each tenth sequence per condition.
    groups=defaultdict(list)
    for r in all_rows:
        if r['official_split']=='train':groups[r['condition']].append(r['sequence'])
    val={s for names in groups.values() for s in sorted(names)[9::10]}
    splits=defaultdict(list);audits=[];all_intervals=[];source_rows={};errors=[]
    for row in rows:
        started=time.monotonic();times,sources,samples,rgb_stats,labels=metadata(row);sig=signature(sources)
        cache=out/'sequences'/f'{row["sequence"]}.json'
        if cache.exists():
            saved=json.loads(cache.read_text())
            if saved['source_signature']!=sig or saved['schema']!=SCHEMA:raise ValueError('Existing sequence cache mismatch')
        else:
            try:
                starts,ends,counts,audit=scan(row['file'],times)
            except (ValueError,OSError) as exc:
                errors.append(dict(sequence=row['sequence'],path=str(row['file']),error=str(exc)))
                (out/'SOURCE_ERRORS.json').write_text(json.dumps(errors,indent=2))
                if not args.continue_errors:raise
                print(json.dumps(dict(source_error=errors[-1])),flush=True)
                continue
            for s,a,b,c in zip(samples,starts,ends,counts):s.update(event_start=int(a),event_end=int(b),event_count=int(c))
            audit.update(sequence=row['sequence'],condition=row['condition'],official_split=row['official_split'],
                samples=len(samples),empty_boxes=sum(s['bbox_count']==0 for s in samples),zero_event_windows=int((counts==0).sum()),
                box_count=len(labels['annotations']),rgb_checks=rgb_stats,interval_quantiles_us=np.quantile(np.diff(times),[0,.5,.95,.99,1]).tolist())
            saved=dict(schema=SCHEMA,source_signature=sig,sources=sources,samples=samples,audit=audit)
            tmp=cache.with_suffix('.tmp');tmp.write_text(json.dumps(saved));tmp.replace(cache)
        audit=saved['audit'];audits.append(audit);all_intervals.extend(np.diff(times).tolist())
        split=('val' if row['sequence'] in val else 'train') if row['official_split']=='train' else 'test'
        splits[split].extend(saved['samples'])
        if split=='test':splits['test_'+row['condition']].extend(saved['samples'])
        for s in sources:source_rows[s['path']]=s
        print(json.dumps(dict(sequence=row['sequence'],split=split,samples=len(samples),seconds=time.monotonic()-started,audit=audit)),flush=True)
    # 2x training p99 label gap accommodates one skipped normal frame, resets larger gaps.
    train_gaps=[]
    for row in rows:
        if row['official_split']=='train':train_gaps.extend(np.diff(np.loadtxt(row['timestamps'],delimiter=',',dtype=np.int64,ndmin=2)[:,1]).tolist())
    reset_gap=int(np.ceil(2*np.quantile(train_gaps,.99)/1000)*1000)
    manifest_shas={}
    for split,samples in splits.items():
        used={s[k] for s in samples for k in ('file','label_file','rgb_file')}
        for row in rows:
            if any(s['sequence']==row['sequence'] for s in samples):used.add(str(row['timestamps']))
        sources=[source_rows[k] for k in sorted(used)]
        m=dict(schema=SCHEMA,kind='peod',geometry=GEOMETRY,classes=CLASSES,reset_gap_us=reset_gap,
            sources=sources,source_signature=signature(sources),samples=samples,rgb_pairing='published_frame_timestamp_exact',
            split_rule='official_test; train_condition_sorted_every_tenth_to_val',val_sequences=sorted(val))
        dest=out/f'{split}.json';dest.write_text(json.dumps(m));manifest_shas[split]=sha_file(dest)
    report=dict(schema=SCHEMA,root=str(root),diagnostic_subset=bool(args.limit_sequences),splits={k:len(v) for k,v in splits.items()},
        sequence_count=len(rows),audits=audits,geometry=GEOMETRY,classes=CLASSES,reset_gap_us=reset_gap,
        interval_quantiles_us=np.quantile(all_intervals,[0,.5,.95,.99,1]).tolist(),
        filters=dict(missing_rgb=0,stale_rgb=0,dropped_empty_boxes=0,dropped_zero_events=0),manifest_sha256=manifest_shas,
        raw_dependencies='Original DAT, PNG and annotation JSON remain required; no deletion authorized')
    report['source_errors']=errors
    report['accepted_sequences']=len(audits)
    marker='AUDIT_FAILED.json' if errors else 'COMPLETE.json'
    (out/marker).write_text(json.dumps(report,indent=2));print(marker,out,report['splits'],flush=True)
    if errors:raise SystemExit(2)

if __name__=='__main__':main()
