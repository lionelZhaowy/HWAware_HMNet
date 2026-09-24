#!/usr/bin/env python3
"""Build immutable raw-source indexes; output must be a NEW directory.

GEN1 reuses the existing DAT reader. Eventscape scans exact packet endpoints,
not nominal packet timestamps. MVSEC indexes timestamp chunks without copying
HDF5. SHA256 covers indexes/labels; large input files are bound by size+mtime,
not falsely described as fully content-hashed event archives.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,importlib.util
import numpy as np
import h5py
from hmnet.dataset.task_frames import SCHEMA,dat_map,bound,sha_file

ROOT=Path(__file__).resolve().parents[1]

def source_row(path,hash_content=False):
    path=Path(path).resolve();s=path.stat()
    d=dict(path=str(path),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    if hash_content:d['sha256']=sha_file(path)
    return d


def save(out,split,kind,samples,sources,**extras):
    sources=sorted({r['path']:r for r in sources}.values(),key=lambda r:r['path'])
    signature=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    data=dict(schema=SCHEMA,kind=kind,split=split,samples=samples,sources=sources,source_signature=signature,**extras)
    (out/f'{split}.json').write_text(json.dumps(data,separators=(',',':'))+'\n')
    gaps=[];last={}
    for s in samples:
        if s['sequence'] in last:gaps.append(s['target_us']-last[s['sequence']])
        last[s['sequence']]=s['target_us']
    print(json.dumps(dict(split=split,frames=len(samples),sequences=len(last),gap_quantiles_us=np.quantile(gaps,[0,.5,.95,.99,1]).tolist() if gaps else [])),flush=True)


def gen1_sequence(label):
    path=label.with_name(label.name.replace('_bbox.npy','_td.dat'))
    if not path.is_file():raise FileNotFoundError(path)
    a=dat_map(path);ts=a['t'];labels=np.load(label);field='ts' if 'ts' in labels.dtype.names else 't'
    sources=[source_row(path),source_row(label,True)];samples=[]
    for end in np.unique(labels[field]).astype(np.int64):
        samples.append(dict(file=str(path),label_file=str(label),sequence=path.stem,target_us=int(end),
            event_start=bound(ts,int(end)-50000),event_end=bound(ts,int(end),True)))
    return samples,sources


def gen1(args):
    for split in ('train','val','test'):
        files=sorted((args.source/'detection_dataset_duration_60s_ratio_1.0'/split).glob('*_bbox.npy'))
        if not files:raise FileNotFoundError(split)
        if args.sequences:files=files[:args.sequences]
        samples=[];sources=[]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i,(rows,inputs) in enumerate(pool.map(gen1_sequence,files)):
                samples.extend(rows);sources.extend(inputs)
                if i%25==0:print('GEN1',split,i+1,len(files),flush=True)
        save(args.output,split,'gen1',samples,sources)


def get_times(path):
    # Same truncation convention as the existing Eventscape index converter.
    return np.array([int(float(line.split()[-1])*1e6) for line in path.read_text().splitlines() if line.strip()],np.int64)


def eventscape_sequence(seq):
    samples=[];sources=[]
    df=sorted((seq/'depth/data').glob('*.npy'));rf=sorted((seq/'rgb/data').glob('*.png'))
    dt=get_times(seq/'depth/data/timestamps.txt');rt=get_times(seq/'rgb/data/timestamps.txt')
    if len(df)!=len(dt) or len(rf)!=len(rt):raise ValueError(f'Incomplete image/depth sequence {seq}')
    if np.any(np.diff(dt)<=0) or np.any(np.diff(rt)<=0):raise ValueError('Nonincreasing timestamps')
    sources.extend(source_row(seq/f'{mode}/data/timestamps.txt',True) for mode in ('depth','rgb','events'))
    packets=[]
    for path in sorted((seq/'events/data').glob('*.npz')):
        with np.load(path) as data:
            t=data['t'].astype(np.int64)
            if np.any(np.diff(t)<0):raise ValueError(f'Unsorted event packet {path}')
            if len(t):packets.append((int(t[0]),int(t[-1]),str(path)))
        sources.append(source_row(path))
    if any(a[1]>b[0] for a,b in zip(packets,packets[1:])):raise ValueError('Overlapping/out-of-order packets')
    ends=[p[1] for p in packets];starts=[p[0] for p in packets]
    for j,end in enumerate(dt):
        r=int(np.searchsorted(rt,end,side='right'))-1
        if r<0 or end-rt[r]>50000:raise ValueError(f'No causal RGB within 50ms: {seq}/{j}')
        lo=int(np.searchsorted(ends,end-50000));hi=int(np.searchsorted(starts,end,side='right'))
        samples.append(dict(file=str(seq),sequence=f'{seq.parent.name}/{seq.name}',target_us=int(end),
            label_file=str(df[j]),rgb_file=str(rf[r]),rgb_us=int(rt[r]),event_files=[p[2] for p in packets[lo:hi]]))
    sources.extend(source_row(p) for p in df+rf)
    return samples,sources


def eventscape(args):
    spec=importlib.util.spec_from_file_location('split_reference',ROOT/'experiments/depth/data/eventscape/scripts/make_depth_info.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    groups={k:[] for k in ('train','val','test')}
    for seq in sorted(args.source.glob('Town*/sequence_*')):
        split='train' if seq.parent.name in ('Town01','Town02','Town03') else ('val' if int(seq.name.split('_')[1]) in mod.VAL_IDS else 'test')
        groups[split].append(seq)
    for split,seqs in groups.items():
        if args.sequences:seqs=seqs[:args.sequences]
        samples=[];sources=[]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for seq,(rows,inputs) in zip(seqs,pool.map(eventscape_sequence,seqs)):
                samples.extend(rows);sources.extend(inputs)
                print('Eventscape',split,seq.parent.name,seq.name,len(rows),flush=True)
        save(args.output,split,'eventscape',samples,sources)


def mvsec(args):
    for split in ('day2','day1','night1'):
        path=args.source/f'outdoor_{split}_data.hdf5';gt=args.source/f'outdoor_{split}_gt.hdf5'
        with h5py.File(path,'r') as f,h5py.File(gt,'r') as g:
            events=f['davis/left/events'];start=events[0,2]*1e6
            labels=np.rint(g['davis/left/depth_image_raw_ts'][...]*1e6-start).astype(np.int64)
            # Full timestamp-only scan in bounded chunks; vectorized exact bounds.
            left=np.full(len(labels),len(events),np.int64);right=left.copy();previous=None
            for offset in range(0,len(events),2**20):
                ts=np.rint(events[offset:offset+2**20,2]*1e6-start).astype(np.int64)
                if np.any(np.diff(ts)<0) or (previous is not None and ts[0]<previous):raise ValueError('MVSEC timestamps unsorted')
                previous=ts[-1]
                for target,result,side in ((labels-50000,left,'left'),(labels,right,'right')):
                    indices=np.searchsorted(ts,target,side=side);valid=indices<len(ts)
                    result[valid]=np.minimum(result[valid],offset+indices[valid])
            samples=[dict(file=str(path),label_file=str(gt),sequence=f'outdoor_{split}',target_us=int(t),
                         label_index=i,event_start=int(left[i]),event_end=int(right[i])) for i,t in enumerate(labels)]
        save(args.output,split,'mvsec',samples,[source_row(path),source_row(gt)],start_us=float(start))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('kind',choices=['gen1','eventscape','mvsec'])
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sequences',type=int,default=0,help='Diagnostic only: first N sequences per split')
    p.add_argument('--workers',type=int,default=4,help='Bounded independent sequence I/O workers')
    args=p.parse_args();args.source=args.source.resolve();args.output=args.output.resolve()
    if not 1 <= args.workers <= 8:raise ValueError('Use 1–8 index workers')
    args.output.mkdir(parents=True,exist_ok=False)
    globals()[args.kind](args)
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(kind=args.kind,diagnostic_subset=bool(args.sequences),schema=SCHEMA)))

if __name__=='__main__':main()
