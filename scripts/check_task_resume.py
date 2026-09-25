#!/usr/bin/env python3
"""Real batch8: continuous updates versus process exit + resume at both cursors.

Mid-epoch (--stop-after 2 of 4 batches) fast-forwards inside the sampler; an epoch
boundary (--stop-after 4) advances the epoch without replaying it at all.
"""
import argparse,json,subprocess,os,gc
from pathlib import Path
import numpy as np
import torch
from hmnet.models.efficientvit_tasks import ROOT
from hmnet.dataset.task_frames import sha_file


def equal(a,b,path='root'):
    if isinstance(a,torch.Tensor):
        if not torch.equal(a,b):raise AssertionError(path+f' maxdiff={float((a-b).abs().max())}')
    elif isinstance(a,np.ndarray):
        if not np.array_equal(a,b):raise AssertionError(path)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError(path+' keys')
        for k in a:equal(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise AssertionError(path+' len')
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'/'+str(i))
    elif a!=b:raise AssertionError(path+f': {a} != {b}')


def command(cmd,log):
    with open(log,'w') as f:
        result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'Failed {log}: '+Path(log).read_text()[-3000:])


def scenario(name,out,base,stop_first,cursor,stop_total):
    """Stop at `stop_first`, resume to `stop_total`, compare with an uninterrupted run."""
    continuous=out/(name+'_continuous');split=out/(name+'_split')
    if not (continuous/'checkpoint.pth').exists():
        command(base+['--output',str(continuous),'--stop-after',str(stop_total)],out/(name+'_continuous.log'))
    command(base+['--output',str(split),'--stop-after',str(stop_first)],out/(name+'_split.log'))
    before=torch.load(split/'checkpoint.pth',map_location='cpu',weights_only=False)
    assert before['step']==stop_first,before['step']
    assert (before['data_epoch'],before['data_cursor'])==(0,cursor),(before['data_epoch'],before['data_cursor'])
    assert any(torch.count_nonzero(x) for x in before['temporal_by_rank'][0]['memory'])
    del before;gc.collect()
    command(base+['--output',str(split),'--resume',str(split/'checkpoint.pth'),'--stop-after',str(stop_total)],out/(name+'_resume.log'))
    one=torch.load(continuous/'checkpoint.pth',map_location='cpu',weights_only=False)
    two=torch.load(split/'checkpoint.pth',map_location='cpu',weights_only=False)
    equal(one,two)
    rows=[json.loads(x) for x in (split/'metrics.jsonl').read_text().splitlines()]
    assert all(r['samples']==8 and np.isfinite(r['loss']) and r['amp_skipped_updates']==0 for r in rows)
    return dict(cursor=cursor,steps=[r['step'] for r in rows],losses=[r['loss'] for r in rows],
        continuous_sha256=sha_file(continuous/'checkpoint.pth'),resumed_sha256=sha_file(split/'checkpoint.pth'),
        peak_memory_mib=max(r['peak_memory_mib'] for r in rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--data-root',required=True)
    p.add_argument('--representation',required=True);p.add_argument('--workers',type=int,default=2);p.add_argument('--limit',type=int,default=32)
    a=p.parse_args();label=a.dataset+('_rvt' if a.representation=='rvt_histogram' else '_binary')
    out=ROOT/'artifacts/validation';out.mkdir(parents=True,exist_ok=True)
    base=[str(ROOT/'scripts/hmnet-python'),'scripts/task_temporal.py','train','--dataset',a.dataset,'--representation',a.representation,
        '--data-root',a.data_root,'--limit',str(a.limit),'--batch','8','--workers',str(a.workers)]
    # At limit 32 / batch 8 an epoch is 4 batches: cursor 2 is mid-epoch, 4 completes it.
    scenarios={name:scenario(name,out,base,cursor,cursor,cursor+1)
               for name,cursor in ((label+'_mid',2),(label+'_epoch',4))}
    result=dict(passed=True,dataset=a.dataset,representation=a.representation,batch=8,workers=a.workers,precision='bf16',
        compared='entire checkpoint: parameters/buffers/optimizer/scaler/RNG/cursor/state/contract bitwise equal',
        mid_epoch=scenarios[label+'_mid'],epoch_boundary=scenarios[label+'_epoch'])
    (out/(label+'_resume_comparison.json')).write_text(json.dumps(result,indent=2)+'\n')
    # Identical continuous checkpoints are redundant; preserve hashes and all logs.
    for name in scenarios:(out/(name+'_continuous')/'checkpoint.pth').unlink()
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
