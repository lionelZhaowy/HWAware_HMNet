#!/usr/bin/env python3
"""当前独立工程真实GPU连续/保存恢复精确对照；包括epoch尾累积组。"""
import argparse,json,subprocess,os
from pathlib import Path
import numpy as np
import torch
from hmnet.models.efficientvit_tasks import ROOT
from scripts.peod import parser

def equal(a,b,path='root'):
    if torch.is_tensor(a):
        if not torch.equal(a,b):raise AssertionError(path)
    elif isinstance(a,np.ndarray):
        if not np.array_equal(a,b):raise AssertionError(path)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError(path+' keys')
        for k in a:equal(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b):raise AssertionError(path+' len')
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'.'+str(i))
    elif a!=b:raise AssertionError(path+f' {a} != {b}')

def main():
    defaults=parser().parse_args(['train'])
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',default=defaults.data_root);p.add_argument('--output',required=True)
    p.add_argument('--batch',type=int,default=defaults.batch);p.add_argument('--workers',type=int,default=defaults.workers)
    p.add_argument('--accumulation',type=int,default=defaults.accumulation);p.add_argument('--tail-check',action='store_true');p.add_argument('--smoke-validation',action='store_true');p.add_argument('--recipe',choices=['a0','a1'],default='a0');a=p.parse_args()
    if a.smoke_validation and not a.tail_check:raise ValueError('Smoke validation requires --tail-check')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    # 67 samples produces 9 batches at batch8: 4+4+1 microbatches, final batch3.
    limit=67 if a.tail_check else 0;steps=6 if a.tail_check else 4
    commands=[]
    for mode,end,resume in [('continuous',steps,False),('split',2,False),('split',steps,True)]:
        dest=out/mode
        command=[str(ROOT/'scripts/hmnet-python'),str(ROOT/'scripts/peod.py'),'train','--data-root',a.data_root,
            '--batch',str(a.batch),'--workers',str(a.workers),'--accumulation',str(a.accumulation),'--limit',str(limit),
            '--stop-after',str(end),'--output',str(dest)]
        command+=['--recipe',a.recipe]
        if a.smoke_validation:command+=['--smoke-validation']
        if resume:command+=['--resume',str(dest/'checkpoint.pth')]
        with (out/f'{mode}_{end}.log').open('w') as f:
            completed=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
        commands.append(dict(command=command,exit_code=completed.returncode))
        if completed.returncode:raise RuntimeError(f'{mode}/{end} failed; see log')
    x=torch.load(out/'continuous/checkpoint.pth',map_location='cpu',weights_only=False)
    y=torch.load(out/'split/checkpoint.pth',map_location='cpu',weights_only=False);equal(x,y)
    rows=[json.loads(line) for line in (out/'continuous/metrics.jsonl').read_text().splitlines()]
    assert x['ema']['updates']==x['step'] and all(r['ema_updates']==r['step'] for r in rows)
    if a.smoke_validation:
        evaluated=[r for r in rows if 'dev' in r]
        assert [r['step'] for r in evaluated]==[3,6]
        assert all(all(k in r for k in ['dev_ema','fixed_val','fixed_val_ema']) for r in evaluated)
        for dest in [out/'continuous',out/'split']:
            for name,kind in [('best_checkpoint.pth','raw'),('best_ema_checkpoint.pth','ema')]:
                best=torch.load(dest/name,map_location='cpu',weights_only=False)
                assert best['inference_only'] and best['weight_kind']==kind and 'optimizer' not in best
    report=dict(recipe=a.recipe,smoke_validation=a.smoke_validation,EMA_exact_resume=True,exact_resume=True,compared=list(x),step=x['step'],data_epoch=x['data_epoch'],data_cursor=x['data_cursor'],
        tail_check=a.tail_check,samples=[r['samples'] for r in rows],microbatches=[r['microbatches'] for r in rows],
        contract=x['training_contract'],commands=commands,
        pending_gradients='none: all checkpoints are optimizer-boundary; partial group flushed at epoch end')
    if a.tail_check:
        assert report['samples']==[32,32,3,32,32,3];assert report['microbatches']==[4,4,1,4,4,1]
        assert sum(report['samples'])==134 and report['data_epoch']==1 and report['data_cursor']==9
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(exact_resume=True,tail_check=a.tail_check,step=x['step'])))
    (out/'continuous/checkpoint.pth').unlink()
if __name__=='__main__':main()
