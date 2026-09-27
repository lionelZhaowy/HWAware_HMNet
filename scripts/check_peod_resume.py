#!/usr/bin/env python3
"""Real GPU continuous versus interrupted PEOD training; exact recursive comparison."""
import argparse,json,subprocess,os
from pathlib import Path
import numpy as np
import torch
from hmnet.models.efficientvit_tasks import ROOT

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
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--output',default='artifacts/peod/resume');p.add_argument('--batch',type=int,default=2);p.add_argument('--workers',type=int,default=0);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);reports=[]
    for name,mod,rep in [('rgb','rgb','rvt_histogram'),('dvs','dvs','rvt_histogram'),('fusion','rgbdvs','rvt_histogram'),('binary','rgbdvs','polarity_binary')]:
        for mode,steps,resume in [('continuous',4,False),('split',2,False),('split',4,True)]:
            dest=out/(name+'_'+mode)
            command=[str(ROOT/'scripts/hmnet-python'),str(ROOT/'scripts/task_temporal.py'),'train','--modality',mod,'--representation',rep,
                '--data-root',a.data_root,'--batch',str(a.batch),'--workers',str(a.workers),'--stop-after',str(steps),'--output',str(dest)]
            if resume:command+=['--resume',str(dest/'checkpoint.pth')]
            with (out/f'{name}_{mode}_{steps}.log').open('w') as f:subprocess.run(command,check=True,stdout=f,stderr=subprocess.STDOUT)
        x=torch.load(out/(name+'_continuous/checkpoint.pth'),map_location='cpu',weights_only=False)
        y=torch.load(out/(name+'_split/checkpoint.pth'),map_location='cpu',weights_only=False)
        equal(x,y); reports.append(dict(model=name,exact_resume=True,compared=list(x),step=x['step']))
        (out/'report.json').write_text(json.dumps(reports,indent=2));print(name,'exact resume PASS',flush=True)
        # Redundant continuous optimizer snapshot is rebuildable and already checked.
        (out/(name+'_continuous/checkpoint.pth')).unlink()
        del x,y
    print(out/'report.json')

if __name__=='__main__':main()
