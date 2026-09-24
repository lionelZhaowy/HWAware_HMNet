#!/usr/bin/env python3
"""Reject wrong resume identities and corrupt state without taking an update."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import copy,json
from pathlib import Path
from unittest.mock import patch
import torch
from scripts.task_temporal import parser,configuration
from hmnet.utils.frame_train import run
from hmnet.models.efficientvit_tasks import ROOT


def main():
    args=parser().parse_args(['train','--dataset','gen1','--data-root','artifacts/data_smoke/gen1',
        '--resume','artifacts/validation/gen1_rvt_split/checkpoint.pth','--output','artifacts/validation/rejections',
        '--limit','32','--stop-after','4','--workers','0'])
    args.single=True;args.distributed=False;args.amp=False;args.overwrite=False
    config=configuration(args);load=torch.load;saved=load(args.resume,map_location='cpu',weights_only=False)
    changes={
      'model_task':lambda s:s['training_contract'].__setitem__('task','depth'),
      'representation':lambda s:s['training_contract']['event_input'].__setitem__('representation','polarity_binary'),
      'split':lambda s:s['training_contract']['event_data'].__setitem__('split','val'),
      'data_sha':lambda s:s['training_contract']['event_data'].__setitem__('manifest_sha256','wrong'),
      'budget':lambda s:s['training_contract']['schedule'].__setitem__('total_updates',401),
      'state_shape':lambda s:s['temporal_by_rank'][0]['memory'].__setitem__(0,s['temporal_by_rank'][0]['memory'][0][:1]),
      'state_dtype':lambda s:s['temporal_by_rank'][0]['memory'].__setitem__(0,s['temporal_by_rank'][0]['memory'][0].bfloat16()),
    };results={}
    for name,change in changes.items():
        modified=copy.deepcopy(saved);change(modified)
        def guarded(path,*a,**kw):return modified if str(path)==args.resume else load(path,*a,**kw)
        try:
            with patch('torch.load',guarded):run(config,args)
        except ValueError as error:results[name]=str(error)
        else:raise AssertionError(f'Mismatch accepted: {name}')
        print(name,'rejected',flush=True)
    (ROOT/'artifacts/validation/contract_rejections.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
