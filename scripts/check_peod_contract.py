#!/usr/bin/env python3
"""恢复契约与误覆盖保护负向验收，不执行优化器更新。"""
import argparse,copy,json
from pathlib import Path
import torch
from scripts.peod import parser,configuration
from hmnet.utils.frame_train import run,validate_resume_contract
from hmnet.dataset.peod_frames import PEODFrames

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)['training_contract'];checks=[]
 validate_resume_contract(saved,saved)
 for label,path,value in [('height',['event_input','height'],240),('schema',['event_input','schema'],'old'),('data',['event_data','manifest_sha256'],'changed'),('modality',['modality'],'other'),('representation',['event_input','representation'],'other'),('epochs',['schedule','total_updates'],saved['schedule']['total_updates']+1),('accumulation',['accumulation'],1),('batch',['batch_size'],4),('seed',['seed'],1),('precision',['precision'],'fp32'),('learning_rate',['schedule','base_lr'],.1)]:
  wrong=copy.deepcopy(saved);parent=wrong
  for key in path[:-1]:parent=parent[key]
  parent[path[-1]]=value
  try:validate_resume_contract(saved,wrong)
  except ValueError as exc:checks.append(dict(case=label,rejected=True,reason=str(exc)))
  else:raise AssertionError(label+' accepted')
 args=parser().parse_args(['train','--resume',a.checkpoint,'--epochs','201','--stop-after','5','--output',str(out/'reject_budget')]);args.single=True;args.distributed=False;args.amp=False;args.overwrite=False
 c=configuration(args)
 def forbidden():raise AssertionError('Model constructed before incompatible resume rejected')
 c.get_model=forbidden
 try:run(c,args)
 except ValueError as exc:
  assert 'Resume training contract differs' in str(exc);checks.append(dict(case='real_entry_budget_rejected_before_model',rejected=True))
 else:raise AssertionError('Budget accepted')
 dest=out/'protected';dest.mkdir(exist_ok=True);(dest/'settings.json').write_text('{}')
 args=parser().parse_args(['train','--stop-after','1','--output',str(dest)]);args.single=True;args.distributed=False;args.amp=False;args.overwrite=False;c=configuration(args);c.get_model=forbidden
 try:run(c,args)
 except FileExistsError:checks.append(dict(case='fresh_output_overwrite',rejected=True))
 else:raise AssertionError('Output overwrite accepted')
 old=Path(args.data_root).parent/'hmnet_v12t_240x304_v1'
 try:PEODFrames(old,'train')
 except ValueError:checks.append(dict(case='old_low_resolution_data',rejected=True))
 else:raise AssertionError('Legacy schema accepted')
 (out/'report.json').write_text(json.dumps(dict(passed=True,checks=checks),indent=2));print('PASS',len(checks))
if __name__=='__main__':main()
