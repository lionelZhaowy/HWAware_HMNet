#!/usr/bin/env python3
"""Synchronous M=2 task training/evaluation; no asynchronous RGB or pseudo labels."""
import argparse
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from hmnet.models.efficientvit_tasks import build_frame_task, ROOT
from hmnet.dataset.task_frames import TaskFrames,sha_file
from hmnet.dataset.temporal_frames import SequenceBatchSampler
from hmnet.dataset.custom_collate_fn import collate_keep_dict
from hmnet.utils.temporal_streams import TemporalStreams
from hmnet.utils.frame_train import run,unpack

DATA_NAMES={'gen1':'GEN1','eventscape':'Eventscape','mvsec':'MVSEC'}

def configuration(args):
    defaults=tomllib.loads((ROOT/'experiments/task_temporal/experiment.toml').read_text())
    kind=args.dataset or defaults['dataset'];rep=defaults['representation']
    if args.representation:rep=args.representation
    config=SimpleNamespace(task='detection' if kind=='gen1' else 'depth',dataset=kind,
        event_representation=rep,event_channels=20 if rep=='rvt_histogram' else 2,
        modality='rgbdvs' if kind=='eventscape' else 'dvs',fusion_mode='add',temporal_window=2,
        data_root=args.data_root or f'/data/lab_dataset/RGB_DVS_Fusion/{DATA_NAMES[kind]}/preprocessed/hmnet_v12t_raw_v1',
        deterministic=True,epochs=args.epochs,updates=None,batch_size=args.batch,accumulation=1,workers=args.workers,
        learning_rate=2e-4,weight_decay=.01,precision='bf16',lr_schedule='warmup_cosine',
        min_learning_rate=2e-6,warmup_epochs=5,warmup_start_factor=.1,prefetch_factor=1,
        eval_every_epochs=1,resume=args.resume or '',
        output=args.output or str(ROOT/f'logs/{"detection" if kind=="gen1" else "depth"}/{kind}_{"rvt" if rep=="rvt_histogram" else "binary"}'))
    pretrained=ROOT/'pretrained/efficientvit_b1_r224.pth'
    if not pretrained.exists():raise FileNotFoundError(pretrained)
    config.initialization_contract=dict(official_b1_sha256=sha_file(pretrained),init_from=None)
    # On resume reuse the audited initialization provenance; weights come from resume.
    if config.resume:
        if args.init_from:raise ValueError('Choose resume or init-from')
        saved=torch.load(config.resume,map_location='cpu',weights_only=False)
        config.initialization_contract=saved['training_contract']['initialization']
    elif args.init_from:
        config.initialization_contract['init_from']=dict(path=str(Path(args.init_from).resolve()),sha256=sha_file(args.init_from))
    def model():
        result=build_frame_task(config.task,str(pretrained),mvsec=kind=='mvsec',modality=config.modality,
            temporal_window=2,event_channels=config.event_channels)
        if args.init_from:
            if kind!='mvsec':raise ValueError('init-from is the explicit Eventscape→MVSEC transition')
            source=torch.load(args.init_from,map_location='cpu',weights_only=False);contract=source['training_contract']
            if (contract.get('task')!='depth' or contract['event_data']['kind']!='eventscape'
                or contract['event_input']['representation']!=rep or contract['temporal']['window']!=2):
                raise ValueError('Transfer checkpoint must be matching Eventscape M=2 representation')
            dropped=[k for k in source['state_dict'] if k.startswith(('backbone.rgb_encoder.','backbone.rgb_proj.'))]
            kept={k:v for k,v in source['state_dict'].items() if k not in dropped}
            result.load_state_dict(kept,strict=True)
            print(json.dumps(dict(transfer_source=args.init_from,dropped_rgb_keys=dropped,
                optimizer_and_memory='fresh',new_depth_range=[1.978,80.])),flush=True)
        return result
    config.get_model=model
    config.get_dataset=lambda:TaskFrames(config.data_root,'day2' if kind=='mvsec' else 'train',rep,augment=True,limit=args.limit)
    return config


def load_evaluation_model(config,checkpoint,device):
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);c=state['training_contract']
    if (c.get('task')!=config.task or c['event_data']['kind']!=config.dataset or
        c['event_input']['representation']!=config.event_representation or
        c['modality']!=config.modality or c['temporal']['window']!=2):
        raise ValueError('Evaluation architecture/dataset/representation checkpoint mismatch')
    model=config.get_model().to(device).eval();model.load_state_dict(state['state_dict'],strict=True)
    return model,state


@torch.no_grad()
def evaluate(config,args):
    from experiments.depth.scripts.eval_depth import evaluate_one_sample
    from experiments.detection.scripts.coco_eval import _to_coco_format
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    device=torch.device(args.device);checkpoint=args.checkpoint or str(Path(config.output)/'checkpoint.pth')
    model,state=load_evaluation_model(config,checkpoint,device)
    split=args.split or ('day1' if config.dataset=='mvsec' else 'val')
    data=TaskFrames(config.data_root,split,config.event_representation,limit=args.limit)
    sampler=SequenceBatchSampler(data,args.batch,training=False)
    loader=DataLoader(data,batch_sampler=sampler,collate_fn=collate_keep_dict,num_workers=args.workers)
    streams=TemporalStreams(model.backbone,args.batch,data.reset_gap_us)
    scores=[];gts=[];preds=[];skipped=0;frames=0
    dtype=np.dtype([('t','<i8'),('x','<f4'),('y','<f4'),('w','<f4'),('h','<f4'),('class_id','<i4'),('class_confidence','<f4')])
    for batch in loader:
        values=unpack(batch,config.task);metas=values[1] if config.task=='detection' else values[2]
        inputs=values[:2] if config.task=='detection' else values[:3]
        with torch.amp.autocast(device.type,enabled=args.precision!='fp32',dtype=torch.bfloat16 if args.precision=='bf16' else torch.float16):
            pred,_,current=model.inference(*inputs,temporal_state=streams.select(metas))
        streams.commit(metas,current)
        for i,m in enumerate(metas):
            frames+=1;s=data.samples[m['sample_index']]
            if config.task=='depth':
                # Preserve legacy Eventscape skip_ts=199.9 ms. No MVSEC crop or interpolation.
                if config.dataset=='eventscape' and m['curr_time_org']<=199900:skipped+=1;continue
                gt=batch[1][i]['depth'].numpy().squeeze();p=pred[i].float().cpu().numpy().squeeze()
                lo,hi=(1.978,80.) if config.dataset=='mvsec' else (3.346,1000.)
                if not ((gt>=lo)&(gt<=hi)&np.isfinite(gt)).any():skipped+=1;continue
                with np.errstate(invalid='ignore',divide='ignore'):
                    scores.append(evaluate_one_sample(p,gt,hi,lo,[30,20,10],False))
            else:
                raw=np.load(s['label_file']);field='ts' if 'ts' in raw.dtype.names else 't';raw=raw[raw[field]==s['target_us']]
                gt=np.zeros(len(raw),dtype)
                for k in ('x','y','w','h','class_id'):gt[k]=raw[k]
                gt['t']=s['target_us'];gt['class_confidence']=1.
                d=pred[i];n=len(d['labels']);pr=np.zeros(n,dtype)
                if n:
                    b=d['bboxes'].float().cpu().numpy();pr['x'],pr['y']=b[:,0],b[:,1];pr['w'],pr['h']=b[:,2]-b[:,0],b[:,3]-b[:,1]
                    pr['class_id']=d['labels'].cpu().numpy();pr['class_confidence']=d['scores'].float().cpu().numpy();pr['t']=s['target_us']
                def filtered(x):return x[(x['t']>500000)&(x['w']>=10)&(x['h']>=10)&(x['w']**2+x['h']**2>=900)]
                gt,pr=filtered(gt),filtered(pr)
                # Official Prophesee scoring only includes timestamps with retained GT.
                if len(gt):gts.append(gt);preds.append(pr)
                else:skipped+=1
    if config.task=='depth':
        metrics={k:float(np.nanmean([r[k] for r in scores])) for k in scores[0]} if scores else {}
    else:
        categories=[dict(id=1,name='car',supercategory='none'),dict(id=2,name='pedestrian',supercategory='none')]
        dataset,results=_to_coco_format(gts,preds,categories,height=240,width=304)
        coco=COCO();coco.dataset=dict(dataset,info={});coco.createIndex()
        if results:det=coco.loadRes(results)
        else:
            det=COCO();det.dataset=dict(images=dataset['images'],categories=categories,annotations=[]);det.createIndex()
        ev=COCOeval(coco,det,'bbox');ev.evaluate();ev.accumulate();ev.summarize()
        metrics=dict(mAP=float(ev.stats[0]),AP50=float(ev.stats[1]),AP75=float(ev.stats[2]),classes={})
        for i,name in enumerate(('car','pedestrian')):
            v=ev.eval['precision'][:,:,i,0,-1];metrics['classes'][name]=float(v[v>=0].mean()) if (v>=0).any() else None
    out=Path(args.output or ROOT/f'artifacts/evaluation/{config.dataset}_{config.event_representation}_{split}')
    out.mkdir(parents=True,exist_ok=True)
    report=dict(dataset=config.dataset,split=split,frames=frames,scored_frames=frames-skipped,metrics=metrics,
        checkpoint=str(Path(checkpoint).resolve()),checkpoint_sha256=sha_file(checkpoint),step=state['step'],
        input=data.input_spec,data_contract=data.data_contract,precision=args.precision,trained_accuracy_claim=False,
        protocol='original depth frame mean/cutoffs; or Prophesee 0.5s/30diag/10side, exact label-time predictions (within ±25ms)')
    path=out/'metrics.json';path.write_text(json.dumps(report,indent=2)+'\n');print(path,metrics)


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['train','eval'])
    p.add_argument('--dataset',choices=DATA_NAMES);p.add_argument('--representation',choices=['rvt_histogram','polarity_binary'])
    p.add_argument('--data-root');p.add_argument('--output');p.add_argument('--resume');p.add_argument('--init-from');p.add_argument('--checkpoint');p.add_argument('--split')
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch',type=int,default=8);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--limit',type=int,default=0);p.add_argument('--stop-after',type=int)
    p.add_argument('--precision',choices=['bf16','fp32','fp16'],default='bf16');p.add_argument('--device',default='cuda:0')
    return p


def main():
    args=parser().parse_args();torch.set_num_threads(4)
    if args.limit and args.action=='train' and not args.stop_after:raise ValueError('Subset requires bounded diagnostic stop')
    args.single=True;args.distributed=False;args.amp=False;args.overwrite=False
    config=configuration(args)
    if args.action=='train':run(config,args)
    else:evaluate(config,args)

if __name__=='__main__':main()
