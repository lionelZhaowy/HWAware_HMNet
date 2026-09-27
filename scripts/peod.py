#!/usr/bin/env python3
"""Four PEOD experiments with a shared training and original-coordinate COCO protocol."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import tomllib
import numpy as np
import torch
from torch.utils.data import DataLoader
from hmnet.models.efficientvit_tasks import build_frame_task, ROOT
from hmnet.dataset.peod_frames import PEODFrames
from hmnet.dataset.task_frames import sha_file
from hmnet.dataset.custom_collate_fn import collate_keep_dict
from hmnet.dataset.temporal_frames import SequenceBatchSampler
from hmnet.utils.temporal_streams import TemporalStreams
from hmnet.utils.frame_train import run, unpack


def parser():
    defaults=tomllib.loads((ROOT/'experiments/task_temporal/experiment.toml').read_text())
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['train','eval'])
    p.add_argument('--dataset',choices=['peod'],default='peod')
    p.add_argument('--modality',choices=['rgb','dvs','rgbdvs'],default=defaults.get('modality','rgbdvs'))
    p.add_argument('--representation',choices=['rvt_histogram','polarity_binary'],default=defaults['representation'])
    p.add_argument('--data-root',default='/data/lab_dataset/RGB_DVS_Fusion/PEOD_orig/preprocessed/hmnet_v12t_240x304_v1')
    p.add_argument('--output');p.add_argument('--resume');p.add_argument('--checkpoint');p.add_argument('--split',default='val')
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch',type=int,default=32)
    p.add_argument('--eval-batch',type=int,default=32);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--limit',type=int,default=0);p.add_argument('--stop-after',type=int)
    p.add_argument('--precision',choices=['bf16','fp32'],default='bf16');p.add_argument('--device',default='cuda:0')
    p.add_argument('--dump',help='JSONL predictions in original 1280x720 coordinates; scored without rerunning inference')
    return p


def configuration(a):
    temporal=0 if a.modality=='rgb' else 2
    name=f'peod_{a.modality}_{"none" if a.modality=="rgb" else a.representation}'
    pretrained=ROOT/'pretrained/efficientvit_b1_r224.pth'
    c=SimpleNamespace(task='detection',dataset='peod',modality=a.modality,event_representation=a.representation,
        event_channels=0 if a.modality=='rgb' else (20 if a.representation=='rvt_histogram' else 2),
        temporal_window=temporal,sequential_sampling=True,fusion_mode='add',data_root=a.data_root,
        deterministic=True,epochs=a.epochs,updates=None,batch_size=a.batch,eval_batch_size=a.eval_batch,
        accumulation=1,workers=a.workers,prefetch_factor=1,learning_rate=2e-4,weight_decay=.01,
        precision=a.precision,lr_schedule='warmup_cosine',min_learning_rate=2e-6,warmup_epochs=5,
        warmup_start_factor=.1,eval_every_epochs=1,resume=a.resume or '',output=a.output or str(ROOT/'logs/detection'/name))
    c.initialization_contract=dict(official_b1_sha256=sha_file(pretrained),method='named_component_sha256_seed_v1',seed=a.seed,
        binary_stem='official_RGB_mean_times_3_over_2',init_from=None)
    c.peod_contract=dict(classes=['car','person','bus','truck','2-wheeler','3-wheeler'],workers=a.workers,
        prefetch_factor=1,persistent_workers=a.workers>0,pin_memory=True,eval_batch=a.eval_batch,
        eval_precision=a.precision,score_threshold=.01,nms_iou=.65,gradient_clip=None,
        validation='every_epoch_COCO_mAP',main_checkpoint='last_epoch',best_checkpoint='val_mAP_separate',
        geometry='1280x720_to_304x171_pad_top34_bottom35',rgb='ImageNet_normalization',augmentation='lane_consistent_horizontal_flip',
        optimizer='AdamW_default_betas_eps',evaluation='original_float_COCO_all_frames_no_GEN1_filter')
    c.get_model=lambda:build_frame_task('detection',str(pretrained),modality=a.modality,temporal_window=temporal,
            event_channels=c.event_channels if temporal else 20,num_classes=6,stable_seed=a.seed)
    c.get_dataset=lambda:PEODFrames(a.data_root,'train',a.representation,a.modality,True,a.limit)
    # A bounded smoke does not perform a full validation sweep; the protocol remains fixed.
    if not a.stop_after:
        c.get_validation_dataset=lambda:PEODFrames(a.data_root,'val',a.representation,a.modality)
        c.evaluate_validation=lambda model,data:score_model(model,data,a.eval_batch,a.workers,a.precision)[0]
    return c


def score_dump(rows,classes):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    images=[];annotations=[];predictions=[]
    for i,row in enumerate(rows):
        images.append(dict(id=i,width=1280,height=720))
        for a in row['gt']:
            annotations.append(dict(a,id=len(annotations)+1,image_id=i))
        for a in row['predictions']:predictions.append(dict(a,image_id=i))
    categories=[dict(id=i,name=name) for i,name in enumerate(classes)]
    coco=COCO();coco.dataset=dict(info={},images=images,annotations=annotations,categories=categories);coco.createIndex()
    if predictions:det=coco.loadRes(predictions)
    else:
        det=COCO();det.dataset=dict(images=images,annotations=[],categories=categories);det.createIndex()
    ev=COCOeval(coco,det,'bbox');ev.evaluate();ev.accumulate();ev.summarize()
    metrics=dict(mAP=float(ev.stats[0]),AP50=float(ev.stats[1]),AP75=float(ev.stats[2]),classes={})
    for i,name in enumerate(classes):
        values=ev.eval['precision'][:,:,i,0,-1];values=values[values>=0]
        metrics['classes'][name]=float(values.mean()) if len(values) else None
    return metrics


@torch.no_grad()
def score_model(model,data,batch,workers,precision,dump=None):
    model.eval();device=next(model.parameters()).device
    sampler=SequenceBatchSampler(data,batch,training=False)
    loader=DataLoader(data,batch_sampler=sampler,collate_fn=data.collate_fn,num_workers=workers,
        pin_memory=True,**(dict(prefetch_factor=1) if workers else {}))
    streams=TemporalStreams(model.backbone,batch,data.reset_gap_us) if model.backbone.temporal_window else None
    rows=[]
    for packed in loader:
        values=unpack(packed,'detection');metas=values[1]
        with torch.amp.autocast(device.type,enabled=precision!='fp32',dtype=torch.bfloat16):
            output=model.inference(*values[:2],**(dict(temporal_state=streams.select(metas)) if streams else {}))
        predictions=output[0]
        if streams:streams.commit(metas,output[2])
        for pred,meta in zip(predictions,metas):
            index=meta['sample_index'];s=data.samples[index];boxes=pred['bboxes'].float().cpu().numpy().copy()
            if len(boxes):
                boxes[:,[1,3]]-=34;boxes/=.2375
                boxes[:,[0,2]]=boxes[:,[0,2]].clip(0,1280);boxes[:,[1,3]]=boxes[:,[1,3]].clip(0,720)
            preds=[]
            for b,score,label in zip(boxes,pred['scores'].float().cpu().tolist(),pred['labels'].cpu().tolist()):
                x,y,x2,y2=map(float,b)
                if x2>x and y2>y:preds.append(dict(category_id=int(label),bbox=[x,y,x2-x,y2-y],score=score))
            gt=[dict(category_id=a['category_id'],bbox=a['bbox'],area=a['area'],iscrowd=a.get('iscrowd',0),ignore=a.get('ignore',0)) for a in data.annotations(index)]
            rows.append(dict(index=index,sequence=s['sequence'],target_us=s['target_us'],condition=s['condition'],gt=gt,predictions=preds))
    rows.sort(key=lambda r:r['index'])
    if len(rows)!=len(data) or len({r['index'] for r in rows})!=len(data):raise AssertionError('Incomplete/duplicate evaluation')
    if dump:
        path=Path(dump);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
    return score_dump(rows,data.manifest['classes']),rows


def evaluate(c,a):
    checkpoint=a.checkpoint or str(Path(c.output)/'checkpoint.pth')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);saved=state['training_contract']
    data=PEODFrames(a.data_root,a.split,a.representation,a.modality,limit=a.limit)
    train_contract=PEODFrames(a.data_root,'train',a.representation,a.modality,limit=saved['event_data']['limit']).data_contract
    expected_window=c.temporal_window
    if (saved['task']!='detection' or saved['modality']!=a.modality or saved['event_input']!=data.input_spec or
        saved['event_data']!=train_contract or saved.get('temporal',{}).get('window',0)!=expected_window or
        saved['initialization']!=c.initialization_contract):raise ValueError('PEOD evaluation checkpoint contract differs')
    model=c.get_model().to(a.device).eval();model.load_state_dict(state['state_dict'],strict=True)
    out=Path(a.output or ROOT/f'artifacts/evaluation/{a.split}');out.mkdir(parents=True,exist_ok=True)
    dump=a.dump or str(out/'predictions.jsonl')
    metrics,rows=score_model(model,data,a.eval_batch,a.workers,a.precision,dump)
    report=dict(metrics=metrics,frames=len(data),scored_frames=len(rows),split=a.split,step=state['step'],
        checkpoint=str(Path(checkpoint).resolve()),checkpoint_sha256=sha_file(checkpoint),data_contract=data.data_contract,
        input=data.input_spec,precision=a.precision,eval_batch=a.eval_batch,dump=str(Path(dump).resolve()),dump_sha256=sha_file(dump),
        protocol='COCO bbox original 1280x720 floating GT, all frames including empty GT, no GEN1 filters',subsets={})
    if a.split=='test':
        for subset in ('normal','challenge'):report['subsets'][subset]=score_dump([r for r in rows if r['condition']==subset],data.manifest['classes'])
    (out/'metrics.json').write_text(json.dumps(report,indent=2));print(out/'metrics.json')


def main():
    a=parser().parse_args();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if a.limit and a.action=='train' and not a.stop_after:raise ValueError('Subset training requires bounded stop-after')
    a.single=True;a.distributed=False;a.amp=False;a.overwrite=False
    c=configuration(a)
    if a.action=='train':run(c,a)
    else:evaluate(c,a)

if __name__=='__main__':main()
