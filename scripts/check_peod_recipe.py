#!/usr/bin/env python3
"""RecipeV2 CPU invariants: float geometry, EMA, locked splits and resume contracts."""
import argparse
import copy
import json
import math
from pathlib import Path
import torch
from scripts.peod import parser,configuration
from hmnet.models.base.head.task_head.det_head_yolox import YOLOXHead,IOUloss,bboxes_iou
from hmnet.dataset.peod_recipe import PEODRecipeFrames
from hmnet.dataset.temporal_frames import SequenceBatchSampler
from hmnet.utils.model_ema import ModelEMA
from hmnet.utils.frame_train import resolve_training_updates,training_schedule,validate_resume_contract
from hmnet.utils.recipe_checkpoints import atomic_save


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(42)
    head=YOLOXHead(6,in_channels=[16]*3,feat_channels=16)
    boxes=torch.tensor([[10.25,20.5,34.75,61.5]]);labels=torch.tensor([1])
    target=head._to_tensor_gt([boxes],[labels]);want=torch.tensor([[[1.,22.5,41.,24.5,41.]]])
    assert target.dtype==torch.float32 and torch.equal(target,want)
    empty=head._to_tensor_gt([torch.empty(0,4)],[torch.empty(0,dtype=torch.long)])
    assert empty.shape==(1,0,5) and empty.dtype==torch.float32
    raw=tuple(tuple(torch.zeros(2,ch,h,w,dtype=torch.bfloat16,requires_grad=True) for h,w in [(4,4),(2,2),(1,1)]) for ch in [4,1,6])
    empty_loss,_=head.loss(raw,[torch.empty(0,4)]*2,[torch.empty(0,dtype=torch.long)]*2,[torch.empty(0,dtype=torch.bool)]*2)
    assert empty_loss.dtype==torch.float32 and torch.isfinite(empty_loss)
    empty_loss.backward();assert all(x.grad is not None and torch.isfinite(x.grad).all() for group in raw for x in group)
    cxcywh=torch.tensor([[22.5,41.,24.5,41.],[.75,1.25,1.5,2.5],[1271.375,711.75,9.25,7.5]])
    assert torch.allclose(IOUloss()(cxcywh,cxcywh),torch.zeros(3),atol=1e-6)
    assert torch.allclose(bboxes_iou(cxcywh,cxcywh,False).diag(),torch.ones(3),atol=1e-6)
    shifted=(cxcywh+torch.tensor([.125,.25,0.,0.])).requires_grad_();loss=IOUloss()(shifted,cxcywh).sum();loss.backward();assert torch.isfinite(shifted.grad).all()
    prediction=torch.zeros(1,280,11,dtype=torch.bfloat16);prediction[:,:,0]=.1
    result=head.decode_outputs(prediction,[(1,160),(1,80),(1,40)],prediction.dtype)
    assert result.dtype==torch.float32
    assert result[0,159,0].item()==(159+float(torch.tensor(.1,dtype=torch.bfloat16)))*8
    toy=torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.BatchNorm1d(3));rng=torch.get_rng_state().clone();ema=ModelEMA(toy)
    assert torch.equal(rng,torch.get_rng_state())
    before={k:v.clone() for k,v in ema.model.state_dict().items()}
    with torch.no_grad():
        for param in toy.parameters():param.add_(1)
        toy[1].running_mean.fill_(3);toy[1].num_batches_tracked.fill_(7)
    ema.update(toy);decay=.9998*(1-math.exp(-1/2000))
    for k,v in ema.model.state_dict().items():
        if v.is_floating_point():assert torch.allclose(v,before[k]*decay+toy.state_dict()[k]*(1-decay),atol=1e-7)
        else:assert torch.equal(v,toy.state_dict()[k])
    restored=ModelEMA(toy);restored.load_state_dict(ema.state_dict());assert restored.updates==1
    for k,v in ema.model.state_dict().items():assert torch.equal(v,restored.model.state_dict()[k])
    args=parser().parse_args(['train']);c=configuration(args);data={s:PEODRecipeFrames(args.data_root,s,args.representation,args.modality,selection=args.selection) for s in ['train','dev','val','test']}
    lengths={s:len(d) for s,d in data.items()};assert lengths==dict(train=44415,dev=10393,val=4820,test=11847)
    groups={s:{v['sequence'] for v in d.samples} for s,d in data.items()}
    assert all(not(groups[x]&groups[y]) for i,x in enumerate(groups) for y in list(groups)[i+1:])
    sampler=SequenceBatchSampler(data['train'],8,training=True);seen=[key[0] for batch in sampler for key in batch]
    assert len(seen)==len(set(seen))==44415
    total=resolve_training_updates(c,len(sampler));schedule=training_schedule(c,len(sampler),total)
    assert len(sampler)==5552 and total==83280 and schedule['warmup_updates']==6940
    assert c.get_model is not None and c.ema_contract['decay']==.9998
    b=configuration(parser().parse_args(['train','--recipe','a1']));assert c.learning_rate==5e-4 and b.learning_rate==1e-4
    assert c.initialization_contract==b.initialization_contract
    old=dict(fusion_mode='none',peod={});new=dict(fusion_mode='none',peod=c.peod_contract)
    try:validate_resume_contract(old,new)
    except ValueError:pass
    else:raise AssertionError('Legacy precision contract accepted')
    # A space rejection must preserve an existing checkpoint byte-for-byte.
    temp=out/'space_guard.pth';atomic_save({'x':torch.ones(1)},temp);saved=temp.read_bytes()
    try:atomic_save({'x':torch.zeros(1)},temp,reserve_bytes=10**30)
    except OSError:pass
    else:raise AssertionError('Space reserve was ignored')
    assert temp.read_bytes()==saved;temp.unlink()
    report=dict(passed=True,float_targets=True,empty_GT_loss_and_gradients_finite=True,fractional_identical_IoU_one=True,finite_geometry_gradient=True,
        BF16_logits_FP32_pixel_decode=True,EMA_FP32_parameters_and_buffers=True,EMA_rng_preserved=True,
        split_frames=lengths,split_sequences={s:len(g) for s,g in groups.items()},disjoint=True,all_train_frames_once=True,
        batches_per_epoch=len(sampler),updates_per_epoch=total//60,total_updates=total,schedule=schedule,
        selection_sha256=c.peod_contract['selection_sha256'],legacy_resume_rejected=True,space_guard_preserves_previous=True)
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
