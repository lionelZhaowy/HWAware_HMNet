#!/usr/bin/env python3
"""Bounded CPU regression: causal cache, two-step gradients, BN and exact resume."""
import argparse,copy,io,json,tempfile
import h5py
import numpy as np
from pathlib import Path
import torch
from hmnet.models.efficientvit_tasks import build_frame_task
from hmnet.models.async_frame import AsyncPredictor,masked_ce
from hmnet.utils.temporal_streams import TemporalStreams
from hmnet.utils.activation_checkpoint import temporary_bn_buffers
from hmnet.dataset.dsec_async import DSECAsync
from hmnet.dataset.temporal_frames import SequenceBatchSampler
from scripts.pseudo_dsec_async import agreement_mask
from scripts.prepare_dsec_async import time_grid
from scripts.prepare_dsec_b1 import read_window
from hmnet.models.base.event_repr.rvt_histogram import RVTHistogram


def same(a,b):
    if torch.is_tensor(a):torch.testing.assert_close(a,b,atol=0,rtol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:same(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b,(a,b)


def build(channels=20):
    return build_frame_task('segmentation',modality='rgbdvs',fusion_mode='add',temporal_window=2,event_channels=channels)


def train_step(model,opt,bank,events,images,targets,time):
    metas=[]
    for i in range(len(images)):
        steps=[dict(height=64,width=96,rgb_id=f'{i}/old',rgb_valid=True,rgb_time=time-50000,rgb_age_us=(t+1)*25000) for t in range(2)]
        metas.append(dict(steps=steps,sequence=f's{i}',stream_slot=i,reset=time==50000,
                          flipped=False,curr_time_org=time,pseudo_weight=.1))
    model.train();opt.zero_grad(set_to_none=True)
    result=model(events,images,metas,targets,temporal_state=bank.select(metas))
    result['loss'].backward();opt.step();bank.commit(metas,result['temporal_state'])
    assert all(x.grad_fn is None for x in bank.memory)
    return float(result['loss'].detach())


def main(args):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(42)
    report={}
    # Common modules must receive identical initialization despite stem width.
    torch.manual_seed(42);model=build(20)
    torch.manual_seed(42);other=build(10)
    skip='backbone.event_encoder.input_stem.op_list.0.conv.weight'
    for key,value in model.state_dict().items():
        if key!=skip:same(value,other.state_dict()[key])
    del other
    report['shared_initialization']='all parameters/buffers except event stem bitwise equal'
    model.eval();backbone=model.backbone
    e=torch.rand(1,20,64,96);r=torch.rand(1,3,64,96);state=backbone.zero_temporal_state(1)
    with torch.no_grad():
        expected,_=backbone(e,r,state)
        actual,_=backbone.async_step(e,backbone.encode_rgb(r),state)
        same(expected,actual)
    predictor=AsyncPredictor(model);calls=[]
    handle=backbone.rgb_encoder.register_forward_hook(lambda *x:calls.append(1))
    meta=dict(height=64,width=96,sequence='a',curr_time_org=25000,rgb_time=0,rgb_id='a/0',reset=False)
    first=predictor.step(e,meta,r)
    predictor.step(e,dict(meta,curr_time_org=50000),None)
    assert len(calls)==1
    try:predictor.step(e,dict(meta,curr_time_org=75000,rgb_time=100000,rgb_id='future'),r)
    except ValueError:pass
    else:raise AssertionError('future RGB accepted')
    predictor.reset();same(first,predictor.step(e,meta,r))
    predictor.reset();assert torch.isfinite(predictor.step(e,dict(meta,rgb_id=None,rgb_time=-1),None)).all()
    # Arbitrary 30Hz image arrivals on a 5ms DVS clock; no integer-ratio assumption.
    predictor.reset();calls.clear()
    for t in range(0,100001,5000):
        rt=t//33333*33333
        new=predictor.rgb_time!=rt
        predictor.step(e,dict(meta,curr_time_org=t,rgb_time=rt,rgb_id=f'a/{rt}'),r if new else None)
    assert len(calls)==4
    handle.remove();report['inference']='exact cache equality; no re-encode on DVS-only step; future rejection; reset/cold start; 30Hz arrivals'
    # Gradient through first summaries, not merely extra forward/head BN updates.
    model.train();events=torch.rand(2,2,20,64,96,requires_grad=True)
    images=[torch.rand(2,3,64,96,requires_grad=True) for _ in range(2)]
    targets=[torch.stack([torch.full((64,96),255),torch.randint(11,(64,96))]) for _ in range(2)]
    captured=[];original=backbone.event_step
    def capture(*x):
        features,state=original(*x)
        for v in state:v.retain_grad()
        captured.append(state)
        return features,state
    backbone.event_step=capture
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4);bank=TemporalStreams(backbone,2)
    bn_before={k:int(v) for k,v in model.state_dict().items() if k.endswith('num_batches_tracked')}
    train_step(model,opt,bank,events,images,targets,50000)
    for key,old_count in bn_before.items():
        expected_increment=2 if key.startswith(('backbone.event_encoder.','backbone.event_proj.')) else 1
        assert int(model.state_dict()[key])-old_count==expected_increment,(key,old_count,model.state_dict()[key])
    assert any(v.grad is not None and v.grad.abs().sum()>0 for v in captured[0])
    assert events.grad[:,0].abs().sum()>0 and all(x.grad[:,...].abs().sum()>0 for x in images)
    backbone.event_step=original
    report['gradient']='step-2 GT reaches step-1 DVS summaries/events and cached RGB; bank detached after optimizer'
    # Real phase-2 loss path and empty confidence mask.
    z=torch.randn(2,11,8,9,requires_grad=True);gt=torch.full((2,8,9),255)
    loss=masked_ce(z,gt);assert float(loss)==0;loss.backward();assert torch.isfinite(z.grad).all()
    a=torch.full((1,11,2,2),-20.);a[:,3]=20
    labels,mask=agreement_mask(a,a,.95);assert mask.all() and (labels==3).all()
    b=a.clone();b[:,3]=-20;b[:,4]=20
    labels,mask=agreement_mask(a,b,.95);assert not mask.any() and (labels==255).all()
    report['pseudo']='same-time teacher agreement, confidence, empty-mask finite zero'
    events=events.detach();images=[x.detach() for x in images]
    targets=[torch.randint(11,(2,64,96)) for _ in range(2)]
    payload=io.BytesIO();torch.save(dict(model=model.state_dict(),opt=opt.state_dict(),bank=bank.state_dict(),rng=torch.get_rng_state()),payload)
    expected_loss=train_step(model,opt,bank,events,images,targets,100000)
    payload.seek(0);saved=torch.load(payload,weights_only=False)
    restored=build();restored.load_state_dict(saved['model'],strict=True)
    opt2=torch.optim.AdamW(restored.parameters());opt2.load_state_dict(saved['opt'])
    bank2=TemporalStreams(restored.backbone,2);bank2.load_state_dict(saved['bank'])
    torch.set_rng_state(saved['rng']);actual_loss=train_step(restored,opt2,bank2,events,images,targets,100000)
    assert actual_loss==expected_loss
    same(model.state_dict(),restored.state_dict());same(opt.state_dict(),opt2.state_dict());same(bank.state_dict(),bank2.state_dict())
    report['resume']='bitwise weights, BN, optimizer, state and loss after phase-2-style next update'
    # Only real forward steps update BN; recomputation leaves buffers intact.
    saved_bn={k:v.clone() for k,v in backbone.state_dict().items() if 'running_' in k or 'num_batches' in k}
    with temporary_bn_buffers(backbone):backbone.encode_rgb(torch.rand(2,3,64,96))
    for k,v in saved_bn.items():same(v,backbone.state_dict()[k])
    report['bn']='temporary recompute buffers restored exactly'
    data=DSECAsync(args.data_root,'train',clips=True,augment=True)
    keys=list(SequenceBatchSampler(data,32));indices=[key[0] for batch in keys for key in batch]
    assert sorted(indices)==list(range(len(data)))
    for key in keys[0][:4]:
        inputs,targets,metadata=data[key];steps=metadata['image_meta']['steps']
        assert inputs['events'].shape==(2,20,440,640)
        assert not steps[0]['has_gt'] and steps[1]['has_gt']
        assert steps[0]['curr_time_org']<steps[1]['curr_time_org']
        assert all(not m['rgb_valid'] or m['rgb_time']<=m['curr_time_org'] for m in steps)
    report['data']='one GT exposure/epoch; two ordered event steps; causal RGB; raw-derived real cache'
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'events.h5';times=np.array([0,1,25000,50000,50001],dtype=np.int64)
        with h5py.File(path,'w') as f:
            f['t_offset']=1000000;f['events/t']=times
            f['events/x']=np.ones(5,dtype=np.int16);f['events/y']=np.ones(5,dtype=np.int16)
            f['events/p']=np.array([0,0,1,0,1],dtype=np.int8)
            f['ms_to_idx']=np.searchsorted(times,np.arange(52)*1000)
        with h5py.File(path) as f:
            events=read_window(f,1050000,50000);events=events[events[:,0]>0]
        assert events[:,0].tolist()==[1,25000,50000]
        hist,_=RVTHistogram()(torch.from_numpy(events),dict(height=4,width=4))
        assert hist[:,1,1].nonzero().flatten().tolist()==[0,9,14] and hist.sum()==3
        empty,_=RVTHistogram(bins=5)(torch.empty(0,4,dtype=torch.int64),dict(height=4,width=4))
        assert empty.shape==(10,4,4) and empty.sum()==0
    try:DSECAsync(args.data_root,'train',window_us=25000,bins=5)
    except ValueError:pass
    else:raise AssertionError('B cache accepted by C')
    report['representation']='int64 offset, half-open boundary, future event exclusion, polarity/bin mapping, empty event, B/C cache mismatch'
    result=dict(passed=True,checks=report)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',required=True);p.add_argument('--output',required=True);main(p.parse_args())
