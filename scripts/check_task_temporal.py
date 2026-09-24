#!/usr/bin/env python3
"""Representation, initialization, task-gradient and memory invariants."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from hmnet.dataset.task_frames import represent
from hmnet.models.efficientvit_tasks import build_frame_task,ROOT
from hmnet.utils.temporal_streams import TemporalStreams


def checks(args):
    torch.set_num_threads(2);device=torch.device(args.device)
    report={};h,w=8,10
    cases=[np.empty((0,4),np.int64),np.array([[0,2,3,0]],np.int64),
        np.array([[0,2,3,0],[50000,2,3,1]],np.int64),
        np.tile(np.array([[0,2,3,1]],np.int64),(256,1))]
    for ev in cases:
        b=represent(ev,'polarity_binary',h,w);r=represent(ev,'rvt_histogram',h,w)
        want=torch.zeros_like(b)
        for _,x,y,p in ev:want[p,y,x]=1
        assert torch.equal(b,want) and set(b.unique().tolist())<={0.,1.}
        assert r.shape==(20,h,w)
    assert represent(cases[-1],'polarity_binary',h,w).sum()==1
    assert represent(cases[-1],'rvt_histogram',h,w).sum()==0  # preserved uint8 overflow
    report['representation_cases']=len(cases)
    for kind,task,modality,size in [('gen1','detection','dvs',(240,304)),('eventscape','depth','rgbdvs',(256,512)),('mvsec','depth','dvs',(260,346))]:
        torch.manual_seed(42)
        a=build_frame_task(task,str(ROOT/'pretrained/efficientvit_b1_r224.pth'),mvsec=kind=='mvsec',modality=modality,temporal_window=2)
        rng=torch.get_rng_state();torch.manual_seed(42)
        b=build_frame_task(task,str(ROOT/'pretrained/efficientvit_b1_r224.pth'),mvsec=kind=='mvsec',modality=modality,temporal_window=2,event_channels=2)
        assert torch.equal(rng,torch.get_rng_state())
        stem='backbone.event_encoder.input_stem.op_list.0.conv.weight'
        assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items() if k!=stem)
        del a
        model=b.to(device);model.train();hh,ww=size
        events=torch.randint(2,(2,2,hh,ww),device=device).float()
        images=[torch.randn(3,hh,ww,device=device) for _ in range(2)] if modality=='rgbdvs' else [None,None]
        metas=[dict(height=hh,width=ww,stream_slot=i,reset=True,sequence=str(i),flipped=False,curr_time_org=50000) for i in range(2)]
        bank=TemporalStreams(model.backbone,2,1500000 if kind=='gen1' else 75000)
        targets=[torch.full((1,hh,ww),10.,device=device) for _ in range(2)]
        args_forward=(events,metas,[torch.tensor([[20.,20.,80.,90.]],device=device),torch.empty((0,4),device=device)],
            [torch.zeros(1,dtype=torch.long,device=device),torch.empty(0,dtype=torch.long,device=device)],
            [torch.zeros(1,dtype=torch.bool,device=device),torch.empty(0,dtype=torch.bool,device=device)]) if task=='detection' else (events,images,metas,targets)
        with torch.autocast(device.type,dtype=torch.bfloat16):result=model(*args_forward,temporal_state=bank.select(metas))
        assert torch.isfinite(result['loss']);result['loss'].backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.backbone.event_encoder.parameters())
        bank.commit(metas,result['temporal_state']);assert all(s.dtype==torch.float32 for s in bank.memory)
        assert sum(s.abs().sum() for s in bank.memory)>0
        loss=float(result['loss']);del result;model.zero_grad(set_to_none=True);model.eval()
        following=[dict(m,reset=False,curr_time_org=100000) for m in metas]
        with torch.no_grad():
            f,st=model.backbone(events,torch.stack(images) if modality=='rgbdvs' else None,bank.select(following))
            fresh,z=model.backbone(events,torch.stack(images) if modality=='rgbdvs' else None,model.backbone.zero_temporal_state(2))
        shapes=[list(x.shape) for x in f]
        assert [x.shape[1] for x in f]==[256]*4
        change=max(float((x-y).abs().max()) for x,y in zip(f,fresh));assert change>0
        reset=[dict(m,sequence='new'+m['sequence']) for m in following]
        assert all(torch.count_nonzero(x)==0 for x in bank.select(reset))
        if task=='depth':
            # An unsupervised observation advances all seven states without loss.
            with torch.no_grad():empty=model(events,images,following,[torch.zeros_like(t) for t in targets],temporal_state=bank.select(following))
            assert empty['skip_step'] and len(empty['temporal_state'])==7
        report[kind]=dict(nonstem_initialization_equal=True,rng_equal=True,loss=loss,shapes=shapes,history_max_delta=change,state_shapes=[list(s.shape) for s in st],state_dtype='float32',empty_detection_background=True if task=='detection' else None)
        del model,b,bank,f,st,fresh,z,events,images,args_forward
        if device.type=='cuda':torch.cuda.empty_cache()
        print(kind,'passed',flush=True)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--output',default='artifacts/validation/invariants.json');checks(p.parse_args())
