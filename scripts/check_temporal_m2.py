#!/usr/bin/env python3
"""CPU regression checks for M=2 algebra, gradients, stream order and isolation."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from hmnet.models.base.backbone.temporal_litemla import attention_step, encoder_step
from hmnet.models.base.backbone.vendor.efficientvit.models.nn.ops import LiteMLA
from hmnet.models.base.backbone.efficientvit_b1 import EfficientViTB1
from hmnet.dataset.temporal_frames import SequenceBatchSampler
from hmnet.utils.temporal_streams import TemporalStreams


def main():
    torch.set_num_threads(1); torch.manual_seed(42)
    results={}
    m=LiteMLA(32,32,dim=16).eval()
    x=torch.randn(1,32,8,10,requires_grad=True)
    z=torch.zeros(1,4,17,16)
    y,c=attention_step(m,x,z)
    torch.testing.assert_close(y,m(x),atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(y,attention_step(m,x,z)[0],atol=0,rtol=0)
    history=torch.randn_like(z)*.1; history[:,:,-1].abs_()
    y2,c2=attention_step(m,x,history)
    torch.testing.assert_close(c,c2,atol=0,rtol=0)
    assert not torch.allclose(y,y2)
    # Explicit token-row attention with the previous frame's actual K/V.
    previous_x=torch.randn_like(x)
    def qkv(value):
        base=m.qkv(value);packed=torch.cat([base]+[op(base) for op in m.aggreg],1)
        q,k,v=packed.reshape(1,4,48,-1).split(16,2)
        return q.relu(),k.relu(),v
    q,k,v=qkv(x);_,kp,vp=qkv(previous_x)
    _,cp=attention_step(m,previous_x,z)
    scores=torch.cat([k,kp],-1).transpose(-1,-2)@q
    expected=(torch.cat([v,vp],-1)@scores)/(scores.sum(-2,keepdim=True)+m.eps)
    expected=m.proj(expected.reshape(1,64,8,10))
    actual,_=attention_step(m,x,cp)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    actual.square().mean().backward()
    assert x.grad is not None and x.grad.abs().sum()>0 and m.qkv.conv.weight.grad.abs().sum()>0
    # Reset is functional: no implicit state survives a forward.
    torch.testing.assert_close(attention_step(m,x,z)[0],y,atol=0,rtol=0)
    results['attention']='M=2 explicit-token equivalence; gradient; current-only state; reset passed'
    for mode in ['add','cross_stage_post_mbconv_no_feedback']:
        torch.manual_seed(31)
        base=EfficientViTB1(modality='rgbdvs',fusion_mode=mode).eval()
        torch.manual_seed(31)
        temporal=EfficientViTB1(modality='rgbdvs',fusion_mode=mode,temporal_window=2).eval()
        assert base.state_dict().keys()==temporal.state_dict().keys()
        for name,t in base.state_dict().items():
            torch.testing.assert_close(t,temporal.state_dict()[name],atol=0,rtol=0)
        e=torch.randn(1,20,128,160);r=torch.randn(1,3,128,160)
        state=temporal.zero_temporal_state(1)
        with torch.no_grad():
            old=base(e,r);new,ns=temporal(e,r,state)
            for u,v in zip(old,new):torch.testing.assert_close(u,v,atol=3e-6,rtol=3e-5)
            assert len(ns)==7 and all(v.dtype==torch.float32 for v in ns)
            assert sum(v.numel()*v.element_size() for v in ns)==187*1024
            _,changed_rgb=temporal(e,r+1,state)
            for u,v in zip(ns,changed_rgb):torch.testing.assert_close(u,v,atol=0,rtol=0)
            _, different_history=temporal(torch.randn_like(e)*3,r,state)
            second,_=temporal(e,r,different_history)
            assert any(not torch.allclose(u,v) for u,v in zip(new,second))
            reset,_=temporal(e,r,state)
            for u,v in zip(new,reset):torch.testing.assert_close(u,v,atol=0,rtol=0)
        bank=TemporalStreams(temporal,2)
        metas=[dict(stream_slot=1,sequence='a',curr_time_org=50000,flipped=False,reset=True)]
        bank.commit(metas,ns)
        nextmeta=[dict(metas[0],curr_time_org=100000,reset=False)]
        selected=bank.select(nextmeta)
        for u,v in zip(selected,ns):torch.testing.assert_close(u,v,atol=0,rtol=0)
        other=[dict(nextmeta[0],sequence='b')]
        assert all(torch.count_nonzero(v)==0 for v in bank.select(other))
        saved=bank.state_dict();restored=TemporalStreams(temporal,2);restored.load_state_dict(saved)
        for u,v in zip(bank.select(nextmeta),restored.select(nextmeta)):
            torch.testing.assert_close(u,v,atol=0,rtol=0)
        results[mode]='parent state_dict/init; zero-history equivalence; 7 FP32 states; RGB independence; stream reset/save/restore passed'
    samples=[]
    for seq,n in [('a',23),('b',35),('c',19)]:
        samples.extend(dict(sequence=seq,target_us=i*50000,file=f'{seq}/{i}') for i in range(n))
    class Dataset:
        augment=True
        def __init__(self):self.samples=samples
        def __len__(self):return len(self.samples)
    data=Dataset()
    def verify(batches):
        visited=[];last={}
        for batch in batches:
            for i,slot,reset,flip in batch:
                s=samples[i];visited.append(i)
                if not reset:
                    old,oldflip=last[slot]
                    assert s['sequence']==old['sequence'] and s['target_us']-old['target_us']==50000
                    assert oldflip==flip
                last[slot]=(s,flip)
        return visited
    sampler=SequenceBatchSampler(data,4)
    full=list(sampler); assert len(full)==20 and len(full[-1])==1
    assert sorted(verify(full))==list(range(len(data)))
    assert full==list(sampler)
    sampler.set_epoch(1); assert list(sampler)!=full
    # DDP partitions only concurrent lanes; no frame duplication or interleaving a lane.
    ddp=[]
    for rank in range(2):ddp.extend(verify(SequenceBatchSampler(data,3*2,rank,2)))
    assert sorted(ddp)==list(range(len(data)))
    for rank in range(2):
        batches=list(SequenceBatchSampler(data,4,rank,2,training=False))
        verify(batches)
        # No internal scene splits: one cold start per assigned full sequence.
        assert sum(key[2] for batch in batches for key in batch)==len(['a','b','c'][rank::2])
    results['sampler']='one visit/frame; consecutive lanes; consistent flip; deterministic epochs; rank partition; whole-sequence validation passed'
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
