#!/usr/bin/env python3
"""Input ablation invariants, canonical initialization and optional real-cache audit."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import h5py
import numpy as np
import torch
from hmnet.models.base.event_repr.polarity import polarity_counts, input_spec, validate_input_spec, validate_counts
from hmnet.models.efficientvit_tasks import build_frame_task, ROOT
from hmnet.dataset.dsec_frames import DSECFrames
from hmnet.dataset.temporal_frames import SequenceBatchSampler, manifest_signature
from scripts.prepare_dsec_b1 import read_window


def reject(fn):
    try: fn()
    except (ValueError, OverflowError): return
    raise AssertionError("Invalid input was accepted")


def main(args):
    torch.set_num_threads(2)
    checks = []
    empty = np.empty((0,4), np.int64)
    assert not polarity_counts(empty,2,3).any()
    events = np.array([[10,1,0,0]]*300 + [[11,1,0,1]]*513 + [[12,2,1,1]], np.int64)
    counts = polarity_counts(events,2,3)
    assert counts.dtype == np.int32 and counts[0,0,1] == 300 and counts[1,0,1] == 513
    assert counts[1,1,2] == 1 and counts.sum() == len(events)
    assert (counts > 0).sum() == 3
    reject(lambda: polarity_counts(events.astype(np.int32),2,3))
    reject(lambda: polarity_counts(np.array([[0,3,0,0]],np.int64),2,3))
    reject(lambda: polarity_counts(np.array([[0,0,0,2]],np.int64),2,3))
    validate_counts(counts,2,3)
    reject(lambda: validate_counts(counts.astype(np.float32),2,3))
    reject(lambda: validate_counts(-counts,2,3))
    reject(lambda: validate_counts(np.full((2,2,3),2**24+1,np.int32),2,3))
    checks.append("empty, multiplicity>255, polarity separation, binary occupancy, invalid events/count dtype/range")
    with tempfile.TemporaryDirectory() as d:
        with h5py.File(Path(d)/"events.h5","w") as f:
            t=np.array([9999,10000,10001,59999,60000,60001],np.int64)
            f['t_offset']=123456;f['events/t']=t
            f['events/x']=np.zeros(len(t),np.int16)
            f['events/y']=np.array([0,0,439,440,0,0],np.int16)
            f['events/p']=np.array([0,0,1,0,1,0],np.int8)
            f['ms_to_idx']=np.searchsorted(t,np.arange(62)*1000)
            selected=read_window(f,123456+60000)
            assert selected[:,0].tolist()==[10000,10001,59999,60000]
            cropped=polarity_counts(selected[selected[:,2]<440])
            assert cropped.sum()==3 and cropped[1,439,0]==1
    checks.append("parent closed 50ms interval, t_offset, y<440 crop")
    for rep in ("polarity_binary","polarity_count"):
        validate_input_spec(input_spec(rep),rep)
        reject(lambda: validate_input_spec(None,rep))
        other="polarity_count" if rep=="polarity_binary" else "polarity_binary"
        reject(lambda: validate_input_spec(input_spec(other),rep))
    validate_input_spec(None,"rvt_histogram")
    checks.append("same-shape cross-representation and missing contract rejection")
    baseline_factory=build_frame_task
    if args.parent_project:
        location=Path(args.parent_project)/"hmnet/models/efficientvit_tasks.py"
        spec=importlib.util.spec_from_file_location("original_parent_task_factory",location)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        baseline_factory=module.build_frame_task
    kwargs=dict(pretrained=str(ROOT/'pretrained/efficientvit_b1_r224.pth'),
                modality='rgbdvs',fusion_mode='add',temporal_window=2)
    torch.manual_seed(42); parent=baseline_factory('segmentation',**kwargs)
    parent_rng=torch.get_rng_state().clone()
    torch.manual_seed(42); binary=build_frame_task('segmentation',event_channels=2,**kwargs)
    assert torch.equal(parent_rng,torch.get_rng_state())
    torch.manual_seed(42); count=build_frame_task('segmentation',event_channels=2,**kwargs)
    stem='backbone.event_encoder.input_stem.op_list.0.conv.weight'
    same=0
    for key,value in parent.state_dict().items():
        actual=binary.state_dict()[key]
        if key!=stem:
            assert torch.equal(value,actual),key;same+=1
        else: assert value.shape[1]==20 and actual.shape[1]==2
        assert torch.equal(actual,count.state_dict()[key]),key
    torch.testing.assert_close(binary.state_dict()[stem],
        binary.state_dict()['backbone.rgb_encoder.input_stem.op_list.0.conv.weight'].mean(1,keepdim=True).repeat(1,2,1,1)*1.5,
        atol=0,rtol=0)
    for a,b in zip(parent.backbone.zero_temporal_state(1),binary.backbone.zero_temporal_state(1)):
        assert a.shape==b.shape and a.dtype==b.dtype==torch.float32
    checks.append(f"{same} non-stem tensors equal parent; both ablations identical; RNG and seven state shapes preserved")
    report=dict(passed=True, checks=checks, parent_project=args.parent_project,
                parameters=sum(p.numel() for p in binary.parameters()),
                parent_parameters=sum(p.numel() for p in parent.parameters()),
                input_spec=input_spec('polarity_binary'),
                pretrained_sha256=hashlib.sha256((ROOT/'pretrained/efficientvit_b1_r224.pth').read_bytes()).hexdigest())
    if args.cache:
        b=DSECFrames(args.cache,'train',augment=True,representation='polarity_binary')
        c=DSECFrames(args.cache,'train',augment=True,representation='polarity_count')
        r=DSECFrames(b.asset_root,'train',augment=True)
        if not b.data_contract['diagnostic_subset']:
            assert manifest_signature(b)==manifest_signature(r)
            assert list(SequenceBatchSampler(b,32))==list(SequenceBatchSampler(r,32))
        for i in np.linspace(0,len(b)-1,min(8,len(b)),dtype=int):
            # Explicit temporal key fixes augmentation and allows an exact RGB/GT comparison.
            key=(int(i),0,True,True)
            bd,bt,bm=b[key];cd,ct,cm=c[key]
            parent_index=next(j for j,s in enumerate(r.samples) if s['file']==b.samples[i]['file'])
            rd,rt,rm=r[(parent_index,0,True,True)]
            assert torch.equal(bd['events'],(cd['events']>0).float())
            assert torch.equal(bd['images'],cd['images']) and torch.equal(bd['images'],rd['images'])
            assert torch.equal(bt['labels'],ct['labels']) and torch.equal(bt['labels'],rt['labels'])
            assert bm['image_meta']['curr_time_org']==rm['image_meta']['curr_time_org']
            with np.load(r.root/r.samples[parent_index]['file']) as f:
                assert torch.equal(rd['events'],torch.from_numpy(f['histogram'].copy()).float().flip(-1))
        checks.append("real binary=count>0, identical RGB/GT/timestamps/augmentation, original RVT loading unchanged")
        report['data_contract']=b.data_contract
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-project');p.add_argument('--cache')
    p.add_argument('--output',default='artifacts/polarity_implementation/input_checks.json')
    main(p.parse_args())
