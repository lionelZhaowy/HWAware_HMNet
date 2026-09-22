#!/usr/bin/env python3
"""Export explicit-state FP32 M=2 graphs; independent PyTorch/ORT state trajectories.

Mixed precision is the TRAINING contract. These diagnostic graphs use FP32
inference, not BF16 execution, quantization or Dremi deployment validation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import onnx
import onnxruntime as ort
from onnxsim import simplify
import torch
from hmnet.dataset.dsec_frames import DSECFrames
from hmnet.models.efficientvit_tasks import build_frame_task
from scripts.export_b1_onnx import FrameGraph


class TemporalGraph(FrameGraph):
    def __init__(self, model, backbone_only=False):
        super().__init__(model, "segmentation", 440, 640, backbone_only)

    def forward(self, event_hist, rgb, *previous):
        features, current = self.model.backbone(event_hist, rgb, previous)
        outputs = (tuple(features) if self.backbone_only else
                   (self.model.seg_head(self.model.neck(list(features)), self.metas),))
        return (*outputs, *current)


def compare(want, got, name, graph, case, logits=False, atol=1e-3):
    finite=bool(np.isfinite(want).all() and np.isfinite(got).all())
    error=np.abs(want-got)
    passed=finite and bool(np.allclose(want,got,atol=atol,rtol=1e-4))
    result=dict(case=case,graph=graph,output=name,finite=finite,passed=passed,
        max_abs_error=float(error.max()),mean_abs_error=float(error.mean()),
        mismatch_fraction=float((~np.isfinite(error) | (error>atol+1e-4*np.abs(want))).mean()),
        atol=atol,rtol=1e-4)
    if logits:
        result['argmax_agreement']=float((want.argmax(1)==got.argmax(1)).mean())
        result['passed'] &= result['argmax_agreement'] >= .9999
    return result


def export(args):
    torch.set_num_threads(2);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    contract=ckpt.get('training_contract',{})
    temporal=contract.get('temporal',{})
    if temporal.get('window')!=2 or temporal.get('branch')!='dvs':
        raise ValueError('A matching M=2 DVS temporal training checkpoint is required')
    mode=contract['fusion_mode']
    model=build_frame_task('segmentation',modality='rgbdvs',fusion_mode=mode,temporal_window=2).eval()
    model.load_state_dict(ckpt['state_dict'],strict=True)
    wrapper=TemporalGraph(model,args.backbone_only).eval()
    data=DSECFrames(args.data_root,'dev',limit=3)
    frames=[]
    for i in range(len(data)):
        inputs,_,_=data[i];frames.append((inputs['events'][None],inputs['images'][None]))
    if len(frames)!=3 or len({s['sequence'] for s in data.samples})!=1:
        raise ValueError('Need three consecutive dev frames from one sequence')
    memory=model.backbone.zero_temporal_state(1)
    input_names=['event_hist','rgb']+[f'previous_{i}' for i in range(7)]
    output_names=(['f4','f8','f16','f32'] if args.backbone_only else ['logits'])+[f'current_{i}' for i in range(7)]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    stem='segmentation_temporal'+('_backbone' if args.backbone_only else '')
    path=out/(stem+'.onnx');simple=out/(stem+'.sim.onnx')
    with torch.no_grad():
        torch.onnx.export(wrapper,(*frames[0],*memory),str(path),input_names=input_names,
            output_names=output_names,opset_version=17,do_constant_folding=True,dynamo=False)
    original=onnx.load(path);onnx.checker.check_model(original,full_check=True)
    simplified,ok=simplify(original,check_n=0)
    if not ok:raise RuntimeError('onnxsim failed')
    onnx.checker.check_model(simplified,full_check=True);onnx.save(simplified,simple)
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    sessions=[ort.InferenceSession(str(p),options,providers=['CPUExecutionProvider']) for p in (path,simple)]
    checks=[]
    zeros=tuple(torch.zeros_like(t) for t in frames[0])
    cases=[('real_sequence',frames),('all_zero',[zeros,zeros,zeros]),
           ('empty_event',[(torch.zeros_like(e),r) for e,r in frames]),
           ('reset_replay',[frames[0]])]
    for case,sequence in cases:
        torch_state=tuple(x.clone() for x in memory)
        ort_states=[[x.numpy().copy() for x in memory] for _ in sessions]
        for step,frame in enumerate(sequence):
            with torch.no_grad():expected=wrapper(*frame,*torch_state)
            values=[]
            for sess,state in zip(sessions,ort_states):
                feed=dict(zip(input_names,[x.numpy() for x in frame]+state))
                values.append(sess.run(output_names,feed))
            for j,name in enumerate(output_names):
                want=expected[j].numpy()
                for label,got in zip(['original','simplified'],[v[j] for v in values]):
                    checks.append(compare(want,got,name,label,f'{case}/{step}',name=='logits'))
                checks.append(compare(values[0][j],values[1][j],name,'simplification',
                                      f'{case}/{step}',atol=1e-5))
            torch_state=tuple(x.detach() for x in expected[-7:])
            ort_states=[v[-7:] for v in values]
    failures=[f"{c['case']}/{c['graph']}/{c['output']}" for c in checks if not c['passed']]
    report=dict(passed=not failures,failures=failures,checks=checks,fusion_mode=mode,
        temporal=temporal,checkpoint=str(Path(args.checkpoint).resolve()),step=ckpt.get('step'),
        checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        inference_precision='FP32',training_precision=contract['precision'],
        samples=[s['file'] for s in data.samples],state_feedback='independent trajectories for each backend',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_worktree_status=subprocess.check_output(['git','status','--short'],text=True),
        source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ['hmnet/models/base/backbone/temporal_litemla.py',
             'hmnet/models/base/backbone/efficientvit_b1.py',
             'hmnet/models/base/backbone/cross_modal_litemla.py',
             'hmnet/models/efficientvit_tasks.py','scripts/export_temporal_onnx.py']},
        inputs={n:list(t.shape) for n,t in zip(input_names,(*frames[0],*memory))},
        outputs={n:list(t.shape) for n,t in zip(output_names,expected)},
        nodes={label:dict(Counter(n.op_type for n in graph.graph.node)) for label,graph in
               [('original',original),('simplified',simplified)]})
    (out/(stem+'.report.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
    if failures:raise RuntimeError('Temporal ONNX numerical checks failed; diagnostic graphs/reports retained')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--data-root',default='/home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/dsec_b1')
    p.add_argument('--backbone-only',action='store_true')
    export(p.parse_args())
