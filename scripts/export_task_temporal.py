#!/usr/bin/env python3
"""Explicit seven-state M=2 ONNX, with independent backend state trajectories."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort
from hmnet.models.efficientvit_tasks import build_frame_task,ROOT
from hmnet.dataset.task_frames import TaskFrames,sha_file
from scripts.export_temporal_onnx import compare

class Graph(torch.nn.Module):
    def __init__(self,model,task,height,width,backbone=False):
        super().__init__();self.model=model;self.task=task;self.backbone_only=backbone
        self.metas=[dict(height=height,width=width)]
    def forward(self,events,*args):
        rgb=args[0] if self.model.backbone.use_rgb else None
        previous=args[1:] if self.model.backbone.use_rgb else args
        features,state=self.model.backbone(events,rgb,previous)
        if self.backbone_only:outputs=tuple(features)
        else:
            features=self.model.neck(list(features))
            outputs=(self.model.bbox_head.inference(features) if self.task=='detection' else self.model.reg_head.inference(features,self.metas),)
        return (*outputs,*state)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--data-root',required=True);p.add_argument('--split',required=True);p.add_argument('--output',required=True)
    p.add_argument('--backbone-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);c=saved['training_contract']
    if c.get('temporal',{}).get('window')!=2:raise ValueError('Explicit M=2 checkpoint required')
    kind=c['event_data']['kind'];rep=c['event_input']['representation'];task=c['task']
    data=TaskFrames(a.data_root,a.split,rep,limit=3)
    if data.kind!=kind or data.modality!=c['modality']:raise ValueError('Export data/model mismatch')
    if len(data)<3 or len({s['sequence'] for s in data.samples})!=1:raise ValueError('Need three same-sequence real observations')
    model=build_frame_task(task,mvsec=kind=='mvsec',modality=c['modality'],temporal_window=2,event_channels=c['event_input']['channels']).eval()
    model.load_state_dict(saved['state_dict'],strict=True)
    wrapper=Graph(model,task,data.height,data.width,a.backbone_only).eval();frames=[]
    for i in range(3):
        inputs,_,_=data[i]
        frames.append((inputs[None],) if task=='detection' else ((inputs['events'][None],inputs['images'][None]) if model.backbone.use_rgb else (inputs['events'][None],)))
    states=model.backbone.zero_temporal_state(1)
    inputs=['event_hist']+(['rgb'] if model.backbone.use_rgb else [])+[f'previous_{i}' for i in range(7)]
    outputs=(['f4','f8','f16','f32'] if a.backbone_only else ['detections_xywh_objectness_classes' if task=='detection' else 'depth_meters'])+[f'current_{i}' for i in range(7)]
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);path=out/('backbone.onnx' if a.backbone_only else 'model.onnx')
    with torch.no_grad():torch.onnx.export(wrapper,(*frames[0],*states),str(path),input_names=inputs,output_names=outputs,opset_version=17,do_constant_folding=True,dynamo=False)
    onnx.checker.check_model(onnx.load(path),full_check=True)
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    zeros=tuple(torch.zeros_like(x) for x in frames[0]);checks=[]
    cases=[('real_sequence',frames),('all_zero',[zeros]*3),('reset_replay',[frames[0]])]
    if model.backbone.use_rgb:cases.append(('empty_event',[(torch.zeros_like(f[0]),f[1]) for f in frames]))
    for name,sequence in cases:
        ptstate=states;ortstate=[x.numpy() for x in states]
        for i,frame in enumerate(sequence):
            # Respect real sequence gap/reset policy on both backends.
            reset=(name in ('real_sequence','empty_event') and i>0 and
                   data.samples[i]['target_us']-data.samples[i-1]['target_us']>data.reset_gap_us)
            if reset:ptstate=states;ortstate=[x.numpy() for x in states]
            with torch.no_grad():want=wrapper(*frame,*ptstate)
            got=session.run(outputs,dict(zip(inputs,[x.numpy() for x in frame]+ortstate)))
            checks.extend(compare(x.numpy(),y,n,'original',f'{name}/{i}') for n,x,y in zip(outputs,want,got))
            ptstate=tuple(x.detach() for x in want[-7:]);ortstate=got[-7:]
    report=dict(structure_passed=True,numerical_passed=all(x['passed'] for x in checks),deployment_accepted=False,
        failures=[f"{x['case']}/{x['output']}" for x in checks if not x['passed']],checks=checks,
        task=task,dataset=kind,modality=c['modality'],event_input=c['event_input'],temporal=c['temporal'],
        weights=dict(checkpoint=str(Path(a.checkpoint).resolve()),sha256=sha_file(a.checkpoint),step=saved['step'],provenance='bounded diagnostic updates, not converged task weights'),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)):sha_file(p) for base in ('hmnet','scripts') for p in (ROOT/base).rglob('*.py')},
        precision='FP32 inference; BF16 training is separate',state_feedback='independent backend trajectories',
        inputs={n:list(x.shape) for n,x in zip(inputs,(*frames[0],*states))},outputs={n:list(x.shape) for n,x in zip(outputs,want)})
    path.with_suffix('.report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(path,report['numerical_passed'],report['failures'],flush=True)
    if not report['numerical_passed']:raise SystemExit(2)

if __name__=='__main__':main()
