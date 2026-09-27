#!/usr/bin/env python3
"""Export PEOD RGB/DVS/fusion at 240x304, retaining explicit FP32 DVS memory."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort
from hmnet.models.efficientvit_tasks import build_frame_task,ROOT
from hmnet.dataset.peod_frames import PEODFrames
from hmnet.dataset.task_frames import sha_file
from scripts.export_temporal_onnx import compare

class Graph(torch.nn.Module):
    def __init__(self,model,backbone):
        super().__init__();self.model=model;self.backbone=backbone
    def forward(self,*args):
        b=self.model.backbone;i=0;event=None;rgb=None
        if b.use_events:event=args[i];i+=1
        if b.use_rgb:rgb=args[i];i+=1
        f=b(event,rgb,tuple(args[i:]) if b.temporal_window else None)
        if b.temporal_window:f,state=f
        else:state=()
        outputs=tuple(f) if self.backbone else (self.model.bbox_head.inference(self.model.neck(list(f))),)
        return (*outputs,*state)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--data-root',required=True)
    p.add_argument('--output',required=True);p.add_argument('--backbone-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);c=saved['training_contract']
    if c['event_data']['kind']!='peod':raise ValueError('PEOD checkpoint required')
    modality=c['modality'];rep=c['event_input']['representation'];window=c.get('temporal',{}).get('window',0)
    data=PEODFrames(a.data_root,'train','rvt_histogram' if modality=='rgb' else rep,modality,limit=3)
    if data.manifest['geometry']!=c['event_data']['geometry']:raise ValueError('Export geometry differs')
    model=build_frame_task('detection',modality=modality,temporal_window=window,
        event_channels=20 if modality=='rgb' else c['event_input']['channels'],num_classes=6).eval()
    model.load_state_dict(saved['state_dict'],strict=True);wrapper=Graph(model,a.backbone_only).eval()
    frames=[]
    for i in range(3):
        d,_,_=data[i];frames.append(tuple(d[k][None] for k in ('events','images') if d[k] is not None))
    states=model.backbone.zero_temporal_state(1) if window else ()
    inputs=(['event_hist'] if model.backbone.use_events else [])+(['rgb'] if model.backbone.use_rgb else [])+[f'previous_{i}' for i in range(len(states))]
    outputs=(['f4','f8','f16','f32'] if a.backbone_only else ['detections_xywh_objectness_classes'])+[f'current_{i}' for i in range(len(states))]
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);path=out/('backbone.onnx' if a.backbone_only else 'model.onnx')
    with torch.no_grad():torch.onnx.export(wrapper,(*frames[0],*states),str(path),input_names=inputs,output_names=outputs,
        opset_version=17,do_constant_folding=True,dynamo=False)
    graph=onnx.load(path);onnx.checker.check_model(graph,full_check=True)
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    zero=tuple(torch.zeros_like(x) for x in frames[0]);cases=[('real_sequence',frames),('all_zero',[zero]*3),('reset_replay',[frames[0]])]
    if modality=='rgbdvs':cases.append(('empty_event',[(torch.zeros_like(f[0]),f[1]) for f in frames]))
    checks=[]
    for name,sequence in cases:
        ptstate=states;ortstate=[x.numpy() for x in states]
        for i,frame in enumerate(sequence):
            if i and data.samples[i]['target_us']-data.samples[i-1]['target_us']>data.reset_gap_us:
                ptstate=states;ortstate=[x.numpy() for x in states]
            with torch.no_grad():want=wrapper(*frame,*ptstate)
            got=session.run(outputs,dict(zip(inputs,[x.numpy() for x in frame]+ortstate)))
            checks.extend(compare(x.numpy(),y,n,'original',f'{name}/{i}') for n,x,y in zip(outputs,want,got))
            if states:ptstate=tuple(x.detach() for x in want[-len(states):]);ortstate=got[-len(states):]
    report=dict(structure_passed=True,numerical_passed=all(x['passed'] for x in checks),deployment_accepted=False,
        checks=checks,failures=[f"{x['case']}/{x['output']}" for x in checks if not x['passed']],
        input_shapes={n:list(x.shape) for n,x in zip(inputs,(*frames[0],*states))},output_shapes={n:list(x.shape) for n,x in zip(outputs,want)},
        modality=modality,temporal_window=window,state_feedback='independent_PyTorch_ORT_trajectories',precision='CPU_FP32',
        weights=dict(path=str(Path(a.checkpoint).resolve()),sha256=sha_file(a.checkpoint),step=saved['step'],source='bounded_smoke_official_B1_initialization'),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)):sha_file(p) for base in ('hmnet','scripts') for p in (ROOT/base).rglob('*.py')},
        nodes=len(graph.graph.node),batch_normalization_nodes=sum(n.op_type=='BatchNormalization' for n in graph.graph.node))
    path.with_suffix('.report.json').write_text(json.dumps(report,indent=2));print(path,report['numerical_passed'],flush=True)
    if not report['numerical_passed']:raise SystemExit(2)

if __name__=='__main__':main()
