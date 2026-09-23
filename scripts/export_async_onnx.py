#!/usr/bin/env python3
"""FP32 RGB-update, DVS-prediction and backbone graphs with explicit cache/state.

Numerical failure does not prevent retaining all graphs and diagnostic reports.
Tolerance is inherited from temporal export: atol=1e-3, rtol=1e-4.
"""
import argparse,json,subprocess
from pathlib import Path
from collections import Counter
import numpy as np
import onnx
import onnxruntime as ort
from onnxsim import simplify
import torch
from torch import nn
from hmnet.dataset.dsec_async import DSECAsync
from hmnet.models.efficientvit_tasks import build_frame_task
from hmnet.utils.async_checkpoint import load_async_model,file_sha256
from scripts.export_temporal_onnx import compare


class RGBGraph(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,rgb):return self.model.backbone.encode_rgb(rgb)


class EventGraph(nn.Module):
    def __init__(self,model,backbone=False):
        super().__init__();self.model=model;self.backbone=backbone
        self.meta=[dict(height=440,width=640)]
    def forward(self,event_hist,*inputs):
        features,state=self.model.backbone.async_step(event_hist,inputs[:4],inputs[4:])
        output=features if self.backbone else (self.model.seg_head(self.model.neck(list(features)),self.meta),)
        return (*output,*state)


def export_graph(wrapper,inputs,names,outputs,path):
    with torch.no_grad():
        torch.onnx.export(wrapper,inputs,str(path),opset_version=17,input_names=names,
                          output_names=outputs,do_constant_folding=True,dynamo=False)
    original=onnx.load(path)
    with torch.no_grad():actual_outputs=wrapper(*inputs)
    # Fixed-size diagnostic graphs: annotate the measured output shapes, rather
    # than reporting protobuf dim_value=0 for exporter-generated symbolic dims.
    for info,value in zip(original.graph.output,actual_outputs):
        for dim,size in zip(info.type.tensor_type.shape.dim,value.shape):dim.dim_value=int(size)
    onnx.checker.check_model(original,full_check=True);onnx.save(original,path)
    simple,valid=simplify(original,check_n=0)
    if not valid:raise RuntimeError('Simplification failed')
    onnx.checker.check_model(simple,full_check=True)
    simple_path=path.with_name(path.stem+'.sim.onnx');onnx.save(simple,simple_path)
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    sessions=[ort.InferenceSession(str(p),options,providers=['CPUExecutionProvider']) for p in (path,simple_path)]
    return sessions,dict(inputs={n:list(t.shape) for n,t in zip(names,inputs)},
        outputs={o.name:[d.dim_value for d in o.type.tensor_type.shape.dim] for o in original.graph.output},
        ops=dict(Counter(n.op_type for n in original.graph.node)),onnx_sha256=file_sha256(path))


def main(args):
    torch.set_num_threads(2);torch.manual_seed(42)
    if args.checkpoint:
        model,contract=load_async_model(args.checkpoint)
        ac=contract['asynchronous'];window,bins=ac['window_us'],ac['bins']
        source=dict(kind='early smoke checkpoint or user-provided trained checkpoint',path=args.checkpoint,
                    sha256=file_sha256(args.checkpoint))
    else:
        window,bins=(50000,10) if args.variant=='B' else (25000,5)
        model=build_frame_task('segmentation',modality='rgbdvs',fusion_mode='add',temporal_window=2,event_channels=2*bins).eval()
        source=dict(kind='random initialization, seed42; structure diagnostic only')
    data=DSECAsync(args.data_root,'dev',window_us=window,bins=bins,limit=4)
    if len(data)<4:raise ValueError('Need four real consecutive event steps')
    frames=[data[i] for i in range(4)]
    e=frames[0][0]['events'][None];rgb=next(x[0]['images'][None] for x in frames if x[2]['image_meta']['rgb_valid'])
    model.eval();memory=model.backbone.zero_temporal_state(1)
    with torch.no_grad():cache=model.backbone.encode_rgb(rgb)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cache_names=[f'rgb_cache_{i}' for i in range(4)]
    state_names=[f'previous_{i}' for i in range(7)]
    next_names=[f'current_{i}' for i in range(7)]
    rgb_sessions,rgb_info=export_graph(RGBGraph(model),(rgb,),['rgb'],cache_names,out/'rgb_update.onnx')
    infos={'rgb_update':rgb_info};checks=[]
    for kind,backbone_only in [('event_prediction',False),('event_backbone',True)]:
        wrapper=EventGraph(model,backbone_only).eval()
        names=['event_hist']+cache_names+state_names
        outputs=(['f4','f8','f16','f32'] if backbone_only else ['logits'])+next_names
        sessions,infos[kind]=export_graph(wrapper,(e,*cache,*memory),names,outputs,out/(kind+'.onnx'))
        for case in ('real_sequence','all_zero','cold_zero_cache','empty_event','reset_replay'):
            torch_state=tuple(torch.zeros_like(x) for x in memory)
            torch_cache=tuple(torch.zeros_like(x) for x in cache)
            ort_states=[[x.numpy().copy() for x in torch_state] for _ in sessions]
            ort_caches=[[x.numpy().copy() for x in torch_cache] for _ in sessions]
            old_id=None
            count=1 if case=='reset_replay' else len(frames)
            for step,frame in enumerate(frames[:count]):
                meta=frame[2]['image_meta'];events=frame[0]['events'][None]
                if case in ('all_zero','cold_zero_cache','empty_event'):events=torch.zeros_like(events)
                update_rgb = ((case=='all_zero' and step==0) or
                              (case not in ('all_zero','cold_zero_cache') and meta['rgb_valid'] and meta['rgb_id']!=old_id))
                if update_rgb:
                    image=torch.zeros_like(rgb) if case=='all_zero' else frame[0]['images'][None]
                    with torch.no_grad():torch_cache=model.backbone.encode_rgb(image)
                    ort_caches=[session.run(cache_names,{'rgb':image.numpy()}) for session in rgb_sessions]
                    for j,want in enumerate(torch_cache):
                        for backend,values in zip(('original','simplified'),ort_caches):
                            checks.append(compare(want.numpy(),values[j],cache_names[j],backend,f'{kind}/{case}/{step}/rgb'))
                    old_id=meta['rgb_id']
                with torch.no_grad():expected=wrapper(events,*torch_cache,*torch_state)
                actual=[session.run(outputs,dict(zip(names,[events.numpy()]+cached+state)))
                        for session,cached,state in zip(sessions,ort_caches,ort_states)]
                for j,name in enumerate(outputs):
                    for backend,values in zip(('original','simplified'),actual):
                        checks.append(compare(expected[j].numpy(),values[j],name,backend,f'{kind}/{case}/{step}',name=='logits'))
                    checks.append(compare(actual[0][j],actual[1][j],name,'simplification',f'{kind}/{case}/{step}',atol=1e-5))
                torch_state=tuple(x.detach() for x in expected[-7:]);ort_states=[v[-7:] for v in actual]
        del sessions
    failures=[f'{x["case"]}/{x["graph"]}/{x["output"]}' for x in checks if not x['passed']]
    files=['hmnet/models/base/backbone/efficientvit_b1.py','hmnet/models/efficientvit_tasks.py',
           'hmnet/models/async_frame.py','hmnet/models/base/backbone/temporal_litemla.py','scripts/export_async_onnx.py']
    result=dict(structure_passed=True,numerical_passed=not failures,failures=failures,checks=checks,
        graphs=infos,weights=source,window_us=window,bins=bins,inference_precision='FP32',
        state_feedback='independent PyTorch/original ORT/simplified ORT RGB caches and 7-state trajectories',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_worktree_status=subprocess.check_output(['git','status','--short'],text=True),
        source_sha256={name:file_sha256(name) for name in files})
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','graphs')},indent=2))
    if failures:raise SystemExit('Numerical checks failed; original thresholds and all reports/graphs retained')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint');p.add_argument('--variant',choices=('B','C'),default='B')
    p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    main(p.parse_args())
