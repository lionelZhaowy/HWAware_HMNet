#!/usr/bin/env python3
"""共有组件权重核对、原生特征/检测网格、高目标真实更新和时序有效性。"""
import argparse,json,hashlib,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import torch
from scripts.peod import parser,configuration,score_dump
from hmnet.dataset.peod_frames import PEODFrames,collate_peod
from hmnet.dataset.temporal_frames import SequenceBatchSampler
from hmnet.utils.frame_train import unpack,resolve_training_updates,training_schedule,learning_rate_at
from hmnet.utils.temporal_streams import TemporalStreams

def digest(named):
 h=hashlib.sha256()
 for name,t in sorted(named):h.update(name.encode());h.update(t.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 args=parser().parse_args(['train','--stop-after','2']);c=configuration(args);data=c.get_dataset();model=c.get_model().cuda()
 initial={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}
 modules={k:digest([(n.removeprefix(k+'.'),v) for n,v in initial.items() if n.startswith(k+'.')]) for k in ('backbone.rgb_encoder','backbone.event_encoder','backbone.rgb_proj','backbone.event_proj','neck','bbox_head') if any(n.startswith(k+'.') for n in initial)}
 nonstem=digest([(n.removeprefix('backbone.event_encoder.'),v) for n,v in initial.items() if n.startswith('backbone.event_encoder.') and n!='backbone.event_encoder.input_stem.op_list.0.conv.weight'])
 streams=TemporalStreams(model.backbone,8,data.reset_gap_us) if c.temporal_window else None
 candidates=sorted(range(len(data)),key=lambda i:data.samples[i]['bbox_count'],reverse=True)[:8]
 optimizer=torch.optim.AdamW(model.parameters(),lr=c.learning_rate,weight_decay=c.weight_decay);torch.cuda.reset_peak_memory_stats()
 losses=[];tested_indices=[];bn_start={n:int(t) for n,t in model.named_buffers() if n.endswith('num_batches_tracked')}
 for step in range(2):
  tested_indices.append(list(candidates))
  packed=collate_peod([data[(i,slot,step==0,False)] for slot,i in enumerate(candidates)]);values=unpack(packed,'detection');metas=values[1]
  model.train();optimizer.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):result=model(*values,**({'temporal_state':streams.select(metas)} if streams else {}))
  assert torch.isfinite(result['loss']);result['loss'].backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
  if streams:streams.commit(metas,result.pop('temporal_state'));assert all(s.dtype==torch.float32 and not s.requires_grad for s in streams.memory)
  optimizer.step();losses.append(float(result['loss'].detach()))
  candidates=[min(i+1,len(data)-1) for i in candidates]
 bn_end={n:int(t) for n,t in model.named_buffers() if n.endswith('num_batches_tracked')};assert all(bn_end[n]-v==2 for n,v in bn_start.items())
 updates={name:sum(not torch.equal(initial[n],p.detach().cpu()) for n,p in model.named_parameters() if n.startswith(name+'.')) for name in modules}
 assert all(v>0 for v in updates.values())
 pair=max((i for i in range(len(data)-1) if data.samples[i]['sequence']==data.samples[i+1]['sequence'] and 0<data.samples[i+1]['target_us']-data.samples[i]['target_us']<=data.reset_gap_us),key=lambda i:data.samples[i]['event_count'])
 model.eval();d,_,_=data[pair];event=d['events'][None].cuda() if d['events'] is not None else None;rgb=d['images'][None].cuda() if d['images'] is not None else None
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  if c.temporal_window:
   features,state=model.backbone(event,rgb,model.backbone.zero_temporal_state(1));second,_,_=data[pair+1];event2=second['events'][None].cuda();rgb2=second['images'][None].cuda() if second['images'] is not None else None
   continued,nextstate=model.backbone(event2,rgb2,state)
   reset,_=model.backbone(event2,rgb2,model.backbone.zero_temporal_state(1));history_effect=[float((x-y).abs().max()) for x,y in zip(continued,reset)];assert max(history_effect)>0
  else:features=model.backbone(None,rgb);state=();history_effect=None;assert not hasattr(model.backbone,'event_encoder') or model.backbone.event_encoder is None
  pyramid=model.neck(list(features));raw=model.bbox_head.inference(pyramid)
 assert [list(f.shape[-2:]) for f in features]==[[180,320],[90,160],[45,80],[23,40]]
 assert [list(f.shape[-2:]) for f in pyramid]==[[90,160],[45,80],[23,40]]
 assert list(raw.shape)==[1,18920,11],raw.shape
 sampler=SequenceBatchSampler(data,8,training=True);seen=[key[0] for batch in sampler for key in batch];assert len(seen)==len(data)==len(set(seen))
 total=resolve_training_updates(c,len(sampler));schedule=training_schedule(c,len(sampler),total);assert total==342600 and schedule['warmup_updates']==8565
 # Published GT as perfect predictions; clipped training geometry scored separately.
 rows=[];roundtrip=[];val=PEODFrames(args.data_root,'val',modality='rgb')
 for i,s in enumerate(val.samples):
  gt=[dict(category_id=x['category_id'],bbox=x['bbox'],area=x['area'],iscrowd=x.get('iscrowd',0),ignore=x.get('ignore',0)) for x in val.annotations(i)]
  pred=[dict(category_id=x['category_id'],bbox=x['bbox'],score=1.) for x in gt]
  rows.append(dict(index=i,sequence=s['sequence'],gt=gt,predictions=pred))
  predictions=[]
  for x in gt:
   xx,yy,w,h=x['bbox'];x1,x2=max(0.,xx),min(1280.,xx+w);y1,y2=max(0.,yy),min(720.,yy+h)
   if x2>x1 and y2>y1:predictions.append(dict(category_id=x['category_id'],bbox=[x1,y1,x2-x1,y2-y1],score=1.))
  roundtrip.append(dict(index=i,sequence=s['sequence'],gt=gt,predictions=predictions))
 oracle=score_dump(rows,val.manifest['classes']);assert oracle['mAP']>.99999
 clipped=score_dump(roundtrip,val.manifest['classes'])
 report=dict(passed=True,modality=c.modality,representation=c.event_representation,parameters=sum(p.numel() for p in model.parameters()),
  initial_components=modules,event_encoder_nonstem_sha256=nonstem,updated_parameters=updates,losses=losses,
  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
  high_target_sample_indices=tested_indices,high_target_counts=[[data.samples[i]['bbox_count'] for i in indices] for indices in tested_indices],
  features=[list(f.shape) for f in features],pyramid=[list(f.shape) for f in pyramid],raw_detections=list(raw.shape),
  temporal_pair_indices=[pair,pair+1],state_shapes=[list(s.shape) for s in state],state_dtype=[str(s.dtype) for s in state],history_effect=history_effect,
  bn_updates_per_two_forwards=2,activation_recomputation=False,all_samples_once=True,batches_per_epoch=len(sampler),
  updates_per_epoch=1713,total_updates=total,schedule=schedule,first_lr=learning_rate_at(1,schedule),last_lr=learning_rate_at(total,schedule),
  geometry_oracle=oracle,geometry_clip_roundtrip=clipped,validation_frames=len(val))
 (out/'report.json').write_text(json.dumps(report,indent=2));print('PASS',c.modality,report['parameters'],flush=True)
if __name__=='__main__':main()
