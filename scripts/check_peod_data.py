#!/usr/bin/env python3
"""独立NumPy原生表示、坐标与共同样本核验；显示不进入输入。"""
import argparse,json,hashlib,io,time,zlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from hmnet.dataset.peod_frames import PEODFrames
from hmnet.dataset.task_frames import represent


def oracle(ev):
    r=np.zeros(20*720*1280,np.uint8);b=np.zeros(2*720*1280,np.uint8)
    if len(ev):
        t,x,y,p=ev.T
        bins=np.floor((t-t[0]).astype(np.float32)/np.float32(max(int(t[-1]-t[0]),1))*10).astype(np.int64).clip(0,9)
        idx=x+1280*y+720*1280*(bins+10*p)
        np.add.at(r,idx,np.uint8(1));np.minimum(r,10,out=r)
        b[x+1280*y+720*1280*p]=1
    return r.reshape(20,720,1280),b.reshape(2,720,1280)


def boundaries():
    t=100000
    raw=np.array([(t+1,1,1,0),(t,1279,719,1),(t-50000,0,0,0),(t-50001,3,3,0),(t,1,1,0)],np.int64)
    packed=np.empty(len(raw),dtype=[('t','<u4'),('_','<u4')]);packed['t']=raw[:,0];packed['_']=raw[:,1]|(raw[:,2]<<14)|(raw[:,3]<<28)
    d=PEODFrames.__new__(PEODFrames);d.samples=[dict(file='synthetic',event_start=0,event_end=5,event_count=3,target_us=t)];d._events=lambda _:packed
    ev=d.raw(0);assert ev.tolist()==[[50000,0,0,0],[100000,1279,719,1],[100000,1,1,0]]
    for kind in ('rvt_histogram','polarity_binary'):
        assert represent(np.empty((0,4),np.int64),kind,720,1280).sum()==0
        assert represent(ev[:1],kind,720,1280).sum()==1
        for bad in (np.array([[1,-1,0,0]],np.int64),np.array([[1,1280,0,1]],np.int64)):
            try:represent(bad,kind,720,1280)
            except ValueError:pass
            else:raise AssertionError('Invalid coordinate accepted')
    high=np.tile(np.array([[1,0,0,0]],np.int64),(256,1))
    assert represent(high,'rvt_histogram',720,1280).sum()==0
    assert represent(high,'polarity_binary',720,1280).sum()==1
    return dict(closed_endpoints=True,stable_regression_sort=True,zero_single_both_polarities=True,
        out_of_bounds_rejected=True,rvt_uint8_wrap_preserved=True,binary_from_raw=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(1);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    report=dict(boundaries=boundaries(),splits={},frames=[],compression=[],failures=[]);tiles=[];started=time.perf_counter()
    for split in ('train','val','test'):
        data=PEODFrames(a.root,split);binary=PEODFrames(a.root,split,'polarity_binary')
        legacy=json.loads((Path(a.root).parent/'hmnet_v12t_240x304_v1'/f'{split}.json').read_text())
        assert data.samples==legacy['samples']
        assert data.reset_gap_us==legacy['reset_gap_us']
        groups=defaultdict(list)
        for i,s in enumerate(data.samples):groups[s['sequence']].append(i)
        selected={i for indices in groups.values() for i in (indices[0],indices[len(indices)//2],indices[-1])}
        selected.add(max(range(len(data)),key=lambda i:data.samples[i]['event_count']))
        selected.add(max(range(len(data)),key=lambda i:data.samples[i]['bbox_count']))
        empty=next((i for i,s in enumerate(data.samples) if s['bbox_count']==0),None)
        if empty is not None:selected.add(empty)
        zero=next((i for i,s in enumerate(data.samples) if s['event_count']==0),None)
        if zero is not None:selected.add(zero)
        report['splits'][split]=dict(frames=len(data),sequences=len(groups),sample_fingerprint=hashlib.sha256(json.dumps(data.samples,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            empty_frames=sum(s['bbox_count']==0 for s in data.samples),zero_event_frames=sum(s['event_count']==0 for s in data.samples),max_events=max(s['event_count'] for s in data.samples),max_boxes=max(s['bbox_count'] for s in data.samples))
        for number,i in enumerate(sorted(selected)):
            s=data.samples[i];ev=data.raw(i);expected,b=oracle(ev)
            # Decode/filter DAT independently from the raw bit fields; verify exact windows.
            raw=data._events(s['file'])[s['event_start']:s['event_end']];t=raw['t'].astype(np.int64);raw=raw[(t>=s['target_us']-50000)&(t<=s['target_us'])]
            q=raw['_'];decoded=np.column_stack([raw['t'].astype(np.int64),q&16383,(q>>14)&16383,(q>>28)&1]).astype(np.int64)
            decoded=decoded[np.argsort(decoded[:,0],kind='stable')];assert np.array_equal(ev,decoded)
            values,target,meta=data[i];v2,_,_=binary[i]
            assert np.array_equal(values['events'].numpy(),expected)
            assert np.array_equal(v2['events'].numpy(),b)
            with Image.open(s['rgb_file']) as im:rgb=np.array(im)
            x=torch.from_numpy(rgb.transpose(2,0,1).copy()).float()/255
            x=(x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
            assert torch.equal(x,values['images'])
            boxes=[];classes=[]
            for annotation in data.annotations(i):
                x,y,w,h=annotation['bbox'];box=np.array([x,y,x+w,y+h],np.float64);box[[0,2]]=box[[0,2]].clip(0,1280);box[[1,3]]=box[[1,3]].clip(0,720)
                if box[2]>box[0] and box[3]>box[1]:boxes.append(box);classes.append(annotation['category_id'])
            assert np.array_equal(target['bboxes'].numpy(),np.array(boxes,dtype=np.float32).reshape(-1,4))
            assert target['labels'].tolist()==classes
            if number==0:
                flipped,tgt,_=data[(i,0,True,True)]
                assert torch.equal(flipped['events'],values['events'].flip(-1)) and torch.equal(flipped['images'],values['images'].flip(-1))
                recovered=tgt['bboxes'].clone();recovered[:,[0,2]]=1280-recovered[:,[2,0]];assert torch.allclose(recovered,target['bboxes'],atol=.0001,rtol=0)
            report['frames'].append(dict(split=split,index=i,sequence=s['sequence'],events=len(ev),boxes=len(boxes),rgb_mean=float(rgb.mean()),passed=True))
            if number%max(1,len(selected)//8)==0:
                sizes=dict(split=split,index=i,rvt_bytes=len(zlib.compress(expected.tobytes(),1)),binary_bytes=len(zlib.compress(np.packbits(b,axis=-1).tobytes(),1)),rgb_bytes=len(zlib.compress(rgb.tobytes(),1)),raw_png_bytes=Path(s['rgb_file']).stat().st_size)
                report['compression'].append(sizes)
            if number==0 or (s['condition']=='challenge' and len(tiles)<6):
                if len(tiles)<6:
                    base=Image.fromarray(rgb);display=rgb.copy();positive=b[1].astype(bool);negative=b[0].astype(bool)
                    display[negative]=[255,50,50];display[positive]=[50,100,255]
                    overlay=Image.fromarray(display)
                    for image in (base,overlay):
                        draw=ImageDraw.Draw(image)
                        for box in boxes:draw.rectangle(box.tolist(),outline=(0,255,0),width=2)
                    base.save(out/f'{split}_{i}_rgb_gt.png');overlay.save(out/f'{split}_{i}_event_gt.png')
                    crop=(480,200,800,520);base.crop(crop).save(out/f'{split}_{i}_crop_1to1_rgb.png');overlay.crop(crop).save(out/f'{split}_{i}_crop_1to1_events.png')
                    tile=Image.new('RGB',(1280,386));tile.paste(base.resize((640,360)),(0,26));tile.paste(overlay.resize((640,360)),(640,26));ImageDraw.Draw(tile).text((5,5),f'{split} {s["sequence"]} index={i} RGB / events+GT; display only',fill='white');tiles.append(tile)
            if number%30==0:print(split,number,len(selected),flush=True)
    sheet=Image.new('RGB',(1280,len(tiles)*386))
    for i,tile in enumerate(tiles):sheet.paste(tile,(0,i*386))
    sheet.save(out/'native_overlays.png')
    report.update(passed=True,seconds=time.perf_counter()-started,display_only='red p0 blue p1 existence, downscaled contact sheet plus exact 1:1 crops')
    mean=np.mean([sum(s[k] for k in ('rvt_bytes','binary_bytes','rgb_bytes')) for s in report['compression']])
    report['estimated_full_cache_gib']=float(mean*71475/2**30)
    (out/'report.json').write_text(json.dumps(report,indent=2));print('PASS',len(report['frames']),report['estimated_full_cache_gib'],flush=True)
if __name__=='__main__':main()
