#!/usr/bin/env python3
"""Boundary/oracle/cache checks and multi-sequence alignment contact sheet."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from hmnet.dataset.peod_frames import PEODFrames
from hmnet.dataset.task_frames import represent

def boundaries():
    t=100000
    raw=np.array([(t+1,1,1,0),(t,1279,719,1),(t-50000,0,0,0),(t-50001,3,3,0),(t,1,1,0)],dtype=np.int64)
    packed=np.empty(len(raw),dtype=[('t','<u4'),('_','<u4')]);packed['t']=raw[:,0];packed['_']=raw[:,1]|(raw[:,2]<<14)|(raw[:,3]<<28)
    d=PEODFrames.__new__(PEODFrames);d.samples=[dict(file='synthetic',event_start=0,event_end=5,event_count=3,target_us=t)]
    d._events=lambda _:packed
    ev=d.raw(0)
    assert ev.tolist()==[[50000,0,34,0],[100000,303,204,1],[100000,0,34,0]]
    for kind in ('rvt_histogram','polarity_binary'):
        assert represent(np.empty((0,4),np.int64),kind,240,304).sum()==0
        single=represent(ev[:1],kind,240,304);assert single.sum()==1
        both=represent(ev,kind,240,304);assert torch.isfinite(both).all()
        for bad in (np.array([[1,-1,0,0]],np.int64),np.array([[1,304,0,1]],np.int64)):
            try:represent(bad,kind,240,304)
            except ValueError:pass
            else:raise AssertionError('Out-of-grid input accepted')
    collision=represent(ev,'polarity_binary',240,304)
    assert collision[0,34,0]==1 and collision[1,204,303]==1 and collision.sum()==2
    high=np.tile(np.array([[1,0,0,0]],np.int64),(256,1))
    assert represent(high,'rvt_histogram',240,304).sum()==0
    assert represent(high,'polarity_binary',240,304).sum()==1
    return dict(closed_endpoints=True,stable_regression_sort=True,coordinate_floor_collision=True,
        zero_single_both_polarities=True,out_of_bounds_rejected=True,rvt_uint8_wrap_preserved=True,binary_not_derived_from_rvt=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',default='artifacts/peod/data_checks');a=p.parse_args()
    torch.set_num_threads(1);root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);report=boundaries()
    canvases=[];cache_checks=[];empty=0
    paths=sorted((root/'sequences').glob('*.json'))
    selected=paths[::max(1,len(paths)//6)][:6]
    for path in selected:
        row=json.loads(path.read_text());d=PEODFrames.__new__(PEODFrames);d.root=root;d.samples=row['samples']
        i=len(d.samples)//2;s=d.samples[i];ev=d.raw(i)
        hist=represent(ev,'rvt_histogram',240,304);binary=represent(ev,'polarity_binary',240,304)
        cache=d._cache(s['sequence'])
        with Image.open(s['rgb_file']) as im:rgb=im.resize((304,171),Image.Resampling.BILINEAR)
        if cache is not None:
            assert np.array_equal(hist.numpy(),cache['rvt'][i])
            assert np.array_equal(binary.numpy(),np.unpackbits(cache['binary_packed'][i],axis=-1))
            assert np.array_equal(np.asarray(rgb),cache['rgb'][i]);cache_checks.append(s['sequence'])
        base=Image.new('RGB',(304,240));base.paste(rgb,(0,34));overlay=np.asarray(base).copy()
        edge=binary.numpy().any(0);overlay[edge]=(overlay[edge].astype(float)*.55+np.array([0,255,255])*.45).astype(np.uint8)
        tile=Image.new('RGB',(608,266));tile.paste(base,(0,26));tile.paste(Image.fromarray(overlay),(304,26));draw=ImageDraw.Draw(tile)
        draw.text((4,5),f'{s["sequence"]} t={s["target_us"]/1e6:.6f}s; RGB | events+RGB',fill='white')
        labels=json.loads(Path(s['label_file']).read_text())
        for label in labels['annotations']:
            if label['image_id']!=s['image_id']:continue
            x,y,w,h=label['bbox'];box=[x*.2375,y*.2375+60,(x+w)*.2375,(y+h)*.2375+60]
            draw.rectangle(box,outline='orange');draw.rectangle([box[0]+304,box[1],box[2]+304,box[3]],outline='orange')
        canvases.append(tile)
    sheet=Image.new('RGB',(608,266*len(canvases)))
    for i,tile in enumerate(canvases):sheet.paste(tile,(0,266*i))
    sheet.save(out/'alignment.png');report['cache_exact_sequences']=cache_checks;report['alignment_sequences']=[p.stem for p in selected]
    (out/'report.json').write_text(json.dumps(report,indent=2));print(report)

if __name__=='__main__':main()
