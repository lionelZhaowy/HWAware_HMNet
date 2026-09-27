#!/usr/bin/env python3
"""Render four frozen prediction dumps on exactly the same PEOD observations.

Dump boxes are used directly; this script never reruns inference. Video cadence
is 10 displayed observations/s, not acquisition time; actual time is overlaid.
"""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import cv2
from hmnet.dataset.peod_frames import PEODFrames
from hmnet.dataset.task_frames import sha_file

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',required=True)
    p.add_argument('--split',default='test');p.add_argument('--dumps',nargs=4,required=True,metavar=('RGB','DVS','FUSION_RVT','FUSION_BINARY'))
    p.add_argument('--output',required=True);p.add_argument('--score',type=float,default=.25);p.add_argument('--videos',action='store_true');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);data=PEODFrames(a.data_root,a.split,'polarity_binary','rgbdvs')
    dumps=[[json.loads(l) for l in Path(path).read_text().splitlines()] for path in a.dumps]
    keys=[(r['index'],r['sequence'],r['target_us']) for r in dumps[0]]
    if any([(r['index'],r['sequence'],r['target_us']) for r in rows]!=keys for rows in dumps[1:]):raise ValueError('Dumps use different samples/order')
    if len(keys)!=len(data):raise ValueError('Full split dumps required')
    names=['RGB only','DVS RVT','RGB+DVS RVT','RGB+DVS Binary'];writers={};frame_counts={};selected=set(np.linspace(0,len(keys)-1,8,dtype=int));still=[]
    for pos,(index,seq,t) in enumerate(keys):
        if not a.videos and pos not in selected:continue
        s=data.samples[index]
        if (s['sequence'],s['target_us'])!=(seq,t):raise ValueError('Manifest/dump mapping differs')
        with Image.open(s['rgb_file']) as im:rgb=im.resize((608,342),Image.Resampling.BILINEAR)
        event=data.raw(index,source_grid=True);ev=np.zeros((720,1280,3),np.uint8)
        if len(event):
            off=event[event[:,3]==0];on=event[event[:,3]==1]
            ev[off[:,2],off[:,1],0]=255;ev[on[:,2],on[:,1],1]=255
        event_image=Image.fromarray(ev).resize((608,342),Image.Resampling.NEAREST)
        canvas=Image.new('RGB',(1216,3*374));draw=ImageDraw.Draw(canvas)
        canvas.paste(rgb,(0,32));canvas.paste(event_image,(608,32))
        draw.text((5,7),f'RGB / GT  {seq}  t={t/1e6:.6f}s',fill='white');draw.text((613,7),'Raw 50ms event occupancy / GT',fill='white')
        for col in range(2):
            for gt in dumps[0][pos]['gt']:
                x,y,w,h=gt['bbox'];draw.rectangle([col*608+x*.475,32+y*.475,col*608+(x+w)*.475,32+(y+h)*.475],outline='orange',width=2)
        for i,rows in enumerate(dumps):
            x0=(i%2)*608;y0=(1+i//2)*374;canvas.paste(rgb,(x0,y0+32));draw.text((x0+5,y0+7),names[i],fill='white')
            for pr in rows[pos]['predictions']:
                if pr['score']<a.score:continue
                x,y,w,h=pr['bbox'];box=[x0+x*.475,y0+32+y*.475,x0+(x+w)*.475,y0+32+(y+h)*.475]
                draw.rectangle(box,outline='lime',width=2);draw.text((box[0],box[1]),f'{pr["category_id"]}:{pr["score"]:.2f}',fill='lime')
        if pos in selected:
            path=out/f'frame_{index:06d}.png';canvas.save(path);still.append(path.name)
        if a.videos:
            if seq not in writers:
                for previous in writers.values():previous.release()
                writer=cv2.VideoWriter(str(out/f'{seq}.mp4'),cv2.VideoWriter_fourcc(*'avc1'),10.,canvas.size)
                if not writer.isOpened():raise RuntimeError('H264 encoder unavailable')
                writers[seq]=writer;frame_counts[seq]=0
            writers[seq].write(np.asarray(canvas)[:,:,::-1]);frame_counts[seq]+=1
    for writer in writers.values():writer.release()
    report=dict(split=a.split,dumps={str(Path(p).resolve()):sha_file(p) for p in a.dumps},frames=len(keys),
        display_threshold=a.score,selection='8 uniform sample indices, no disagreement selection',video_fps=10,
        playback='10 observations/s; not original elapsed time; actual timestamp displayed',videos=frame_counts,stills=still)
    (out/'provenance.json').write_text(json.dumps(report,indent=2))
    html='<meta charset="utf-8"><title>PEOD frozen predictions</title><p>10 observations/s; timestamps show acquisition time.</p>'
    html+=''.join(f'<h3>{name}</h3><img width="100%" src="{name}">' for name in still)
    html+=''.join(f'<h3>{name}</h3><video controls width="100%" src="{name}.mp4"></video>' for name in frame_counts)
    (out/'index.html').write_text(html)

if __name__=='__main__':main()
