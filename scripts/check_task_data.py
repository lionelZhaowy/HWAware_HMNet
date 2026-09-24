#!/usr/bin/env python3
"""Independent closed-window/overflow oracle and real split checks."""
import json,subprocess
from pathlib import Path
import numpy as np
from hmnet.dataset.task_frames import TaskFrames,represent
from hmnet.models.efficientvit_tasks import ROOT


def main():
    out=ROOT/'artifacts/window_fixture';out.mkdir(exist_ok=False)
    events=np.array([(999,2,3,0),(1000,2,3,0),(51000,2,3,1),(51001,4,3,1)],np.int64)
    for split in ('train','val','test'):
        d=out/'source/detection_dataset_duration_60s_ratio_1.0'/split;d.mkdir(parents=True)
        v=np.zeros(4,dtype=[('t','<u4'),('_','<i4')]);v['t']=events[:,0];v['_']=events[:,1]+(events[:,2]<<14)+(events[:,3]<<28)
        with (d/'fixture_td.dat').open('wb') as f:
            f.write(b'% Height 240\n% Width 304\n'+bytes([0,8]));v.tofile(f)
        labels=np.zeros(1,dtype=[('ts','<u8'),('x','<f4'),('y','<f4'),('w','<f4'),('h','<f4'),('class_id','u1')]);labels['ts']=51000
        for k,v in [('x',10),('y',10),('w',40),('h',40)]:labels[k]=v
        np.save(d/'fixture_bbox.npy',labels)
    subprocess.run([str(ROOT/'scripts/hmnet-python'),str(ROOT/'scripts/prepare_task_frames.py'),'gen1','--source',str(out/'source'),'--output',str(out/'index'),'--sequences','1'],check=True)
    report={}
    for split in ('train','val','test'):
        data=TaskFrames(out/'index',split);assert np.array_equal(data.raw(0),events[1:3])
        r=data[0][0];b=TaskFrames(out/'index',split,'polarity_binary')[0][0]
        assert r.sum()==2 and r[0,3,2]==1 and r[19,3,2]==1
        assert b.sum()==2 and b[0,3,2]==1 and b[1,3,2]==1
    report['boundary_fixture']='both endpoints included; outside endpoints excluded; polarity/bin order checked on every split'
    for count in (0,1,256,10000):
        ev=np.tile(np.array([[123,2,3,1]],np.int64),(count,1))
        assert represent(ev,'polarity_binary',8,10).sum()==bool(count)
        assert represent(ev,'rvt_histogram',8,10).sum()==min(10,count%256)
    report['high_count_oracle']='0/1/256/10000: raw occupancy exact, RVT uint8 wrap then cutoff preserved'
    for kind,root,splits in [('gen1',ROOT/'artifacts/data_smoke/gen1',('train','val','test')),('eventscape',ROOT/'artifacts/data_smoke/eventscape',('train','val','test')),('mvsec',Path('/data/lab_dataset/RGB_DVS_Fusion/MVSEC/preprocessed/hmnet_v12t_raw_v1'),('day2','day1','night1'))]:
        entries=[]
        for split in splits:
            r=TaskFrames(root,split);b=TaskFrames(root,split,'polarity_binary')
            for i in (0,min(1,len(r)-1)):
                raw=r.raw(i);assert np.array_equal(raw,b.raw(i))
                end=r.samples[i]['target_us'];assert not len(raw) or (raw[:,0].min()>=end-50000 and raw[:,0].max()<=end)
                x,_,_=b[i];binary=x if kind=='gen1' else x['events'];expected=np.zeros((2,b.height,b.width),np.float32)
                if len(raw):expected[raw[:,3],raw[:,2],raw[:,1]]=1
                assert np.array_equal(binary.numpy(),expected)
                entries.append(dict(split=split,index=i,raw_events=len(raw),grid=[b.height,b.width]))
        report[kind]=entries
    (ROOT/'artifacts/validation/data_checks.json').write_text(json.dumps(report,indent=2)+'\n');print('data invariants passed',flush=True)

if __name__=='__main__':main()
