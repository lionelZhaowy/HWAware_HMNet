"""Label-anchored raw GEN1/Eventscape/MVSEC, one shared representation contract.

Indexes contain original sources and exact event ranges; no histogram-derived
binary occupancy, resampling, or guessed RGB. All times are integer microseconds.
"""
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import bisect
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from hmnet.models.base.event_repr.rvt_histogram import RVTHistogram
from hmnet.models.base.event_repr.polarity import polarity_counts, input_spec
from hmnet.dataset.vendor.dat_events_tools import parse_header, load_td_data
from hmnet.dataset.gen1_windows import GEN1_WINDOW_INDEX

SCHEMA = 'task_raw_label_windows_v1'
SIZES = {'gen1': (240,304), 'eventscape': (256,512), 'mvsec': (260,346)}


def sha_file(path):
    digest=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''): digest.update(chunk)
    return digest.hexdigest()


def dat_map(path):
    with open(path,'rb') as f: offset,kind,size,shape=parse_header(f)
    if size!=8 or shape!=[240,304]: raise ValueError(f'Unexpected GEN1 DAT {path}: {size}, {shape}')
    return np.memmap(path,mode='r',offset=offset,dtype=[('t','<u4'),('_','<i4')])


def bound(array,value,right=False):
    """Binary search strided/mapped timestamp arrays without making a full copy."""
    lo,hi=0,len(array)
    while lo<hi:
        mid=(lo+hi)//2
        if array[mid]<value or (right and array[mid]==value):lo=mid+1
        else:hi=mid
    return lo


def represent(events,kind,height,width):
    events=np.asarray(events,dtype=np.int64)
    if kind=='polarity_binary':
        return torch.from_numpy((polarity_counts(events,height,width)>0).astype(np.float32))
    if kind!='rvt_histogram':raise ValueError(kind)
    return RVTHistogram()(torch.from_numpy(events.copy()),dict(height=height,width=width))[0]


class TaskFrames(Dataset):
    def __init__(self,root,split,representation='rvt_histogram',augment=False,limit=0):
        self.root=Path(root); path=self.root/f'{split}.json'
        self.manifest=json.loads(path.read_text())
        completion=self.root/'COMPLETE.json'
        if not completion.exists():raise ValueError('Dataset index is incomplete')
        diagnostic=json.loads(completion.read_text())['diagnostic_subset']
        if self.manifest['schema']!=SCHEMA:raise ValueError('Unknown raw index schema')
        self.kind=self.manifest['kind'];self.split=split
        if self.kind=='gen1' and self.manifest.get('gen1_window_index')!=GEN1_WINDOW_INDEX:
            raise ValueError('Legacy GEN1 binary-search index is unsafe for unordered DAT timestamps; rebuild into hmnet_v12t_raw_v2')
        self.representation=representation;self.augment=augment
        self.height,self.width=SIZES[self.kind]
        self.modality='rgbdvs' if self.kind=='eventscape' else 'dvs'
        self.reset_gap_us=1500000 if self.kind=='gen1' else 75000
        self.samples=self.manifest['samples'][:limit or None]
        if not self.samples:raise ValueError('Empty dataset')
        self.input_spec=dict(input_spec(representation),height=self.height,width=self.width,
                             schema=SCHEMA,kind=self.kind,output_dtype='float32')
        self.data_contract=dict(schema=SCHEMA,kind=self.kind,split=split,modality=self.modality,
            manifest_sha256=sha_file(path),source_signature=self.manifest['source_signature'],
            diagnostic_subset=bool(limit) or diagnostic,limit=limit,reset_gap_us=self.reset_gap_us,
            rgb_pairing='latest_not_after_gt_max_50ms' if self.modality=='rgbdvs' else None,
            depth='meters_raw_no_interpolation',size=[self.height,self.width])
        if self.kind=='gen1':
            self.data_contract['gen1_window_index']=GEN1_WINDOW_INDEX
        self._verify_sources()

    def _verify_sources(self):
        for row in self.manifest['sources']:
            stat=Path(row['path']).stat()
            if stat.st_size!=row['bytes'] or stat.st_mtime_ns!=row['mtime_ns']:
                raise ValueError(f"Source file changed since indexing: {row['path']}")
        signature=hashlib.sha256(json.dumps(self.manifest['sources'],sort_keys=True).encode()).hexdigest()
        if signature!=self.manifest['source_signature']:raise ValueError('Source signature differs')

    def __len__(self):return len(self.samples)

    @lru_cache(maxsize=8)
    def _labels(self,path):return np.load(path)

    @lru_cache(maxsize=4)
    def _h5(self,path):return h5py.File(path,'r')

    def raw(self,index):
        s=self.samples[index];start,end=s['target_us']-50000,s['target_us']
        if self.kind=='gen1':
            a=load_td_data(s['file'],s['event_end']-s['event_start'],s['event_start'])
            ev=np.column_stack([a[k].astype(np.int64) for k in ('t','x','y','p')])
        elif self.kind=='eventscape':
            parts=[]
            for path in s['event_files']:
                with np.load(path) as a:
                    parts.append(np.column_stack([a[k].astype(np.int64) for k in ('t','x','y','p')]))
            ev=np.concatenate(parts) if parts else np.empty((0,4),np.int64)
        else:
            a=self._h5(s['file'])['davis/left/events'][s['event_start']:s['event_end']]
            # Match parent time origin before rounding to integer microseconds.
            ts=np.rint(a[:,2]*1e6-self.manifest['start_us']).astype(np.int64)
            pol=a[:,3].astype(np.int64)
            if np.any((pol!=-1)&(pol!=1)):raise ValueError('MVSEC expected signed polarity')
            ev=np.column_stack([ts,a[:,:2].astype(np.int64),(pol+1)//2])
        ev=ev[(ev[:,0]>=start)&(ev[:,0]<=end)]
        if self.kind=='gen1':
            if len(ev)!=s['event_count']:
                raise ValueError(f"GEN1 window event count mismatch: {s['file']} at {end}")
            if len(ev)>1 and np.any(ev[1:,0]<ev[:-1,0]):
                ev=ev[np.argsort(ev[:,0],kind='stable')]
        return np.ascontiguousarray(ev,dtype=np.int64)

    def __getitem__(self,key):
        index,slot,reset,flip=key if isinstance(key,tuple) else (key,0,True,False)
        s=self.samples[index];events=represent(self.raw(index),self.representation,self.height,self.width)
        image=None
        meta=dict(height=self.height,width=self.width,img_shape=[self.width,self.height,events.shape[0]],
            pad_shape=[self.width,self.height,events.shape[0]],filename=s['file'],sequence=s['sequence'],
            curr_time_org=s['target_us'],curr_time_crop=50000,delta_t=50000,stream_slot=slot,
            reset=reset,flipped=flip,sample_index=index)
        if self.kind=='gen1':
            a=self._labels(s['label_file']);field='ts' if 'ts' in a.dtype.names else 't';a=a[a[field]==s['target_us']]
            # Same invalid-box removal and integer xywh conversion as GEN1Frames.
            x,y,w,h=[a[k] for k in ('x','y','w','h')]
            cw=(x+w).clip(0,self.width)-x.clip(0,self.width)
            ch=(y+h).clip(0,self.height)-y.clip(0,self.height)
            valid=(cw>0)&(ch>0)&(cw!=self.width)&(ch!=self.height)
            a=a[valid]; boxes=np.column_stack([a[k].astype(np.int64) for k in ('x','y','w','h')])
            boxes[:,2:]+=boxes[:,:2]
            boxes=torch.from_numpy(boxes).float();wh=boxes[:,2:]-boxes[:,:2]
            target=dict(bboxes=boxes,labels=torch.from_numpy(a['class_id'].astype(np.int64)),
                ignore_mask=(wh.square().sum(1)<900)|(wh.min(1).values<10))
            if flip:
                x1=boxes[:,0].clone();boxes[:,0]=self.width-boxes[:,2];boxes[:,2]=self.width-x1
        else:
            if self.kind=='eventscape':
                depth=np.load(s['label_file']).astype(np.float32)
                rgb=np.asarray(Image.open(s['rgb_file']).convert('RGB'),dtype=np.float32).transpose(2,0,1)/255.
                image=(torch.from_numpy(rgb.copy())-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
                meta['image_path']=s['rgb_file'];meta['rgb_age_us']=s['target_us']-s['rgb_us']
                meta.update(min_depth=3.346,max_depth=1000.)
            else:
                depth=self._h5(s['label_file'])['davis/left/depth_image_raw'][s['label_index']].astype(np.float32)
                meta.update(min_depth=1.978,max_depth=80.)
            if depth.shape!=(self.height,self.width):raise ValueError('Depth grid differs')
            target={'depth':torch.from_numpy(depth[None])}
            if image is not None and image.shape!=(3,self.height,self.width):raise ValueError('RGB grid differs')
            if flip:
                target['depth']=target['depth'].flip(-1)
                if image is not None:image=image.flip(-1)
        if flip:events=events.flip(-1)
        data=events if self.kind=='gen1' else dict(events=events,images=image)
        return data,target,dict(image_meta=meta)
