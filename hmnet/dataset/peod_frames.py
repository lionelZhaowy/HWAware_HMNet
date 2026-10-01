"""Published pixel-aligned PEOD, shared samples for RGB/DVS/fusion experiments."""
from functools import lru_cache
import json
import h5py
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from hmnet.dataset.task_frames import sha_file, represent
from hmnet.dataset.vendor.dat_events_tools import parse_header
from hmnet.models.base.event_repr.polarity import input_spec

SCHEMA='peod_closed_50ms_native_720p_v1'

def collate_peod(batch):
    """Stack dense modalities into shared, pinnable worker batches; preserve all values/order."""
    data,targets,metadata=zip(*batch)
    dense={key:(torch.utils.data.default_collate([d[key] for d in data])
                if data[0][key] is not None else None) for key in ("events","images")}
    return dense,list(targets),list(metadata)


class PEODFrames(Dataset):
    collate_fn = staticmethod(collate_peod)
    def __init__(self,root,split,representation='rvt_histogram',modality='rgbdvs',augment=False,limit=0):
        if modality not in ('rgb','dvs','rgbdvs'):raise ValueError(modality)
        self.root=Path(root);path=self.root/f'{split}.json'
        completion=json.loads((self.root/'COMPLETE.json').read_text())
        if completion['manifest_sha256'][split]!=sha_file(path):raise ValueError('Manifest fingerprint differs')
        self.manifest=json.loads(path.read_text())
        if self.manifest['schema']!=SCHEMA or self.manifest['geometry']['input_hw']!=[720,1280]:
            raise ValueError('Native PEOD 720P schema required; low-resolution cache is incompatible')
        self.samples=self.manifest['samples'][:limit or None]
        self.kind='peod';self.split=split;self.modality=modality;self.representation=representation;self.augment=augment
        self.height,self.width=720,1280;self.reset_gap_us=self.manifest['reset_gap_us']
        if not self.samples:raise ValueError('Empty PEOD split')
        for row in self.manifest['sources']:
            st=Path(row['path']).stat()
            if (st.st_size,st.st_mtime_ns)!=(row['bytes'],row['mtime_ns']):raise ValueError(f'Source changed: {row["path"]}')
        self.input_spec=(dict(representation='none',channels=0) if modality=='rgb' else dict(input_spec(representation)))
        self.input_spec.update(height=720,width=1280,kind='peod',schema=SCHEMA,output_dtype='float32')
        self.data_contract=dict(schema=SCHEMA,kind='peod',split=split,modality=modality,
            manifest_sha256=sha_file(path),source_signature=self.manifest['source_signature'],
            diagnostic_subset=bool(limit) or completion['diagnostic_subset'],limit=limit,reset_gap_us=self.reset_gap_us,
            rgb_pairing=self.manifest['rgb_pairing'],geometry=self.manifest['geometry'],classes=self.manifest['classes'],
            annotation_protocol='published_COCO_float_GT_no_GEN1_filters_all_frames')

    def __len__(self):return len(self.samples)

    @lru_cache(maxsize=128)
    def _events(self,path):
        with open(path,'rb') as f:offset,kind,size,shape=parse_header(f)
        if size!=8 or shape!=[720,1280]:raise ValueError('PEOD DAT geometry differs')
        return np.memmap(path,mode='r',offset=offset,dtype=[('t','<u4'),('_','<u4')])

    @lru_cache(maxsize=128)
    def _labels(self,path):
        data=json.loads(Path(path).read_text());by_id={im['id']:[] for im in data['images']}
        for a in data['annotations']:by_id[a['image_id']].append(a)
        return by_id

    @lru_cache(maxsize=128)
    def _cache(self,sequence):
        path=self.root/'inputs'/f'{sequence}.h5'
        if not path.exists(): return None
        expected=json.loads((self.root/'sequences'/f'{sequence}.json').read_text())['source_signature']
        f=h5py.File(path,'r')
        if not f.attrs.get('complete') or f.attrs['schema']!=SCHEMA or f.attrs['source_signature']!=expected:
            f.close();raise ValueError('Invalid input cache')
        if (f['rvt'].shape[1:]!=(20,720,1280) or f['binary_packed'].shape[1:]!=(2,720,160)
                or f['rgb'].shape[1:]!=(720,1280,3) or any(f[k].dtype!=np.uint8 for k in ('rvt','binary_packed','rgb'))):
            f.close();raise ValueError('Native input cache dimensions/dtype differ')
        return f

    def annotations(self,index):
        s=self.samples[index];return self._labels(s['label_file'])[s['image_id']]

    def raw(self,index,source_grid=True):
        s=self.samples[index];a=self._events(s['file'])[s['event_start']:s['event_end']]
        t=a['t'].astype(np.int64);a=a[(t>=s['target_us']-50000)&(t<=s['target_us'])]
        if len(a)!=s['event_count']:raise ValueError('Exact window count differs')
        packed=a['_'];ev=np.column_stack([a['t'].astype(np.int64),packed&16383,(packed>>14)&16383,(packed>>28)&1]).astype(np.int64)
        if len(ev)>1 and np.any(ev[1:,0]<ev[:-1,0]):ev=ev[np.argsort(ev[:,0],kind='stable')]
        if not source_grid:raise ValueError('Native PEOD does not remap event coordinates')
        return np.ascontiguousarray(ev)

    def __getitem__(self,key):
        index,slot,reset,flip=key if isinstance(key,tuple) else (key,0,True,False)
        s=self.samples[index];events=None;rgb=None
        cache=self._cache(s['sequence']);ci=None
        if cache is not None:
            ci=int(np.searchsorted(cache['target_us'][:],s['target_us']))
            if int(cache['target_us'][ci])!=s['target_us']:raise ValueError('Cache timestamp differs')
        if self.modality!='rgb':
            if cache is None:events=represent(self.raw(index),self.representation,720,1280)
            elif self.representation=='rvt_histogram':events=torch.from_numpy(cache['rvt'][ci].astype(np.float32))
            else:events=torch.from_numpy(np.unpackbits(cache['binary_packed'][ci],axis=-1).astype(np.float32))
        if self.modality!='dvs':
            if cache is not None:resized=cache['rgb'][ci]
            else:
                with Image.open(s['rgb_file']) as im:
                    if im.mode!='RGB' or im.size!=(1280,720):raise ValueError('Native RGB geometry differs')
                    resized=np.asarray(im)
            a=resized
            rgb=torch.from_numpy(a.transpose(2,0,1).copy()).float()/255
            rgb=(rgb-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
        annotations=self.annotations(index)
        boxes=[];classes=[];ignore=[]
        for a in annotations:
            x,y,w,h=a['bbox'];b=np.array([x,y,x+w,y+h],np.float64)
            b[[0,2]]=b[[0,2]].clip(0,1280);b[[1,3]]=b[[1,3]].clip(0,720)
            if b[2]<=b[0] or b[3]<=b[1]:continue
            boxes.append(b);classes.append(a['category_id']);ignore.append(bool(a.get('ignore',0) or a.get('iscrowd',0)))
        boxes=torch.tensor(np.asarray(boxes).reshape(-1,4),dtype=torch.float32)
        if flip:
            if events is not None:events=events.flip(-1)
            if rgb is not None:rgb=rgb.flip(-1)
            x=boxes[:,0].clone();boxes[:,0]=1280-boxes[:,2];boxes[:,2]=1280-x
        target=dict(bboxes=boxes,labels=torch.tensor(classes,dtype=torch.long),ignore_mask=torch.tensor(ignore,dtype=torch.bool))
        meta=dict(height=720,width=1280,img_shape=[1280,720,3],pad_shape=[1280,720,3],filename=s['file'],
            sequence=s['sequence'],curr_time_org=s['target_us'],curr_time_crop=50000,delta_t=50000,
            stream_slot=slot,reset=reset,flipped=flip,sample_index=index,image_path=s['rgb_file'],rgb_age_us=0)
        return dict(events=events,images=rgb),target,dict(image_meta=meta)
