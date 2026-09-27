#!/usr/bin/env python3
"""Bounded lossless shared input cache; both representations derive from DAT.

Enforces a total cache budget and free-space reserve BEFORE each sample write.
Atomic sequence publication; raw sources remain required for validation/audit.
"""
import argparse,json,time,shutil
from pathlib import Path
import h5py
import numpy as np
import torch
from PIL import Image
from hmnet.dataset.peod_frames import PEODFrames,SCHEMA
from hmnet.dataset.task_frames import represent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='/data/lab_dataset/RGB_DVS_Fusion/PEOD_orig/preprocessed/hmnet_v12t_240x304_v1')
    p.add_argument('--limit-sequences',type=int,default=1)
    p.add_argument('--follow',action='store_true',help='Wait for new sequence indices, with the same hard space bounds')
    p.add_argument('--max-cache-gib',type=float,default=.5);p.add_argument('--reserve-gib',type=float,default=50.)
    a=p.parse_args();torch.set_num_threads(1);root=Path(a.root);out=root/'inputs';out.mkdir(exist_ok=True)
    budget=int(a.max_cache_gib*2**30);reserve=int(a.reserve_gib*2**30)
    def pending():
        seen=set()
        while True:
            paths=sorted((root/'sequences').glob('*.json'))[:a.limit_sequences or None]
            for path in paths:
                if path not in seen:
                    yield path
                    seen.add(path)
            if not a.follow or ((root/'COMPLETE.json').exists() and len(seen)==json.loads((root/'COMPLETE.json').read_text())['sequence_count']):break
            time.sleep(5)
    for path in pending():
        row=json.loads(path.read_text());dest=out/path.with_suffix('.h5').name
        if dest.exists():
            with h5py.File(dest,'r') as f:
                if f.attrs['source_signature']!=row['source_signature'] or f.attrs['schema']!=SCHEMA:raise ValueError('Stale input cache')
            continue
        data=PEODFrames.__new__(PEODFrames);data.samples=row['samples'];n=len(data.samples)
        temp=dest.with_suffix('.tmp.h5');started=time.monotonic()
        used=sum(x.stat().st_size for x in out.glob('*.h5'))
        if temp.exists():used-=temp.stat().st_size
        try:
            with h5py.File(temp,'w') as f:
                f.attrs['source_signature']=row['source_signature'];f.attrs['schema']=SCHEMA
                f.attrs['representation']='frozen_RVT20_uint8_and_raw_Binary2_packbits_v1'
                hist=f.create_dataset('rvt',(n,20,240,304),dtype='u1',chunks=(1,20,240,304),compression='gzip',compression_opts=1)
                binary=f.create_dataset('binary_packed',(n,2,240,38),dtype='u1',chunks=(1,2,240,38),compression='gzip',compression_opts=1)
                rgb=f.create_dataset('rgb',(n,171,304,3),dtype='u1',chunks=(1,171,304,3),compression='lzf')
                f.create_dataset('target_us',data=[s['target_us'] for s in data.samples])
                for i,s in enumerate(data.samples):
                    # At most 2 MiB uncompressed data is added per sample; 8 MiB guard includes HDF5 metadata.
                    f.flush();current=temp.stat().st_size
                    if used+current+8*2**20>budget or shutil.disk_usage(out).free<reserve+8*2**20:
                        raise RuntimeError(f'Cache capacity guard at {path.stem}/{i}: stop without changing originals')
                    ev=data.raw(i);rvt=represent(ev,'rvt_histogram',240,304).numpy().astype(np.uint8)
                    b=represent(ev,'polarity_binary',240,304).numpy().astype(np.uint8)
                    hist[i]=rvt;binary[i]=np.packbits(b,axis=-1)
                    with Image.open(s['rgb_file']) as im:rgb[i]=np.asarray(im.resize((304,171),Image.Resampling.BILINEAR))
                f.attrs['complete']=True
            temp.replace(dest)
        except BaseException:
            # Only the incomplete file created by this cache invocation is removed.
            temp.unlink(missing_ok=True)
            raise
        print(json.dumps(dict(sequence=path.stem,samples=n,bytes=dest.stat().st_size,seconds=time.monotonic()-started)),flush=True)

if __name__=='__main__':main()
