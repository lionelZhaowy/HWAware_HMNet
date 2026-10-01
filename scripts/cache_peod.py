#!/usr/bin/env python3
"""Bounded lossless shared input cache; both representations derive from DAT.

Enforces a total cache budget and free-space reserve BEFORE each sample write.
Atomic sequence publication; raw sources remain required for validation/audit.
"""
import argparse,json,time,shutil,fcntl,os,tempfile
from contextlib import contextmanager
from pathlib import Path
import h5py
import numpy as np
import torch
from PIL import Image
from hmnet.dataset.peod_frames import PEODFrames,SCHEMA
from hmnet.dataset.task_frames import represent

@contextmanager
def cache_writer_lock(out):
    """One writer per shared cache, across all four experiment checkouts.

    Keep the lock file inode: unlinking it permits a second independent lock.
    The kernel releases flock automatically when a process exits.
    """
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.cache_writer.lock').open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.seek(0)
            raise RuntimeError('共享缓存已有写入任务，拒绝重复启动；持锁信息：'+lock.read().strip()) from None
        try:
            lock.seek(0);lock.truncate()
            lock.write(json.dumps(dict(pid=os.getpid(),started=time.time())))
            lock.flush()
            yield
        finally:
            fcntl.flock(lock.fileno(),fcntl.LOCK_UN)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='/data/lab_dataset/RGB_DVS_Fusion/PEOD_orig/preprocessed/hmnet_v12t_720x1280_v1')
    p.add_argument('--limit-sequences',type=int,default=1)
    p.add_argument('--follow',action='store_true',help='Wait for new sequence indices, with the same hard space bounds')
    p.add_argument('--max-cache-gib',type=float,default=.5);p.add_argument('--reserve-gib',type=float,default=50.)
    a=p.parse_args();torch.set_num_threads(1);root=Path(a.root);out=root/'inputs';out.mkdir(exist_ok=True)
    with cache_writer_lock(out):
        build_cache(a,root,out)
        completion=json.loads((root/'COMPLETE.json').read_text())
        paths=sorted((root/'sequences').glob('*.json'))
        complete=len(paths)==completion['sequence_count']
        total=0
        for index in paths:
            row=json.loads(index.read_text());dest=out/index.with_suffix('.h5').name
            if not dest.exists():complete=False;continue
            with h5py.File(dest,'r') as f:
                if not f.attrs.get('complete') or f.attrs['schema']!=SCHEMA or f.attrs['source_signature']!=row['source_signature']:
                    raise ValueError('Invalid completed cache')
            total+=dest.stat().st_size
        if complete:
            from scripts.prepare_peod import atomic_json
            atomic_json(out/'CACHE_COMPLETE.json',dict(schema=SCHEMA,sequence_count=len(paths),bytes=total,
                index_complete_sha256=__import__('hmnet.dataset.task_frames',fromlist=['sha_file']).sha_file(root/'COMPLETE.json')))
        print(json.dumps(dict(cache_complete=complete,cached_bytes=total,optional=True)),flush=True)

def check_capacity(used,current,free,budget,reserve,where):
    guard=32*2**20
    reasons=[]
    if used+current+guard>budget:
        reasons.append(f'缓存总量上限：已完成及临时缓存={(used+current)/2**30:.3f} GiB，单次写入预留=32 MiB，上限={budget/2**30:.3f} GiB；核对磁盘空间后调整 --max-cache-gib')
    if free<reserve+guard:
        reasons.append(f'磁盘剩余空间保护：可用={free/2**30:.3f} GiB，最低保留={reserve/2**30:.3f} GiB，另需32 MiB写入余量；请释放空间或更换存储位置')
    if reasons:
        raise RuntimeError(f'Cache capacity guard at {where}: '+ '; '.join(reasons)+'；原始数据和完整缓存保持不变')

def build_cache(a,root,out):
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
                if not f.attrs.get('complete',False) or f.attrs['source_signature']!=row['source_signature'] or f.attrs['schema']!=SCHEMA or len(f['target_us'])!=len(row['samples']):raise ValueError('Stale input cache')
            continue
        data=PEODFrames.__new__(PEODFrames);data.samples=row['samples'];n=len(data.samples)
        started=time.monotonic()
        used=sum(x.stat().st_size for x in out.glob('*.h5'))
        # Exclusive unique file; cleanup can never unlink another writer's file.
        fd,name=tempfile.mkstemp(prefix=dest.stem+'.',suffix='.tmp.h5',dir=out)
        os.close(fd);temp=Path(name)
        try:
            with h5py.File(temp,'w') as f:
                f.attrs['source_signature']=row['source_signature'];f.attrs['schema']=SCHEMA
                f.attrs['representation']='frozen_RVT20_uint8_and_raw_Binary2_packbits_v1'
                hist=f.create_dataset('rvt',(n,20,720,1280),dtype='u1',chunks=(1,20,720,1280),compression='gzip',compression_opts=1)
                binary=f.create_dataset('binary_packed',(n,2,720,160),dtype='u1',chunks=(1,2,720,160),compression='gzip',compression_opts=1)
                rgb=f.create_dataset('rgb',(n,720,1280,3),dtype='u1',chunks=(1,720,1280,3),compression='gzip',compression_opts=1)
                f.create_dataset('target_us',data=[s['target_us'] for s in data.samples])
                for i,s in enumerate(data.samples):
                    # At most 23 MiB uncompressed data is added per sample; 32 MiB guard includes HDF5 metadata.
                    f.flush();current=temp.stat().st_size
                    check_capacity(used,current,shutil.disk_usage(out).free,budget,reserve,f'{path.stem}/{i}')
                    ev=data.raw(i);rvt=represent(ev,'rvt_histogram',720,1280).numpy().astype(np.uint8)
                    b=represent(ev,'polarity_binary',720,1280).numpy().astype(np.uint8)
                    hist[i]=rvt;binary[i]=np.packbits(b,axis=-1)
                    with Image.open(s['rgb_file']) as im:rgb[i]=np.asarray(im)
                f.attrs['complete']=True
            temp.replace(dest)
        except BaseException:
            # Only the incomplete file created by this cache invocation is removed.
            temp.unlink(missing_ok=True)
            raise
        print(json.dumps(dict(sequence=path.stem,samples=n,bytes=dest.stat().st_size,seconds=time.monotonic()-started)),flush=True)

if __name__=='__main__':main()
