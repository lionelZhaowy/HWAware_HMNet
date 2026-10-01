#!/usr/bin/env python3
"""建立原生720P索引；复用已完整扫描的50ms物理索引，不复用缩放输入。"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from hmnet.dataset.peod_frames import SCHEMA
from hmnet.dataset.task_frames import sha_file

GEOMETRY=dict(source_hw=[720,1280],input_hw=[720,1280],scale=1.,pad_ltrb=[0,0,0,0],
    event_rounding='identity_original_DAT_integer_coordinates',rgb_interpolation='none',
    bbox='float_xyxy_clip_source_identity',feature_hw=[[180,320],[90,160],[45,80],[23,40]],
    internal_padding='only_existing_same_convolutions; no_added_image_padding')

def atomic_json(path,value):
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        Path(name).unlink(missing_ok=True)

def sample_fingerprint(samples):
    return hashlib.sha256(json.dumps(samples,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='/data/lab_dataset/RGB_DVS_Fusion/PEOD_orig')
    p.add_argument('--index-source');p.add_argument('--output');a=p.parse_args()
    root=Path(a.root).resolve();source=Path(a.index_source or root/'preprocessed/hmnet_v12t_240x304_v1')
    out=Path(a.output or root/'preprocessed/hmnet_v12t_720x1280_v1')
    if source.resolve()==out.resolve():raise ValueError('New native index must not overwrite legacy index')
    complete=json.loads((source/'COMPLETE.json').read_text())
    if complete['schema']!='peod_closed_50ms_letterbox_v1' or complete['diagnostic_subset']:
        raise ValueError('Full audited legacy physical index required; no diagnostic subset')
    if {s:complete['splits'][s] for s in ('train','val','test')}!={'train':54808,'val':4820,'test':11847}:
        raise ValueError('Unexpected shared sample counts')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.index_writer.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Reject mismatched completed artifacts; restart is idempotent.
        if (out/'COMPLETE.json').exists():
            saved=json.loads((out/'COMPLETE.json').read_text())
            if saved['index_source_complete_sha256']!=sha_file(source/'COMPLETE.json'):
                raise ValueError('Existing native index provenance differs')
            for split,digest in saved['manifest_sha256'].items():
                if sha_file(out/f'{split}.json')!=digest:raise ValueError('Existing manifest changed')
        manifests={};sources={};fingerprints={};shas={}
        for split,digest in complete['manifest_sha256'].items():
            path=source/f'{split}.json'
            if sha_file(path)!=digest:raise ValueError('Audited source manifest changed')
            m=json.loads(path.read_text())
            for row in m['sources']:sources[row['path']]=row
            fingerprints[split]=sample_fingerprint(m['samples'])
            m.update(schema=SCHEMA,geometry=GEOMETRY,index_provenance=dict(
                source_manifest_sha256=digest,shared_samples_sha256=fingerprints[split],
                reuse='exact physical event_start/end/count, published RGB pairing and split; no resized tensors'))
            manifests[split]=m
        for row in sources.values():
            st=Path(row['path']).stat()
            if (st.st_size,st.st_mtime_ns)!=(row['bytes'],row['mtime_ns']):
                raise ValueError(f'Source changed: {row["path"]}')
        (out/'sequences').mkdir(exist_ok=True)
        for path in sorted((source/'sequences').glob('*.json')):
            row=json.loads(path.read_text());row.update(schema=SCHEMA,geometry=GEOMETRY,
                index_source_sha256=sha_file(path),shared_samples_sha256=sample_fingerprint(row['samples']))
            atomic_json(out/'sequences'/path.name,row)
        for split,m in manifests.items():
            atomic_json(out/f'{split}.json',m);shas[split]=sha_file(out/f'{split}.json')
        report=dict(complete,schema=SCHEMA,geometry=GEOMETRY,manifest_sha256=shas,
            shared_samples_sha256=fingerprints,index_source=str(source.resolve()),
            index_source_complete_sha256=sha_file(source/'COMPLETE.json'),source_files_verified=len(sources),
            optional_input_cache='none required; direct raw PNG/DAT reading is fully supported',
            raw_dependencies='Original DAT, PNG and annotation JSON are required even with optional caches')
        atomic_json(out/'COMPLETE.json',report)
    print(json.dumps(dict(output=str(out),splits=report['splits'],schema=SCHEMA,shared_samples_sha256=fingerprints)))
if __name__=='__main__':main()
