#!/usr/bin/env python3
"""Synthetic pseudo fixture through the real phase-2 trainer (NOT teacher validation)."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from hmnet.utils.async_config import AsyncSettings
from hmnet.utils.frame_train import run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--resume',default='')
    p.add_argument('--stop-after',type=int,default=1)
    args=p.parse_args();out=Path(args.output).resolve()
    if 'artifacts' not in out.parts:raise ValueError('Synthetic diagnostics must live under artifacts/')
    manifest=json.loads((Path(args.data_root)/'manifest.json').read_text())
    fixture=out.parent/'synthetic_pseudo_fixture'
    fixture.mkdir(parents=True,exist_ok=True)
    if not (fixture/'manifest.json').exists():
        files={}
        for row in manifest['samples']:
            if row['split']!='train' or row['has_gt']:continue
            key=f'{row["sequence"]}/{row["target_us"]}';relative=key+'.npy'
            path=fixture/relative;path.parent.mkdir(parents=True,exist_ok=True)
            # Synthetic sparse class IDs exercise accepted and ignored pixels.
            label=np.full((440,640),255,dtype=np.uint8);label[100:140,100:140]=0
            np.save(path,label);files[key]=relative
        (fixture/'manifest.json').write_text(json.dumps(dict(format='dsec_async_pseudo_v1',audit_passed=True,
            diagnostic_only=True,notice='synthetic fixture, no teacher-quality claim; never use for formal training',
            grid_sha256=manifest['grid_sha256'],files=files,split='train'),indent=2))
    config=AsyncSettings();config.cache=args.data_root;config.output=str(out)
    config.pseudo_root=str(fixture);config.init_from=args.checkpoint if not args.resume else None
    config.resume=args.resume;config.epochs=20;config.warmup_epochs=1
    config.learning_rate=2e-5;config.workers=2;config.validation_limit=4
    config.diagnostic_pseudo=True
    runtime_args=SimpleNamespace(seed=42,single=True,distributed=False,amp=False,precision='bf16',
        overwrite=False,stop_after=args.stop_after)
    run(config,runtime_args)
