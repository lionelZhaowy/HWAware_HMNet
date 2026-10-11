"""Atomic checkpoints with a bounded retention policy and disk-space reserve."""
from pathlib import Path
import shutil
import torch


def atomic_save(state, path, reserve_bytes=512*2**20):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    seen=set()
    def size(value):
        if torch.is_tensor(value):
            key=(str(value.device),value.untyped_storage().data_ptr())
            if key in seen:return 0
            seen.add(key);return value.untyped_storage().nbytes()
        if isinstance(value,dict):return sum(size(v) for v in value.values())
        if isinstance(value,(list,tuple)):return sum(size(v) for v in value)
        return 0
    estimate=int(size(state)*1.1)+16*2**20
    if shutil.disk_usage(path.parent).free < reserve_bytes+estimate:
        raise OSError(f'Insufficient checkpoint space: need {estimate+reserve_bytes} free bytes at {path.parent}; previous checkpoint preserved')
    temp=path.with_name(path.stem+'.tmp'+path.suffix)
    torch.save(state,temp);temp.replace(path)


def inference_checkpoint(checkpoint, ema=False):
    """Small best/milestone files; only checkpoint.pth may be used for exact resume."""
    return dict(state_dict=checkpoint['ema']['state_dict'] if ema else checkpoint['state_dict'],
        training_contract=checkpoint['training_contract'],step=checkpoint['step'],
        data_epoch=checkpoint['data_epoch'],data_cursor=checkpoint['data_cursor'],
        weight_kind='ema' if ema else 'raw',inference_only=True,
        selection_split='dev',best_miou=checkpoint['best_miou'],best_step=checkpoint['best_step'],
        best_ema_mAP=checkpoint['best_ema_mAP'],best_ema_step=checkpoint['best_ema_step'])
