"""Strict inference loading shared by evaluation, teacher generation and ONNX."""
import hashlib
from pathlib import Path
import torch
from hmnet.models.efficientvit_tasks import build_frame_task


def file_sha256(path):
    digest=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def load_async_model(checkpoint,device="cpu",expected_variant=None,phase=None):
    saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
    contract=saved.get("training_contract",{})
    asynchronous=contract.get("asynchronous",{})
    shape=(asynchronous.get("window_us"),asynchronous.get("bins"))
    if shape not in ((50000,10),(25000,5)) or contract.get("fusion_mode")!="add":
        raise ValueError("Expected asynchronous SimpleAdd B/C checkpoint")
    variant="B" if shape==(50000,10) else "C"
    if expected_variant and variant!=expected_variant:raise ValueError("Wrong event representation")
    if phase is not None and asynchronous.get("phase")!=phase:raise ValueError("Wrong teacher/training phase")
    model=build_frame_task("segmentation",modality="rgbdvs",fusion_mode="add",temporal_window=2,event_channels=shape[1]*2)
    model.load_state_dict(saved["state_dict"],strict=True)
    model.to(device).eval()
    return model,contract
