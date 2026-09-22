#!/usr/bin/env python3
"""Deterministic CPU save/restore regression (small real-data crop, not an accuracy test)."""
import argparse, io, json
from pathlib import Path
import torch
from hmnet.models.efficientvit_tasks import build_frame_task
from hmnet.dataset.dsec_frames import DSECFrames
from hmnet.dataset.temporal_frames import SequenceBatchSampler
from hmnet.utils.temporal_streams import TemporalStreams


def build(mode):
    return build_frame_task('segmentation',modality='rgbdvs',fusion_mode=mode,temporal_window=2)


def step(model,opt,bank,batch):
    data,targets,metadata=zip(*batch)
    # 2 streams and 128x160 crops keep this deterministic CPU check economical.
    e=torch.stack([x['events'][:,:128,:160] for x in data])
    rgb=[x['images'][:,:128,:160] for x in data]
    gt=[x['labels'][:128,:160] for x in targets]
    metas=[dict(x['image_meta'],height=128,width=160) for x in metadata]
    model.train();opt.zero_grad(set_to_none=True)
    result=model(e,rgb,metas,gt,temporal_state=bank.select(metas))
    bank.commit(metas,result['temporal_state'])
    result['loss'].backward();opt.step()
    return float(result['loss'].detach())


def compare(a,b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    else:assert a==b,(a,b)


def main(args):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(42)
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    contract=ckpt['training_contract'];data=DSECFrames(contract['data_root'],'train',augment=True)
    sampler=SequenceBatchSampler(data,contract['batch_size'],seed=contract['seed'])
    sampler.set_epoch(ckpt['data_epoch']);keys=list(sampler)
    cursor=ckpt['data_cursor'];assert cursor<len(keys)
    batch=[data[k] for k in keys[cursor][:2]]
    model=build(contract['fusion_mode']);model.load_state_dict(ckpt['state_dict'],strict=True)
    opt=torch.optim.AdamW(model.parameters());opt.load_state_dict(ckpt['optimizer'])
    bank=TemporalStreams(model.backbone,contract['batch_size']);bank.load_state_dict(ckpt['temporal_by_rank'][0])
    # Serialize the running weights/optimizer/BN and bank before the next step.
    buf=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),
                                    bank=bank.state_dict(),rng=torch.get_rng_state()),buf)
    expected_loss=step(model,opt,bank,batch)
    buf.seek(0);saved=torch.load(buf,map_location='cpu',weights_only=False)
    restored=build(contract['fusion_mode']);restored.load_state_dict(saved['model'],strict=True)
    resumed_opt=torch.optim.AdamW(restored.parameters());resumed_opt.load_state_dict(saved['optimizer'])
    resumed_bank=TemporalStreams(restored.backbone,contract['batch_size']);resumed_bank.load_state_dict(saved['bank'])
    torch.set_rng_state(saved['rng'])
    actual_loss=step(restored,resumed_opt,resumed_bank,batch)
    compare(model.state_dict(),restored.state_dict());compare(opt.state_dict(),resumed_opt.state_dict());compare(bank.state_dict(),resumed_bank.state_dict())
    assert actual_loss==expected_loss
    report=dict(passed=True,loss=actual_loss,comparison='bitwise CPU weights, BN, optimizer, state and loss',
                input='2 real chronological lane samples cropped to 128x160',checkpoint=args.checkpoint)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);main(p.parse_args())
