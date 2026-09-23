#!/usr/bin/env python3
"""Every-step streaming segmentation; only genuine GT enters accuracy metrics."""
import argparse,json
from pathlib import Path
import cv2
import torch
from hmnet.dataset.dsec_async import DSECAsync
from hmnet.utils.async_checkpoint import load_async_model,file_sha256
from hmnet.utils.async_evaluation import evaluate_async

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True);p.add_argument("--data-root",required=True)
    p.add_argument("--output",required=True);p.add_argument("--device",default="cuda:0")
    p.add_argument("--split",choices=("train","dev","test"),default="dev")
    p.add_argument("--rgb-delay-frames",type=int,choices=(0,1),default=0)
    p.add_argument("--rgb-keep-every",type=int,choices=(1,2),default=1,help="2: controlled 20-to-10Hz RGB thinning")
    p.add_argument("--workers",type=int,default=2);p.add_argument("--limit",type=int)
    p.add_argument("--save-predictions",action="store_true")
    args=p.parse_args();torch.set_num_threads(2)
    model,contract=load_async_model(args.checkpoint,args.device)
    ac=contract["asynchronous"]
    data=DSECAsync(args.data_root,args.split,window_us=ac["window_us"],bins=ac["bins"],
                   rgb_delay_frames=args.rgb_delay_frames,rgb_keep_every=args.rgb_keep_every,limit=args.limit)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    times=(out/"prediction_times.jsonl").open("w")
    def record(logits,gt,meta):
        times.write(json.dumps(meta)+"\n")
        if args.save_predictions:
            dest=out/"predictions"/meta["sequence"];dest.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(dest/f'{meta["curr_time_org"]}.png'),logits.argmax(1)[0].byte().cpu().numpy())
    try:result=evaluate_async(model,data,args.workers,on_prediction=record)
    finally:times.close()
    result.update(checkpoint=args.checkpoint,checkpoint_sha256=file_sha256(args.checkpoint),
                  dataset_sha256=data.signature,rgb_delay_frames=args.rgb_delay_frames,rgb_keep_every=args.rgb_keep_every)
    (out/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
