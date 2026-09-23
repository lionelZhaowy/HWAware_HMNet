#!/usr/bin/env python3
"""Shared asynchronous stage-1 / stage-2 CLI; delegates to the existing trainer."""
import argparse
import os
from pathlib import Path
from hmnet.utils.async_config import AsyncSettings,variant
from hmnet.utils.frame_train import run

def parse_settings(argv=None):
    """Resolve same-stage resume after selecting the final output directory."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",type=int,choices=(1,2),default=1)
    p.add_argument("--data-root")
    p.add_argument("--output")
    recovery=p.add_mutually_exclusive_group()
    recovery.add_argument("--resume",nargs="?",const=True,default=None,metavar="CHECKPOINT",
                          help="Restore CHECKPOINT, or output/checkpoint.pth when no path is supplied")
    recovery.add_argument("--init-from")
    p.add_argument("--pseudo-root")
    p.add_argument("--epochs",type=int)
    p.add_argument("--pseudo-weight",type=float,default=.2)
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--stop-after",type=int)
    p.add_argument("--validation-limit",type=int,help="diagnostics only; included in run settings")
    p.add_argument("--deterministic",action="store_true")
    p.add_argument("--single",action="store_true")
    p.add_argument("--distributed",action="store_true")
    p.add_argument("--precision",choices=("bf16","fp32"),default="bf16")
    args=p.parse_args(argv)
    if args.resume == "":p.error("--resume checkpoint path must not be empty")
    if args.resume is None:args.resume=""
    args.amp=False;args.overwrite=False
    if args.deterministic:os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    if args.epochs is not None and args.epochs <= 0:p.error("epochs must be positive")
    if args.workers < 0:p.error("workers must be nonnegative")
    if not 0 <= args.pseudo_weight <= 1:p.error("pseudo-weight must be in [0,1]")
    config=AsyncSettings()
    config.workers=args.workers;config.resume=args.resume;config.init_from=args.init_from
    config.deterministic=args.deterministic;config.validation_limit=args.validation_limit
    if args.data_root:config.cache=args.data_root
    if args.epochs:config.epochs=args.epochs
    if args.stage==2:
        if not args.pseudo_root:p.error("stage 2 requires --pseudo-root")
        if args.epochs is None:p.error("stage 2 requires an explicit extra --epochs budget")
        config.pseudo_root=args.pseudo_root;config.pseudo_weight=args.pseudo_weight
        config.learning_rate=2e-5;config.min_learning_rate=2e-6;config.warmup_epochs=1
        config.output=str(Path(config.output).with_name(f"efficientvit_b1_{variant.VERSION}_stage2"))
    elif args.pseudo_root or args.init_from:p.error("stage 1 cannot consume stage-2 initialization/pseudo labels")
    if args.output:config.output=args.output
    if args.resume:
        checkpoint=(Path(config.output)/"checkpoint.pth" if args.resume is True
                    else Path(args.resume))
        checkpoint=checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            p.error(f"Resume checkpoint does not exist or is not a file: {checkpoint}. "
                    "Use --output RUN_DIR or --resume CHECKPOINT for a different run; "
                    "omit --resume only to start a new run.")
        # The trainer still performs strict contract/state restoration. Never
        # fall back to best_checkpoint.pth, another stage, or a fresh training run.
        config.resume=args.resume=str(checkpoint)
    return config,args


if __name__ == "__main__":
    config,args=parse_settings()
    if config.resume:print(f"[resume] checkpoint={config.resume}",flush=True)
    run(config,args)
