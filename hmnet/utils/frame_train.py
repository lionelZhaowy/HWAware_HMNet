"""Single-GPU independent-window training, dispatched by the existing task CLIs."""

import json
import math
import random
import shutil
from collections import defaultdict
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from hmnet.dataset.custom_collate_fn import collate_keep_dict
from hmnet.utils.common import fix_seed


def unpack(batch, task):
    data, targets, metadata = batch
    metas = [m["image_meta"] for m in metadata]
    if task == "detection":
        events = (
            data
            if torch.is_tensor(data)
            else torch.stack([d["events"] if isinstance(d, dict) else d for d in data])
        )
        return (
            events,
            metas,
            [t["bboxes"] for t in targets],
            [t["labels"] for t in targets],
            [t["ignore_mask"] for t in targets],
        )
    return (
        torch.stack([d["events"] for d in data]),
        [d["images"] for d in data],
        metas,
        [t["labels" if task == "segmentation" else "depth"] for t in targets],
    )


@torch.no_grad()
def evaluate_seg(model, dataset, batch_size=2, workers=0, prefetch_factor=1):
    model.eval()
    confusion = torch.zeros(11, 11, dtype=torch.int64)
    for batch in DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_keep_dict,
        num_workers=workers,
        pin_memory=True,
        **({"prefetch_factor": prefetch_factor} if workers > 0 else {}),
    ):
        events, images, metas, labels = unpack(batch, "segmentation")
        pred, _ = model.inference(events, images, metas)
        pred = pred.argmax(1).cpu()
        gt = torch.stack(labels)
        valid = (gt >= 0) & (gt < 11)
        confusion += torch.bincount(
            (gt[valid] * 11 + pred[valid]).flatten(), minlength=121
        ).reshape(11, 11)
    union = confusion.sum(0) + confusion.sum(1) - confusion.diag()
    iou = confusion.diag().double() / union.clamp_min(1)
    return dict(
        miou=float(iou[union > 0].mean()),
        class_iou=[float(iou[i]) if union[i] > 0 else None for i in range(11)],
        confusion=confusion.tolist(),
    )


def write_tensorboard(writer, record):
    """Mirror optimizer-update metrics into the run's TensorBoard event file."""
    step = record["step"]
    for key in ("loss", "lr", "amp_scale", "amp_skipped_updates", "data_epochs"):
        writer.add_scalar(f"train/{key}", record[key], step)
    for index, lr in enumerate(record["learning_rates"]):
        writer.add_scalar(f"train/lr_group_{index}", lr, step)
    for name, loss in record["loss_components"].items():
        tag = "_".join(name.split())
        writer.add_scalar(f"train/loss_components/{tag}", loss, step)
    for key in ("seconds", "samples_per_second", "peak_memory_mib"):
        writer.add_scalar(f"performance/{key}", record[key], step)
    for split in ("dev", "fixed_train"):
        if split not in record:
            continue
        writer.add_scalar(f"{split}/mIoU", record[split]["miou"], step)
        for class_id, iou in enumerate(record[split]["class_iou"]):
            if iou is not None:
                writer.add_scalar(f"{split}/IoU_class_{class_id:02d}", iou, step)


def configure_frame_training(config, args):
    """Apply a small set of explicit CLI overrides; do not read legacy B1 env vars."""
    if args.epochs is not None and args.updates is not None:
        raise ValueError("Choose --epochs or --updates, not both")
    if args.epochs is not None:
        config.epochs, config.updates = args.epochs, None
    if args.updates is not None:
        config.updates = args.updates
    for name in ("resume", "output"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(config, name, value)
    if args.data_root is not None:
        setattr(
            config,
            "cache" if config.task == "segmentation" else "data_root",
            args.data_root,
        )


def resolve_training_updates(config, batches_per_epoch):
    """Convert epochs using the actual loader length; keep explicit update budgets."""
    if type(config.accumulation) is not int or config.accumulation <= 0:
        raise ValueError("accumulation must be a positive integer")
    if batches_per_epoch <= 0:
        raise ValueError("Empty frame dataset")
    if config.updates is not None:
        if type(config.updates) is not int or config.updates <= 0:
            raise ValueError("updates must be None or a positive integer")
        return config.updates
    epochs = getattr(config, "epochs", None)
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be a positive integer when updates is None")
    # Accumulation continues across data epochs. The last update may consume up
    # to accumulation-1 extra batches; skipped/overflowed batches do not count.
    return (batches_per_epoch * epochs + config.accumulation - 1) // config.accumulation


def training_schedule(config, batches_per_epoch, max_updates):
    """Store the complete schedule in checkpoints, including its fixed end point."""
    kind = getattr(config, "lr_schedule", "constant")
    if kind not in ("constant", "warmup_cosine"):
        raise ValueError(f"Unknown lr_schedule: {kind}")
    base = float(config.learning_rate)
    minimum = float(getattr(config, "min_learning_rate", base))
    warmup = math.ceil(
        batches_per_epoch * getattr(config, "warmup_epochs", 0) / config.accumulation
    )
    factor = float(getattr(config, "warmup_start_factor", 0.1))
    if base <= 0 or not 0 <= minimum <= base or not 0 < factor <= 1:
        raise ValueError("Invalid learning rate, minimum or warmup_start_factor")
    if kind == "warmup_cosine" and not 0 <= warmup < max_updates:
        raise ValueError("Warmup must be shorter than the total training budget")
    return dict(
        kind=kind,
        base_lr=base,
        min_lr=minimum,
        warmup_updates=warmup,
        warmup_start_factor=factor,
        total_updates=max_updates,
    )


def learning_rate_at(update, schedule):
    """LR for a 1-based successful optimizer update; AMP retries reuse the same LR."""
    if schedule["kind"] == "constant":
        return schedule["base_lr"]
    warmup = schedule["warmup_updates"]
    if update <= warmup:
        progress = (update - 1) / max(1, warmup - 1)
        return schedule["base_lr"] * (
            schedule["warmup_start_factor"]
            + (1 - schedule["warmup_start_factor"]) * progress
        )
    progress = (update - warmup) / (schedule["total_updates"] - warmup)
    return schedule["min_lr"] + 0.5 * (schedule["base_lr"] - schedule["min_lr"]) * (
        1 + math.cos(math.pi * min(1.0, progress))
    )


def run(config, args):
    if args.distributed:
        raise ValueError(
            "First-version frame training is single GPU; run with --single"
        )
    fix_seed(args.seed)
    torch.set_num_threads(4)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = output / "tensorboard"
    has_run = (
        (output / "checkpoint.pth").exists()
        or (output / "metrics.jsonl").exists()
        or tensorboard_dir.exists()
    )
    if has_run and not config.resume and not args.overwrite:
        raise FileExistsError(f"{output}: use --output, --resume, or --overwrite")
    dataset = config.get_dataset()
    # Seed each epoch explicitly. Resume checkpoints include the epoch/batch cursor
    # and random states, so an optimizer update never resumes midway through accumulation.
    generator = torch.Generator()
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        # Bound host memory when several modality experiments run together.
        **(
            {"prefetch_factor": getattr(config, "prefetch_factor", 1)}
            if config.workers > 0
            else {}
        ),
        generator=generator,
        collate_fn=collate_keep_dict,
        pin_memory=True,
        drop_last=False,
        # DSEC workers must receive the new epoch for reproducible augmentation.
        persistent_workers=config.workers > 0 and not hasattr(dataset, "set_epoch"),
    )
    max_updates = resolve_training_updates(config, len(loader))
    schedule = training_schedule(config, len(loader), max_updates)
    eval_every = getattr(config, "eval_every_epochs", None)
    if eval_every is not None and eval_every <= 0:
        raise ValueError("eval_every_epochs must be positive")
    eval_interval = (
        max(1, math.ceil(len(loader) * eval_every / config.accumulation))
        if eval_every is not None
        else 50
    )
    # Changing these during resume would invalidate data order or the LR curve.
    contract = dict(
        schedule=schedule,
        modality=getattr(config, "modality", "dvs"),
        fusion_mode=getattr(config, "fusion_mode", "add"),
        batch_size=config.batch_size,
        accumulation=config.accumulation,
        train_samples=len(dataset),
        weight_decay=config.weight_decay,
        amp=args.amp,
        seed=args.seed,
        data_root=str(getattr(config, "cache", getattr(config, "data_root", ""))),
    )
    print(
        json.dumps(
            dict(
                train_samples=len(dataset),
                batches_per_epoch=len(loader),
                epochs=getattr(config, "epochs", None),
                requested_updates=config.updates,
                resolved_updates=max_updates,
                schedule=schedule,
                modality=getattr(config, "modality", "dvs"),
                fusion_mode=getattr(config, "fusion_mode", "add"),
                batch_size=config.batch_size,
                accumulation=config.accumulation,
            )
        ),
        flush=True,
    )
    model = config.get_model().cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp, init_scale=1024.0)
    step, epoch, cursor = 0, 0, 0
    best_miou, best_step = -1.0, 0
    stage_parent = None
    if getattr(config, "start_new_stage", False) and not config.resume:
        raise ValueError("A new training stage requires --resume with a full checkpoint")
    if config.resume:
        ckpt = torch.load(config.resume, map_location="cpu", weights_only=False)
        # Historical checkpoints predate the architecture field and use addition.
        previous = dict(ckpt.get("training_contract", {}))
        previous.setdefault("fusion_mode", "add")
        if previous["fusion_mode"] != contract["fusion_mode"]:
            raise ValueError("Resume fusion architecture differs; start a new experiment")
        # A new stage explicitly changes the LR budget but retains the optimizer,
        # sampling cursor and RNG. Ordinary resume still checks the full contract.
        new_stage = getattr(config, "start_new_stage", False) and not ckpt.get("stage_parent")
        if new_stage:
            if output.resolve() == Path(config.resume).resolve().parent or has_run:
                raise ValueError("A new stage requires a separate, empty output directory")
            if {k: v for k, v in previous.items() if k != "schedule"} != {
                k: v for k, v in contract.items() if k != "schedule"
            }:
                raise ValueError("New stage may change LR/budget only; data and training settings must match")
            stage_parent = dict(
                checkpoint=str(Path(config.resume).resolve()),
                step=ckpt["step"],
                data_epoch=ckpt["data_epoch"],
                data_cursor=ckpt["data_cursor"],
                training_contract=previous,
                best_miou=ckpt.get("best_miou"),
                best_step=ckpt.get("best_step"),
            )
        elif schedule["kind"] != "constant" and previous != contract:
            raise ValueError(
                "Resume training contract differs (modality/data/batch/seed/LR budget). "
                "Use the original settings; old constant-LR runs are separate experiments."
            )
        best_miou = ckpt.get("best_miou", -1.0)
        best_step = ckpt.get("best_step", 0)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        step, epoch, cursor = ckpt["step"], ckpt["data_epoch"], ckpt["data_cursor"]
        random.setstate(ckpt["python_rng"])
        np.random.set_state(ckpt["numpy_rng"])
        torch.set_rng_state(ckpt["torch_rng"])
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        if new_stage:
            # Local steps measure the additional stage, not the total data epochs.
            # The historical best remains in the parent run; select a new stage best.
            step, best_miou, best_step = 0, -1.0, 0
        else:
            stage_parent = ckpt.get("stage_parent")
        del ckpt
    if step >= max_updates:
        raise ValueError(
            f"Checkpoint already has {step} updates; target is {max_updates}. "
            "Increase total --epochs/--updates to continue, or run test.py."
        )
    validation = (
        config.get_validation_dataset()
        if hasattr(config, "get_validation_dataset")
        else None
    )
    history = output / "metrics.jsonl"
    if config.resume and history.exists():
        # A crash can leave metrics newer than the last saved optimizer update.
        # Keep JSON history aligned with TensorBoard's purge_step on continuation.
        rows = [
            json.loads(line)
            for line in history.read_text().splitlines()
            if line.strip()
        ]
        history.write_text(
            "".join(json.dumps(row) + "\n" for row in rows if row["step"] <= step)
        )
    elif not config.resume:
        history.write_text("")
        if tensorboard_dir.exists():
            # Only reached after the caller explicitly requested --overwrite.
            shutil.rmtree(tensorboard_dir)
    (output / "settings.json").write_text(
        json.dumps(
            dict(
                seed=args.seed,
                task=config.task,
                updates=max_updates,
                requested_updates=config.updates,
                epochs=getattr(config, "epochs", None),
                batches_per_epoch=len(loader),
                batch=config.batch_size,
                accumulation=config.accumulation,
                lr=config.learning_rate,
                lr_schedule=schedule,
                modality=getattr(config, "modality", "dvs"),
                fusion_mode=getattr(config, "fusion_mode", "add"),
                eval_interval_updates=eval_interval,
                weight_decay=config.weight_decay,
                amp=args.amp,
                data_root=str(
                    getattr(config, "cache", getattr(config, "data_root", ""))
                ),
                train_samples=len(dataset),
                dev_samples=len(validation) if validation else 0,
                tensorboard_dir=str(tensorboard_dir),
                resume=config.resume,
                stage_parent=stage_parent,
            ),
            indent=2,
        )
    )

    def batches(start_epoch, start_cursor):
        data_epoch = start_epoch
        while True:
            generator.manual_seed(args.seed + data_epoch)
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(data_epoch, args.seed)
            for idx, batch in enumerate(loader):
                if data_epoch == start_epoch and idx < start_cursor:
                    continue
                yield batch, data_epoch, idx + 1
            data_epoch += 1

    iterator = batches(epoch, cursor)
    torch.cuda.reset_peak_memory_stats()
    amp_skipped_updates = 0
    # Use optimizer updates as the x-axis, not microbatches. Purging after resume
    # hides stale points beyond the checkpoint; context exit flushes on exceptions.
    with SummaryWriter(
        str(tensorboard_dir),
        purge_step=step + 1 if config.resume else None,
        flush_secs=30,
    ) as writer:
        while step < max_updates:
            # Apply BEFORE optimizer.step: the logged LR is the one actually used.
            lr = learning_rate_at(step + 1, schedule)
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            loss_sum, samples, accepted, attempts = 0.0, 0, 0, 0
            component_sums = defaultdict(float)
            while accepted < config.accumulation:
                batch, epoch, cursor = next(iterator)
                attempts += 1
                if attempts > max(len(loader) * 2, config.accumulation * 4):
                    raise RuntimeError(
                        "No sufficient supervised windows: check labels and depth ranges"
                    )
                with torch.amp.autocast("cuda", enabled=args.amp):
                    result = model(*unpack(batch, config.task))
                    loss = result["loss"]
                if result.get("skip_step", False):
                    continue
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at update {step}")
                scaler.scale(loss / config.accumulation).backward()
                loss_sum += float(loss.detach())
                for name, value in result.get("log_vars", {}).items():
                    component_sums[name] += float(value)
                samples += result["num_samples"]
                accepted += 1
            scaler.unscale_(optimizer)
            if not all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in model.parameters()
            ):
                if args.amp:
                    scaler.step(optimizer)
                    scaler.update()
                    amp_skipped_updates += 1
                    print("AMP overflow: skipped update, reduced scale", flush=True)
                    continue
                raise FloatingPointError("Non-finite FP32 gradient")
            scaler.step(optimizer)
            scaler.update()
            step += 1
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            record = dict(
                step=step,
                epoch=epoch + 1,
                data_epochs=epoch + cursor / len(loader),
                stage_epochs=step * config.accumulation / len(loader),
                loss=loss_sum / accepted,
                seconds=seconds,
                lr=optimizer.param_groups[0]["lr"],
                learning_rates=[group["lr"] for group in optimizer.param_groups],
                loss_components={
                    name: value / accepted for name, value in component_sums.items()
                },
                amp_scale=scaler.get_scale(),
                amp_skipped_updates=amp_skipped_updates,
                samples_per_second=samples / seconds,
                peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20,
            )
            is_best = False
            if validation is not None and (
                step == 1 or step % eval_interval == 0 or step == max_updates
            ):
                record["dev"] = evaluate_seg(
                    model,
                    validation,
                    getattr(config, "eval_batch_size", config.batch_size),
                    workers=getattr(config, "workers", 0),
                    prefetch_factor=getattr(config, "prefetch_factor", 1),
                )
                is_best = record["dev"]["miou"] > best_miou
                if is_best:
                    best_miou, best_step = record["dev"]["miou"], step
                record.update(best_miou=best_miou, best_step=best_step)
                if getattr(config, "overfit", 0):
                    record["fixed_train"] = evaluate_seg(
                        model,
                        dataset,
                        config.batch_size,
                        workers=getattr(config, "workers", 0),
                        prefetch_factor=getattr(config, "prefetch_factor", 1),
                    )
            with history.open("a") as f:
                f.write(json.dumps(record) + "\n")
            write_tensorboard(writer, record)
            print(json.dumps(record), flush=True)
            if is_best or step % eval_interval == 0 or step == max_updates:
                checkpoint = dict(
                    state_dict=model.state_dict(),
                    training_contract=contract,
                    stage_parent=stage_parent,
                    best_miou=best_miou,
                    best_step=best_step,
                    optimizer=optimizer.state_dict(),
                    scaler=scaler.state_dict(),
                    step=step,
                    data_epoch=epoch,
                    data_cursor=cursor,
                    python_rng=random.getstate(),
                    numpy_rng=np.random.get_state(),
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(),
                )
                temp = output / "checkpoint.tmp.pth"
                torch.save(checkpoint, temp)
                temp.replace(output / "checkpoint.pth")
                if is_best:
                    # Keep a full resume-able best checkpoint, chosen only on dev.
                    best_temp = output / "best_checkpoint.tmp.pth"
                    shutil.copyfile(output / "checkpoint.pth", best_temp)
                    best_temp.replace(output / "best_checkpoint.pth")
                writer.flush()
    return model


def run_seg_evaluation(config, args):
    """Existing segmentation test CLI: data_list=train/dev/test, data_root=frame cache."""
    from hmnet.dataset.dsec_frames import DSECFrames

    if (
        args.mode != "single_process"
        or args.fast
        or args.fuse_right
        or args.test_chunks != "1/1"
    ):
        raise ValueError(
            "Frame cache evaluation requires single_process, left RGB and a complete train/dev/test split"
        )
    if getattr(args, "output", None):
        config.output = args.output
    args.data_list = args.data_list or "dev"
    args.data_root = args.data_root or config.cache
    if args.data_list not in ("train", "dev", "test"):
        raise ValueError(
            "For the B1 frame configuration, data_list must be train, dev or test"
        )
    device = torch.device("cpu" if args.cpu else f"cuda:{int(args.gpuid)}")
    torch.set_num_threads(2)
    model = config.get_model().to(device)
    checkpoint = args.pretrained or str(Path(config.output) / "checkpoint.pth")
    if not args.random_init:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("state_dict", state), strict=True)
    dataset = DSECFrames(args.data_root, args.data_list)
    with torch.amp.autocast(device.type, enabled=args.fp16):
        metrics = evaluate_seg(
            model,
            dataset,
            config.batch_size,
            workers=getattr(config, "workers", 0),
            prefetch_factor=getattr(config, "prefetch_factor", 1),
        )
    out = Path(config.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"evaluation_{args.data_list}.json").write_text(
        json.dumps(metrics, indent=2)
    )
    print(json.dumps(metrics), flush=True)
