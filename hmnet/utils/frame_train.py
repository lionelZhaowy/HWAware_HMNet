"""Frame/sequence training: FP32/BF16/FP16, single GPU or segmentation DDP."""

import json
import math
import random
import shutil
from collections import defaultdict
from contextlib import nullcontext
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, RandomSampler
from hmnet.utils.frame_distributed import DistributedRun, GlobalBatchSampler, validate_segmentation_ddp
from torch.utils.tensorboard import SummaryWriter
from hmnet.dataset.custom_collate_fn import collate_keep_dict
from hmnet.utils.common import fix_seed
from hmnet.dataset.temporal_frames import SequenceBatchSampler, manifest_signature
from hmnet.utils.temporal_streams import TemporalStreams
from hmnet.models.base.event_repr.polarity import input_spec, validate_input_spec


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
def evaluate_seg(model, dataset, batch_size=2, workers=0, prefetch_factor=1, runtime=None):
    model.eval()
    confusion = torch.zeros(11, 11, dtype=torch.int64)
    temporal = bool(getattr(model.backbone, "temporal_window", 0))
    sequence_sampler = (SequenceBatchSampler(dataset, batch_size,
        runtime.rank if runtime else 0, runtime.world_size if runtime else 1,
        training=False) if temporal else None)
    streams = TemporalStreams(model.backbone, batch_size) if temporal else None
    sampler = None
    if runtime is not None and runtime.distributed:
        sampler = range(runtime.rank, len(dataset), runtime.world_size)
        batch_size = max(1, math.ceil(batch_size / runtime.world_size))
        workers = runtime.workers(workers)
    for batch in DataLoader(
        dataset,
        **(dict(batch_sampler=sequence_sampler) if temporal else
           dict(batch_size=batch_size, sampler=sampler)),
        collate_fn=collate_keep_dict,
        num_workers=workers,
        pin_memory=True,
        **({"prefetch_factor": prefetch_factor} if workers > 0 else {}),
    ):
        events, images, metas, labels = unpack(batch, "segmentation")
        if temporal:
            pred, _, current = model.inference(events, images, metas,
                                               temporal_state=streams.select(metas))
            streams.commit(metas, current)
        else:
            pred, _ = model.inference(events, images, metas)
        pred = pred.argmax(1).cpu()
        gt = torch.stack(labels)
        valid = (gt >= 0) & (gt < 11)
        confusion += torch.bincount(
            (gt[valid] * 11 + pred[valid]).flatten(), minlength=121
        ).reshape(11, 11)
    if runtime is not None and runtime.distributed:
        confusion = confusion.to(runtime.device)
        dist.all_reduce(confusion)
        confusion = confusion.cpu()
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
    runtime = DistributedRun(args)
    try:
        return _run(config, args, runtime)
    finally:
        runtime.close()


def _run(config, args, runtime):
    precision = getattr(args, "precision", None) or ("fp16" if args.amp else getattr(config, "precision", "fp32"))
    if getattr(args, "precision", None) == "fp32" and args.amp:
        raise ValueError("--precision fp32 cannot be combined with --amp")
    amp = precision != "fp32"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support BF16")
    if runtime.distributed and config.task != "segmentation":
        raise ValueError("Frame DDP is validated for segmentation only")
    # Explicit full FP32 includes convolution/matmul: no implicit TF32 shortcut.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fix_seed(args.seed)
    torch.set_num_threads(4)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = output / "tensorboard"
    has_run = ((output / "checkpoint.pth").exists()
               or (output / "metrics.jsonl").exists() or tensorboard_dir.exists())
    if has_run and not config.resume and not args.overwrite:
        raise FileExistsError(f"{output}: use --output, --resume, or --overwrite")
    dataset = config.get_dataset()
    generator = torch.Generator()
    temporal = bool(getattr(config, "temporal_window", 0))
    if temporal and (config.task != "segmentation" or config.accumulation != 1):
        raise ValueError("Temporal first experiment requires segmentation and accumulation=1")
    sampler = (SequenceBatchSampler(dataset, config.batch_size, runtime.rank,
                                   runtime.world_size, training=True, seed=args.seed)
               if temporal else GlobalBatchSampler(
                   RandomSampler(dataset, generator=generator), config.batch_size,
                   runtime.rank, runtime.world_size))
    workers = runtime.workers(config.workers)
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=workers,
        **({"prefetch_factor": getattr(config, "prefetch_factor", 1)} if workers else {}),
        generator=generator, collate_fn=collate_keep_dict, pin_memory=True,
        persistent_workers=workers > 0 and not hasattr(dataset, "set_epoch"),
    )
    max_updates = resolve_training_updates(config, len(loader))
    schedule = training_schedule(config, len(loader), max_updates)
    eval_every = getattr(config, "eval_every_epochs", None)
    if eval_every is not None and eval_every <= 0:
        raise ValueError("eval_every_epochs must be positive")
    eval_interval = (max(1, math.ceil(len(loader) * eval_every / config.accumulation))
                     if eval_every is not None else 50)
    fusion_mode = (config.fusion_mode
                   if getattr(config, "modality", "dvs") == "rgbdvs" else "none")
    contract = dict(
        schedule=schedule, modality=getattr(config, "modality", "dvs"),
        fusion_mode=fusion_mode, batch_size=config.batch_size,
        accumulation=config.accumulation, train_samples=len(dataset),
        weight_decay=config.weight_decay, amp=amp, precision=precision,
        world_size=runtime.world_size, sync_bn=runtime.distributed, tf32=False,
        seed=args.seed,
        data_root=str(getattr(config, "cache", getattr(config, "data_root", ""))),
    )
    representation = getattr(config, "event_representation", "rvt_histogram")
    if representation != "rvt_histogram":
        validate_input_spec(getattr(dataset, "input_spec", None), representation)
        if getattr(config, "event_channels", None) != dataset.input_spec["channels"]:
            raise ValueError("Config event channels differ from dataset representation")
        if (dataset.data_contract["diagnostic_subset"] and not getattr(args, "stop_after", None)
                and not getattr(config, "overfit", 0)):
            raise ValueError("Diagnostic subset cache cannot start a full training run")
        contract["event_input"] = input_spec(representation)
        contract["event_data"] = dataset.data_contract
    if temporal:
        contract["temporal"] = dict(window=2, branch="dvs", state_dtype="float32",
            reduction_dtype="float32", tbptt=1, sampler="balanced_sequence_lanes_v1",
            reset_gap_us=75000, manifest_sha256=manifest_signature(dataset))
    if runtime.primary:
        print(json.dumps(dict(training_contract=contract, batches_per_epoch=len(loader),
                              resolved_updates=max_updates, local_batch=config.batch_size // runtime.world_size)), flush=True)
    raw_model = config.get_model().to(runtime.device)
    if raw_model.backbone.fusion_mode != fusion_mode:
        raise ValueError("Config fusion mode and instantiated backbone disagree")
    if runtime.distributed:
        validate_segmentation_ddp(raw_model)
        raw_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(raw_model)
        model = DistributedDataParallel(raw_model, device_ids=[runtime.local_rank],
                                        broadcast_buffers=False)
    else:
        model = raw_model
    streams = TemporalStreams(raw_model.backbone, config.batch_size) if temporal else None
    # Distinct dropout streams, reproducible via per-rank checkpoint RNG.
    fix_seed(args.seed + runtime.rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16", init_scale=1024.0)
    step, epoch, cursor = 0, 0, 0
    best_miou, best_step = -1.0, 0
    stage_parent = None
    if getattr(config, "start_new_stage", False) and not config.resume:
        raise ValueError("A new training stage requires --resume with a full checkpoint")
    if config.resume:
        ckpt = torch.load(config.resume, map_location="cpu", weights_only=False)
        previous = dict(ckpt.get("training_contract", {}))
        if previous.get("fusion_mode") != contract["fusion_mode"]:
            raise ValueError("Resume fusion architecture differs; start a new experiment")
        new_stage = getattr(config, "start_new_stage", False) and not ckpt.get("stage_parent")
        expected = {k: v for k, v in contract.items() if not new_stage or k != "schedule"}
        actual = {k: v for k, v in previous.items() if not new_stage or k != "schedule"}
        if actual != expected:
            raise ValueError("Resume training contract differs (event representation/precision/world size/BN/data/batch/seed/LR); start a separate experiment")
        if new_stage:
            if output.resolve() == Path(config.resume).resolve().parent or has_run:
                raise ValueError("A new stage requires a separate, empty output directory")
            stage_parent = dict(checkpoint=str(Path(config.resume).resolve()),
                                step=ckpt["step"], data_epoch=ckpt["data_epoch"],
                                data_cursor=ckpt["data_cursor"], training_contract=previous,
                                best_miou=ckpt.get("best_miou"), best_step=ckpt.get("best_step"))
        else:
            stage_parent = ckpt.get("stage_parent")
        raw_model.load_state_dict(ckpt["state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        step, epoch, cursor = ckpt["step"], ckpt["data_epoch"], ckpt["data_cursor"]
        best_miou, best_step = ckpt.get("best_miou", -1.), ckpt.get("best_step", 0)
        runtime.restore_rng(ckpt["rng_by_rank"])
        if temporal:
            saved_streams = ckpt.get("temporal_by_rank", [])
            if len(saved_streams) != runtime.world_size:
                raise ValueError("Missing or mismatched temporal resume state")
            streams.load_state_dict(saved_streams[runtime.rank])
        if new_stage:
            step, best_miou, best_step = 0, -1., 0
        del ckpt
    if step >= max_updates:
        raise ValueError(f"Checkpoint already has {step} updates; target is {max_updates}")
    validation = config.get_validation_dataset() if hasattr(config, "get_validation_dataset") else None
    if representation != "rvt_histogram" and validation is not None:
        validate_input_spec(getattr(validation, "input_spec", None), representation)
    history = output / "metrics.jsonl"
    if runtime.primary:
        if config.resume and history.exists():
            rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
            history.write_text("".join(json.dumps(row) + "\n" for row in rows if row["step"] <= step))
        elif not config.resume:
            history.write_text("")
            if tensorboard_dir.exists():
                shutil.rmtree(tensorboard_dir)
        (output / "settings.json").write_text(json.dumps(dict(
            seed=args.seed, task=config.task, updates=max_updates,
            requested_updates=config.updates, epochs=getattr(config, "epochs", None),
            batches_per_epoch=len(loader), batch=config.batch_size,
            local_batch=config.batch_size // runtime.world_size,
            workers=config.workers, accumulation=config.accumulation,
            lr=config.learning_rate, lr_schedule=schedule, modality=contract["modality"],
            fusion_mode=fusion_mode, eval_interval_updates=eval_interval,
            weight_decay=config.weight_decay, amp=amp, precision=precision,
            world_size=runtime.world_size, sync_bn=runtime.distributed, tf32=False,
            training_contract=contract, data_root=contract["data_root"],
            train_samples=len(dataset), dev_samples=len(validation) if validation else 0,
            tensorboard_dir=str(tensorboard_dir), resume=config.resume, stage_parent=stage_parent,
        ), indent=2))
    runtime.barrier()

    def batches(start_epoch, start_cursor):
        data_epoch = start_epoch
        while True:
            generator.manual_seed(args.seed + data_epoch)
            if temporal:
                sampler.set_epoch(data_epoch)
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
    writer_context = (SummaryWriter(str(tensorboard_dir), purge_step=step+1 if config.resume else None,
                                    flush_secs=30) if runtime.primary else nullcontext(None))
    with writer_context as writer:
        while step < max_updates:
            lr = learning_rate_at(step + 1, schedule)
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            loss_sum, samples, accepted, attempts = 0., 0, 0, 0
            component_sums = defaultdict(float)
            while accepted < config.accumulation:
                batch, epoch, cursor = next(iterator)
                attempts += 1
                if attempts > max(len(loader)*2, config.accumulation*4):
                    raise RuntimeError("No sufficient supervised windows: check labels and depth ranges")
                weight = 1.
                if runtime.distributed:
                    pixels = sum(int((target["labels"] != 255).sum()) for target in batch[1])
                    total_pixels = runtime.reduce([pixels])[0]
                    if total_pixels == 0:
                        continue
                    # Avoid a rank skipping forward while its peers enter SyncBN.
                    if not runtime.all_true(pixels > 0):
                        raise ValueError("DDP batch has a rank with no valid labels; use fewer GPUs")
                    weight = runtime.world_size * pixels / total_pixels
                with torch.amp.autocast("cuda", enabled=amp, dtype=amp_dtype):
                    unpacked = unpack(batch, config.task)
                    result = model(*unpacked, **(dict(temporal_state=streams.select(unpacked[2]))
                                                if temporal else {}))
                    loss = result["loss"]
                if temporal:
                    streams.commit(unpacked[2], result["temporal_state"])
                    # The external bank owns detached state; do not retain the
                    # output-state autograd graph across optimizer updates.
                    del result["temporal_state"]
                if result.get("skip_step", False):
                    continue
                if not runtime.all_true(bool(torch.isfinite(loss))):
                    raise FloatingPointError(f"Non-finite {precision} loss at update {step}")
                scaler.scale(loss * weight / config.accumulation).backward()
                loss_sum += float(loss.detach()) * weight
                for name, value in result.get("log_vars", {}).items():
                    component_sums[name] += float(value) * weight
                samples += result["num_samples"]
                accepted += 1
            scaler.unscale_(optimizer)
            finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            if not runtime.all_true(bool(finite)):
                if precision == "fp16":
                    scaler.step(optimizer)
                    scaler.update()
                    amp_skipped_updates += 1
                    if runtime.primary:
                        print("AMP overflow: skipped update, reduced scale", flush=True)
                    continue
                raise FloatingPointError(f"Non-finite {precision} gradient")
            scaler.step(optimizer)
            scaler.update()
            step += 1
            torch.cuda.synchronize()
            seconds = runtime.reduce([time.perf_counter()-started], dist.ReduceOp.MAX)[0]
            names = sorted(component_sums)
            sums = runtime.reduce([loss_sum, samples] + [component_sums[name] for name in names])
            record = dict(
                step=step, epoch=epoch+1, data_epochs=epoch+cursor/len(loader),
                stage_epochs=step*config.accumulation/len(loader),
                loss=sums[0]/runtime.world_size/accepted, seconds=seconds,
                lr=lr, learning_rates=[group["lr"] for group in optimizer.param_groups],
                loss_components={name: value/runtime.world_size/accepted for name, value in zip(names, sums[2:])},
                amp_scale=scaler.get_scale(), amp_skipped_updates=amp_skipped_updates,
                samples_per_second=sums[1]/seconds, samples=int(sums[1]),
                peak_memory_mib=runtime.reduce([torch.cuda.max_memory_allocated()/2**20], dist.ReduceOp.MAX)[0],
            )
            is_best = False
            if validation is not None and (step == 1 or step % eval_interval == 0 or step == max_updates):
                record["dev"] = evaluate_seg(
                    raw_model, validation, getattr(config, "eval_batch_size", config.batch_size),
                    workers=config.workers, prefetch_factor=getattr(config, "prefetch_factor", 1), runtime=runtime)
                is_best = record["dev"]["miou"] > best_miou
                if is_best:
                    best_miou, best_step = record["dev"]["miou"], step
                record.update(best_miou=best_miou, best_step=best_step)
                if getattr(config, "overfit", 0):
                    record["fixed_train"] = evaluate_seg(
                        raw_model, dataset, config.batch_size, workers=config.workers,
                        prefetch_factor=getattr(config, "prefetch_factor", 1), runtime=runtime)
            if runtime.primary:
                with history.open("a") as f:
                    f.write(json.dumps(record) + "\n")
                write_tensorboard(writer, record)
                print(json.dumps(record), flush=True)
            if (is_best or step % eval_interval == 0 or step == max_updates
                    or (getattr(args, "stop_after", None) is not None and step >= args.stop_after)):
                rng_by_rank = runtime.gather_rng()
                temporal_by_rank = None
                if temporal:
                    local_state = streams.state_dict()
                    temporal_by_rank = [None]*runtime.world_size
                    if runtime.distributed:
                        dist.all_gather_object(temporal_by_rank, local_state)
                    else:
                        temporal_by_rank[0] = local_state
                if runtime.primary:
                    checkpoint = dict(
                        state_dict=raw_model.state_dict(), training_contract=contract,
                        stage_parent=stage_parent, best_miou=best_miou, best_step=best_step,
                        optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), step=step,
                        data_epoch=epoch, data_cursor=cursor, rng_by_rank=rng_by_rank,
                        temporal_by_rank=temporal_by_rank,
                    )
                    temp = output / "checkpoint.tmp.pth"
                    torch.save(checkpoint, temp)
                    temp.replace(output / "checkpoint.pth")
                    if is_best:
                        shutil.copyfile(output / "checkpoint.pth", output / "best_checkpoint.tmp.pth")
                        (output / "best_checkpoint.tmp.pth").replace(output / "best_checkpoint.pth")
                    writer.flush()
                runtime.barrier()
            # Optional bounded diagnostics stop without changing the LR budget.
            if getattr(args, "stop_after", None) is not None and step >= args.stop_after:
                break
    # Close the generator before returning so bounded diagnostics stop loader workers.
    iterator.close()
    return raw_model


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
        saved_contract = state.get("training_contract", {})
        validate_input_spec(saved_contract.get("event_input"),
                            getattr(config, "event_representation", "rvt_histogram"))
        saved_window = saved_contract.get("temporal", {}).get("window", 0)
        if saved_window != getattr(model.backbone, "temporal_window", 0):
            raise ValueError("Evaluation temporal architecture differs from checkpoint")
        saved_mode = saved_contract.get("fusion_mode")
        if saved_mode is not None and saved_mode != model.backbone.fusion_mode:
            raise ValueError("Evaluation fusion mode differs from checkpoint contract")
        model.load_state_dict(state.get("state_dict", state), strict=True)
    dataset = DSECFrames(args.data_root, args.data_list,
                         representation=getattr(config, "event_representation", "rvt_histogram"))
    with torch.amp.autocast(device.type, enabled=args.fp16):
        metrics = evaluate_seg(
            model,
            dataset,
            config.batch_size,
            workers=getattr(config, "workers", 0),
            prefetch_factor=getattr(config, "prefetch_factor", 1),
        )
    metrics["event_input"] = dataset.input_spec
    metrics["event_data"] = dataset.data_contract
    out = Path(config.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"evaluation_{args.data_list}.json").write_text(
        json.dumps(metrics, indent=2)
    )
    print(json.dumps(metrics), flush=True)
