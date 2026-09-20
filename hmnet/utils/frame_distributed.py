"""Frame-training DDP helpers: global batches, collectives and per-rank RNG."""
import math
import os
import random
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist


class GlobalBatchSampler:
    """Split the SAME shuffled global batch; never pad or repeat tail samples."""
    def __init__(self, sampler, batch_size, rank=0, world_size=1):
        if batch_size % world_size:
            raise ValueError("Global batch_size must be divisible by world_size")
        tail = len(sampler) % batch_size
        if len(sampler) < world_size or 0 < tail < world_size:
            raise ValueError("Every rank needs a sample in the final global batch; use fewer GPUs")
        self.sampler, self.batch_size = sampler, batch_size
        self.rank, self.world_size = rank, world_size

    def __len__(self):
        return math.ceil(len(self.sampler) / self.batch_size)

    def __iter__(self):
        batch = []
        for index in self.sampler:
            batch.append(index)
            if len(batch) == self.batch_size:
                yield batch[self.rank::self.world_size]
                batch = []
        if batch:
            yield batch[self.rank::self.world_size]


class DistributedRun:
    def __init__(self, args):
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.distributed = self.world_size > 1
        if args.distributed != self.distributed or (self.distributed and args.single):
            raise ValueError("Use torchrun --nproc_per_node=N with --distributed, or --single without torchrun")
        self.device = torch.device("cuda", self.local_rank)
        torch.cuda.set_device(self.device)
        if self.distributed:
            dist.init_process_group("nccl", timeout=timedelta(minutes=60))

    @property
    def primary(self):
        return self.rank == 0

    def reduce(self, values, op=dist.ReduceOp.SUM):
        tensor = torch.as_tensor(values, dtype=torch.float64, device=self.device)
        if self.distributed:
            dist.all_reduce(tensor, op=op)
        return tensor.cpu().tolist()

    def all_true(self, value):
        return bool(self.reduce([int(value)], dist.ReduceOp.MIN)[0])

    def barrier(self):
        if self.distributed:
            dist.barrier(device_ids=[self.local_rank])

    def close(self):
        if self.distributed:
            dist.destroy_process_group()

    def workers(self, total):
        return total // self.world_size + int(self.rank < total % self.world_size)

    def rng_state(self):
        return dict(python=random.getstate(), numpy=np.random.get_state(),
                    torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(self.device))

    def gather_rng(self):
        local = self.rng_state()
        if not self.distributed:
            return [local]
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, local)
        return gathered

    def restore_rng(self, states):
        if len(states) != self.world_size:
            raise ValueError("Resume world size differs from saved RNG states")
        state = states[self.rank]
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state(state["cuda"], self.device)


def validate_segmentation_ddp(model):
    # Pixel-count weighting below is exact for the current unweighted mean CE
    # (main + auxiliary at full GT resolution), not arbitrary future losses.
    for head in (model.seg_head, model.aux_head):
        criterion = head.criterion
        ce = criterion.ce.ce_loss
        if (ce.weight is not None or ce.reduction != "mean" or ce.ignore_index != 255
                or criterion.id2trainid is not None
                or any(getattr(criterion, name) != 0 for name in (
                    "coef_rce", "coef_dice", "coef_jaccard", "coef_logdice",
                    "coef_logjaccard", "coef_lovasz", "coef_miou"))):
            raise ValueError("DDP currently requires unweighted mean segmentation CE with ignore=255")
