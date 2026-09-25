"""Deterministic sequence lanes; every frame is visited once, never padded.

Training concatenates shuffled whole sequences then partitions them into B
balanced contiguous lanes. All batches have B current frames except the tail.
A lane boundary, scene boundary or timestamp gap explicitly resets history.
Validation assigns whole sequences to lanes/ranks, never splits a scene into
independent chunks simply to fill the configured batch size.
"""
from collections import defaultdict
import hashlib
import json
import math
import torch


def manifest_signature(dataset):
    fields = [(s["file"], s["sequence"], s["target_us"]) for s in dataset.samples]
    return hashlib.sha256(json.dumps(fields, separators=(",", ":")).encode()).hexdigest()


class SequenceBatchSampler:
    def __init__(self, dataset, batch_size, rank=0, world_size=1, training=True, seed=42):
        if batch_size < 1 or batch_size % world_size:
            raise ValueError("Global sequence batch must be divisible by world size")
        self.dataset, self.batch_size = dataset, batch_size
        self.rank, self.world_size, self.training, self.seed = rank, world_size, training, seed
        self.epoch, self.skip = 0, 0
        groups = defaultdict(list)
        for i, s in enumerate(dataset.samples):
            groups[s["sequence"]].append(i)
        self.groups = [sorted(groups[k], key=lambda i: dataset.samples[i]["target_us"])
                       for k in sorted(groups)]
        for indices in self.groups:
            times = [dataset.samples[i]["target_us"] for i in indices]
            if any(a >= b for a, b in zip(times, times[1:])):
                raise ValueError("Duplicate/non-increasing timestamps within a sequence")
        tail = len(dataset) % batch_size
        if training and (len(dataset) < world_size or 0 < tail < world_size):
            raise ValueError("Final batch cannot feed all ranks; use fewer GPUs")

    def set_epoch(self, epoch, skip=0):
        """`skip` batches of `epoch` were already consumed before a resume.

        Swallowing them here keeps the fast-forward pure index arithmetic, so the
        loader never asks a worker to decode a batch the trainer would discard.
        """
        if skip < 0 or (self.training and skip > len(self)):
            raise ValueError("Resume cursor is outside the epoch")
        self.epoch, self.skip = epoch, skip

    def __len__(self):
        if self.training:
            return math.ceil(len(self.dataset)/self.batch_size)
        return sum(1 for _ in self.__iter__())

    def __iter__(self):
        rng = torch.Generator().manual_seed(self.seed + self.epoch)
        if self.training:
            order = torch.randperm(len(self.groups), generator=rng).tolist()
            flat = [i for g in order for i in self.groups[g]]
            n, extra = divmod(len(flat), self.batch_size)
            lanes, start = [], 0
            for lane in range(self.batch_size):
                end = start+n+int(lane < extra)
                lanes.append(flat[start:end]); start=end
            # One consistent transform per uninterrupted lane, both modalities.
            flips = torch.randint(2, (self.batch_size,), generator=rng).tolist()
            previous = [None]*self.batch_size
            skipped = 0
            for t in range(max(map(len, lanes))):
                batch=[]
                for slot, indices in enumerate(lanes):
                    if t >= len(indices) or slot % self.world_size != self.rank:
                        continue
                    i=indices[t]; sample=self.dataset.samples[i]; old=previous[slot]
                    reset=(old is None or old[0] != sample["sequence"] or
                           not 0 < sample["target_us"]-old[1] <= getattr(self.dataset, "reset_gap_us", 75000))
                    batch.append((i, slot, reset, bool(flips[slot]) and self.dataset.augment))
                    previous[slot]=(sample["sequence"], sample["target_us"])
                if not batch:
                    continue
                # Resumed batches still advance `previous`, so the first yielded
                # batch carries the exact reset flags a full replay would produce.
                if skipped < self.skip:
                    skipped += 1
                    continue
                yield batch
        else:
            # Whole sequences are independent; ranks with no sequence still join
            # the final metric all-reduce but do no model forward collectives.
            pending=iter(self.groups[self.rank::self.world_size])
            count=max(1,self.batch_size//self.world_size)
            active=[next(pending,None) for _ in range(count)]; positions=[0]*count
            while any(x is not None for x in active):
                batch=[]
                for slot, indices in enumerate(active):
                    if indices is None: continue
                    pos=positions[slot];i=indices[pos]
                    reset=pos==0 or not 0 < (self.dataset.samples[i]["target_us"]-
                              self.dataset.samples[indices[pos-1]]["target_us"]) <= getattr(self.dataset, "reset_gap_us", 75000)
                    batch.append((i,slot,reset,False))
                    positions[slot]+=1
                    if positions[slot]==len(indices):
                        active[slot]=next(pending,None);positions[slot]=0
                yield batch
