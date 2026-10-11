"""Locked sequence-only development views; original PEOD manifests/cache stay read-only."""
import json
import tomllib
from pathlib import Path
from hmnet.dataset.peod_frames import PEODFrames
from hmnet.dataset.task_frames import sha_file
from hmnet.dataset.temporal_frames import manifest_signature

PRECISION_CONTRACT = 'peod_float_targets_fp32_decode_assignment_iou_v2'


class PEODRecipeFrames(PEODFrames):
    def __init__(self, root, split, representation='rvt_histogram', modality='rgbdvs',
                 augment=False, limit=0, selection=None):
        if split not in ('train', 'dev', 'val', 'test'):
            raise ValueError('Recipe splits: train/dev/val/test')
        if selection is None:
            raise ValueError('A locked development selection is required')
        selection = Path(selection)
        spec = tomllib.loads(selection.read_text())
        if spec['schema'] != 'peod_sequence_development_v1':
            raise ValueError('Development selection schema differs')
        base = 'train' if split in ('train', 'dev') else split
        root = Path(root)
        for name, expected in spec['parent_sha256'].items():
            if sha_file(root/f'{name}.json') != expected:
                raise ValueError(f'Parent manifest changed: {name}')
        super().__init__(root, base, representation, modality, augment, 0)
        train = set(spec['train_sequences']); dev = set(spec['dev_sequences'])
        if len(train) != 70 or len(dev) != 18 or train & dev:
            raise ValueError('Expected 70/18 disjoint sequence groups')
        manifests = {name: json.loads((root/f'{name}.json').read_text()) for name in ('train','val','test')}
        groups = {name: {s['sequence'] for s in m['samples']} for name,m in manifests.items()}
        if train | dev != groups['train'] or (train | dev) & (groups['val'] | groups['test']):
            raise ValueError('Incomplete split or sequence leakage')
        if split in ('train', 'dev'):
            allowed = train if split == 'train' else dev
            self.samples = [s for s in self.samples if s['sequence'] in allowed]
        full_samples = len(self.samples)
        self.samples = self.samples[:limit or None]
        if not self.samples:
            raise ValueError('Empty development view')
        self.split = split
        self.data_contract.update(split=split, diagnostic_subset=bool(limit), limit=limit,
            recipe_split=dict(selection_sha256=sha_file(selection), schema=spec['schema'],
                parent_split=base, full_samples=full_samples,
                sample_identity_sha256=manifest_signature(self),
                grouping='whole_published_DAT_sequence; cross-sequence recording IDs unavailable'))
