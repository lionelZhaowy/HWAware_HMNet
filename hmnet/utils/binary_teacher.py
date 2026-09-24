"""Frozen synchronous binary teacher, evaluated on two causal 50ms streams."""
import json
from pathlib import Path
import numpy as np
import torch
from hmnet.models.efficientvit_tasks import build_frame_task
from hmnet.models.base.event_repr.polarity import validate_input_spec
from hmnet.utils.async_checkpoint import file_sha256


def load_binary_teacher(checkpoint, device):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    contract = saved.get('training_contract', {})
    validate_input_spec(contract.get('event_input'), 'polarity_binary')
    temporal = contract.get('temporal', {})
    if (contract.get('modality') != 'rgbdvs' or contract.get('fusion_mode') != 'add'
            or temporal.get('window') != 2 or temporal.get('branch') != 'dvs'
            or temporal.get('state_dtype') != 'float32' or contract.get('asynchronous')):
        raise ValueError('Expected synchronous binary SimpleAdd / DVS M=2 teacher')
    model = build_frame_task('segmentation', modality='rgbdvs', fusion_mode='add',
                             temporal_window=2, event_channels=2)
    model.load_state_dict(saved['state_dict'], strict=True)
    return model.to(device).eval(), contract


class BinaryTeacherInputs:
    def __init__(self, root, grid, contract):
        self.root = Path(root)
        manifest_path = self.root/'manifest.json'
        self.manifest = json.loads(manifest_path.read_text())
        expected = dict(format='dsec_async_binary_teacher_v1', complete=True,
                        grid_sha256=grid['grid_sha256'], window_us=50000,
                        boundary='[start,end]', channels=2, dtype='uint8',
                        representation='polarity_binary', polarity_order=[0, 1],
                        temporal_policy='two_interleaved_50ms_streams_v1')
        if any(self.manifest.get(k) != v for k, v in expected.items()):
            raise ValueError('Binary teacher cache contract differs')
        if self.manifest['polarity_manifest_sha256'] != contract['event_data']['manifest_sha256']:
            raise ValueError('Binary cache was not verified against teacher training inputs')
        keys = {f"{s['sequence']}/{s['target_us']}" for s in grid['samples']}
        if keys != set(self.manifest['files']):
            raise ValueError('Incomplete binary teacher time grid')
        self.signature = file_sha256(manifest_path)

    def read(self, meta):
        key = f"{meta['sequence']}/{meta['curr_time_org']}"
        record = self.manifest['files'][key]
        path = self.root/record['file']
        if file_sha256(path) != record['sha256']:
            raise ValueError(f'Binary input hash mismatch: {key}')
        with np.load(path, allow_pickle=False) as values:
            binary = values['binary']
        if (binary.shape != (2, 440, 640) or binary.dtype != np.uint8
                or np.any(binary > 1)):
            raise ValueError('Invalid binary teacher input')
        return torch.from_numpy(binary.copy()).float()[None]


class BinaryTeacherPredictor:
    """Keep GT and midpoint histories separate to preserve training cadence.

    Both lanes use only past events/RGB. has_gt selects a time-grid phase,
    never supplies semantic labels. A gap or sequence reset clears both lanes.
    RGB projections are shared read-only across lanes for this frozen Add model.
    """
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.states = [None, None]
        self.times = [None, None]
        self.cache = self.ids = self.rgb_time = None
        self.sequence = self.last_time = None

    @torch.no_grad()
    def step(self, events, meta, rgb=None):
        if self.model.training:
            raise ValueError('Frozen teacher must use eval()')
        time = int(meta['curr_time_org'])
        if (meta.get('reset', False) or self.sequence != meta['sequence']
                or (self.last_time is not None and time-self.last_time > 37500)):
            self.reset()
        if self.last_time is not None and time <= self.last_time:
            raise ValueError('Teacher event time must increase')
        lane = int(bool(meta['has_gt']))
        device = next(self.model.parameters()).device
        events = events.to(device)
        if events.shape[0] != 1 or events.shape[1] != 2:
            raise ValueError('Binary teacher expects [1,2,H,W]')
        if self.states[lane] is None or time-self.times[lane] > 75000:
            self.states[lane] = self.model.backbone.zero_temporal_state(1)
        if rgb is not None:
            source_time = int(meta['rgb_time'])
            if source_time > time or (self.rgb_time is not None and source_time < self.rgb_time):
                raise ValueError('Teacher RGB must be causal and monotonic')
            if self.ids != [meta['rgb_id']]:
                self.cache = self.model.backbone.encode_rgb(rgb.to(device))
                self.ids = [meta['rgb_id']]
                self.rgb_time = source_time
        if self.cache is None:
            h, w = events.shape[-2:]
            self.cache = tuple(events.new_zeros(1, 256, (h+s-1)//s, (w+s-1)//s)
                               for s in (4, 8, 16, 32))
        features, self.states[lane] = self.model.backbone.async_step(events, self.cache, self.states[lane])
        self.times[lane] = time
        self.sequence, self.last_time = meta['sequence'], time
        return self.model.seg_head(self.model.neck(list(features)), [meta])
