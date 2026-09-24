"""CPU regression checks for binary pseudo teachers; no dataset/GPU required."""
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
import torch
from hmnet.models.efficientvit_tasks import build_frame_task
from hmnet.models.base.event_repr.polarity import polarity_counts
from hmnet.utils.binary_teacher import BinaryTeacherPredictor, load_binary_teacher
from scripts.prepare_dsec_b1 import read_window
from scripts.pseudo_dsec_async import confidence_mask, agreement_mask, validate_audit


class BinaryTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.model = build_frame_task('segmentation', modality='rgbdvs', fusion_mode='add',
                                     temporal_window=2, event_channels=2).eval()

    def meta(self, i, **changes):
        value = dict(curr_time_org=100000+i*25000, sequence='seq', has_gt=bool(i % 2),
                     reset=i == 0, rgb_time=100000+(i//2)*50000, rgb_id=f'rgb{i//2}',
                     height=32, width=32)
        value.update(changes)
        return value

    def test_interleaved_matches_two_independent_synchronous_streams(self):
        predictor = BinaryTeacherPredictor(self.model)
        states = [self.model.backbone.zero_temporal_state(1) for _ in range(2)]
        images = [torch.randn(1, 3, 32, 32) for _ in range(3)]
        with torch.inference_mode():
            for i in range(6):
                events = torch.randint(2, (1, 2, 32, 32)).float()
                meta = self.meta(i)
                actual = predictor.step(events, meta, images[i//2] if i % 2 == 0 else None)
                features, states[i % 2] = self.model.backbone(events, images[i//2], states[i % 2])
                expected = self.model.seg_head(self.model.neck(list(features)), [meta])
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
                for actual_state, expected_state in zip(predictor.states[i % 2], states[i % 2]):
                    torch.testing.assert_close(actual_state, expected_state, rtol=1e-5, atol=1e-5)

    def test_sequence_and_gap_clear_both_lanes(self):
        for changes in ({'sequence': 'next'}, {'curr_time_org': 500000, 'rgb_time': 500000}):
            predictor = BinaryTeacherPredictor(self.model)
            events = torch.ones(1, 2, 32, 32)
            image = torch.randn(1, 3, 32, 32)
            predictor.step(events, self.meta(0), image)
            predictor.step(events, self.meta(1), image)
            meta = self.meta(2, **changes)
            actual = predictor.step(events, meta, image)
            expected = BinaryTeacherPredictor(self.model).step(events, meta, image)
            torch.testing.assert_close(actual, expected)
            self.assertIsNone(predictor.states[1])

    def test_future_rgb_and_reverse_time_rejected(self):
        events = torch.ones(1, 2, 32, 32); image = torch.ones(1, 3, 32, 32)
        with self.assertRaisesRegex(ValueError, 'causal'):
            BinaryTeacherPredictor(self.model).step(events, self.meta(0, rgb_time=100001), image)
        predictor = BinaryTeacherPredictor(self.model)
        predictor.step(events, self.meta(0), image)
        with self.assertRaisesRegex(ValueError, 'increase'):
            predictor.step(events, self.meta(1, curr_time_org=99999), image)

    def test_counts_do_not_wrap_at_256(self):
        events = np.zeros((256, 4), dtype=np.int64)
        counts = polarity_counts(events, height=1, width=1)
        self.assertEqual(int(counts[0, 0, 0]), 256)
        self.assertTrue((counts > 0)[0, 0, 0])

    def test_closed_window_includes_left_and_right_only(self):
        with tempfile.TemporaryDirectory() as directory:
            with h5py.File(Path(directory)/'events.h5', 'w') as f:
                f['t_offset'] = 1000
                f['ms_to_idx'] = np.array([0, 4, 4])
                f['events/t'] = np.array([9, 10, 20, 21])
                for key in ('x', 'y', 'p'):
                    f[f'events/{key}'] = np.zeros(4, dtype=np.int64)
                events = read_window(f, 1020, 10)
                np.testing.assert_array_equal(events[:, 0], [10, 20])

    def test_single_and_agreement_confidence_masks(self):
        a = torch.tensor([[[[10., 0., 10.]], [[0., 0., 0.]]]])
        b = torch.tensor([[[[10., 0., 0.]], [[0., 0., 10.]]]])
        label, _ = confidence_mask([a], .95)
        self.assertEqual(label.tolist(), [[[0, 255, 0]]])
        label, _ = agreement_mask(a, b, .95)
        self.assertEqual(label.tolist(), [[[0, 255, 255]]])

    def test_audit_rejects_changed_mode_cache_teacher_or_threshold(self):
        import copy
        teachers = {'binary': {'checkpoint': '/frozen/best.pth', 'sha256': 'teacher-sha'}}
        protocol = {'teacher_mode': 'binary', 'binary_manifest_sha256': 'data-sha'}
        audit = dict(passed=True, teachers=teachers, teacher_protocol=protocol,
                     confidence=.95, grid_sha256='grid')
        validate_audit(audit, teachers, protocol, .95, 'grid')
        for key, value in [('passed', False), ('teachers', {}), ('confidence', .9),
                           ('teacher_protocol', {'teacher_mode': 'binary_c'}),
                           ('teacher_protocol', dict(protocol, binary_manifest_sha256='changed')),
                           ('grid_sha256', 'other-grid')]:
            stale = copy.deepcopy(audit); stale[key] = value
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                validate_audit(stale, teachers, protocol, .95, 'grid')

    def test_wrong_teacher_contract_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'wrong.pth'
            torch.save(dict(training_contract={}), path)
            with self.assertRaisesRegex(ValueError, 'Event input contract'):
                load_binary_teacher(path, 'cpu')


if __name__ == '__main__':
    unittest.main()
