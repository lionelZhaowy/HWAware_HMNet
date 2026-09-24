"""CPU checks for strict defaults, research opt-in and immutable audit identity."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from hmnet.dataset.dsec_async import DSECAsync
from hmnet.utils.async_config import AsyncSettings
from hmnet.utils.pseudo_audit import audit_failures, validate_audit, validate_pseudo_audit
from scripts.train_async import parse_settings


class PseudoAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cache = self.root/'cache'; self.cache.mkdir()
        self.pseudo = self.root/'pseudo'; self.pseudo.mkdir()
        self.window, self.bins = AsyncSettings.window_us, AsyncSettings.bins
        grid = dict(format='dsec_async_v1', window_us=self.window, bins=self.bins,
                    boundary='(start,end]', count_cutoff=10, fastmode=True, grid_sha256='grid',
                    samples=[dict(sequence='s', split='train', has_gt=False, target_us=25),
                             dict(sequence='s', split='train', has_gt=True, target_us=50)])
        (self.cache/'manifest.json').write_text(json.dumps(grid))
        self.audit = dict(format='dsec_async_audit_v1', passed=False,
                          teachers={'B': {'sha256': 'b'}, 'C': {'sha256': 'c'}},
                          teacher_protocol={'teacher_mode': 'bc'}, confidence=.95, grid_sha256='grid',
                          precision=.98, coverage=.89, accepted_by_gt_class=[10000, 5000, 2000],
                          class_precision=[.99, .95, .3],
                          thresholds=dict(min_precision=.9, min_coverage=.1,
                                          min_class_precision=.7, min_class_pixels=1000))
        payload = json.dumps(self.audit).encode()
        (self.pseudo/'teacher_audit.json').write_bytes(payload)
        self.manifest = dict(format='dsec_async_pseudo_v1', audit_passed=False,
                             audit_file='teacher_audit.json', audit_sha256=hashlib.sha256(payload).hexdigest(),
                             teachers=self.audit['teachers'], teacher_protocol=self.audit['teacher_protocol'],
                             confidence=.95, grid_sha256='grid', split='train', files={'s/25': '25.npy'},
                             audit_override=dict(enabled=True, reason='Explicit experiment',
                                                 failed_checks=['gt_class_2_precision']))
        self.write_manifest()

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self):
        (self.pseudo/'manifest.json').write_text(json.dumps(self.manifest))

    def dataset(self, allow=False, pseudo=True):
        return DSECAsync(self.cache, 'train', clips=True, window_us=self.window, bins=self.bins,
                         pseudo_root=self.pseudo if pseudo else None, allow_failed_pseudo_audit=allow)

    def check_audit(self, audit=None, allow=False, reason=''):
        validate_audit(audit or self.audit, self.audit['teachers'], self.audit['teacher_protocol'],
                       .95, 'grid', allow_failed=allow, reason=reason)

    def test_failed_generation_rejected_by_default(self):
        with self.assertRaisesRegex(ValueError, 'Failed audit'):
            self.check_audit()

    def test_opt_in_needs_reason_and_preserves_failure(self):
        with self.assertRaises(ValueError): self.check_audit(allow=True)
        self.check_audit(allow=True, reason='Explicit experiment')
        self.assertFalse(self.audit['passed'])
        self.assertEqual(audit_failures(self.audit), ['gt_class_2_precision'])

    def test_opt_in_does_not_bypass_identity_checks(self):
        for key, value in [('teachers', {}), ('teacher_protocol', {}), ('confidence', .90),
                           ('grid_sha256', 'changed'), ('passed', None)]:
            audit = copy.deepcopy(self.audit); audit[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check_audit(audit, allow=True, reason='Explicit experiment')

    def test_training_requires_separate_opt_in(self):
        with self.assertRaisesRegex(ValueError, 'allow-failed-pseudo-audit'): self.dataset()
        dataset = self.dataset(allow=True)
        self.assertEqual(dataset.contract()['phase'], 2)
        self.assertEqual(dataset.pseudo_files, {'s/25': '25.npy'})
        self.assertFalse(json.loads((self.pseudo/'manifest.json').read_text())['audit_passed'])

    def test_tampered_original_audit_rejected(self):
        (self.pseudo/'teacher_audit.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'SHA'): self.dataset(allow=True)

    def test_missing_reason_or_failure_record_rejected(self):
        for field, value in [('reason', ''), ('enabled', False), ('failed_checks', [])]:
            manifest = copy.deepcopy(self.manifest);manifest['audit_override'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_pseudo_audit(manifest, self.pseudo, allow_failed=True)

    def test_legacy_passed_manifest_remains_usable(self):
        self.manifest = dict(format='dsec_async_pseudo_v1', audit_passed=True, grid_sha256='grid',
                             split='train', files={})
        self.write_manifest()
        self.assertEqual(self.dataset().contract()['phase'], 2)

    def test_diagnostic_fixture_still_rejected(self):
        self.manifest['diagnostic_only'] = True;self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Synthetic'): self.dataset(allow=True)

    def test_manifest_digest_binds_resume_provenance(self):
        before = self.dataset(allow=True).contract()['pseudo_sha256']
        self.manifest['audit_override']['reason'] = 'Different experiment';self.write_manifest()
        after = self.dataset(allow=True).contract()['pseudo_sha256']
        self.assertNotEqual(before, after)

    def test_phase_one_contract_unchanged(self):
        self.assertEqual(self.dataset(pseudo=False).contract(), self.dataset(allow=True, pseudo=False).contract())

    def test_cli_start_and_resume_propagate_opt_in(self):
        checkpoint = self.root/'checkpoint.pth';checkpoint.touch()
        shared = ['--single', '--stage', '2', '--epochs', '20', '--data-root', str(self.cache),
                  '--pseudo-root', str(self.pseudo), '--allow-failed-pseudo-audit']
        fresh, _ = parse_settings(shared+['--init-from', str(checkpoint)])
        resumed, _ = parse_settings(shared+['--resume', str(checkpoint)])
        self.assertEqual(fresh.get_dataset().contract(), resumed.get_dataset().contract())
        without, _ = parse_settings([v for v in shared if v != '--allow-failed-pseudo-audit'])
        with self.assertRaises(ValueError): without.get_dataset()


if __name__ == '__main__':
    unittest.main()
