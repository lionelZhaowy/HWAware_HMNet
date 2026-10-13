#!/usr/bin/env python3
"""CPU regression checks for interruption-safe history rewind; no training outputs."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from hmnet.utils.resume_history import rewind_metrics_history
import hmnet.utils.resume_history as module


class HistoryRecovery(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'metrics.jsonl'

    def test_torn_tail_and_uncheckpointed_steps(self):
        raw = b'{"step":1,"loss":0.25}\n{"step":2}\n{"step":3}\n{"step":4,"e'
        self.path.write_bytes(raw)
        r = rewind_metrics_history(self.path, 2)
        self.assertEqual(self.path.read_bytes(), b'{"step":1,"loss":0.25}\n{"step":2}\n')
        self.assertEqual(r['discarded_partial_final_line'], 4)
        self.assertEqual(r['removed_after_checkpoint'], 1)
        self.assertEqual(Path(r['backup']).read_bytes(), raw)
        self.assertFalse(rewind_metrics_history(self.path, 2)['changed'])

    def test_valid_unterminated_last_record_is_retained(self):
        self.path.write_bytes(b'{"step":1}')
        rewind_metrics_history(self.path, 1)
        self.assertEqual(self.path.read_bytes(), b'{"step":1}\n')

    def test_torn_utf8_tail(self):
        self.path.write_bytes(b'{"step":1}\n{"step":2,"text":"\xe4\xb8')
        r=rewind_metrics_history(self.path,1)
        self.assertEqual(r['discarded_partial_final_line'],2)
        self.assertEqual(json.loads(self.path.read_bytes())['step'],1)

    def test_interior_or_complete_corruption_fails_without_writes(self):
        for raw in [b'{"step":1}\n{bad}\n{"step":2}\n', b'{"step":1}\n{"step":2,"e\n', b'{bad}\n{"step":2,"e']:
            with self.subTest(raw=raw):
                self.path.write_bytes(raw)
                with self.assertRaises(ValueError):rewind_metrics_history(self.path,1)
                self.assertEqual(self.path.read_bytes(),raw)
                self.assertFalse((self.path.parent/'history_backups').exists())

    def test_invalid_step_schema_fails(self):
        for value in [True,-1,'2',None]:
            raw=json.dumps(dict(step=value)).encode()+b'\n';self.path.write_bytes(raw)
            with self.assertRaises(ValueError):rewind_metrics_history(self.path,1)
            self.assertEqual(self.path.read_bytes(),raw)

    def test_atomic_replace_failure_preserves_original(self):
        raw=b'{"step":1}\n{"step":2}\n';self.path.write_bytes(raw)
        replace=module.os.replace
        def fail_metrics(src,dest):
            if Path(dest)==self.path:raise OSError('simulated full disk / replace failure')
            return replace(src,dest)
        with patch.object(module.os,'replace',side_effect=fail_metrics):
            with self.assertRaises(OSError):rewind_metrics_history(self.path,1)
        self.assertEqual(self.path.read_bytes(),raw)
        self.assertEqual(next((self.path.parent/'history_backups').glob('*.bak')).read_bytes(),raw)
        self.assertFalse(list(self.path.parent.rglob('*.tmp')))
        self.assertEqual(rewind_metrics_history(self.path,1)['kept_records'],1)

    def test_backup_failure_preserves_original(self):
        raw=b'{"step":1}\n{"step":2}';self.path.write_bytes(raw)
        with patch.object(module.os,'replace',side_effect=OSError('simulated backup failure')):
            with self.assertRaises(OSError):rewind_metrics_history(self.path,1)
        self.assertEqual(self.path.read_bytes(),raw)

    def test_noop_keeps_history_without_backup(self):
        raw=b'{"step":1}\n';self.path.write_bytes(raw)
        self.assertFalse(rewind_metrics_history(self.path,1)['changed'])
        self.assertFalse((self.path.parent/'history_backups').exists())


if __name__=='__main__':unittest.main()
