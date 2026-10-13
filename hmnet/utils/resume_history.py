"""Recover an interrupted JSONL append without hiding interior corruption."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def _atomic_write(path, data):
    """Keep the previous file intact if writing/flushing the replacement fails."""
    fd, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):os.unlink(name)


def rewind_metrics_history(path, step, backup_dir=None):
    """Restore history to checkpoint step; checkpoint/RNG/model state are untouched.

    Only an invalid final physical line without a line terminator may be dropped.
    Invalid complete/interior records and invalid step fields fail before writes.
    Back up original bytes before atomically publishing the retained JSONL prefix.
    """
    if type(step) is not int or step < 0:raise ValueError('Invalid checkpoint step')
    path = Path(path)
    original = path.read_bytes()
    lines = original.splitlines(keepends=True)
    kept, dropped, partial_line = [], 0, None
    for index, line in enumerate(lines):
        if not line.strip():continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            if index == len(lines)-1 and not line.endswith((b'\n', b'\r')):
                partial_line = index+1
                continue
            raise ValueError(f'Corrupt metrics history at line {index+1}; original file preserved: {path}') from exc
        if not isinstance(row, dict) or type(row.get('step')) is not int or row['step'] < 0:
            raise ValueError(f'Invalid metrics step at line {index+1}; original file preserved: {path}')
        if row['step'] <= step:
            kept.append(line if line.endswith(b'\n') else line+b'\n')
        else:
            dropped += 1
    restored = b''.join(kept)
    result = dict(checkpoint_step=step, kept_records=len(kept), removed_after_checkpoint=dropped,
                  discarded_partial_final_line=partial_line, backup=None, changed=restored != original)
    if not result['changed']:return result
    backup_dir = Path(backup_dir) if backup_dir is not None else path.parent/'history_backups'
    backup_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(original).hexdigest()
    backup = backup_dir/f'{path.name}.before_resume_step_{step}.{digest}.bak'
    if backup.exists():
        if backup.read_bytes() != original:raise ValueError(f'Backup content differs: {backup}')
    else:
        _atomic_write(backup, original)
    # Detect an append/replacement during inspection; concurrent writers are unsupported.
    if path.read_bytes() != original:raise RuntimeError('Metrics history changed during resume; stop the other writer')
    _atomic_write(path, restored)
    result['backup'] = str(backup)
    return result
