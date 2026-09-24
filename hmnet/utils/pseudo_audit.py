"""Audit provenance and explicit research opt-in; never relabel failure as pass."""
import hashlib
import json
from pathlib import Path


def validate_audit(audit, provenance, protocol, confidence, grid_sha256,
                   allow_failed=False, reason=''):
    # The opt-in relaxes quality rejection only, never identity/alignment checks.
    if (audit.get('format') != 'dsec_async_audit_v1'
            or audit.get('teachers') != provenance or audit.get('confidence') != confidence
            or audit.get('teacher_protocol') != protocol or audit.get('grid_sha256') != grid_sha256
            or type(audit.get('passed')) is not bool):
        raise ValueError('Audit teacher/confidence/protocol/grid mismatch; rerun audit')
    if not audit['passed'] and not (allow_failed and isinstance(reason, str) and reason.strip()):
        raise ValueError('Failed audit: explicit exploratory opt-in and reason required (quality mismatch)')


def audit_failures(audit):
    thresholds = audit['thresholds']
    failed = []
    if audit['precision'] < thresholds['min_precision']:
        failed.append('overall_precision')
    if audit['coverage'] < thresholds['min_coverage']:
        failed.append('overall_coverage')
    if not sum(audit['accepted_by_gt_class']):
        failed.append('no_accepted_pixels')
    for index, (count, precision) in enumerate(zip(audit['accepted_by_gt_class'], audit['class_precision'])):
        if count >= thresholds['min_class_pixels'] and (precision is None or precision < thresholds['min_class_precision']):
            failed.append(f'gt_class_{index}_precision')
    return failed


def validate_pseudo_audit(pseudo, root, allow_failed=False):
    if pseudo.get('format') != 'dsec_async_pseudo_v1':
        raise ValueError('Unsupported pseudo label format')
    # Backwards compatibility for previously passing audits/diagnostic fixtures.
    if pseudo.get('audit_passed') is True:
        return
    override = pseudo.get('audit_override', {})
    reason = override.get('reason', '')
    if (pseudo.get('audit_passed') is not False or not allow_failed
            or override.get('enabled') is not True or not isinstance(reason, str) or not reason.strip()):
        raise ValueError('Failed pseudo audit requires --allow-failed-pseudo-audit and recorded exploration reason')
    if pseudo.get('audit_file') != 'teacher_audit.json':
        raise ValueError('Exploratory pseudo labels must preserve the original audit')
    payload = (Path(root)/'teacher_audit.json').read_bytes()
    if hashlib.sha256(payload).hexdigest() != pseudo.get('audit_sha256'):
        raise ValueError('Original audit SHA mismatch')
    audit = json.loads(payload)
    validate_audit(audit, pseudo['teachers'], pseudo['teacher_protocol'], pseudo['confidence'],
                   pseudo['grid_sha256'], allow_failed=True, reason=reason)
    if audit['passed'] is not False or override.get('failed_checks') != audit_failures(audit):
        raise ValueError('Exploratory audit result/failure record mismatch')
