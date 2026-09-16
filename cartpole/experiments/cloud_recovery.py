"""E6d-r1: repairable audit journals and independent durable loss accounting."""
import hashlib
import json
from pathlib import Path

from cartpole.experiments.cloud_io import atomic_bytes, atomic_json, sha256


def read_journal(path, *, repair=False):
    path = Path(path)
    if not path.exists():
        return []
    original = path.read_bytes(); lines = original.splitlines(keepends=True)
    rows = []; replacement = None
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('journal records must be JSON objects')
        except (ValueError, UnicodeError) as exc:
            if not repair or i != len(lines)-1 or line.endswith(b'\n'):
                raise ValueError(f'corrupt journal {path}, line {i+1}; not an incomplete final fragment') from exc
            replacement = b''.join(lines[:i])
            break
        rows.append(row)
        if i == len(lines)-1 and not line.endswith(b'\n'):
            if not repair:
                raise ValueError(f'unterminated journal record: {path}')
            replacement = original+b'\n'
    if replacement is not None:
        digest = hashlib.sha256(original).hexdigest()
        directory = path.parent/'journal_repairs'; directory.mkdir(exist_ok=True)
        backup = directory/(path.name+'.'+digest+'.original')
        receipt = backup.with_suffix('.json')
        # Receipt/backup precede replacement; a crash can repeat this operation
        # by content identity without losing the original bytes or repair event.
        if not backup.exists():
            atomic_bytes(backup, original)
        if sha256(backup) != digest:
            raise ValueError('journal repair backup checksum mismatch')
        if not receipt.exists():
            atomic_json(receipt, dict(event='journal_repaired', repair_id=path.name+'.'+digest,
                        journal=path.name, original_sha256=digest, backup=str(backup.relative_to(path.parent)),
                        original_bytes=len(original), repaired_bytes=len(replacement),
                        complete_records=len(rows), discarded_incomplete_tail=len(replacement) < len(original)))
        atomic_bytes(path, replacement)
    return rows


def recover_journals(run):
    run = Path(run)
    for path in sorted(run.glob('*.jsonl')):
        read_journal(path, repair=True)
    history = read_journal(run/'history.jsonl')
    known = {r.get('repair_id') for r in history if r.get('event') == 'journal_repaired'}
    from cartpole.experiments.cloud_io import append_json
    for path in sorted((run/'journal_repairs').glob('*.json')):
        record = json.loads(path.read_text())
        if record['repair_id'] not in known:
            append_json(run/'history.jsonl', record)
    return read_journal(run/'history.jsonl')


def reconcile_losses(run, checkpoint, progress, segments):
    """Idempotent per-segment charges, committed BEFORE sealing or new resume.

    Never derive money/transition accounting from a possibly torn JSONL row.
    Losing this atomic state is an explicit failure, not a budget reset.
    """
    run = Path(run); path = run/'recovery_state.json'
    if not path.exists():
        raise ValueError('missing E6d-r1 recovery accounting; legacy runs require their saved code')
    data = json.loads(path.read_text())
    if data['schema'] != 1:
        raise ValueError('unsupported recovery accounting')
    before = loss_totals(data)
    minimum = checkpoint.get('recovery_accounting', {})
    if any(before[key] < minimum.get(key, 0) for key in before):
        raise ValueError('recovery accounting regressed behind checkpoint')
    charges = data['segments']
    for name, segment in segments.items():
        if 'duration_seconds' not in segment:
            entry = charges.setdefault(name, dict(discarded_known=0, uncertain=0))
            entry['uncertain'] = max(entry['uncertain'], 1000)
    if progress:
        name = progress['segment']
        if name not in segments:
            raise ValueError('progress names an unknown segment')
        discarded = max(0, progress['transitions']-checkpoint['transitions'])
        if discarded:
            entry = charges.setdefault(name, dict(discarded_known=0, uncertain=0))
            entry['discarded_known'] = max(entry['discarded_known'], discarded)
    atomic_json(path, data)
    return data, {k: loss_totals(data)[k]-before[k] for k in before}


def loss_totals(data):
    for row in data['segments'].values():
        if any(type(row.get(k)) is not int or row[k] < 0 for k in ('discarded_known','uncertain')):
            raise ValueError('invalid durable loss counters; refusing a budget refund')
    return {key: sum(row[key] for row in data['segments'].values())
            for key in ('discarded_known', 'uncertain')}
