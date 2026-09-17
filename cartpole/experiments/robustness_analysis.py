"""Stdlib-only analysis of structured-robustness-dev-v1; never executes policies.

--review DIR_OR_TAR [--recovery TAR] validates/imports a complete export.
--normalized FILE recomputes scientific aggregates without archives or ML libraries.
--check is read-only (including stdout-only results); it excludes --output.
--output NEW_DIRECTORY explicitly writes normalized.json and analysis.json and refuses
an existing destination. No files are written otherwise. Run with python -B to
suppress Python's own bytecode cache. Archive members are streamed, never extracted.
Output is NOT transactional: an I/O failure may leave an incomplete new directory;
existing directories are never overwritten. Atomic publication is a low-priority
follow-up, not a property of this interface.

Normalized schema v2 requires an envelope SHA binding provenance and scientific
data. The scientific SHA excludes provenance/export clocks. The envelope detects
unrehashed changes; it is not an authenticity signature. With normalized input,
source archives are not reopened: source_provenance is not_revalidated, even when
the envelope is valid. Old v1 normalized files must be reimported explicitly.

Validation levels are deliberately separate: manifest/receipt integrity, compact
consistency, and opt-in raw metric recomputation (--verify-raw-metrics). Without
that flag, saved metrics are not rederived from trajectories. Raw mode decodes one
bounded gzip episode at a time; no models, environments or pickle are loaded.

Aggregation conventions adapted from submission/scripts/reproduce_robustness.py
and the independent robustness audit; definitions checked against common.metrics,
safety_metrics, cloud_evaluation and robustness_protocol. No imports from those
modules: a clean clone works with python -S -B and the standard library alone.
"""
import argparse
from collections import Counter, defaultdict
from contextvars import ContextVar
from dataclasses import dataclass, field
from fractions import Fraction
import gzip
import io
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
import zlib

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'structured-robustness-dev-v1'
SCHEMA = 'structured-robustness-normalized-v2'
GROUPS = ('nominal', 'position', 'velocity', 'angle', 'angular_velocity', 'boundary', 'joint')
MODELS = tuple(f'{a}_seed{s}' for a in ('SAC', 'TQC') for s in range(3)) + ('classical',)
FIELDS = ('cart_position', 'cart_velocity', 'pole_angle', 'pole_angular_velocity')
CORE = ('plan.json', 'start_request.json', 'queue/queue_state.json', 'structured_report.json')
EVALUATOR_SOURCES = ('cartpole/experiments/parallel_evaluation.py',
    'cartpole/experiments/inference_compatibility.py', 'cartpole/experiments/cloud_evaluation.py',
    'cartpole/control/swing_up.py', 'cartpole/control/lqr.py')
MAX_MEMBER = 64 * 1024**2
MAX_TOTAL = 3 * 1024**3
# Scoped to a single explicit raw run. Legacy compact consistency comparisons
# also register tolerated differences; default compact/normalized APIs are intact.
NUMERICAL_AUDIT = ContextVar('robustness_numerical_audit', default=None)
# Exact metric schema for VERSION (source exports) and SCHEMA (normalized data).
METRIC_KEYS = frozenset(('completion', 'duration', 'first_hold_confirmation', 'first_hold_start',
    'hold_final', 'hold_longest', 'max_abs_applied_acceleration', 'max_abs_omega',
    'max_abs_position_interval', 'max_abs_requested_acceleration', 'max_abs_velocity',
    'reward', 'safety', 'success_episode'))
COMPLETION_KEYS = frozenset(('reason', 'time', 'terminated', 'truncated', 'transition_count', 'stop_reasons'))
SAFETY_KEYS = frozenset(('filter_enabled', 'decisions', 'accepted_decisions', 'intervention_count',
    'intervention_fraction', 'intervention_abs_sum', 'intervention_abs_max', 'intervention_abs_mean',
    'refusal_count', 'refusal_reasons', 'max_abs_position_interval', 'desired_excess_max',
    'exceeds_024', 'exceeds_024_resolved', 'intervals_exceeding_024', 'position_limit_events',
    'max_abs_endpoint_position_residual', 'max_abs_endpoint_velocity_residual', 'definitions'))
SAFETY_DEFINITIONS = {
    'position': 'max of actual Drake endpoints and analytical held-u quadratic including interior vertex',
    'exceedance': 'strict >.24; separate resolved >.24+1e-12 m; neither alters execution',
    'intervention_fraction': 'changed accepted filter decisions / all accepted filter decisions; refusals separate',
    'hold': 'original consecutive recorded states; cart safety alone is not task success'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def sha_value(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'invalid SHA256')
    return value


def parse_json(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('nonfinite JSON value: ' + value)
    return json.loads(payload, object_pairs_hook=pairs, parse_constant=nonfinite)


def read_json(path):
    return parse_json(Path(path).read_bytes())


def number(value, name, minimum=None):
    require(type(value) in (int, float) and math.isfinite(value), 'nonfinite/non-numeric ' + name)
    require(minimum is None or value >= minimum, 'negative ' + name)
    return value


def integer(value, name):
    require(type(value) is int and value >= 0, 'invalid count: ' + name)
    return value


def same(actual, expected, name):
    """Exact structure/types; numeric tolerance only for floating aggregates."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), name + ': keys differ')
        for key in expected:
            same(actual[key], expected[key], name + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), name + ': length differs')
        for i, (a, b) in enumerate(zip(actual, expected)):
            same(a, b, name + '/' + str(i))
    elif type(expected) is float:
        collector = NUMERICAL_AUDIT.get()
        if collector is not None:
            collector.check(actual, expected, 'consistency/' + name)
            return
        number(actual, name)
        require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), name + ': value differs')
    else:
        require(type(actual) is type(expected) and actual == expected, name + ': value/type differs')


def exact(actual, expected, name):
    require(canonical(actual) == canonical(expected), name + ': identity differs')


def keys(value, expected, name):
    require(isinstance(value, dict) and set(value) == set(expected), name + ': schema keys differ')


def seal_envelope(dataset):
    dataset['envelope_sha256'] = digest({k: v for k, v in dataset.items() if k != 'envelope_sha256'})
    return dataset


def validate_envelope(dataset):
    keys(dataset, ('schema', 'stage', 'protocol_sha256', 'episodes', 'scientific_sha256',
                   'provenance', 'envelope_sha256'), 'normalized envelope')
    provenance = dataset['provenance']
    require(isinstance(provenance, dict) and set(provenance) in ({'review'}, {'review', 'recovery'}),
            'provenance source roster')
    for record in provenance.values():
        keys(record, ('archive_sha256', 'expected_archive_sha_verified', 'manifest_files',
                      'export_manifest_sha256'), 'provenance record')
        if record['archive_sha256'] is not None:
            sha_value(record['archive_sha256'])
        require(type(record['expected_archive_sha_verified']) is bool, 'provenance verification flag')
        require(record['archive_sha256'] is not None or not record['expected_archive_sha_verified'],
                'provenance verification without archive')
        require(integer(record['manifest_files'], 'manifest files') > 0, 'empty provenance manifest')
        sha_value(record['export_manifest_sha256'])
    require(sha_value(dataset['envelope_sha256']) == digest(
        {k: v for k, v in dataset.items() if k != 'envelope_sha256'}), 'envelope SHA mismatch')


def safe_name(name):
    require(isinstance(name, str) and name and '\\' not in name, 'unsafe member/path')
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and str(p) == name, 'unsafe member/path: ' + name)
    return name


def unique(rows, key, label):
    result = {}
    for row in rows:
        k = key(row)
        require(k not in result, 'duplicate ' + label + ': ' + str(k))
        result[k] = row
    return result


class Protocol:
    """Read pinned JSON, not NumPy regeneration or a mutable model pointer."""
    def __init__(self, root=ROOT):
        self.root = Path(root)
        directory = self.root / 'configs/robustness'
        manifest = read_json(directory / 'manifest.json')
        require(set(manifest) == {'states.json', 'bindings.json'}, 'frozen manifest roster')
        for name, value in manifest.items():
            require(file_sha(directory / name) == sha_value(value), 'frozen SHA: ' + name)
        self.sha256 = file_sha(directory / 'manifest.json')
        states = read_json(directory / 'states.json')
        bindings = read_json(directory / 'bindings.json')
        require(states['version'] == bindings['version'] == VERSION and states['final'] is False, 'protocol version')
        self.cases = unique(states['cases'], lambda c: c['id'], 'frozen case')
        require(set(self.cases) == {f'{g}_{i:02d}' for g in GROUPS for i in range(20)}, 'frozen case roster')
        require(len({c['state_sha256'] for c in self.cases.values()}) == 140, 'duplicate frozen state')
        for cid, case in self.cases.items():
            require(case['subset'] == cid.rsplit('_', 1)[0] and case['seed'] is None, 'frozen case subset/seed')
            require(case['generator_seed'] == 2026091601 and case['generation_index'] == int(cid.rsplit('_', 1)[1]), 'generator identity')
            state = case['initial_state']
            require(set(state) == set(FIELDS), 'state fields')
            for key, value in state.items():
                number(value, key)
            require(digest(state) == case['state_sha256'], 'state SHA')
            x, v = Fraction(state['cart_position']), Fraction(state['cart_velocity'])
            limit = Fraction(.24 - 1e-6)
            require(abs(x) <= limit and abs(v) <= Fraction(2 - 1e-6)
                    and x + max(v, 0)**2 / 8 <= limit and -x + max(-v, 0)**2 / 8 <= limit, 'state outside K')
        self.models = unique(bindings['models'], lambda m: m['id'], 'frozen model')
        require(set(self.models) == set(MODELS), 'frozen model roster')
        for mid, model in self.models.items():
            expected = ('classical', None) if mid == 'classical' else (mid.split('_')[0], int(mid[-1]))
            exact([model['algorithm'], model['seed']], list(expected), 'model algorithm/seed')
        self.sources = bindings['current_sources']
        self.check_sources(self.sources)

    def check_environment(self, environment, algorithm, mode):
        # Frozen JSON contracts are readable without importing the ML stack.
        algorithm = 'SAC' if algorithm == 'classical' else algorithm
        config = read_json(self.root / f'configs/cloud/{algorithm.lower()}_training_v2_{mode}.json')
        expected = config['training_contract']
        exact({k: environment[k] for k in ('id', 'sha256', 'specification')},
              {k: expected[k] for k in ('id', 'sha256', 'specification')}, 'environment contract')
        require(environment['sha256'] == digest(environment['specification']), 'environment SHA')
        require(environment['sources'].keys() == expected['sources'].keys(), 'environment source roster')
        self.check_sources(environment['sources'])

    def check_sources(self, sources):
        require(bool(sources), 'empty source manifest')
        for name, value in sources.items():
            require(file_sha(self.root / safe_name(name)) == sha_value(value), 'source SHA: ' + name)

    def jobs(self, stage):
        require(stage in ('main', 'timing'), 'unknown/mixed stage')
        return [dict(id=f'{m}_{mode}_{g}', model_id=m, algorithm=self.models[m]['algorithm'],
                     seed=self.models[m]['seed'], mode=mode, group=g,
                     case_ids=[f'{g}_{i:02d}' for i in range(20)])
                for m in MODELS for mode in ('off', 'on') for g in GROUPS
                if stage == 'main' or (m == 'SAC_seed0' and g == 'nominal')]

    def model_sha(self, mid):
        model = self.models[mid]
        return model['nominal_sha256' if mid == 'classical' else 'model_sha256']


def flags(metrics):
    s = metrics['safety']
    return dict(technical_error=False, filter_refusal=s['refusal_count'] > 0,
                constraint_event=any(r in ('position_limit', 'velocity_limit') for r in metrics['completion']['stop_reasons']),
                desired_exceedance=s['exceeds_024_resolved'],
                working_position_exceedance=metrics['max_abs_position_interval'] > .25 + 1e-12,
                working_velocity_exceedance=metrics['max_abs_velocity'] > 2 + 1e-12,
                hard_position_exceedance=metrics['max_abs_position_interval'] > .27 + 1e-12,
                hard_velocity_exceedance=metrics['max_abs_velocity'] > 2.5 + 1e-12,
                hard_acceleration_exceedance=metrics['max_abs_applied_acceleration'] > 5 + 1e-12,
                hold_failure=not metrics['success_episode'], success=metrics['success_episode'])


def validate_metrics(m, mode):
    keys(m, METRIC_KEYS, 'metrics')
    keys(m['completion'], COMPLETION_KEYS, 'completion')
    keys(m['safety'], SAFETY_KEYS, 'safety')
    exact(m['safety']['definitions'], SAFETY_DEFINITIONS, 'safety.definitions')
    for key in ('hold_final', 'hold_longest', 'duration', 'max_abs_requested_acceleration',
                'max_abs_applied_acceleration', 'max_abs_position_interval', 'max_abs_velocity', 'max_abs_omega'):
        number(m[key], key, 0)
    number(m['reward'], 'reward')
    require(0 <= m['hold_final'] <= m['hold_longest'] <= m['duration'] <= 10, 'hold/duration order')
    c = m['completion']
    require(c['reason'] in ('horizon', 'terminal'), 'non-complete episode')
    require(type(c['stop_reasons']) is list and all(isinstance(x, str) for x in c['stop_reasons']), 'stop reasons')
    same(c['terminated'], c['reason'] == 'terminal', 'terminated')
    same(c['truncated'], c['reason'] == 'horizon', 'truncated')
    same(c['time'], m['duration'], 'completion time')
    n = integer(c['transition_count'], 'transitions')
    require(c['reason'] != 'horizon' or (abs(m['duration'] - 10) <= 1e-12 and not c['stop_reasons']), 'horizon semantics')
    same(m['success_episode'], c['reason'] == 'horizon' and m['hold_final'] >= 2 - 1e-12, 'success')
    first, confirm = m['first_hold_start'], m['first_hold_confirmation']
    require((first is None) == (confirm is None), 'incomplete first hold')
    require((first is not None) == (m['hold_longest'] >= 2 - 1e-12), 'first/longest hold')
    if first is not None:
        number(first, 'first hold', 0); number(confirm, 'hold confirmation', 0)
        require(first <= confirm <= m['duration'] and confirm - first >= 2 - 1e-12, 'hold confirmation time')
    s = m['safety']
    same(s['filter_enabled'], mode == 'on', 'filter_enabled')
    for k in ('decisions', 'accepted_decisions', 'intervention_count', 'refusal_count',
              'intervals_exceeding_024', 'position_limit_events'):
        integer(s[k], k)
    require(s['accepted_decisions'] + s['refusal_count'] == s['decisions'], 'decision denominator')
    require(s['intervention_count'] <= s['accepted_decisions'] <= n and s['decisions'] <= n, 'decision counts')
    require((mode == 'on' and s['decisions'] == n) or (mode == 'off' and s['decisions'] == 0), 'mode/decisions')
    same(s['intervention_fraction'], s['intervention_count'] / s['accepted_decisions'] if s['accepted_decisions'] else None, 'intervention fraction')
    for k, v in s['refusal_reasons'].items():
        require(isinstance(k, str), 'refusal name'); integer(v, 'refusal reason')
    require(sum(s['refusal_reasons'].values()) == s['refusal_count'], 'refusal counts')
    for k in ('intervention_abs_sum', 'intervention_abs_max', 'intervention_abs_mean',
              'max_abs_position_interval', 'desired_excess_max', 'max_abs_endpoint_position_residual',
              'max_abs_endpoint_velocity_residual'):
        number(s[k], k, 0)
    count = s['intervention_count']
    total, maximum, mean = (s['intervention_abs_' + k] for k in ('sum', 'max', 'mean'))
    # Same binary64 tolerance as aggregate comparison: 1e-12 absolute and relative.
    # Empty statistics are exact zeros; one sample has sum == mean == maximum.
    if count == 0:
        require(total == maximum == mean == 0, 'nonzero correction statistics at count=0')
    else:
        same(mean, total / count, 'correction mean')
        def less_equal(left, right, label):
            require(left <= right or math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12), label)
        less_equal(maximum, total, 'correction maximum exceeds sum')
        less_equal(mean, maximum, 'correction mean exceeds maximum')
        less_equal(total, count * maximum, 'correction sum exceeds count*maximum')
        if count == 1:
            same(total, float(maximum), 'count=1 correction sum/max')
            same(mean, float(maximum), 'count=1 correction mean/max')
    same(s['desired_excess_max'], max(0., s['max_abs_position_interval'] - .24), 'excess')
    same(s['exceeds_024'], s['max_abs_position_interval'] > .24, 'strict exceedance')
    same(s['exceeds_024_resolved'], s['max_abs_position_interval'] > .24 + 1e-12, 'resolved exceedance')
    require(s['intervals_exceeding_024'] <= n, 'interval count')
    require(bool(s['intervals_exceeding_024']) == s['exceeds_024'], 'interval exceedance count/flag')
    require(s['position_limit_events'] == c['stop_reasons'].count('position_limit'), 'position event count')
    # Two independently rounded maxima are retained, never collapsed to one.
    require(s['max_abs_position_interval'] + 1e-12 >= m['max_abs_position_interval'], 'interval maxima inconsistent')


def scientific_rows(rows, stage, protocol):
    jobs = protocol.jobs(stage)
    expected = {(j['model_id'], c, j['mode']): j for j in jobs for c in j['case_ids']}
    indexed = unique(rows, lambda r: (r['model_id'], r['case_id'], r['filter']), 'episode')
    require(indexed.keys() == expected.keys(), 'missing/unexpected episode or incomplete pair')
    for key, row in indexed.items():
        require(set(row) == {'stage', 'job_id', 'model_id', 'case_id', 'training_seed', 'filter',
                            'group', 'state_sha256', 'model_sha256', 'metrics', 'flags', 'log_sha256'},
                'normalized episode fields')
        job = expected[key]; case = protocol.cases[row['case_id']]
        exact([row['stage'], row['job_id'], row['training_seed'], row['group'], row['state_sha256'], row['model_sha256']],
              [stage, job['id'], job['seed'], job['group'], case['state_sha256'], protocol.model_sha(job['model_id'])], 'episode binding')
        sha_value(row['log_sha256'])
        validate_metrics(row['metrics'], row['filter'])
        same(row['flags'], flags(row['metrics']), 'flags')
    return [indexed[k] for k in sorted(indexed)]


def aggregate(rows):
    """Deterministic math.fsum; denominator is accepted decisions, not episodes."""
    rows = sorted(rows, key=lambda r: (r['model_id'], r['case_id'], r['filter']))
    metrics = [r['metrics'] for r in rows]; safe = [m['safety'] for m in metrics]
    accepted = sum(s['accepted_decisions'] for s in safe)
    interventions = sum(s['intervention_count'] for s in safe)
    result = dict(completed=len(rows), expected=len(rows), missing=0,
                  successes=sum(r['flags']['success'] for r in rows),
                  success_rate=sum(r['flags']['success'] for r in rows) / len(rows),
                  provisional=False, technical_attempt_logs=[],
                  accepted_decisions=accepted, interventions=interventions,
                  intervention_fraction=interventions / accepted if accepted else None,
                  correction_abs_sum=math.fsum(s['intervention_abs_sum'] for s in safe),
                  correction_abs_max=max(s['intervention_abs_max'] for s in safe),
                  stop_reasons=dict(sorted(Counter(v for m in metrics for v in m['completion']['stop_reasons']).items())))
    for k in ('filter_refusal', 'constraint_event', 'desired_exceedance', 'working_position_exceedance',
              'working_velocity_exceedance', 'hard_position_exceedance', 'hard_velocity_exceedance', 'hard_acceleration_exceedance'):
        result[k] = sum(r['flags'][k] for r in rows)
    for k in ('hold_final', 'hold_longest', 'reward', 'duration'):
        result['mean_' + k] = math.fsum(m[k] for m in metrics) / len(rows)
    return result


def analyze(dataset, protocol):
    validate_envelope(dataset)
    require(dataset['schema'] == SCHEMA and dataset['protocol_sha256'] == protocol.sha256, 'normalized schema/protocol')
    rows = scientific_rows(dataset['episodes'], dataset['stage'], protocol)
    require(dataset['scientific_sha256'] == digest(dict(stage=dataset['stage'], protocol_sha256=protocol.sha256, episodes=rows)), 'normalized scientific SHA')
    grouped = defaultdict(list); paired = defaultdict(dict)
    for r in rows:
        grouped[r['job_id']].append(r); paired[r['model_id'], r['case_id']][r['filter']] = r
    groups = []
    for job in protocol.jobs(dataset['stage']):
        a = grouped[job['id']]; m = [r['metrics'] for r in a]
        groups.append(dict(job_id=job['id'], model_id=job['model_id'], training_seed=job['seed'],
                           filter=job['mode'], group=job['group'], **aggregate(a),
                           horizons=sum(v['completion']['reason'] == 'horizon' for v in m),
                           hold_failures=sum(not v['success_episode'] for v in m),
                           strict_exceedance=sum(v['safety']['exceeds_024'] for v in m),
                           max_abs_position_interval=max(v['max_abs_position_interval'] for v in m),
                           safety_max_abs_position_interval=max(v['safety']['max_abs_position_interval'] for v in m),
                           max_abs_requested_acceleration=max(v['max_abs_requested_acceleration'] for v in m),
                           max_abs_applied_acceleration=max(v['max_abs_applied_acceleration'] for v in m),
                           decisions=sum(v['safety']['decisions'] for v in m),
                           refusal_count=sum(v['safety']['refusal_count'] for v in m)))
    pairs = []
    for (model, cid), modes in sorted(paired.items()):
        pair = dict(model_id=model, state_id=cid, complete=True)
        for mode in ('off', 'on'):
            r = modes[mode]; m = r['metrics']
            pair[mode] = dict(success=r['flags']['success'], hold_final=m['hold_final'], hold_longest=m['hold_longest'], desired_exceedance=r['flags']['desired_exceedance'])
        pairs.append(pair)
    return dict(stage=dataset['stage'], episodes=len(rows), jobs=len(groups), pairs=pairs, groups=groups,
                pooled_group_success=None, final=False, training=False,
                scientific_sha256=dataset['scientific_sha256'])


@dataclass
class Export:
    hashes: dict
    documents: dict
    manifest: dict
    evidence: dict
    raw: dict = field(default_factory=dict)


def load_export(path, expected_sha=None, *, verify_raw=False):
    """Hash every byte; optionally decode bounded logs in memory, never extract.

    MAX_MEMBER still limits compressed members and each expanded episode to
    64 MiB. A separate 24 GiB cumulative expansion cap covers a full 1960-case
    audit without retaining expanded logs. The outer 3 GiB cap is unchanged.
    """
    path = Path(path); hashes = {}; documents = {}; total = 0; raw = {}; raw_total = 0
    def member(name, stream, size):
        nonlocal total, raw_total
        safe_name(name); require(name not in hashes, 'duplicate archive member: ' + name)
        total += size
        require(size <= MAX_MEMBER and total <= MAX_TOTAL and len(hashes) < 50000, 'export size limit')
        h = hashlib.sha256(); chunks = []; read_size = 0
        # Operational/session JSON is not needed. Do not deserialize arbitrary payloads.
        retain = name in CORE or name == 'EXPORT_MANIFEST.json' or name.endswith((
            '/job_manifest.json', '/job_receipt.json', '/specification.json', '/report.json',
            '/receipt.json', '/completed.json', '/operation.json'))
        is_log = verify_raw and name.endswith('/episode.json.gz')
        retain = retain or is_log
        for block in iter(lambda: stream.read(1024**2), b''):
            read_size += len(block)
            require(read_size <= size, 'member grew during read')
            h.update(block)
            if retain:
                chunks.append(block)
        require(read_size == size, 'truncated export member')
        hashes[name] = h.hexdigest()
        if is_log:
            try:
                decoded = decode_raw_log(b''.join(chunks))
                raw_total += decoded[1]
                require(raw_total <= 24 * 1024**3, 'total expanded raw size limit')
                raw[name] = recompute_raw(decoded[0])
            except (ValueError, KeyError, TypeError, OSError, EOFError, zlib.error) as exc:
                raise ValueError(name + ': ' + str(exc)) from exc
        elif retain:
            documents[name] = parse_json(b''.join(chunks))
    archive_sha = None; outer_verified = False
    if path.is_dir():
        require(expected_sha is None, 'archive SHA supplied for a directory')
        for file in sorted(path.rglob('*')):
            require(not file.is_symlink(), 'symlink in export')
            if file.is_file():
                with file.open('rb') as stream:
                    member(file.relative_to(path).as_posix(), stream, file.stat().st_size)
    else:
        archive_sha = file_sha(path)
        sidecar = Path(str(path) + '.sha256')
        if sidecar.exists():
            # Accept sha256sum text/binary records, including relocated absolute paths;
            # the basename must still identify this archive. Never follow that path.
            lines = sidecar.read_text().splitlines()
            require(len(lines) == 1, 'archive sidecar must contain one record')
            match = re.fullmatch(r'([0-9a-f]{64}) [ *](.+)', lines[0])
            require(match is not None, 'invalid archive sidecar format')
            side_sha, side_name = match.groups()
            require(PurePosixPath(side_name).name == path.name, 'archive sidecar filename mismatch')
            require(archive_sha == sha_value(side_sha), 'archive sidecar SHA mismatch')
            outer_verified = True
        if expected_sha is not None:
            require(archive_sha == sha_value(expected_sha), 'archive SHA mismatch')
            outer_verified = True
        with tarfile.open(path, 'r|gz') as archive:
            for info in archive:
                if info.isdir():
                    safe_name(info.name.rstrip('/')); continue
                require(info.isfile(), 'non-regular archive member')
                with archive.extractfile(info) as stream:
                    member(info.name, stream, info.size)
    manifest = documents['EXPORT_MANIFEST.json']
    require(manifest['schema'] == 'cart-safety-bundle-v1' and manifest['version'] == VERSION, 'export schema/version')
    require(set(hashes) == set(manifest['files']) | {'EXPORT_MANIFEST.json'}, 'export member roster')
    for name, value in manifest['files'].items():
        require(hashes[safe_name(name)] == sha_value(value), 'export SHA: ' + name)
    require(all(n in documents for n in CORE), 'missing core export member')
    return Export(hashes, documents, manifest, dict(archive_sha256=archive_sha,
                  expected_archive_sha_verified=outer_verified, manifest_files=len(manifest['files']),
                  export_manifest_sha256=hashes['EXPORT_MANIFEST.json']), raw)


def import_compact(export, protocol):
    d = export.documents; p = d['plan.json']; q = d['queue/queue_state.json']; report = d['structured_report.json']
    stage = p['stage']; jobs = protocol.jobs(stage)
    require(p['version'] == report['version'] == VERSION and report['stage'] == stage, 'export version/stage')
    exact([p['inputs_manifest_sha256'], p['horizon'], p['dt'], p['training_jobs'], p['final'], p['selection'], p['device']],
          [protocol.sha256, 10., .01, 0, False, False, 'cpu'], 'plan science')
    indexed = unique(p['jobs'], lambda j: j['id'], 'job')
    require(set(indexed) == {j['id'] for j in jobs}, 'plan job roster')
    for j in jobs:
        exact({k: indexed[j['id']][k] for k in j}, j, 'plan job')
        require(indexed[j['id']]['kind'] == 'evaluation', 'training job in evaluation plan')
    require(p['expected_episodes'] == len(jobs) * 20 and p['cases_per_chunk'] == 20, 'plan episode count')
    checked = unique(p['checked_models'], lambda m: m['id'], 'checked model')
    require(set(checked) == set(MODELS) - {'classical'}, 'checked checkpoint roster')
    for mid, record in checked.items():
        model = protocol.models[mid]
        require(record['files'] == len(model['files']), 'checkpoint payload count')
        require(PurePosixPath(record['checkpoint']) == PurePosixPath(p['source_base']) / model['checkpoint_relative'],
                'checkpoint generation locator differs')
    exact(q['frozen']['jobs'], p['jobs'], 'queue job bindings')
    exact(q['frozen']['limits'], p['limits'], 'queue resources')
    require(q['frozen']['identity']['plan_sha256'] == export.hashes['plan.json'], 'queue plan SHA')
    require(q['frozen']['identity']['package_sha256'] == p['package_sha256'] == export.manifest['package_sha256'], 'package binding')
    start = d['start_request.json']
    require(start['explicit'] is True and start['plan_sha256'] == export.hashes['plan.json'], 'start plan binding')
    exact(start['session_identity'], q['frozen']['session_identity'], 'start session binding')
    require(q['status'] == 'complete' and not q['active'] and not q['failures'], 'incomplete/failed queue')
    require(set(q['completed']) == set(q['attempts']) == set(indexed), 'queue identity roster')
    require(all(type(v) is int and v == 1 for v in q['attempts'].values()), 'only single-attempt complete exports supported')
    rows = []
    for row in report['episodes']:
        cid = row['case']['id']; require(cid in protocol.cases, 'unknown case')
        exact(row['case'], protocol.cases[cid], 'nested case')
        require(not row['earlier_technical_attempts'], 'earlier failed attempts require separate review')
        rows.append(dict(stage=stage, job_id=row['job_id'], model_id=row['model_id'], case_id=cid,
                         training_seed=row['training_seed'], filter=row['filter'], group=row['group'],
                         state_sha256=row['case']['state_sha256'], model_sha256=protocol.model_sha(row['model_id']),
                         metrics=row['metrics'], flags=row['flags'], log_sha256=row['log_sha256']))
    rows = scientific_rows(rows, stage, protocol)
    science = dict(stage=stage, protocol_sha256=protocol.sha256, episodes=rows)
    normalized = seal_envelope(dict(schema=SCHEMA, **science, scientific_sha256=digest(science),
                                    provenance={'review': export.evidence}))
    result = analyze(normalized, protocol)
    groups = unique(report['groups'], lambda r: r['job_id'], 'group')
    require(set(groups) == set(indexed), 'group identity roster')
    by_job = defaultdict(list)
    for r in rows:
        by_job[r['job_id']].append(r)
    for j in jobs:
        expected = dict(job_id=j['id'], model_id=j['model_id'], training_seed=j['seed'], filter=j['mode'], group=j['group'], **aggregate(by_job[j['id']]))
        same(groups[j['id']], expected, 'group summary')
    pairs = unique(report['paired'], lambda r: (r['model_id'], r['state_id']), 'pair')
    same([pairs[k] for k in sorted(pairs)], result['pairs'], 'paired records')
    status = report['status']
    for key, value in dict(stage=stage, queue='complete', completed_jobs=len(jobs), total_jobs=len(jobs),
                           completed_episodes=len(rows), expected_episodes=len(rows)).items():
        same(status[key], value, 'status/' + key)
    statuses = unique(status['rows'], lambda r: r['id'], 'status job')
    require(set(statuses) == set(indexed), 'status job roster')
    for j in jobs:
        sr = statuses[j['id']]
        for key, value in dict(group=j['group'], training_seed=j['seed'], filter=j['mode'], state='completed', episodes=20, expected=20, error=None).items():
            same(sr[key], value, 'job status/' + key)
    for key, value in dict(completed_episode_count=len(rows), pooled_group_success=None, final=False, training=False).items():
        same(report[key], value, key)
    require(report['timing_all_fresh_single_attempt'] is True, 'timing attempts/reuse')
    seconds = number(report['measured_completed_chunk_seconds'], 'chunk seconds', 0)
    require(seconds > 0, 'zero chunk time')
    same(report['episodes_per_second'], len(rows) / seconds, 'episodes_per_second')
    return normalized


def recovery_bindings(export, protocol):
    """Receipt/log SHA validation only; gzip episode payloads are NOT recomputed."""
    require(export.manifest['full'] is True, 'not a recovery export')
    d, h = export.documents, export.hashes
    p = d['plan.json']; q = d['queue/queue_state.json']; total_seconds = []
    compact = {(r['job_id'], r['case']['id']): r for r in d['structured_report.json']['episodes']}
    for job in p['jobs']:
        base = 'queue/jobs/' + job['id']; ep = base + '/evaluation'; spec = d[ep + '/specification.json']
        def child(parent, name):
            return parent + '/' + safe_name(name)
        for key, sha_key in [('receipt', 'receipt_sha256'), ('artifact_manifest', 'artifact_manifest_sha256')]:
            require(h[child('queue', q['completed'][job['id']][key])] == q['completed'][job['id']][sha_key], 'queue receipt SHA')
        op = d[child('queue', q['completed'][job['id']]['receipt'])]
        require(q['completed'][job['id']]['artifact_manifest'] == 'jobs/' + job['id'] + '/job_manifest.json', 'queue artifact locator')
        require(op['status'] == op['reason'] == 'completed' and op['returncode'] == 0 and op['all_owned_processes_stopped'] is True, 'guard completion')
        job_manifest = d[base + '/job_manifest.json']
        require(set(job_manifest) == {n[len(base)+1:] for n in h if n.startswith(base+'/')
                                      and n != base+'/job_manifest.json'}, 'job manifest roster')
        for name, value in job_manifest.items():
            require(h[child(base, name)] == sha_value(value), 'job manifest SHA')
        jr = d[base + '/job_receipt.json']; er = d[ep + '/report.json']
        exact(jr['job'], job, 'receipt job')
        require(jr['status'] == 'complete' and jr['training'] is False and jr['plan_sha256'] == h['plan.json'], 'job receipt')
        require(jr['episodes'] == jr['new'] == 20 and jr['reused'] == 0, 'job episode counts')
        require(jr['report_sha256'] == h[ep + '/report.json'] and er['specification_sha256'] == h[ep + '/specification.json'], 'report SHA')
        require(er['complete'] is True and er['error'] is None and er['expected'] == er['completed'] == er['new_episodes'] == 20 and er['reused_episodes'] == 0 and not er['missing_ids'], 'incomplete evaluation report')
        exact(spec['cases'], [protocol.cases[c] for c in job['case_ids']], 'spec cases')
        require(spec['cases_sha256'] == digest(spec['cases']) and spec['mode'] == job['mode'] and spec['horizon'] == 10 and spec['device'] == 'cpu' and spec['diagnostic'] is True, 'spec binding')
        source = spec['source']; model = protocol.models[job['model_id']]
        key = 'nominal_sha256' if job['model_id'] == 'classical' else 'model_sha256'
        require(source[key] == protocol.model_sha(job['model_id']), 'checkpoint binding')
        exact([source['algorithm'], source['seed']], [job['algorithm'], job['seed']], 'source identity')
        if key == 'model_sha256':
            require(source['checkpoint_manifest_sha256'] == model['checkpoint_manifest_sha256'], 'checkpoint manifest binding')
            exact([source['transitions'], source['updates'], source['training_contract_id']],
                  [model['transitions'], model['updates'], f"cartpole-request-v2-{job['algorithm']}-on"], 'checkpoint generation identity')
        require(set(spec['evaluator_sources']) == set(EVALUATOR_SOURCES), 'evaluator source roster')
        protocol.check_sources(spec['evaluator_sources'])
        protocol.check_environment(spec['environment'], job['algorithm'], job['mode'])
        for i, cid in enumerate(job['case_ids']):
            prefix = ep + '/cases/' + cid; pointer = d[prefix + '/completed.json']
            rp = child(prefix, pointer['receipt']); require(h[rp] == pointer['sha256'], 'case receipt SHA')
            receipt = d[rp]; lp = child(str(PurePosixPath(rp).parent), receipt['log_file'])
            require(h[lp] == receipt['log_sha256'], 'episode log SHA')
            require(receipt['specification_sha256'] == h[ep + '/specification.json'] and receipt['index'] == i, 'case spec/index')
            exact(receipt['case'], protocol.cases[cid], 'receipt case')
            cr = compact[job['id'], cid]
            require(cr['log_sha256'] == receipt['log_sha256'], 'compact log SHA')
            exact({k: receipt['metrics'][k] for k in cr['metrics']}, cr['metrics'], 'compact/receipt metrics')
            paths = [n for n in h if n.startswith(prefix + '/')]
            attempts = {n[len(prefix)+1:].split('/')[0] for n in paths if '/attempt_' in n}
            require(attempts == {'attempt_00000'} and not any(n.endswith(('failure.json', 'partial_episode.json.gz')) for n in paths), 'duplicate/failed case attempt')
        total_seconds.append(number(jr['seconds'], 'job seconds', 0))
    same(d['structured_report.json']['measured_completed_chunk_seconds'], math.fsum(total_seconds), 'chunk wall time')


def decode_raw_log(payload):
    """Bound nested gzip expansion separately from the outer tar's size limits."""
    with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
        expanded = stream.read(MAX_MEMBER + 1)
    require(len(expanded) <= MAX_MEMBER, 'expanded episode size limit')
    return parse_json(expanded), len(expanded)


class RawComparison:
    """Same 1e-12 absolute/relative tolerance as compact comparisons, with evidence.

    Relative error uses max(abs(actual),abs(expected)); both zero gives zero.
    Discrete values/flags and structures are exact. No tolerance on identities.
    """
    def __init__(self):
        self.errors = {}

    def check(self, actual, expected, path):
        if isinstance(expected, dict):
            keys(actual, expected, path)
            for k in expected:
                self.check(actual[k], expected[k], path + '/' + k)
        elif isinstance(expected, list):
            require(isinstance(actual, list) and len(actual) == len(expected), path + ': length differs')
            for i, (a, b) in enumerate(zip(actual, expected)):
                self.check(a, b, path + '/' + str(i))
        elif type(expected) is float:
            number(actual, path)
            delta = abs(actual - expected)
            scale = max(abs(actual), abs(expected))
            group = path.split('/')[0]
            entry = self.errors.setdefault(group, dict(max_abs=0., max_rel=0., comparisons=0))
            entry['comparisons'] += 1
            entry['max_abs'] = max(entry['max_abs'], delta)
            entry['max_rel'] = max(entry['max_rel'], delta / scale if scale else 0.)
            require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
                    f'{path}: saved={actual!r}, raw={expected!r}, abs_error={delta!r}')
        else:
            same(actual, expected, path)


def cart_boundary_events(x, v, u, delta):
    """Stdlib mirror of constrained._position_exit_time/_boundary_events.

    Fixed frozen limits .25 m / 2 m/s; geometric and time resolution 1e-12.
    Snap only the analysis inputs, never the saved Drake endpoints. Preserve
    candidate order, tangency exclusion, t=0/t=delta and simultaneous events.
    No project import, solver, filter certification or pendulum integration.
    """
    tol = 1e-12
    if abs(abs(x)-.25) <= tol:
        x = math.copysign(.25, x)
    if abs(abs(v)-2.) <= tol:
        v = math.copysign(2., v)

    def position_time(side):
        distance, velocity, acceleration = .25-side*x, side*v, side*u
        if distance < 0:
            return 0.
        if acceleration == 0:
            return distance/velocity if velocity > 0 else None
        if acceleration < 0:
            if velocity <= 0:
                return None
            turning = -velocity/acceleration
            if .5*velocity*turning-distance <= tol:
                return None
            contact = math.sqrt(2*abs(acceleration))*math.sqrt(distance)
            speed = velocity*math.sqrt(max(0., 1-contact/velocity))*math.sqrt(1+contact/velocity)
        else:
            speed = math.hypot(velocity, math.sqrt(2*acceleration)*math.sqrt(distance))
            if velocity < 0:
                return (speed-velocity)/acceleration
        denominator = velocity+speed
        return 2*distance/denominator if denominator else 0.

    candidates = []
    def add(reason, side, time):
        if time is not None and 0 <= time <= delta+tol:
            candidates.append(dict(reason=reason, side=side,
                                   time=0. if time <= tol else min(time, delta)))
    for side in (-1, 1):
        add('position_limit', side, position_time(side))
    for side in (-1, 1):
        distance = 2.-side*v
        if distance < 0:
            add('velocity_limit', side, 0.)
        elif side*u > 0:
            add('velocity_limit', side, distance/(side*u))
    first = min((e['time'] for e in candidates), default=math.inf)
    return [e for e in candidates if e['time']-first <= tol]


def verify_cart_boundary(before, after, tr, comparison):
    """Mirror step's planned duration and reached-event/native-error ordering.

    With positive elapsed time use the applied held input. At t=0 no input was
    applied: the submitted command determines the impending outward departure.
    A native error may stop early; a later predicted event is not an outcome.
    """
    prefix = f"boundary_event/transition[{tr['index']}]"
    h, delta = tr['dt_actual'], tr['dt_requested']
    refused = tr.get('safety_filter', {}).get('feasible') is False
    if refused:
        expected, reasons = [], ['safety_filter_refusal']
        comparison.check(h, 0., prefix+'/refusal_duration')
    else:
        u = tr['applied_acceleration'] if h > 0 else tr['u_commanded']
        number(u, 'boundary command')
        planned = [] if before['error'] else cart_boundary_events(
            before['cart_position'], before['cart_velocity'], u, delta)
        duration = min((e['time'] for e in planned), default=delta)
        if before['error']:
            comparison.check(h, 0., prefix+'/native_initial_duration')
        elif after['error']:
            require(h <= duration+1e-12, prefix+': native stop after planned event')
        else:
            comparison.check(h, duration, prefix+'/duration')
        expected = [e for e in planned if e['time'] <= h+1e-12]
        reasons = list(dict.fromkeys(e['reason'] for e in expected))
        if after['error']:
            reasons.insert(0, 'simulator_error')
    try:
        comparison.check(tr['boundary_events'], expected, prefix+'/events')
        exact(tr['stop_reasons'], reasons, prefix+'/stop_reasons')
    except ValueError as exc:
        raise ValueError(f"{prefix}: saved_events={tr['boundary_events']!r}, "
                         f"computed_events={expected!r}; {exc}") from exc
    return expected, reasons


def recompute_raw(log):
    """Independent arithmetic over recorded states/actions; never use saved metrics
    as inputs. Pendulum extrema/hold are sampled, cart extrema are held-u analytic.
    The two historical x maxima deliberately keep their different rounding rules.
    """
    c = RawComparison(); check = c.check
    require(log['schema_version'] == 'cartpole.episode.v1', 'raw schema')
    states, transitions, meta = log['states'], log['transitions'], log['metadata']
    require(0 < len(transitions) <= 1001 and len(states) == len(transitions)+1, 'raw N+1/count')
    for i, row in enumerate(states):
        same(row['index'], i, 'state index')
        for key in (*FIELDS, 'time', 'cart_acceleration'):
            number(row[key], key)
        integer(row['error'], 'state error')
        require(row['time'] >= 0 and (i == 0 or row['time'] >= states[i-1]['time']), 'state time order')
    check(states[0]['time'], 0., 'timing/initial')
    check(states[0]['cart_acceleration'], 0., 'action_mapping/initial_acceleration')
    same(states[0]['error'], 0, 'initial error')
    flags_on = 'safety_filter' in meta
    corrections, refusals, rewards, effort = [], Counter(), [], []
    accepted = decisions = exceeds = position_events = 0
    endpoint_max = max(abs(s['cart_position']) for s in states)
    legacy_max = safety_max = endpoint_max
    residual_x = residual_v = 0.
    for i, tr in enumerate(transitions):
        before, after = states[i:i+2]
        label = f'transition[{i}]'
        exact([tr['index'], tr['from_state'], tr['to_state']], [i, i, i+1], label + ' indices')
        h = number(tr['dt_actual'], 'dt actual', 0)
        requested_dt = number(tr['dt_requested'], 'dt requested', 0)
        require(0 < requested_dt <= .01+1e-12 and h <= requested_dt+1e-12, 'duration range')
        check(tr['time_start'], before['time'], 'timing/start')
        check(tr['time_end'], after['time'], 'timing/end')
        check(h, after['time']-before['time'], 'timing/dt')
        u0 = number(tr['u_requested'], 'requested acceleration')
        check(tr['action_requested'] * 4., u0, 'action_mapping/request')
        u, command = tr['applied_acceleration'], tr['u_commanded']
        limited = max(-4., min(4., u0))
        same('safety_filter' in tr, flags_on, 'filter decision presence')
        decision = tr.get('safety_filter')
        if decision is None:
            check(command, limited, 'action_mapping/working_limiter')
        else:
            decisions += 1
            check(decision['u_proposed'], u0, 'filter_decision/proposed')
            check(decision['u_filtered'], command, 'filter_decision/filtered')
            require(type(decision['feasible']) is bool, 'feasible flag')
            if decision['feasible']:
                accepted += 1
                number(command, 'command')
                same(decision['intervened'], command != u0, 'intervention flag')
                check(decision['correction'], command-u0, 'correction/signed')
                check(decision['correction_abs'], abs(command-u0), 'correction/absolute')
                if command != u0:
                    corrections.append(abs(command-u0))
            else:
                require(command is None and h == 0 and decision['reason'], 'filter refusal shape')
                exact(tr['stop_reasons'], ['safety_filter_refusal'], 'refusal reason')
                exact([decision['intervened'], decision['correction'], decision['correction_abs']],
                      [None, None, None], 'refusal correction')
                refusals[decision['reason']] += 1
        same(tr['action_clipped'], u0 != limited if decision is None or decision['feasible'] else False,
             'clipping flag')
        check(tr['action_commanded'], command / 4. if command is not None else None, 'action_mapping/command')
        if h == 0:
            require(u is None, 'zero transition applied input')
            exact({k: before[k] for k in (*FIELDS, 'cart_acceleration')},
                  {k: after[k] for k in (*FIELDS, 'cart_acceleration')}, 'zero transition state')
        else:
            number(u, 'applied acceleration')
            require(abs(u) <= 4., 'applied exceeds working limit')
            check(u, command, 'action_mapping/applied_command')
            check(u, after['cart_acceleration'], 'action_mapping/acceleration_telemetry')
        same(tr['simulator_error'], after['error'], 'native error telemetry')
        events, reasons = verify_cart_boundary(before, after, tr, c)
        terminated = bool(reasons)
        same(tr['terminated'], terminated, 'transition terminal')
        same(tr['gym_terminated'], terminated, 'Gym terminal')
        truncated = not terminated and abs(after['time']-10.) <= 1e-12
        same(tr['gym_truncated'], truncated, 'Gym truncation')
        require(not (terminated or truncated) or i == len(transitions)-1, 'post-completion transition')
        require(after['time'] <= 10.+1e-12, 'post-horizon state')
        position_events += sum(e['reason'] == 'position_limit' for e in events)
        x, v = before['cart_position'], before['cart_velocity']
        values = [x, after['cart_position']]; vertex = None; dx = dv = 0.
        if h > 0:
            predicted = math.fsum((x, v*h, .5*u*h*h))
            values.append(predicted)
            dx, dv = after['cart_position']-predicted, after['cart_velocity']-(v+u*h)
            if u != 0 and 0 < -v/u < h:
                vertex = -v/u
                values.append(math.fsum((x, v*vertex, .5*u*vertex*vertex)))
                legacy_max = max(legacy_max, abs(x+v*vertex+u*vertex**2/2))
        maximum = max(map(abs, values))
        safety_max = max(safety_max, maximum); exceeds += maximum > .24
        residual_x = max(residual_x, abs(dx)); residual_v = max(residual_v, abs(dv))
        if 'cart_interval' in tr:
            check(tr['cart_interval'], dict(max_abs_position_interval=maximum, vertex_time=vertex,
                desired_excess=max(0., maximum-.24), exceeds_024=maximum > .24,
                exceeds_024_resolved=maximum > .24+1e-12,
                endpoint_position_residual=dx, endpoint_velocity_residual=dv), 'cart_interval')
        # Endpoint R1 quadrature, including exactly one unscaled terminal penalty.
        factors = None; alignment = 0.
        if h > 0:
            factors = dict(height=(1-math.cos(after['pole_angle']))/2,
                position=(.25/math.hypot(.25, after['cart_position']))**2,
                velocity=(2./math.hypot(2., after['cart_velocity']))**2,
                angular_velocity=(2./math.hypot(2., after['pole_angular_velocity']))**2,
                acceleration=(40./math.hypot(40., u))**2)
            alignment = h/.01 * math.prod(factors.values())
            effort.append(u*u*h)
        terms = dict(alignment=alignment, failure=-5. if terminated else 0.)
        reward = math.fsum(terms.values()); rewards.append(reward)
        same(tr['reward_version'], 'R1-v0', 'reward version')
        check(tr['reward'], reward, 'transition_reward')
        check(tr['reward_scale'], h/.01, 'reward_scale')
        check(tr['reward_terms'], terms, 'reward_terms')
        check(tr['reward_factors'], factors, 'reward_factors')
    last = transitions[-1]
    reason = 'terminal' if last['terminated'] else 'horizon'
    require(last['terminated'] or last['gym_truncated'], 'unfinished episode')
    completion = dict(reason=reason, time=states[-1]['time'], terminated=last['terminated'],
        truncated=not last['terminated'], transition_count=len(transitions), stop_reasons=last['stop_reasons'])
    check(log['completion'], completion, 'completion')
    intervals = []; start = end = first = confirmation = None; qualifying = 0
    angle_errors = []
    for s in states:
        theta = s['pole_angle']-math.pi
        error = math.atan2(math.sin(theta), math.cos(theta)); angle_errors.append(error)
        inside = (abs(error) <= math.pi/18 and abs(s['pole_angular_velocity']) <= .5
                  and abs(s['cart_position']) <= .2 and abs(s['cart_velocity']) <= .2)
        qualifying += inside
        if inside:
            if start is None:
                start = s['time']
            end = s['time']
            if first is None and end-start >= 2.-1e-12:
                first, confirmation = start, end
        elif start is not None:
            intervals.append(dict(start=start, end=end, duration=end-start)); start = end = None
    final_hold = 0. if start is None else end-start
    if start is not None:
        intervals.append(dict(start=start, end=end, duration=final_hold))
    longest = max((v['duration'] for v in intervals), default=0.)
    success = reason == 'horizon' and final_hold >= 2.-1e-12
    phases = [t.get('control_phase') for t in transitions]
    switches = [dict(time=transitions[i]['time_start'], from_phase=phases[i-1], to_phase=phases[i])
                for i in range(1, len(phases)) if phases[i] != phases[i-1]]
    hold = dict(success_episode=success, first_hold_start=first, first_hold_confirmation=confirmation,
                hold_final=final_hold, hold_longest=longest)
    evaluation = dict(**hold, required_hold_duration=2.,
        criterion='consecutive recorded states; first/last qualifying timestamps; time tolerance 1e-12 s',
        hold_intervals=intervals, mode_switch_count=len(switches), mode_switches=switches)
    check(log['evaluation'], evaluation, 'evaluation')
    check(log['gym_final_hold_metrics'], dict(**hold, in_upright_region=start is not None), 'gym_hold')
    def tr_max(key):
        return max((abs(t[key]) for t in transitions if t[key] is not None), default=0.)
    base = dict(duration=states[-1]['time'], max_abs_position=endpoint_max,
        max_abs_velocity=max(abs(s['cart_velocity']) for s in states),
        max_abs_requested_acceleration=tr_max('u_requested'), max_abs_commanded_acceleration=tr_max('u_commanded'),
        max_abs_applied_acceleration=tr_max('applied_acceleration'), final_angle_error=angle_errors[-1],
        all_samples_in_upright_region=qualifying == len(states),
        clipped_transition_count=sum(t['action_clipped'] for t in transitions))
    check(log['metrics'], base, 'log_metrics')
    total_reward = math.fsum(rewards)
    check(log['gym_return'], total_reward, 'gym_return')
    correction_sum = math.fsum(corrections)
    safety = dict(filter_enabled=flags_on, decisions=decisions, accepted_decisions=accepted,
        intervention_count=len(corrections), intervention_fraction=len(corrections)/accepted if accepted else None,
        intervention_abs_sum=correction_sum, intervention_abs_max=max(corrections, default=0.),
        intervention_abs_mean=correction_sum/len(corrections) if corrections else 0.,
        refusal_count=sum(refusals.values()), refusal_reasons=dict(refusals),
        max_abs_position_interval=safety_max, desired_excess_max=max(0., safety_max-.24),
        exceeds_024=safety_max > .24, exceeds_024_resolved=safety_max > .24+1e-12,
        intervals_exceeding_024=exceeds, position_limit_events=position_events,
        max_abs_endpoint_position_residual=residual_x, max_abs_endpoint_velocity_residual=residual_v,
        definitions=SAFETY_DEFINITIONS)
    full = dict(**base, **evaluation, completion=completion, reward=total_reward,
        max_abs_omega=max(abs(s['pole_angular_velocity']) for s in states), safety=safety,
        max_abs_position_interval=legacy_max, position_margin_interval=.25-legacy_max,
        velocity_margin_interval=2.-base['max_abs_velocity'],
        interval_margin_definition='analytic cart extrema under held u; original Drake pendulum states',
        control_effort=math.fsum(effort),
        rms_acceleration=math.sqrt(math.fsum(effort)/base['duration']) if base['duration'] else None,
        discounted_reward=math.fsum(math.exp(-.01/5)**i*r for i, r in enumerate(rewards)),
        boundary_failure=any(r.endswith('_limit') for r in completion['stop_reasons']))
    return dict(metadata=meta, initial_state={k: states[0][k] for k in FIELDS}, metrics=full,
                qualifying_upright_samples=qualifying, errors=c.errors)


def verify_raw_bindings(export, protocol):
    """Bind independently recomputed records to every layer, then summarize them.

    Classical metadata is verified as actually executed, not rewritten into the
    SAC-like contract incorrectly recorded in its specification. This caveat is
    returned even if all common science fields and metrics pass.
    """
    require(export.manifest['full'] is True and export.raw, 'raw audit requires full recovery')
    d = export.documents; h = export.hashes; used = set(); c = RawComparison()
    rows = []; warnings = Counter(); errors = {}; qualifying = 0
    compact = {(r['job_id'], r['case']['id']): r for r in d['structured_report.json']['episodes']}
    for job in protocol.jobs(d['plan.json']['stage']):
        ep = 'queue/jobs/' + job['id'] + '/evaluation'
        spec = d[ep+'/specification.json']; contract = spec['environment']
        env = contract['specification']
        for cid in job['case_ids']:
            rp = ep+'/cases/'+cid+'/attempt_00000/receipt.json'
            receipt = d[rp]; lp = str(PurePosixPath(rp).parent) + '/' + safe_name(receipt['log_file'])
            require(lp in export.raw and lp not in used, 'missing/duplicate raw episode: ' + lp)
            used.add(lp); raw = export.raw[lp]; meta = raw['metadata']; case = protocol.cases[cid]
            try:
                require(receipt['log_sha256'] == h[lp] == compact[job['id'], cid]['log_sha256'], 'raw log SHA')
                require(receipt['specification_sha256'] == h[ep+'/specification.json'], 'raw specification SHA')
                exact(raw['initial_state'], case['initial_state'], 'raw initial state')
                require(digest(raw['initial_state']) == case['state_sha256'], 'raw state SHA')
                exact({k: meta['initial_state'][k] for k in FIELDS}, raw['initial_state'], 'metadata initial state')
                exact([meta['algorithm'], meta['training_seed'], meta['evaluation_case_id'], meta['filter_mode']],
                      [job['algorithm'], job['seed'], cid, job['mode']], 'raw identity')
                same(meta['seed'], int(digest(case)[:8], 16), 'raw evaluation seed')
                same(meta['reset_mode'], 'explicit', 'raw reset mode')
                for key, expected in dict(config=env['physical_config'], integrator=env['integrator'],
                        reward=env['reward'], control_interval=.01, horizon=10., required_hold_duration=2.,
                        upright_thresholds=dict(angle=math.pi/18, angular_velocity=.5, position=.2, velocity=.2)).items():
                    exact(meta[key], expected, 'common science/'+key)
                same('safety_filter' in meta, job['mode'] == 'on', 'metadata filter')
                if job['mode'] == 'on':
                    exact(meta['safety_filter'], env['filter']['limits'], 'filter limits')
                if job['algorithm'] == 'classical':
                    require('training_contract' not in meta and meta['environment'] == 'CourseworkCartPole-v0'
                            and meta.get('reset_admission', 'working') == 'working', 'unexpected classical metadata')
                    warnings['classical_specification_SAC_contract_but_runtime_without_training_contract'] += 1
                else:
                    exact(meta['training_contract'], contract, 'runtime training contract')
                    same(meta['environment'], env['environment_api'], 'runtime environment')
                    same(meta.get('reset_admission', 'working'), env['reset']['admission'], 'runtime reset admission')
                full = dict(id=cid, subset=job['group'], **raw['metrics'])
                c.check(receipt['metrics'], full, 'receipt_metrics')
                metric = {k: raw['metrics'][k] for k in METRIC_KEYS}
                c.check(compact[job['id'], cid]['metrics'], metric, 'compact_metrics')
                exact(compact[job['id'], cid]['flags'], flags(metric), 'raw flags')
                rows.append(dict(job_id=job['id'], case_id=cid, model_id=job['model_id'], filter=job['mode'], metrics=metric))
                qualifying += raw['qualifying_upright_samples']
                for key, stats in raw['errors'].items():
                    e = errors.setdefault(key, dict(max_abs=0., max_rel=0., comparisons=0))
                    for field in ('max_abs', 'max_rel'):
                        e[field] = max(e[field], stats[field])
                    e['comparisons'] += stats['comparisons']
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(lp + ': ' + str(exc)) from exc
    require(used == set(export.raw), 'unexpected raw episodes')
    # The normalized scientific record is constructed and validated separately by
    # import_compact. Compare every raw record explicitly to that representation.
    normalized = import_compact(export, protocol)
    normalized_rows = {(r['job_id'], r['case_id']): r for r in normalized['episodes']}
    for r in rows:
        c.check(normalized_rows[r['job_id'], r['case_id']]['metrics'], r['metrics'], 'normalized_metrics')
    errors.update(c.errors)
    modes = {}
    pairs = defaultdict(dict)
    for r in rows:
        pairs[r['model_id'], r['case_id']][r['filter']] = r['metrics']
    for mode in ('off', 'on'):
        values = [r['metrics'] for r in rows if r['filter'] == mode]
        modes[mode] = dict(episodes=len(values), successes=sum(m['success_episode'] for m in values),
            accepted_decisions=sum(m['safety']['accepted_decisions'] for m in values),
            horizons=sum(m['completion']['reason'] == 'horizon' for m in values),
            horizon_hold_failures=sum(m['completion']['reason'] == 'horizon' and not m['success_episode'] for m in values),
            position_limit=sum(m['safety']['position_limit_events'] for m in values),
            velocity_limit=sum('velocity_limit' in m['completion']['stop_reasons'] for m in values),
            strict_exceedances=sum(m['safety']['exceeds_024'] for m in values),
            resolved_exceedances=sum(m['safety']['exceeds_024_resolved'] for m in values),
            successes_without_resolved_exceedance=sum(m['success_episode'] and not m['safety']['exceeds_024_resolved'] for m in values),
            refusals=sum(m['safety']['refusal_count'] for m in values),
            interventions=sum(m['safety']['intervention_count'] for m in values))
    counts = Counter(); off_only_exceedances = 0
    for pair in pairs.values():
        require(set(pair) == {'off', 'on'}, 'incomplete raw pair')
        off, on = pair['off']['success_episode'], pair['on']['success_episode']
        counts['both_success' if off and on else 'off_only' if off else 'on_only' if on else 'both_fail'] += 1
        off_only_exceedances += off and not on and pair['off']['safety']['exceeds_024_resolved']
    # A digest per common case allows timing/main comparison without retaining raw logs.
    per_pair = {model+'/'+cid: digest(pair) for (model, cid), pair in sorted(pairs.items())}
    return dict(episodes=len(rows), transitions=sum(r['metrics']['completion']['transition_count'] for r in rows),
        jobs=len({r['job_id'] for r in rows}), states=len({r['case_id'] for r in rows}), pairs=len(pairs),
        controllers=len({r['model_id'] for r in rows}), qualifying_upright_samples=qualifying,
        modes=modes, paired={k: counts[k] for k in ('both_success', 'on_only', 'off_only', 'both_fail')},
        off_only_with_off_resolved_exceedance=off_only_exceedances, numerical_errors=errors,
        endpoint_residuals={mode: {k: max(r['metrics']['safety'][k] for r in rows if r['filter'] == mode)
            for k in ('max_abs_endpoint_position_residual', 'max_abs_endpoint_velocity_residual')}
            for mode in ('off', 'on')},
        classical_provenance=dict(status='known_mismatch' if warnings else 'not_applicable', warnings=dict(warnings)),
        scientific_pair_sha256=per_pair,
        limitations=['sampled pendulum hold and omega; no intersample pendulum proof',
                     'filter decisions checked against commands, not reexecuted/certified',
                     'model payloads never deserialized; identity is inherited from verified receipt bindings'])


def compare_exports(a, b):
    for name in CORE[:-1]:
        require(a.hashes[name] == b.hashes[name], 'review/recovery binding differs: ' + name)
    # The ONLY permitted export differences are the two measured session clocks.
    diffs = []
    def walk(x, y, path=''):
        if isinstance(x, dict) and isinstance(y, dict) and x.keys() == y.keys():
            for key in sorted(x):
                walk(x[key], y[key], path + '/' + key)
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, path + '/' + str(i))
        elif canonical(x) != canonical(y):
            require(path in ('/status/session/elapsed_session_seconds', '/status/session/seconds_before_save_reserve'), 'review/recovery discrepancy: ' + path)
            number(x, path); number(y, path)
            diffs.append(dict(path=path, review=x, recovery=y))
    walk(a.documents['structured_report.json'], b.documents['structured_report.json'])
    return diffs


def run(args, protocol=None):
    collector = RawComparison() if getattr(args, 'verify_raw_metrics', False) else None
    token = NUMERICAL_AUDIT.set(collector)
    try:
        return _run(args, protocol, collector)
    finally:
        NUMERICAL_AUDIT.reset(token)


def _run(args, protocol=None, collector=None):
    protocol = protocol or Protocol()
    verify_raw = getattr(args, 'verify_raw_metrics', False)
    require(not (verify_raw and args.normalized), 'raw verification requires recovery logs, not normalized input')
    require(not (args.check and args.output), '--check cannot be combined with --output')
    require(not args.recovery_sha256 or args.recovery, '--recovery-sha256 requires --recovery')
    evidence = dict(archive_integrity='not_run', compact_consistency='not_run',
                    normalized_consistency='not_run', source_provenance='not_revalidated',
                    envelope_integrity='not_run',
                    raw_metric_recomputation='not_run', checkpoint_deserialization='not_run')
    if args.normalized:
        require(not args.recovery and not args.archive_sha256 and not args.recovery_sha256, 'archives require --review')
        dataset = read_json(args.normalized)
    else:
        export = load_export(args.review, args.archive_sha256, verify_raw=verify_raw and not args.recovery)
        dataset = import_compact(export, protocol)
        evidence.update(archive_integrity='passed' if export.evidence['archive_sha256'] else 'not_applicable_directory',
                        export_manifest_integrity='passed', compact_consistency='passed',
                        source_provenance='verified_against_exports', review=export.evidence)
        if export.manifest['full']:
            recovery_bindings(export, protocol); evidence['receipt_bindings'] = 'passed'
        if args.recovery:
            recovery = load_export(args.recovery, args.recovery_sha256, verify_raw=verify_raw)
            recovered = import_compact(recovery, protocol)
            evidence['export_clock_differences'] = compare_exports(export, recovery)
            require(dataset['scientific_sha256'] == recovered['scientific_sha256'], 'recovery science differs')
            recovery_bindings(recovery, protocol)
            evidence.update(recovery=recovery.evidence, receipt_bindings='passed')
        if verify_raw:
            evidence['raw_audit'] = verify_raw_bindings(recovery if args.recovery else export, protocol)
            evidence['raw_metric_recomputation'] = 'passed'
        dataset['provenance'] = {k: evidence[k] for k in ('review', 'recovery') if k in evidence}
        seal_envelope(dataset)
    result = analyze(dataset, protocol)
    evidence.update(normalized_consistency='passed', envelope_integrity='passed')
    report = dict(verification=evidence, analysis=result, aggregates_sha256=digest(result),
                  envelope_sha256=dataset['envelope_sha256'])
    if collector is not None:
        evidence['raw_audit']['numerical_errors'].update(collector.errors)
    if args.output:
        target = Path(args.output)
        target.mkdir(parents=False, exist_ok=False)
        for name, value in [('normalized.json', dataset), ('analysis.json', report)]:
            with (target / name).open('x') as stream:
                stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--review', type=Path, help='extracted compact directory or review/recovery tar.gz')
    source.add_argument('--normalized', type=Path)
    parser.add_argument('--recovery', type=Path)
    parser.add_argument('--verify-raw-metrics', action='store_true', help='explicit, expensive streaming raw audit')
    parser.add_argument('--archive-sha256')
    parser.add_argument('--recovery-sha256')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--output', type=Path, help='new directory; parent must exist')
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (ValueError, KeyError, TypeError, OSError, EOFError, tarfile.TarError) as exc:
        parser.exit(2, 'validation failed: ' + str(exc) + '\n')
    summary = {k: v for k, v in result.items() if k != 'analysis'}
    summary.update({k: result['analysis'][k] for k in ('stage', 'episodes', 'jobs', 'scientific_sha256')})
    summary['pairs'] = len(result['analysis']['pairs'])
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
