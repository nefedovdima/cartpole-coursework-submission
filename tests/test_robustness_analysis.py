"""Small generated fixtures only: no archives from experiments, network or dynamics."""
import argparse
import copy
import gzip
import math
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zlib
from unittest.mock import patch

from cartpole.experiments import robustness_analysis as a


def metrics(mode, transitions=1000, interventions=0):
    safety = dict(filter_enabled=mode == 'on', decisions=transitions if mode == 'on' else 0,
        accepted_decisions=transitions if mode == 'on' else 0,
        intervention_count=interventions, intervention_fraction=interventions/transitions if mode == 'on' else None,
        intervention_abs_sum=float(interventions), intervention_abs_max=1. if interventions else 0.,
        intervention_abs_mean=1. if interventions else 0., refusal_count=0, refusal_reasons={},
        max_abs_position_interval=.1, desired_excess_max=0., exceeds_024=False,
        exceeds_024_resolved=False, intervals_exceeding_024=0, position_limit_events=0,
        max_abs_endpoint_position_residual=0., max_abs_endpoint_velocity_residual=0.,
        definitions=copy.deepcopy(a.SAFETY_DEFINITIONS))
    return dict(completion=dict(reason='horizon', time=10., terminated=False, truncated=True,
            transition_count=transitions, stop_reasons=[]), duration=10., hold_final=3., hold_longest=3.,
        first_hold_start=7., first_hold_confirmation=9., max_abs_requested_acceleration=5.,
        max_abs_applied_acceleration=4., max_abs_position_interval=.1, max_abs_velocity=.2,
        max_abs_omega=7., reward=123., safety=safety, success_episode=True)


def dataset(protocol, stage='timing'):
    rows = []
    for j in protocol.jobs(stage):
        for cid in j['case_ids']:
            m = metrics(j['mode'])
            rows.append(dict(stage=stage, job_id=j['id'], model_id=j['model_id'], case_id=cid,
                training_seed=j['seed'], filter=j['mode'], group=j['group'],
                state_sha256=protocol.cases[cid]['state_sha256'], model_sha256=protocol.model_sha(j['model_id']),
                metrics=m, flags=a.flags(m), log_sha256=hashlib.sha256((j['id']+cid).encode()).hexdigest()))
    return seal(dict(schema=a.SCHEMA, stage=stage, protocol_sha256=protocol.sha256, episodes=rows,
        provenance={'review': dict(archive_sha256=None, expected_archive_sha_verified=False,
                                   manifest_files=4, export_manifest_sha256='e'*64)}))


def seal(d):
    rows = sorted(d['episodes'], key=lambda r: (r['model_id'], r['case_id'], r['filter']))
    d['scientific_sha256'] = a.digest(dict(stage=d['stage'], protocol_sha256=d['protocol_sha256'], episodes=rows))
    return a.seal_envelope(d)


def export_documents(p, data):
    jobs = [dict(j, kind='evaluation') for j in p.jobs(data['stage'])]
    plan = dict(version=a.VERSION, stage=data['stage'], jobs=jobs, limits={},
        inputs_manifest_sha256=p.sha256, horizon=10., dt=.01, training_jobs=0,
        final=False, selection=False, device='cpu', expected_episodes=len(jobs)*20,
        cases_per_chunk=20, package_sha256='a'*64, source_base='/synthetic/models',
        checked_models=[dict(id=mid, files=len(m['files']), checkpoint='/synthetic/models/'+m['checkpoint_relative'])
            for mid,m in p.models.items() if mid != 'classical'])
    queue = dict(frozen=dict(jobs=jobs, limits={}, session_identity={'id': 'fixture'},
        identity=dict(plan_sha256=a.digest(plan), package_sha256='a'*64)), status='complete',
        active={}, failures={}, completed={j['id']: {} for j in jobs}, attempts={j['id']: 1 for j in jobs})
    groups = []
    for j in jobs:
        groups.append(dict(job_id=j['id'], model_id=j['model_id'], training_seed=j['seed'],
            filter=j['mode'], group=j['group'], **a.aggregate([r for r in data['episodes'] if r['job_id'] == j['id']])))
    report_rows = []
    for row in data['episodes']:
        report_rows.append({**{k: row[k] for k in ('job_id', 'model_id', 'training_seed', 'filter', 'group', 'metrics', 'flags', 'log_sha256')},
            'case': copy.deepcopy(p.cases[row['case_id']]), 'earlier_technical_attempts': [], 'seconds': .1, 'write_seconds': .001})
    report = dict(version=a.VERSION, stage=data['stage'], episodes=report_rows, groups=groups,
        paired=a.analyze(data, p)['pairs'], completed_episode_count=len(report_rows),
        pooled_group_success=None, final=False, training=False, measured_completed_chunk_seconds=4.,
        episodes_per_second=len(report_rows)/4, timing_all_fresh_single_attempt=True,
        status=dict(stage=data['stage'], queue='complete', completed_jobs=len(jobs), total_jobs=len(jobs),
            completed_episodes=len(report_rows), expected_episodes=len(report_rows),
            session=dict(elapsed_session_seconds=5., seconds_before_save_reserve=50.),
            rows=[dict(id=j['id'], group=j['group'], training_seed=j['seed'], filter=j['mode'],
                state='completed', episodes=20, expected=20, error=None) for j in jobs]))
    return {'plan.json': plan, 'queue/queue_state.json': queue, 'structured_report.json': report,
        'start_request.json': dict(explicit=True, plan_sha256=a.digest(plan), session_identity={'id': 'fixture'})}


def export_object(documents, payloads=None, full=False):
    payloads = dict(payloads or {})
    payloads.update({k: a.canonical(v) for k, v in documents.items()})
    hashes = {k: hashlib.sha256(v).hexdigest() for k, v in payloads.items()}
    manifest = dict(schema='cart-safety-bundle-v1', version=a.VERSION,
        package_sha256='a'*64, full=full, files=hashes.copy())
    hashes['EXPORT_MANIFEST.json'] = a.digest(manifest)
    evidence = dict(archive_sha256=None, expected_archive_sha_verified=False,
                    manifest_files=len(manifest['files']), export_manifest_sha256=hashes['EXPORT_MANIFEST.json'])
    return a.Export(hashes, {**documents, 'EXPORT_MANIFEST.json': manifest}, manifest, evidence)


def recovery(p, data, raw_records=None):
    docs = export_documents(p, data); payloads = {}; queue = docs['queue/queue_state.json']
    for job in docs['plan.json']['jobs']:
        base = 'queue/jobs/' + job['id']; ep = base + '/evaluation'
        mid = p.models[job['model_id']]
        algorithm = 'SAC' if job['algorithm'] == 'classical' else job['algorithm']
        env = copy.deepcopy(a.read_json(p.root / f"configs/cloud/{algorithm.lower()}_training_v2_{job['mode']}.json")['training_contract'])
        env['sources'] = {k: a.file_sha(p.root/k) for k in env['sources']}
        source = dict(algorithm=job['algorithm'], seed=job['seed'])
        if job['algorithm'] == 'classical':
            source.update(kind='classical', nominal_sha256=mid['nominal_sha256'], transitions=None, updates=None)
        else:
            source.update(model_sha256=mid['model_sha256'], checkpoint_manifest_sha256=mid['checkpoint_manifest_sha256'],
                transitions=mid['transitions'], updates=mid['updates'], training_contract_id=f'cartpole-request-v2-{algorithm}-on')
        spec = dict(cases=[p.cases[c] for c in job['case_ids']], mode=job['mode'], horizon=10., device='cpu', diagnostic=True,
            source=source,
            evaluator_sources={k: a.file_sha(p.root/k) for k in a.EVALUATOR_SOURCES}, environment=env)
        spec['cases_sha256'] = a.digest(spec['cases']); docs[ep+'/specification.json'] = spec
        for i, cid in enumerate(job['case_ids']):
            row = next(r for r in docs['structured_report.json']['episodes'] if r['job_id'] == job['id'] and r['case']['id'] == cid)
            prefix = ep+'/cases/'+cid; rp = prefix+'/attempt_00000/receipt.json'
            # Existing compact fixtures remain opaque; E2E supplies real gzip and full metrics.
            payload, saved_metrics = (raw_records[job['id'], cid] if raw_records is not None else
                                      ((job['id']+cid).encode(), row['metrics']))
            payloads[prefix+'/attempt_00000/episode.json.gz'] = payload
            receipt = dict(specification_sha256=a.digest(spec), index=i, case=p.cases[cid],
                metrics=saved_metrics, log_file='episode.json.gz', log_sha256=hashlib.sha256(payload).hexdigest())
            docs[rp] = receipt
            docs[prefix+'/completed.json'] = dict(receipt='attempt_00000/receipt.json', sha256=a.digest(receipt))
        er = dict(specification_sha256=a.digest(spec), complete=True, error=None, expected=20,
            completed=20, new_episodes=20, reused_episodes=0, missing_ids=[])
        docs[ep+'/report.json'] = er
        docs[base+'/job_receipt.json'] = dict(job=job, status='complete', training=False,
            plan_sha256=a.digest(docs['plan.json']), episodes=20, new=20, reused=0, report_sha256=a.digest(er), seconds=2.)
        h = export_object(docs, payloads).hashes
        jm = {k[len(base)+1:]: v for k, v in h.items() if k.startswith(base+'/')}
        docs[base+'/job_manifest.json'] = jm
        gp = 'queue/guards/'+job['id']+'_000/operation.json'
        op = dict(status='completed', reason='completed', returncode=0, all_owned_processes_stopped=True)
        docs[gp] = op
        queue['completed'][job['id']] = dict(receipt=gp[len('queue/'):], receipt_sha256=a.digest(op),
            artifact_manifest=(base+'/job_manifest.json')[len('queue/'):], artifact_manifest_sha256=a.digest(jm))
    return export_object(docs, payloads, full=True)


def args(**kwargs):
    values = dict(check=True, output=None, normalized=None, review=None, recovery=None,
                  archive_sha256=None, recovery_sha256=None)
    values.update(kwargs)
    return argparse.Namespace(**values)


class ModelFixtureProtocol(a.Protocol):
    """Small main fixtures, using actual frozen identities but just one model/group.

    Only the test protocol's expected job roster is narrowed, never production code.
    This exercises each model's import/receipt path without 1960 synthetic episodes.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def jobs(self, stage):
        return [j for j in super().jobs(stage) if j['model_id'] == self.model and j['group'] == 'nominal']


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = a.Protocol()

    def setUp(self):
        self.data = dataset(self.p)

    def test_counts_and_separation(self):
        result = a.analyze(self.data, self.p)
        self.assertEqual((result['episodes'], result['jobs'], len(result['pairs'])), (40, 2, 20))
        self.assertIsNone(result['pooled_group_success'])
        self.assertEqual(len(self.p.jobs('main')), 98)
        self.assertEqual(len(self.p.cases), 140)

    def test_duplicates_missing_and_unpaired(self):
        for operation in ('duplicate', 'missing', 'pair'):
            d = copy.deepcopy(self.data)
            if operation == 'duplicate': d['episodes'][-1] = copy.deepcopy(d['episodes'][0])
            elif operation == 'missing': d['episodes'].pop()
            else: d['episodes'][0]['filter'] = 'on'
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                a.analyze(seal(d), self.p)

    def test_wrong_frozen_identities_even_with_rehashed_data(self):
        for key, value in dict(model_id='SAC_seed1', case_id='position_00', training_seed=1,
            group='angle', state_sha256='0'*64, model_sha256='1'*64, job_id='wrong', stage='main').items():
            d = copy.deepcopy(self.data); d['episodes'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): a.analyze(seal(d), self.p)

    def test_main_timing_mixture_and_stage_relabel_rejected(self):
        for change in ('row', 'stage'):
            d = copy.deepcopy(self.data)
            if change == 'row': d['episodes'][0]['stage'] = 'main'
            else: d['stage'] = 'main'
            with self.subTest(change=change), self.assertRaises(ValueError): a.analyze(seal(d), self.p)

    def test_flags_and_sha_rejected(self):
        self.data['episodes'][0]['flags']['success'] = False
        with self.assertRaises(ValueError): a.analyze(seal(self.data), self.p)
        self.data = dataset(self.p); self.data['scientific_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'SHA'): a.analyze(self.data, self.p)

    def test_off_fraction_is_null_not_zero(self):
        off = next(r for r in self.data['episodes'] if r['filter'] == 'off')
        self.assertIsNone(a.aggregate([off])['intervention_fraction'])
        off['metrics']['safety']['intervention_fraction'] = 0.
        with self.assertRaisesRegex(ValueError, 'fraction'): a.analyze(seal(self.data), self.p)

    def test_fraction_weights_decisions_not_episode_means(self):
        rows = [copy.deepcopy(self.data['episodes'][0]) for _ in range(2)]
        for i, (n, changed) in enumerate(((10, 0), (90, 90))):
            rows[i].update(case_id=str(i), filter='on', metrics=metrics('on', n, changed))
        self.assertEqual(a.aggregate(rows)['intervention_fraction'], .9)
        self.assertEqual(a.aggregate(rows)['interventions'], 90)

    def test_bad_denominators_and_event_counts(self):
        for key, value in [('accepted_decisions', 999), ('decisions', 999), ('intervention_count', 1001),
                           ('intervention_fraction', .5), ('position_limit_events', 1), ('intervals_exceeding_024', 1)]:
            m = metrics('on'); m['safety'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): a.validate_metrics(m, 'on')

    def test_strict_resolved_and_two_maxima_remain_distinct(self):
        m = metrics('off'); m['max_abs_position_interval'] = .24 + 4e-13
        s = m['safety']; s.update(max_abs_position_interval=.24+5e-13, desired_excess_max=5e-13,
            exceeds_024=True, exceeds_024_resolved=False, intervals_exceeding_024=1)
        a.validate_metrics(m, 'off')
        self.assertFalse(a.flags(m)['desired_exceedance'])
        self.assertNotEqual(m['max_abs_position_interval'], s['max_abs_position_interval'])
        self.assertTrue(s['exceeds_024'])

    def test_horizon_hold_and_safety_are_independent(self):
        m = metrics('on'); m.update(hold_final=0., hold_longest=0., first_hold_start=None,
            first_hold_confirmation=None, success_episode=False)
        a.validate_metrics(m, 'on'); f = a.flags(m)
        self.assertTrue(f['hold_failure']); self.assertFalse(f['constraint_event'])
        self.assertFalse(f['desired_exceedance']); self.assertEqual(m['completion']['reason'], 'horizon')

    def test_nan_inf_bool_and_duplicate_json_keys(self):
        for value in (float('nan'), float('inf'), True):
            m = metrics('on'); m['reward'] = value
            with self.subTest(value=value), self.assertRaises(ValueError): a.validate_metrics(m, 'on')
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.assertRaises(ValueError): a.parse_json(text)

    def test_exact_keys_for_metrics_completion_safety(self):
        for section in ('metrics', 'completion', 'safety'):
            template = metrics('on')
            expected = template if section == 'metrics' else template[section]
            # Exercise every missing key, not just one convenient example.
            for missing in expected:
                m = copy.deepcopy(template)
                value = m if section == 'metrics' else m[section]
                del value[missing]
                with self.subTest(section=section, missing=missing), self.assertRaisesRegex(ValueError, 'schema keys'):
                    a.validate_metrics(m, 'on')
            value = template if section == 'metrics' else template[section]
            value['unknown_future_field'] = 0
            with self.subTest(section=section), self.assertRaisesRegex(ValueError, 'schema keys'):
                a.validate_metrics(template, 'on')

    def test_definitions_cannot_change_or_disappear(self):
        for change in ('text', 'missing', 'extra', 'null'):
            m = metrics('on'); definitions = m['safety']['definitions']
            if change == 'text': definitions['hold'] = 'safety implies success'
            elif change == 'missing': del definitions['exceedance']
            elif change == 'extra': definitions['new_meaning'] = 'unknown'
            else: m['safety']['definitions'] = None
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'safety.definitions'):
                a.validate_metrics(m, 'on')

    def test_impossible_correction_statistics(self):
        for count, total, maximum, mean in ((0, 0., 0., 1e-15), (0, 1e-15, 0., 0.),
                (0, 0., 1e-15, 0.), (1, 2., 1., 2.), (1, 1., 2., 1.),
                (2, 3., 1., 1.5), (2, 2., 1., 1.5), (2, 1., 2., .5)):
            m = metrics('on', interventions=count)
            m['safety'].update(intervention_abs_sum=total, intervention_abs_max=maximum, intervention_abs_mean=mean)
            with self.subTest(values=(count, total, maximum, mean)), self.assertRaisesRegex(ValueError, 'correction'):
                a.validate_metrics(m, 'on')

    def test_valid_correction_zero_single_and_roundoff_tolerance(self):
        for count, total, maximum, mean in ((0, 0., 0., 0.), (1, .3, .3, .3),
                (2, .7, .5, .35), (10, 10.+1e-13, 1., (10.+1e-13)/10),
                (1, 1., 1, 1.)):
            m = metrics('on', interventions=count)
            m['safety'].update(intervention_abs_sum=total, intervention_abs_max=maximum, intervention_abs_mean=mean)
            with self.subTest(values=(count, total, maximum, mean)): a.validate_metrics(m, 'on')

    def test_provenance_tampering_or_removal_is_detected(self):
        for change in ('replace', 'delete', 'empty', 'hash', 'unknown_key', 'missing_source_field'):
            d = copy.deepcopy(self.data)
            if change == 'replace': d['provenance']['review']['export_manifest_sha256'] = '0'*64
            elif change == 'delete': del d['provenance']
            elif change == 'empty': d['provenance'] = {}
            elif change == 'hash': del d['envelope_sha256']
            elif change == 'unknown_key': d['provenance']['review']['trust_me'] = True
            else: del d['provenance']['review']['archive_sha256']
            with self.subTest(change=change), self.assertRaises(ValueError): a.analyze(d, self.p)
        d = copy.deepcopy(self.data); d['provenance'] = {}
        with self.assertRaisesRegex(ValueError, 'provenance'): a.analyze(a.seal_envelope(d), self.p)

    def test_envelope_does_not_replace_scientific_hash_or_version(self):
        d = copy.deepcopy(self.data); d['scientific_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'scientific SHA'): a.analyze(a.seal_envelope(d), self.p)
        d = copy.deepcopy(self.data); d['schema'] = 'structured-robustness-normalized-v1'
        with self.assertRaisesRegex(ValueError, 'schema/protocol'): a.analyze(a.seal_envelope(d), self.p)

    def check_model_recovery(self, mid):
        p = ModelFixtureProtocol(mid); data = dataset(p, 'main'); ex = recovery(p, data)
        normalized = a.import_compact(ex, p); result = a.analyze(normalized, p)
        self.assertEqual(result['episodes'], 40)
        self.assertEqual({g['model_id'] for g in result['groups']}, {mid})
        self.assertEqual({g['filter'] for g in result['groups']}, {'off', 'on'})
        self.assertEqual({r['model_sha256'] for r in normalized['episodes']}, {p.model_sha(mid)})
        a.recovery_bindings(ex, p)
        spec = next(v for k,v in ex.documents.items() if k.endswith('/specification.json'))
        algorithm = 'SAC' if mid == 'classical' else p.models[mid]['algorithm']
        self.assertEqual(spec['environment']['id'], f"cartpole-request-v2-{algorithm}-off")
        key = 'nominal_sha256' if mid == 'classical' else 'model_sha256'
        self.assertEqual(spec['source'][key], p.model_sha(mid))
        broken = copy.deepcopy(ex)
        next(v for k,v in broken.documents.items() if k.endswith('/specification.json'))['source'][key] = '0'*64
        with self.assertRaisesRegex(ValueError, 'checkpoint binding'): a.recovery_bindings(broken, p)
        broken = copy.deepcopy(ex)
        env = next(v for k,v in broken.documents.items() if k.endswith('/specification.json'))['environment']
        env['specification']['filter']['enabled'] = True  # off must stay off
        env['sha256'] = a.digest(env['specification'])
        with self.assertRaisesRegex(ValueError, 'environment contract'): a.recovery_bindings(broken, p)
        if mid != 'classical':
            self.assertEqual(spec['source']['training_contract_id'], f'cartpole-request-v2-{algorithm}-on')
            spec['source']['training_contract_id'] = f'cartpole-request-v2-{algorithm}-off'
            with self.assertRaisesRegex(ValueError, 'checkpoint generation identity'): a.recovery_bindings(ex, p)

    def test_sac_main_checkpoint_and_training_contract(self):
        self.check_model_recovery('SAC_seed1')

    def test_tqc_main_checkpoint_and_training_contract(self):
        self.check_model_recovery('TQC_seed2')

    def test_classical_main_nominal_and_environment_contract(self):
        self.check_model_recovery('classical')

    def test_verification_does_not_promote_saved_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'normalized.json'
            self.data['provenance']['review'].update(archive_sha256='0'*64, expected_archive_sha_verified=True)
            source.write_bytes(a.canonical(a.seal_envelope(self.data)))
            verified = a.run(args(normalized=source), self.p)['verification']
            self.assertEqual(verified['normalized_consistency'], 'passed')
            self.assertEqual(verified['compact_consistency'], 'not_run')
            self.assertEqual(verified['source_provenance'], 'not_revalidated')
            directory = root/'review'; directory.mkdir()
            ex = export_object(export_documents(self.p, self.data))
            for name, value in ex.documents.items():
                path = directory/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(a.canonical(value))
            verified = a.run(args(review=directory), self.p)['verification']
            self.assertEqual(verified['normalized_consistency'], 'passed')
            self.assertEqual(verified['compact_consistency'], 'passed')
            self.assertEqual(verified['source_provenance'], 'verified_against_exports')

    def test_determinism_under_row_order_and_provenance(self):
        expected = a.analyze(self.data, self.p)
        old_envelope = self.data['envelope_sha256']
        self.data['episodes'].reverse()
        self.data['provenance']['review']['export_manifest_sha256'] = 'f'*64
        a.seal_envelope(self.data)  # Deliberate new envelope; scientific payload unchanged.
        self.assertNotEqual(old_envelope, self.data['envelope_sha256'])
        self.assertEqual(a.canonical(a.analyze(self.data, self.p)), a.canonical(expected))

    def test_nested_export_identity_flags_groups_pairs_status(self):
        changes = [lambda d: d['structured_report.json']['episodes'][0]['case'].update(seed=123),
            lambda d: d['structured_report.json']['paired'].pop(),
            lambda d: d['structured_report.json']['groups'][0].update(successes=19),
            lambda d: d['structured_report.json']['status']['rows'][0].update(training_seed=2),
            lambda d: d['plan.json']['jobs'][0].update(algorithm='TQC'),
            lambda d: d['queue/queue_state.json']['attempts'].update(SAC_seed0_off_nominal=2)]
        for change in changes:
            docs = export_documents(self.p, self.data); change(docs)
            with self.subTest(change=change), self.assertRaises(ValueError): a.import_compact(export_object(docs), self.p)

    def test_only_export_clocks_may_differ(self):
        docs = export_documents(self.p, self.data); left = export_object(copy.deepcopy(docs))
        docs['structured_report.json']['status']['session']['elapsed_session_seconds'] += 1
        right = export_object(docs)  # Recompute source SHA as in a genuine new export.
        self.assertEqual(len(a.compare_exports(left, right)), 1)
        n1, n2 = (a.import_compact(x, self.p) for x in (left, right))
        self.assertEqual(n1['scientific_sha256'], n2['scientific_sha256'])
        self.assertEqual(a.analyze(n1, self.p), a.analyze(n2, self.p))
        self.assertNotEqual(n1['envelope_sha256'], n2['envelope_sha256'])
        right.documents['structured_report.json']['episodes'][0]['metrics']['reward'] += .1
        with self.assertRaisesRegex(ValueError, 'discrepancy'): a.compare_exports(left, right)

    def test_recovery_receipts_and_contract(self):
        ex = recovery(self.p, self.data)
        a.import_compact(ex, self.p); a.recovery_bindings(ex, self.p)
        path = next(k for k in ex.documents if k.endswith('/receipt.json'))
        ex.documents[path]['case'] = copy.deepcopy(ex.documents[path]['case'])
        ex.documents[path]['case']['seed'] = 8
        with self.assertRaisesRegex(ValueError, 'receipt case'): a.recovery_bindings(ex, self.p)

    def test_recovery_corruption_and_environment_rejected(self):
        for mutation in ('log', 'model', 'contract', 'source_roster', 'duplicate_attempt'):
            ex = recovery(self.p, self.data)
            spec = next(v for k,v in ex.documents.items() if k.endswith('/specification.json'))
            if mutation == 'log': ex.hashes[next(k for k in ex.hashes if k.endswith('/episode.json.gz'))] = '0'*64
            elif mutation == 'model': spec['source']['model_sha256'] = '0'*64
            elif mutation == 'contract':
                spec['environment']['specification']['horizon'] = 9.
                spec['environment']['sha256'] = a.digest(spec['environment']['specification'])
            elif mutation == 'source_roster': spec['environment']['sources'].pop(next(iter(spec['environment']['sources'])))
            else:
                path = next(k for k in ex.hashes if k.endswith('/episode.json.gz'))
                ex.hashes[path.replace('attempt_00000', 'attempt_00001')] = ex.hashes[path]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): a.recovery_bindings(ex, self.p)

    def test_archive_integrity_roster_and_unsafe_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'export.tar.gz'
            docs = export_documents(self.p, self.data); ex = export_object(docs)
            payload = {k: a.canonical(v) for k,v in ex.documents.items()}
            def write(items):
                with tarfile.open(target, 'w:gz') as tar:
                    for name, value in items:
                        ti = tarfile.TarInfo(name); ti.size = len(value); tar.addfile(ti, io.BytesIO(value))
            write(payload.items()); loaded = a.load_export(target, a.file_sha(target))
            self.assertEqual(a.import_compact(loaded, self.p)['scientific_sha256'], self.data['scientific_sha256'])
            with self.assertRaisesRegex(ValueError, 'archive SHA'): a.load_export(target, '0'*64)
            sidecar = Path(str(target)+'.sha256'); checksum = a.file_sha(target)
            for filename in (target.name, '/relocated/export.tar.gz'):
                for marker in (' ', '*'):
                    sidecar.write_text(checksum+' '+marker+filename+'\n')
                    self.assertTrue(a.load_export(target).evidence['expected_archive_sha_verified'])
            for content in (checksum+'  wrong.tar.gz\n', checksum+'\n', checksum+'  export.tar.gz\nextra\n',
                            '0'*64+'  export.tar.gz\n'):
                sidecar.write_text(content)
                with self.subTest(content=content), self.assertRaisesRegex(ValueError, 'sidecar'): a.load_export(target)
            sidecar.unlink()
            for kind in ('corrupt', 'extra', 'duplicate', 'traversal', 'missing'):
                items = list(payload.items())
                if kind == 'corrupt': items[0] = (items[0][0], b'{}')
                elif kind == 'extra': items.append(('unexpected.json', b'{}'))
                elif kind == 'duplicate': items.append(items[0])
                elif kind == 'traversal': items.append(('../outside', b''))
                else: items.pop(0)
                write(items)
                with self.subTest(kind=kind), self.assertRaises((ValueError, KeyError)): a.load_export(target)

    def test_check_is_read_only_output_explicit_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'normalized.json'; source.write_bytes(a.canonical(self.data))
            before = (source.read_bytes(), source.stat().st_mtime_ns, root.stat().st_mtime_ns)
            report = a.run(args(normalized=source), self.p)
            self.assertEqual(report['verification']['archive_integrity'], 'not_run')
            self.assertEqual(report['verification']['compact_consistency'], 'not_run')
            self.assertEqual(report['verification']['normalized_consistency'], 'passed')
            self.assertEqual(report['verification']['envelope_integrity'], 'passed')
            self.assertEqual(report['verification']['source_provenance'], 'not_revalidated')
            self.assertEqual(report['verification']['raw_metric_recomputation'], 'not_run')
            self.assertEqual(before, (source.read_bytes(), source.stat().st_mtime_ns, root.stat().st_mtime_ns))
            self.assertEqual(list(root.iterdir()), [source])
            target = root/'output'
            with self.assertRaises(ValueError): a.run(args(normalized=source, output=target), self.p)
            a.run(args(check=False, normalized=source, output=target), self.p)
            self.assertEqual(a.run(args(normalized=target/'normalized.json'), self.p)['analysis'], report['analysis'])
            with self.assertRaises(FileExistsError): a.run(args(check=False, normalized=source, output=target), self.p)

    def test_clean_clone_cli_without_site_packages_or_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Only tracked source/config files required by this analyzer, no environment/results.
            paths = set(self.p.sources) | {'cartpole/experiments/robustness_analysis.py'}
            paths.update('configs/robustness/'+x for x in ('manifest.json', 'states.json', 'bindings.json'))
            for relative in paths:
                dest = root/relative; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(a.ROOT/relative, dest)
            source = root/'normalized.json'; source.write_bytes(a.canonical(self.data))
            before = {str(p.relative_to(root)): (p.stat().st_mtime_ns, a.file_sha(p)) for p in root.rglob('*') if p.is_file()}
            cmd = [sys.executable, '-S', '-B', '-m', 'cartpole.experiments.robustness_analysis', '--check', '--normalized', str(source)]
            run = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=20)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout)['episodes'], 40)
            after = {str(p.relative_to(root)): (p.stat().st_mtime_ns, a.file_sha(p)) for p in root.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            self.assertFalse(list(root.rglob('__pycache__')))



def raw_fixture(mode='off', *, state=None, algorithm='SAC', seed=0, refusal=False, command=0., proposed=None, cart_motion=False):
    """Static synthetic endpoint observations, not a simulated trajectory.

    Defaults are exact upright equilibrium. Nonzero v intentionally exercises
    the measured endpoint residual, without claiming integration accuracy.
    """
    p = a.Protocol(); case = copy.deepcopy(p.cases['nominal_00'])
    state = state or dict(cart_position=0., cart_velocity=0., pole_angle=math.pi, pole_angular_velocity=0.)
    case['initial_state'] = state; case['state_sha256'] = a.digest(state)
    contract = copy.deepcopy(a.read_json(p.root / f"configs/cloud/{('SAC' if algorithm == 'classical' else algorithm).lower()}_training_v2_{mode}.json")['training_contract'])
    contract['sources'] = {k: a.file_sha(p.root/k) for k in contract['sources']}
    env = contract['specification']
    meta = dict(algorithm=algorithm, training_seed=None if algorithm == 'classical' else seed,
        evaluation_case_id=case['id'], filter_mode=mode, initial_state=state,
        environment='CourseworkCartPole-v0' if algorithm == 'classical' else env['environment_api'],
        config=env['physical_config'], integrator=env['integrator'], reward=env['reward'],
        control_interval=.01, horizon=10., required_hold_duration=2.,
        upright_thresholds=dict(angle=math.pi/18, angular_velocity=.5, position=.2, velocity=.2),
        seed=int(a.digest(case)[:8],16), reset_mode='explicit')
    if algorithm != 'classical':
        meta['training_contract'] = contract
        if mode == 'on': meta['reset_admission'] = 'filter_inner'
    if mode == 'on': meta['safety_filter'] = env['filter']['limits']
    proposed = command if proposed is None else proposed
    n = 1 if refusal else 1000
    rows = [dict(index=i, time=0. if refusal else i*.01, error=0, cart_acceleration=0. if i==0 or refusal else command, **state) for i in range(n+1)]
    if cart_motion:
        for row in rows:
            row['cart_position'] = state['cart_position'] + state['cart_velocity']*row['time']
    trans=[]
    for i in range(n):
        h=rows[i+1]['time']-rows[i]['time']; terminal=refusal; u=None if refusal else command
        factors=None if refusal else dict(height=(1-math.cos(state['pole_angle']))/2,
            position=(.25/math.hypot(.25,rows[i+1]['cart_position']))**2,
            velocity=(2/math.hypot(2,state['cart_velocity']))**2,
            angular_velocity=(2/math.hypot(2,state['pole_angular_velocity']))**2,
            acceleration=(40/math.hypot(40,command))**2)
        reward=-5. if refusal else h/.01*math.prod(factors.values())
        tr=dict(index=i,from_state=i,to_state=i+1,time_start=rows[i]['time'],time_end=rows[i+1]['time'],
            dt_actual=h,dt_requested=.01,u_requested=proposed,u_commanded=u,applied_acceleration=u,
            action_clipped=False,simulator_error=0,terminated=terminal,
            stop_reasons=['safety_filter_refusal'] if refusal else [],boundary_events=[],
            action_requested=proposed/4,action_commanded=None if refusal else command/4,
            gym_terminated=terminal,gym_truncated=not refusal and i==n-1,
            reward=reward,reward_version='R1-v0',reward_scale=h/.01,reward_factors=factors,
            reward_terms=dict(alignment=0. if refusal else reward,failure=-5. if refusal else 0.))
        if mode=='on':
            tr['safety_filter']=dict(u_proposed=proposed,u_filtered=u,feasible=not refusal,
                intervened=None if refusal else command!=proposed,correction=None if refusal else command-proposed,
                correction_abs=None if refusal else abs(command-proposed),reason='inadmissible_state' if refusal else 'unchanged')
        trans.append(tr)
    duration=rows[-1]['time']; inside=abs(math.atan2(math.sin(state['pole_angle']-math.pi),math.cos(state['pole_angle']-math.pi)))<=math.pi/18 and abs(state['cart_position'])<=.2 and abs(state['cart_velocity'])<=.2 and abs(state['pole_angular_velocity'])<=.5
    hold=dict(success_episode=inside and not refusal,first_hold_start=0. if inside and not refusal else None,
        first_hold_confirmation=2. if inside and not refusal else None,hold_final=duration if inside else 0.,hold_longest=duration if inside else 0.)
    evaluation=dict(**hold,required_hold_duration=2.,criterion='consecutive recorded states; first/last qualifying timestamps; time tolerance 1e-12 s',
        hold_intervals=[dict(start=0.,end=duration,duration=duration)] if inside else [],mode_switch_count=0,mode_switches=[])
    return dict(schema_version='cartpole.episode.v1',metadata=meta,states=rows,transitions=trans,
        completion=dict(reason='terminal' if refusal else 'horizon',terminated=refusal,truncated=not refusal,time=duration,transition_count=n,stop_reasons=trans[-1]['stop_reasons']),
        metrics=dict(duration=duration,max_abs_position=max(abs(r['cart_position']) for r in rows),max_abs_velocity=abs(state['cart_velocity']),
            max_abs_requested_acceleration=abs(proposed),max_abs_commanded_acceleration=0. if refusal else abs(command),max_abs_applied_acceleration=0. if refusal else abs(command),
            final_angle_error=math.atan2(math.sin(state['pole_angle']-math.pi),math.cos(state['pole_angle']-math.pi)),all_samples_in_upright_region=inside,clipped_transition_count=0),
        evaluation=evaluation,gym_final_hold_metrics=dict(**hold,in_upright_region=inside),gym_return=math.fsum(t['reward'] for t in trans))


class RawTests(unittest.TestCase):
    def test_valid_raw_and_fsum(self):
        raw=a.recompute_raw(raw_fixture())
        self.assertEqual(raw['metrics']['reward'],1000.)
        self.assertEqual(raw['metrics']['hold_final'],10.)
        self.assertEqual(raw['qualifying_upright_samples'],1001)

    def test_hold_final_corruption(self):
        log=raw_fixture();log['evaluation']['hold_final']=9.
        with self.assertRaisesRegex(ValueError,'hold_final'):a.recompute_raw(log)

    def test_success_corruption(self):
        log=raw_fixture();log['evaluation']['success_episode']=False
        with self.assertRaisesRegex(ValueError,'success_episode'):a.recompute_raw(log)

    def test_reward_corruption(self):
        log=raw_fixture();log['transitions'][2]['reward']+=.01
        with self.assertRaisesRegex(ValueError,'transition_reward'):a.recompute_raw(log)

    def test_return_corruption(self):
        log=raw_fixture();log['gym_return']+=.01
        with self.assertRaisesRegex(ValueError,'gym_return'):a.recompute_raw(log)

    def test_zero_and_one_intervention(self):
        log=raw_fixture('on');raw=a.recompute_raw(log)
        self.assertEqual(raw['metrics']['safety']['intervention_count'],0)
        tr=log['transitions'][0];tr['u_requested']=1.;tr['action_requested']=.25
        tr['safety_filter'].update(u_proposed=1.,intervened=True,correction=-1.,correction_abs=1.)
        log['metrics']['max_abs_requested_acceleration']=1.
        raw=a.recompute_raw(log);s=raw['metrics']['safety']
        self.assertEqual((s['intervention_count'],s['intervention_abs_sum'],s['intervention_abs_max'],s['intervention_abs_mean']),(1,1.,1.,1.))
        for k,v in [('intervened',False),('correction_abs',2.),('correction',1.)]:
            broken=copy.deepcopy(log);broken['transitions'][0]['safety_filter'][k]=v
            with self.subTest(k=k),self.assertRaises(ValueError):a.recompute_raw(broken)

    def test_refusal_zero_transition(self):
        raw=a.recompute_raw(raw_fixture('on',refusal=True))
        self.assertEqual(raw['metrics']['reward'],-5.)
        self.assertEqual(raw['metrics']['safety']['refusal_count'],1)
        self.assertIsNone(raw['metrics']['safety']['intervention_fraction'])

    def test_vertex_and_distinct_maxima(self):
        log=raw_fixture(command=-4.,state=dict(cart_position=.1,cart_velocity=.02,pole_angle=math.pi,pole_angular_velocity=0.))
        raw=a.recompute_raw(log)
        self.assertAlmostEqual(raw['metrics']['max_abs_position_interval'],.10005)
        log['transitions'][0]['cart_interval']=dict(max_abs_position_interval=.1,vertex_time=.005,
            desired_excess=0.,exceeds_024=False,exceeds_024_resolved=False,endpoint_position_residual=0.,endpoint_velocity_residual=.04)
        with self.assertRaisesRegex(ValueError,'cart_interval/max_abs'):a.recompute_raw(log)

    def test_nan_and_time_and_postterminal(self):
        for section,key,value in [('state','pole_angle',float('nan')),('transition','dt_actual',-.01),('transition','gym_truncated',True)]:
            log=raw_fixture();(log['states'][0] if section=='state' else log['transitions'][0])[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):a.recompute_raw(log)

    def test_nested_gzip_damage_and_bomb_limit(self):
        payload=gzip.compress(a.canonical(raw_fixture()))
        self.assertEqual(a.decode_raw_log(payload)[0]['schema_version'],'cartpole.episode.v1')
        for bad in (payload[:-8],b'not gzip'):
            with self.assertRaises((OSError,EOFError)):a.decode_raw_log(bad)
        with patch.object(a,'MAX_MEMBER',10),self.assertRaisesRegex(ValueError,'size limit'):a.decode_raw_log(payload)

    def test_no_raw_claim_for_normalized(self):
        with self.assertRaisesRegex(ValueError,'requires recovery'):
            a.run(args(normalized=Path('unused'),verify_raw_metrics=True))

    def bound_fixture(self,model='SAC_seed0'):
        p=ModelFixtureProtocol(model);data=dataset(p,'main');ex=recovery(p,data)
        for path in list(ex.hashes):
            if not path.endswith('/episode.json.gz'):continue
            job=next(j for j in p.jobs('main') if '/'+j['id']+'/' in path)
            cid=path.split('/cases/')[1].split('/')[0];case=p.cases[cid]
            log=raw_fixture(job['mode'],state=case['initial_state'],algorithm=job['algorithm'],seed=job['seed'])
            log['metadata'].update(evaluation_case_id=cid,seed=int(a.digest(case)[:8],16))
            raw=a.recompute_raw(log);ex.raw[path]=raw
            rp=str(Path(path).parent)+'/receipt.json'
            ex.documents[rp]['metrics']=dict(id=cid,subset=job['group'],**raw['metrics'])
            row=next(r for r in ex.documents['structured_report.json']['episodes'] if r['job_id']==job['id'] and r['case']['id']==cid)
            row['metrics']={k:raw['metrics'][k] for k in a.METRIC_KEYS};row['flags']=a.flags(row['metrics'])
        # Compact import is covered separately; these tests isolate raw bindings
        # (its aggregate summaries are deliberately still the old fixture's).
        normalized={'episodes':[dict(job_id=r['job_id'],case_id=r['case']['id'],metrics=r['metrics']) for r in ex.documents['structured_report.json']['episodes']]}
        return p,ex,normalized

    def test_raw_bindings_sac_tqc_classical(self):
        for mid in ('SAC_seed0','TQC_seed1','classical'):
            with self.subTest(model=mid):
                p,ex,n=self.bound_fixture(mid)
                with patch.object(a,'import_compact',return_value=n):result=a.verify_raw_bindings(ex,p)
                self.assertEqual(result['episodes'],40)
                self.assertEqual(result['classical_provenance']['status'],'known_mismatch' if mid=='classical' else 'not_applicable')
                if mid=='classical':
                    next(iter(ex.raw.values()))['metadata']['training_contract']={}
                    with patch.object(a,'import_compact',return_value=n),self.assertRaisesRegex(ValueError,'classical metadata'):a.verify_raw_bindings(ex,p)

    def test_raw_identity_receipt_and_metric_corruption(self):
        p,ex,n=self.bound_fixture()
        for mutation in ('initial','log_sha','spec_sha','missing','duplicate','count','sum','max','mean','interval'):
            bad=copy.deepcopy(ex);lp=next(iter(bad.raw));rp=str(Path(lp).parent)+'/receipt.json'
            receipt=bad.documents[rp]
            if mutation=='initial':bad.raw[lp]['initial_state']['cart_position']+=.001
            elif mutation=='log_sha':receipt['log_sha256']='0'*64
            elif mutation=='spec_sha':receipt['specification_sha256']='0'*64
            elif mutation=='missing':del bad.raw[lp]
            elif mutation=='duplicate':bad.raw[lp.replace('attempt_00000','attempt_00001')]=bad.raw[lp]
            elif mutation=='interval':receipt['metrics']['max_abs_position_interval']+=.01
            else:receipt['metrics']['safety']['intervention_'+('count' if mutation=='count' else 'abs_'+mutation)]+=1
            with self.subTest(mutation=mutation),patch.object(a,'import_compact',return_value=n),self.assertRaises(ValueError):a.verify_raw_bindings(bad,p)

    def test_raw_archive_read_only_and_opt_in(self):
        p=a.Protocol();data=dataset(p);docs=export_documents(p,data)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tiny.tar.gz';name='queue/jobs/test/evaluation/cases/x/attempt_00000/episode.json.gz'
            payload=gzip.compress(a.canonical(raw_fixture()))
            ex=export_object(docs,{name:payload},full=True)
            items={k:a.canonical(v) for k,v in ex.documents.items()};items[name]=payload
            with tarfile.open(path,'w:gz') as tar:
                for k,v in items.items():
                    ti=tarfile.TarInfo(k);ti.size=len(v);tar.addfile(ti,io.BytesIO(v))
            before=(path.stat().st_mtime_ns,a.file_sha(path),sorted(Path(tmp).iterdir()))
            self.assertFalse(a.load_export(path).raw)
            self.assertEqual(len(a.load_export(path,verify_raw=True).raw),1)
            self.assertEqual(before,(path.stat().st_mtime_ns,a.file_sha(path),sorted(Path(tmp).iterdir())))
            # Real CLI error path: the extra synthetic log has no receipt roster.
            # It must fail read-only, rather than publish a partial raw claim.
            run=subprocess.run([sys.executable,'-S','-B','-m',
                'cartpole.experiments.robustness_analysis','--check','--review',str(path),
                '--verify-raw-metrics'],cwd=a.ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(run.returncode,2,run.stderr)
            self.assertEqual(run.stdout,'')
            self.assertEqual(before,(path.stat().st_mtime_ns,a.file_sha(path),sorted(Path(tmp).iterdir())))


class BoundaryRawTests(unittest.TestCase):
    def boundary(self, x, v, u, dt, actual=None, native=0, events=None):
        expected=a.cart_boundary_events(x,v,u,dt)
        h=min((e['time'] for e in expected),default=dt) if actual is None else actual
        reached=[e for e in expected if e['time']<=h+1e-12] if events is None else events
        reasons=list(dict.fromkeys(e['reason'] for e in reached))
        if native: reasons.insert(0,'simulator_error')
        before=dict(cart_position=x,cart_velocity=v,error=0)
        after=dict(error=native)
        tr=dict(index=7,dt_actual=h,dt_requested=dt,applied_acceleration=u if h else None,
            u_commanded=u,boundary_events=reached,stop_reasons=reasons)
        return before,after,tr

    def verify(self, values):
        return a.verify_cart_boundary(*values,a.RawComparison())

    def test_false_position_and_velocity_in_center(self):
        for reason in ('position_limit','velocity_limit'):
            values=self.boundary(0.,0.,0.,.01,events=[dict(reason=reason,side=1,time=.01)])
            with self.subTest(reason=reason),self.assertRaisesRegex(ValueError,r'computed_events=\[\]'):
                self.verify(values)

    def test_forged_terminal_log_rejected_even_with_consistent_saved_fields(self):
        log=raw_fixture();tr=log['transitions'][-1]
        tr.update(stop_reasons=['position_limit'],boundary_events=[dict(reason='position_limit',side=1,time=tr['dt_actual'])],
            terminated=True,gym_terminated=True,gym_truncated=False)
        tr['reward']-=5.;tr['reward_terms']['failure']=-5.;log['gym_return']-=5.
        log['completion'].update(reason='terminal',terminated=True,truncated=False,stop_reasons=['position_limit'])
        log['evaluation']['success_episode']=log['gym_final_hold_metrics']['success_episode']=False
        with self.assertRaisesRegex(ValueError,r'transition\[999\].*computed_events=\[\]'):a.recompute_raw(log)

    def test_wrong_side_or_time(self):
        for field,value in (('side',-1),('time',.003)):
            b,s,tr=self.boundary(.249,.2,0.,.01)
            tr['boundary_events'][0][field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'computed_events'):
                self.verify((b,s,tr))

    def test_later_event_not_first_and_wrong_duration(self):
        # Position first at .002 s; velocity reaches 2 only at .005 s.
        x=.25-1.99*.002-.5*2*.002**2
        b,s,tr=self.boundary(x,1.99,2.,.01)
        tr.update(dt_actual=.005,boundary_events=[dict(reason='velocity_limit',side=1,time=.005)],stop_reasons=['velocity_limit'])
        with self.assertRaisesRegex(ValueError,r'transition\[7\].*/duration'):self.verify((b,s,tr))

    def test_simultaneous_events_both_sides(self):
        for sign in (-1,1):
            values=self.boundary(sign*(.25-1.99*.005-.5*2*.005**2),sign*1.99,sign*2.,.01)
            events,reasons=self.verify(values)
            self.assertEqual(reasons,['position_limit','velocity_limit'])
            self.assertEqual([e['side'] for e in events],[sign,sign])
            self.assertTrue(all(abs(e['time']-.005)<1e-12 for e in events))

    def test_t0_outward_and_inward(self):
        for sign in (-1,1):
            events,_=self.verify(self.boundary(sign*.25,sign*.1,0.,.01))
            self.assertEqual(events,[dict(reason='position_limit',side=sign,time=0.)])
            self.assertEqual(a.cart_boundary_events(sign*.25,-sign*.1,0.,.01),[])
            events,_=self.verify(self.boundary(sign*.25,0.,sign*1.,.01))
            self.assertEqual(events[0]['time'],0.)

    def test_event_at_requested_end_and_no_event(self):
        for sign in (-1,1):
            events,_=self.verify(self.boundary(sign*.249,sign*.1,0.,.01))
            self.assertAlmostEqual(events[0]['time'],.01,places=12)
        self.assertEqual(self.verify(self.boundary(0.,0.,0.,.01)),([],[]))

    def test_tangency_and_excursion_return(self):
        self.assertEqual(a.cart_boundary_events(.24,.2,-2.,.2),[])
        events=a.cart_boundary_events(.24,.3,-2.,.3)
        self.assertEqual(len(events),1)
        self.assertLess(events[0]['time'],.15)

    def test_native_early_stop_discards_future_boundary(self):
        values=self.boundary(.249,.2,0.,.01,actual=.001,native=2)
        self.assertEqual(self.verify(values),([],['simulator_error']))
        values[2]['boundary_events']=[dict(reason='position_limit',side=1,time=.005)]
        values[2]['stop_reasons'].append('position_limit')
        with self.assertRaisesRegex(ValueError,r'computed_events=\[\]'):self.verify(values)

    def test_tolerated_numeric_telemetry_is_registered(self):
        log=raw_fixture('on');log['transitions'][0]['safety_filter']['u_proposed']=1e-13
        errors=a.recompute_raw(log)['errors']['filter_decision']
        self.assertEqual(errors['max_abs'],1e-13)
        self.assertEqual(errors['max_rel'],1.)
        self.assertEqual(errors['comparisons'],2000)
        log['transitions'][0]['safety_filter']['u_proposed']=2e-12
        with self.assertRaisesRegex(ValueError,'filter_decision'):a.recompute_raw(log)

    def test_legacy_tolerance_uses_scoped_registry(self):
        collector=a.RawComparison();token=a.NUMERICAL_AUDIT.set(collector)
        try:
            a.same(1e-13,0.,'group summary/mean')
            self.assertEqual(collector.errors['consistency'],dict(max_abs=1e-13,max_rel=1.,comparisons=1))
            with self.assertRaises(ValueError):a.same(2e-12,0.,'group summary/mean')
            with self.assertRaises(ValueError):a.same(True,1,'discrete identity')
        finally:
            a.NUMERICAL_AUDIT.reset(token)
        self.assertIsNone(a.NUMERICAL_AUDIT.get())


def write_synthetic_tar(path, documents, payloads):
    ex=export_object(documents,payloads,full=True)
    contents={**payloads,**{k:a.canonical(v) for k,v in ex.documents.items()}}
    with tarfile.open(path,'w:gz') as archive:
        for name,payload in contents.items():
            member=tarfile.TarInfo(name);member.size=len(payload)
            archive.addfile(member,io.BytesIO(payload))


class EndToEndRawTests(unittest.TestCase):
    def test_complete_raw_cli_and_rehashed_scientific_corruption(self):
        # The actual timing roster: 2 jobs x 20 tiny compressed synthetic logs.
        # Cart endpoints use x0+v0*t, u=0; pendulum observations are synthetic.
        # No protocol weakening, subclass, imports of ML or patched audit stages.
        p=a.Protocol();data=dataset(p);records={}
        for row in data['episodes']:
            case=p.cases[row['case_id']]
            log=raw_fixture(row['filter'],state=case['initial_state'],cart_motion=True)
            log['metadata'].update(evaluation_case_id=case['id'],seed=int(a.digest(case)[:8],16))
            raw=a.recompute_raw(log)['metrics']
            payload=gzip.compress(a.canonical(log),mtime=0)
            records[row['job_id'],row['case_id']]=(payload,dict(id=case['id'],subset=case['subset'],**raw))
            row.update(metrics={k:raw[k] for k in a.METRIC_KEYS},log_sha256=hashlib.sha256(payload).hexdigest())
            row['flags']=a.flags(row['metrics'])
        seal(data);ex=recovery(p,data,records)
        docs={k:v for k,v in ex.documents.items() if k!='EXPORT_MANIFEST.json'}
        payloads={f'queue/jobs/{job}/evaluation/cases/{cid}/attempt_00000/episode.json.gz':value[0]
                  for (job,cid),value in records.items()}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'recovery.tar.gz'
            write_synthetic_tar(target,docs,payloads)
            before=(a.file_sha(target),target.stat().st_mtime_ns,root.stat().st_mtime_ns)
            command=[sys.executable,'-S','-B','-m','cartpole.experiments.robustness_analysis',
                     '--check','--review',str(target),'--verify-raw-metrics']
            run=subprocess.run(command,cwd=a.ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(run.returncode,0,run.stderr)
            report=json.loads(run.stdout)
            self.assertEqual(report['verification']['raw_metric_recomputation'],'passed')
            self.assertEqual(report['verification']['raw_audit']['transitions'],40000)
            self.assertEqual(before,(a.file_sha(target),target.stat().st_mtime_ns,root.stat().st_mtime_ns))
            self.assertEqual(list(root.iterdir()),[target])
            # Corrupt raw reward, but preserve all scientific summaries. Rehash
            # log->receipt->pointer->job manifest->queue->export, not the metrics.
            lp=next(iter(payloads));log=a.decode_raw_log(payloads[lp])[0]
            log['transitions'][0]['reward']+=.1
            payloads[lp]=gzip.compress(a.canonical(log),mtime=0)
            rp=str(Path(lp).parent)+'/receipt.json';docs[rp]['log_sha256']=hashlib.sha256(payloads[lp]).hexdigest()
            pointer=str(Path(lp).parent.parent)+'/completed.json';docs[pointer]['sha256']=a.digest(docs[rp])
            job=lp.split('/')[2];cid=docs[rp]['case']['id']
            cr=next(r for r in docs['structured_report.json']['episodes'] if r['job_id']==job and r['case']['id']==cid)
            cr['log_sha256']=docs[rp]['log_sha256']
            base='queue/jobs/'+job;hashes=export_object(docs,payloads).hashes
            jm={k[len(base)+1:]:v for k,v in hashes.items() if k.startswith(base+'/') and k!=base+'/job_manifest.json'}
            docs[base+'/job_manifest.json']=jm
            docs['queue/queue_state.json']['completed'][job]['artifact_manifest_sha256']=a.digest(jm)
            bad=root/'bad.tar.gz';write_synthetic_tar(bad,docs,payloads)
            # Demonstrate container/compact/receipt integrity independently.
            loaded=a.load_export(bad);a.import_compact(loaded,p);a.recovery_bindings(loaded,p)
            command[command.index(str(target))]=str(bad)
            run=subprocess.run(command,cwd=a.ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(run.returncode,2)
            self.assertIn('transition_reward',run.stderr)
            self.assertIn(lp,run.stderr)
            self.assertNotIn('Traceback',run.stderr)
            self.assertEqual(run.stdout,'')

    def test_valid_gzip_header_corrupt_deflate_cli(self):
        # BFINAL=1, BTYPE=3 is an invalid DEFLATE block, after a valid gzip header.
        bad=bytes.fromhex('1f8b0800000000000003')+b'\x07'+b'\x00'*8
        with self.assertRaises(zlib.error):a.decode_raw_log(bad)
        name='queue/jobs/SAC_seed0_on_nominal/evaluation/cases/nominal_00/attempt_00000/episode.json.gz'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'deflate.tar.gz';output=root/'must-not-exist'
            write_synthetic_tar(target,{}, {name:bad})
            before=(a.file_sha(target),target.stat().st_mtime_ns,root.stat().st_mtime_ns)
            for mode in (['--check'],['--output',str(output)]):
                run=subprocess.run([sys.executable,'-S','-B','-m','cartpole.experiments.robustness_analysis',
                    '--review',str(target),'--verify-raw-metrics',*mode],cwd=a.ROOT,capture_output=True,text=True,timeout=20)
                self.assertEqual(run.returncode,2)
                self.assertIn('validation failed:',run.stderr);self.assertIn(name,run.stderr)
                self.assertNotIn('Traceback',run.stderr);self.assertEqual(run.stdout,'')
                self.assertFalse(output.exists())
                self.assertEqual(before,(a.file_sha(target),target.stat().st_mtime_ns,root.stat().st_mtime_ns))


if __name__ == '__main__':
    unittest.main()
