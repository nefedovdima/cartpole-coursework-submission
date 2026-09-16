"""Shared physical metrics with CUDA-aware isolation and unchanged selection."""
import json
from pathlib import Path
import time

import numpy as np

from cartpole.experiments.cloud_config import VALIDATION_SHA256
from cartpole.experiments.cloud_io import ROOT, atomic_json, sha256
from cartpole.experiments.cloud_learning import isolated_policy
from cartpole.experiments.gym_diagnostics import record_episode
from cartpole.experiments.sac_pilot import aggregate, episode_metrics
from cartpole.rl.pilot_config import make_pilot_env
from cartpole.rl.training_contract import evaluation_action, assert_environment
from cartpole.experiments.safety_metrics import safety_metrics


def validation_cases():
    path = ROOT/'tests/fixtures/validation_states.json'
    if sha256(path) != VALIDATION_SHA256:
        raise ValueError('saved validation set checksum mismatch')
    cases = json.loads(path.read_text())['cases']
    ids = {f'validation_{i}' for i in range(1000, 1020)} | {'bottom', 'plus_002', 'minus_002'}
    if {c['id'] for c in cases} != ids or len(cases) != 23:
        raise ValueError('expected exactly 20 original validation states and three named states')
    return cases


def interval_margins(log):
    """Cart x extrema inside each constant-u interval, without pendulum integration."""
    states=log['states']
    max_x=max(abs(s['cart_position']) for s in states)
    max_v=max(abs(s['cart_velocity']) for s in states)
    for state,tr in zip(states,log['transitions']):
        u,dt=tr['applied_acceleration'],tr['dt_actual']
        if u is not None and u != 0 and dt > 0:
            turning=-state['cart_velocity']/u
            if 0 < turning < dt:
                x=state['cart_position']+state['cart_velocity']*turning+u*turning**2/2
                max_x=max(max_x,abs(x))
    return dict(max_abs_position_interval=max_x,position_margin_interval=.25-max_x,
                velocity_margin_interval=2.-max_v,
                interval_margin_definition='analytic cart extrema under held u; original Drake pendulum states')


def evaluate(model, cases, output, label, *, algorithm, seed, guard=None, details=True,
             env_factory=make_pilot_env, diagnostic=False, checkpoint_identity=None, evaluation_contract=None):
    started = time.perf_counter(); output = Path(output)
    (output/'evaluations').mkdir(exist_ok=True); (output/'logs').mkdir(exist_ok=True)
    rows, complete, error = [], False, None
    report_path = output/'evaluations'/(label+'.json')
    if report_path.exists():
        raise FileExistsError(report_path)
    env = env_factory()
    filtered = getattr(model, 'training_contract', None) is not None
    if filtered:
        expected = model.training_contract
        if evaluation_contract is not None:
            from cartpole.rl.methods import parse_contract
            from cartpole.rl.training_contract import specification, validate_identity
            validate_identity(evaluation_contract)
            trained_algorithm, _ = parse_contract(expected['id'])
            evaluated_algorithm, _ = parse_contract(evaluation_contract['id'])
            if trained_algorithm != evaluated_algorithm or trained_algorithm != algorithm:
                raise ValueError('evaluation adapter algorithm differs from checkpoint')
            expected = evaluation_contract
        assert_environment(env, expected)
    try:
        with isolated_policy(model):
            for case in cases:
                if guard:
                    guard()
                def policy(obs, info):
                    if guard:
                        guard()
                    action = evaluation_action(model, obs, filtered=filtered)
                    if not np.isfinite(action).all():
                        raise FloatingPointError('non-finite deterministic evaluation action')
                    return action
                path = (output/'logs'/f'{label}_{case["id"]}.json') if details and (filtered or case['id'] in {
                    'bottom', 'plus_002', 'minus_002', 'validation_1000'}) else None
                log = record_episode(env, policy, seed=case['seed'], options={'initial_state': case['initial_state']},
                       log_path=path, metadata=dict(algorithm=algorithm, training_seed=seed,
                           trained_transitions=model.num_timesteps, updates=model._n_updates,
                           controller_label=f'{algorithm} · seed {seed} · {model.num_timesteps} переходов',
                           evaluation_label=label, diagnostic=diagnostic))
                rows.append(dict(id=case['id'], subset=case['subset'], **episode_metrics(log), **interval_margins(log),
                                 **({'safety': safety_metrics(log)} if filtered else {})))
            complete = True
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc)); raise
    finally:
        env.close()
        full_set = sorted(cases, key=lambda r: r['id']) == sorted(validation_cases(), key=lambda r: r['id'])
        report = dict(label=label, algorithm=algorithm, training_seed=seed,
                      trained_transitions=model.num_timesteps, updates=model._n_updates,
                      complete=complete, selection_eligible=complete and full_set,
                      scientific_result=not diagnostic,
                      diagnostic=diagnostic, seconds=time.perf_counter()-started, error=error, episodes=rows,
                      training_contract_id=getattr(model, 'training_contract', {}).get('id'),
                      evaluation_contract_id=(evaluation_contract or getattr(model, 'training_contract', {})).get('id'),
                      control_transitions=sum(r['completion']['transition_count'] for r in rows),
                      validation=aggregate([r for r in rows if r['subset'] == 'validation']),
                      named=aggregate([r for r in rows if r['subset'] == 'named']))
        report.update(checkpoint_identity or {})
        if 'evaluation_started_cumulative' in report:
            report['cumulative_seconds'] = report.pop('evaluation_started_cumulative')+report['seconds']
        complete_aggregates(report, rows)
        atomic_json(report_path, report)
    return report


def complete_aggregates(report, rows):
    """Identical aggregation for sequential and persistent-worker evaluations."""
    for subset in ('validation', 'named'):
        selected = [r for r in rows if r['subset'] == subset]
        if report[subset] is not None:
            rms = [r['rms_acceleration'] for r in selected if r['rms_acceleration'] is not None]
            report[subset]['mean_rms_acceleration'] = sum(rms)/len(rms) if rms else None
            report[subset]['minimum_position_margin_interval'] = min(r['position_margin_interval'] for r in selected)
            report[subset]['minimum_velocity_margin_interval'] = min(r['velocity_margin_interval'] for r in selected)
            report[subset]['successful_only'] = aggregate([r for r in selected if r['success_episode']])
            for key in ('first_hold_start', 'first_hold_confirmation'):
                values = [r[key] for r in selected if r[key] is not None]
                report[subset]['mean_'+key] = sum(values)/len(values) if values else None
                report[subset][key+'_count'] = len(values)
    return report
