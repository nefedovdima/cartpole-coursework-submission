"""One bounded E6a CPU pilot; independent validation and fresh-process checks."""
import os
# Set before importing NumPy, Drake or Torch in this CLI process.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import resource
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from cartpole.common import State
from cartpole.control.swing_up import NominalTrajectory
from cartpole.experiments.gym_diagnostics import ClassicalPolicy, record_episode, write_json
from cartpole.experiments.lqr_hold import ROOT, capture_source
from cartpole.rl.env import observation_from_state
from cartpole.rl.pilot_config import make_pilot_config, make_pilot_env
from cartpole.rl.sac import (ObservedSAC, assert_pilot_contract, configure_torch,
                            constructor_settings, finite_parameters, sac_settings)


FIELDS = ('cart_position', 'cart_velocity', 'pole_angle', 'pole_angular_velocity')
DETAIL_CASES = {'bottom', 'plus_002', 'minus_002', 'validation_1000'}


@contextmanager
def preserve_global_rng():
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def validation_cases():
    """PLAN distribution B and seeds 1000..1019; never the final set."""
    env = make_pilot_env()
    cases = []
    try:
        for seed in range(1000, 1020):
            _, info = env.reset(seed=seed)
            cases.append(dict(id=f'validation_{seed}', subset='validation', seed=seed,
                              initial_state={k: info['state'][k] for k in FIELDS}))
    finally:
        env.close()
    for name, theta in [('bottom', 0.), ('plus_002', .02), ('minus_002', -.02)]:
        cases.append(dict(id=name, subset='named', seed=None,
                          initial_state=dict(cart_position=0., cart_velocity=0.,
                                             pole_angle=theta, pole_angular_velocity=0.)))
    return cases


def episode_metrics(log):
    transitions, states = log['transitions'], log['states']
    duration = log['completion']['time']
    effort = math.fsum(tr['applied_acceleration']**2*tr['dt_actual']
                       for tr in transitions if tr['applied_acceleration'] is not None)
    return dict(**log['metrics'], **log['evaluation'], completion=log['completion'],
                reward=log['gym_return'],
                discounted_reward=math.fsum(make_pilot_config()['gamma']**i*tr['reward'] for i, tr in enumerate(transitions)),
                max_abs_omega=max(abs(s['pole_angular_velocity']) for s in states),
                control_effort=effort, rms_acceleration=math.sqrt(effort/duration) if duration else None,
                boundary_failure=any(r in ('position_limit', 'velocity_limit') for r in log['completion']['stop_reasons']))


def aggregate(rows):
    if not rows:
        return None
    lift = [r['first_hold_start'] for r in rows if r['first_hold_start'] is not None]
    return dict(episodes=len(rows), successes=sum(r['success_episode'] for r in rows),
                success_rate=sum(r['success_episode'] for r in rows)/len(rows),
                boundary_failure_rate=sum(r['boundary_failure'] for r in rows)/len(rows),
                termination_rate=sum(r['completion']['terminated'] for r in rows)/len(rows),
                **{f'mean_{key}': math.fsum(r[key] for r in rows)/len(rows) for key in
                   ('duration', 'reward', 'hold_final', 'hold_longest', 'control_effort')},
                lift_count=len(lift), mean_first_hold_start=math.fsum(lift)/len(lift) if lift else None,
                maxima={key: max(r[key] for r in rows) for key in
                        ('max_abs_position', 'max_abs_velocity', 'max_abs_omega', 'max_abs_applied_acceleration')})


def selection_score(evaluation):
    """PLAN physical metrics; a tie selects the later checkpoint, never reward."""
    row = evaluation['validation']
    return (row['success_rate'], row['mean_hold_final'], row['mean_hold_longest'],
            -(row['mean_first_hold_start'] if row['mean_first_hold_start'] is not None else 11.),
            evaluation['trained_transitions'])


class PilotTimeLimit(Exception):
    pass


class TimeBudget:
    def __init__(self, seconds, clock=time.perf_counter):
        self.clock, self.seconds = clock, seconds
        self.start = clock()

    def check(self):
        if self.elapsed >= self.seconds:
            raise PilotTimeLimit(f'{self.seconds:g}-second collection/update/evaluation budget reached')

    @property
    def elapsed(self):
        return self.clock()-self.start


def evaluate(model, cases, label, output, *, git=None, classical=False, budget=None,
             save_details=True):
    started = time.perf_counter()
    rows, complete = [], False
    env = make_pilot_env()
    nominal = NominalTrajectory.load(ROOT/'tests/fixtures/swing_up_nominal.npz') if classical else None
    steps = 0 if model is None else model.num_timesteps
    # Local Env RNG plus restored Python/NumPy/Torch streams make evaluation
    # independent of exploration/replay. The training Env is never reset here.
    try:
        with preserve_global_rng():
            for case in cases:
                if budget:
                    budget.check()
                if classical:
                    policy = ClassicalPolicy(env.config, nominal)
                else:
                    def policy(obs, info):
                        if budget:
                            budget.check()
                        action, _ = model.predict(obs, deterministic=True)
                        if not np.isfinite(action).all():
                            raise FloatingPointError('non-finite deterministic SAC action')
                        return action
                path = output/'logs'/f'{label}_{case["id"]}.json' if save_details and case['id'] in DETAIL_CASES else None
                log = record_episode(env, policy, seed=case['seed'],
                                     options={'initial_state': case['initial_state']}, log_path=path,
                                     metadata={'algorithm': 'classical' if classical else 'SAC',
                                               'case': case['id'], 'evaluation_label': label, 'git': git,
                                               'trained_transitions': steps,
                                               'controller_label': 'Классический подъём и LQR' if classical else f'SAC · {steps} переходов обучения'})
                rows.append(dict(id=case['id'], subset=case['subset'], **episode_metrics(log)))
            complete = True
    finally:
        env.close()
        report = dict(label=label, trained_transitions=steps, complete=complete, episodes=rows,
                      seconds=time.perf_counter()-started,
                      validation=aggregate([r for r in rows if r['subset'] == 'validation']),
                      named=aggregate([r for r in rows if r['subset'] == 'named']))
        write_json(output/'evaluations'/f'{label}.json', report)
    return report


def new_output(output=None, prefix='sac_pilot'):
    output = Path(output) if output else ROOT/'results'/datetime.now(timezone.utc).strftime(prefix+'_%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True, exist_ok=False)
    for name in ('checkpoints', 'evaluations', 'logs', 'verification'):
        (output/name).mkdir()
    return output


def fixed_observations(cases):
    env = make_pilot_env()
    try:
        return np.stack([observation_from_state(State(**r['initial_state']), env.config) for r in cases])
    finally:
        env.close()


def save_checkpoint(model, output, label, cases):
    started = time.perf_counter()
    base = output/'checkpoints'/label
    if base.with_suffix('.zip').exists():
        raise FileExistsError(base)
    model.save(base)
    model.save_replay_buffer(str(base)+'_replay.pkl')
    observations = fixed_observations(cases)
    actions, _ = model.predict(observations, deterministic=True)
    write_json(Path(str(base)+'_probes.json'), dict(observations=observations.tolist(),
               deterministic_actions=actions.tolist(), transitions=model.num_timesteps,
               updates=model._n_updates, replay_size=model.replay_buffer.size()))
    return time.perf_counter()-started


class PilotCallback(BaseCallback):
    """Stop between complete transitions; never drop callback's current sample."""
    def __init__(self, budget, output):
        super().__init__()
        self.budget, self.output = budget, output
        self.warmup_end = None
        self.episodes, self.action_blocks = [], []
        self.block = []
        self.max_height, self.top_samples = 0., 0
        self.max_abs_omega, self.max_abs_angle = 0., 0.

    def _on_rollout_start(self):
        self.budget.check()

    def _on_step(self):
        info = self.locals['infos'][0]
        state = info['state']
        height = (1-math.cos(state['pole_angle']))/2
        self.max_height = max(self.max_height, height)
        self.top_samples += height >= .9
        self.max_abs_omega = max(self.max_abs_omega, abs(state['pole_angular_velocity']))
        self.max_abs_angle = max(self.max_abs_angle, abs(state['pole_angle']))
        self.block.append((info['action_requested'], float(self.locals['rewards'][0])))
        if self.locals['dones'][0]:
            self.episodes.append(dict(transitions=self.num_timesteps,
                                      time=info['time'], stop_reasons=info['stop_reasons'],
                                      hold=info['hold_metrics'], monitor=info.get('episode')))
        # False here would prevent SB3._store_transition from running.
        return True

    def _on_rollout_end(self):
        if self.num_timesteps == 2000:
            self.warmup_end = time.perf_counter()
        if self.num_timesteps % 1000 == 0:
            values = np.array(self.block)
            row = dict(transitions=self.num_timesteps, action_mean=float(values[:, 0].mean()),
                       action_std=float(values[:, 0].std()), action_min=float(values[:, 0].min()),
                       action_max=float(values[:, 0].max()), task_reward_mean=float(values[:, 1].mean()))
            self.action_blocks.append(row)
            self.block.clear()
            print(json.dumps(dict(progress=self.num_timesteps, elapsed=self.budget.elapsed,
                                  updates=self.model._n_updates, **row), ensure_ascii=False), flush=True)
        self.budget.check()


def run_pilot(output=None, max_transitions=20000, max_seconds=1800.):
    if isinstance(max_transitions, bool) or not 1 <= max_transitions <= 20000 or not 0 < max_seconds <= 1800:
        raise ValueError('pilot caps are 20000 transitions and 1800 seconds')
    configure_torch()
    output = new_output(output)
    git = capture_source(output)
    cases = validation_cases()
    write_json(output/'validation_states.json', dict(distribution=make_pilot_config()['reset_distribution'], cases=cases))
    config = dict(environment=make_pilot_config(), sac=sac_settings(),
                  budget={'transitions': max_transitions, 'seconds': max_seconds,
                          'starts': 'immediately before model.learn; includes warmup, updates and intermediate evaluation/save'},
                  checkpoint_rule='validation success_rate, mean hold_final, mean hold_longest, earliest mean lift (missing=11s), then later checkpoint; no reward ranking',
                  thread_environment={key: os.environ[key] for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                  torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                  torch_version=torch.__version__, sb3_version=sb3.__version__, python=sys.version, executable=sys.executable,
                  torch_cuda=torch.version.cuda, git=git)
    env = make_pilot_env()
    env.reset(seed=0)  # Read back the real Drake integrator before SAC setup.
    assert_pilot_contract(env)
    model = ObservedSAC(env=env, **constructor_settings())
    model.set_logger(configure(folder=None, format_strings=[]))
    config['resolved'] = dict(target_entropy=model.target_entropy,
                              actor_optimizer={k:v for k,v in model.actor.optimizer.defaults.items()},
                              critic_optimizer={k:v for k,v in model.critic.optimizer.defaults.items()},
                              features_extractor=type(model.actor.features_extractor).__name__,
                              replay_buffer=type(model.replay_buffer).__name__, n_envs=model.n_envs)
    write_json(output/'config.json', config)
    (output/'policy_architecture.txt').write_text(str(model.policy)+'\n'+torch.__config__.parallel_info())
    (output/'verification/installed_sac_train.py').write_text(inspect.getsource(sb3.SAC.train))
    evaluations = {}
    save_seconds, intermediate_eval_seconds, intermediate_save_seconds = 0., 0., 0.
    evaluations['classical'] = evaluate(None, cases, 'classical', output, git=git, classical=True)
    evaluations['untrained'] = evaluate(model, cases, 'untrained', output, git=git)
    save_seconds += save_checkpoint(model, output, 'untrained', cases)
    budget = TimeBudget(max_seconds)
    callback = PilotCallback(budget, output)
    def after_update(current):
        nonlocal intermediate_eval_seconds, intermediate_save_seconds
        budget.check()
        if current.num_timesteps == 10000:
            saved_rng = copy.deepcopy(env.np_random.bit_generator.state)
            started = time.perf_counter()
            try:
                evaluations['step_10000'] = evaluate(current, cases, 'step_10000', output, git=git, budget=budget)
            finally:
                intermediate_eval_seconds += time.perf_counter()-started
            assert env.np_random.bit_generator.state == saved_rng
            intermediate_save_seconds += save_checkpoint(current, output, 'step_10000', cases)
            budget.check()
    model.after_update = after_update
    reason = 'transition_limit'
    try:
        model.learn(total_timesteps=max_transitions, callback=callback, log_interval=None, progress_bar=False)
    except PilotTimeLimit:
        reason = 'wall_time_limit'
    finally:
        collection_seconds = budget.elapsed
        model.after_update = None
    if model.replay_buffer.size() != model.num_timesteps:
        raise AssertionError('collected transition missing from standard replay buffer')
    if not finite_parameters(model):
        raise FloatingPointError('non-finite final parameters')
    save_seconds += save_checkpoint(model, output, 'last', cases)
    evaluations['last'] = evaluate(model, cases, 'last', output, git=git)
    candidates = {k:v for k,v in evaluations.items() if k != 'classical' and v['complete']}
    best = max(candidates, key=lambda k: selection_score(candidates[k]))
    for suffix in ('.zip', '_replay.pkl', '_probes.json'):
        shutil.copyfile(output/'checkpoints'/f'{best}{suffix}', output/'checkpoints'/f'best{suffix}')
    warmup_seconds = (callback.warmup_end-budget.start) if callback.warmup_end is not None else collection_seconds
    post_wall = max(0., collection_seconds-warmup_seconds)
    post_compute = max(0., post_wall-intermediate_eval_seconds-intermediate_save_seconds)
    report = dict(training_started=True, completion_reason=reason, transitions=model.num_timesteps, gradient_updates=model._n_updates,
                  replay_size=model.replay_buffer.size(), best_checkpoint=best,
                  evaluations={key:{k:v for k,v in row.items() if k != 'episodes'} for key,row in evaluations.items()},
                  timing=dict(collection_window_seconds=collection_seconds, warmup_seconds=warmup_seconds,
                              warmup_transitions=min(model.num_timesteps, 2000),
                              warmup_sps=min(model.num_timesteps, 2000)/warmup_seconds,
                              post_warmup_wall_seconds=post_wall, post_warmup_compute_seconds=post_compute,
                              post_warmup_sps=(model.num_timesteps-2000)/post_compute if post_compute else None,
                              gradient_update_seconds=model.update_seconds,
                              intermediate_evaluation_seconds=intermediate_eval_seconds,
                              all_evaluation_seconds=sum(v['seconds'] for v in evaluations.values()),
                              save_seconds=save_seconds+intermediate_save_seconds),
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  training_diagnostics=dict(max_height=callback.max_height, samples_height_at_least_point9=callback.top_samples,
                                            max_abs_omega=callback.max_abs_omega, max_abs_angle=callback.max_abs_angle),
                  validation_seed_range=[1000, 1019], final_test_set_used=False,
                  log_scope='detailed logs only for three named cases and validation_1000; no training trajectory JSON')
    write_json(output/'training_episodes.json', callback.episodes)
    write_json(output/'action_blocks.json', callback.action_blocks)
    with (output/'training_updates.jsonl').open('x') as stream:
        for row in model.diagnostics:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    write_json(output/'summary.json', report)
    model.get_env().close()
    print(json.dumps(dict(output=str(output), transitions=report['transitions'], reason=reason,
                          best=best, validation=report['evaluations']['last']['validation']), ensure_ascii=False), flush=True)
    return output


def smoke(output=None):
    configure_torch()
    output = new_output(output, 'sac_smoke')
    env = make_pilot_env()
    env.reset(seed=4242)
    assert_pilot_contract(env)
    args = constructor_settings()
    args.update(learning_starts=8, batch_size=16, buffer_size=128, seed=4242)
    model = ObservedSAC(env=env, **args)
    before = {key:v.detach().clone() for key,v in model.actor.state_dict().items()}
    started = time.perf_counter()
    model.learn(64, log_interval=None)
    report = dict(role='separate smoke, never continued as the pilot', transitions=model.num_timesteps,
                  updates=model._n_updates, seconds=time.perf_counter()-started,
                  changed_parameters=any(not torch.equal(before[k],v) for k,v in model.actor.state_dict().items()),
                  finite_parameters=finite_parameters(model), diagnostics=model.diagnostics)
    assert report['updates'] == 56 and report['changed_parameters'] and report['finite_parameters']
    write_json(output/'smoke.json', report)
    model.get_env().close()
    return output


def assert_metrics_close(actual, expected, path='metrics'):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(path+' keys differ')
        for key in expected:
            assert_metrics_close(actual[key], expected[key], path+'.'+key)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(path+' lengths differ')
        for a,b in zip(actual, expected):
            assert_metrics_close(a,b,path)
    elif isinstance(expected, (int,float)) and not isinstance(expected, bool):
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError(f'{path}: {actual} != {expected}')
    elif actual != expected:
        raise AssertionError(path+' differs')


def verify_saved(run_dir, output=None):
    """Run in a fresh process; one extra update on a loaded copy, no collection."""
    configure_torch()
    run_dir, output = Path(run_dir), new_output(output, 'sac_load_check')
    summary = json.loads((run_dir/'summary.json').read_text())
    cases = json.loads((run_dir/'validation_states.json').read_text())['cases']
    checks = {}
    for label in ('best', 'last'):
        source = summary['best_checkpoint'] if label == 'best' else 'last'
        base = run_dir/'checkpoints'/label
        model = ObservedSAC.load(base.with_suffix('.zip'), device='cpu')
        probes = json.loads(Path(str(base)+'_probes.json').read_text())
        actions,_ = model.predict(np.array(probes['observations'], dtype=np.float32), deterministic=True)
        np.testing.assert_array_equal(actions, np.array(probes['deterministic_actions'],dtype=np.float32))
        actual = evaluate(model, cases, label, output, save_details=False)
        expected = json.loads((run_dir/'evaluations'/f'{source}.json').read_text())
        assert_metrics_close(actual['episodes'], expected['episodes'])
        model.load_replay_buffer(str(base)+'_replay.pkl')
        assert model.replay_buffer.size() == probes['replay_size']
        checks[label] = dict(actions_exact=True, episode_metrics_match=True,
                             replay_size=model.replay_buffer.size(), evaluated_episodes=len(cases))
        if label == 'last':
            before = {k:v.detach().clone() for k,v in model.actor.state_dict().items()}
            updates = model._n_updates
            model.set_logger(configure(folder=None, format_strings=[]))
            model.train(gradient_steps=1, batch_size=256)
            assert model._n_updates == updates+1 and finite_parameters(model)
            assert any(not torch.equal(before[k],v) for k,v in model.actor.state_dict().items())
            model.save(output/'checkpoints/resumed_copy.zip')
            checks[label]['one_update_on_copy'] = True
    write_json(output/'save_load_check.json', dict(checks=checks, original_run=str(run_dir.resolve()),
               process_id=os.getpid(), executable=sys.executable,
               bitwise_continuation_claimed=False, extra_collection_steps=0, extra_gradient_updates=1))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'smoke', 'verify'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--max-transitions', type=int, default=20000)
    parser.add_argument('--max-seconds', type=float, default=1800.)
    args = parser.parse_args()
    if args.mode == 'run':
        result = run_pilot(args.output_dir, args.max_transitions, args.max_seconds)
    elif args.mode == 'smoke':
        result = smoke(args.output_dir)
    else:
        if args.run_dir is None:
            parser.error('verify requires --run-dir')
        result = verify_saved(args.run_dir, args.output_dir)
    print(result.resolve(), flush=True)


if __name__ == '__main__':
    main()
