"""Opt-in Gymnasium logging and bounded diagnostics, with no learner or Solve."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import statistics
import time
import warnings

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np

from cartpole.common import State
from cartpole.common.episode_log import append_transition, finish_log, new_log, record_state, save_log
from cartpole.control.swing_up import NominalTrajectory, TrackingThenBalance
from cartpole.experiments.lqr_hold import ROOT, capture_source, dependency_versions, run_episode
from cartpole.rl import CartPoleEnv
from cartpole.rl.env import PHYSICAL_FIELDS
from cartpole.rl.reward import RewardR0


def write_json(path, payload):
    with Path(path).open('x') as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def record_episode(env, policy, *, seed=None, options=None, metadata=None, log_path=None, on_failure=None):
    """Separate recorder: step itself stores only its latest transition.

    A policy accepts (observation, info). Optional phase/raw_acceleration
    attributes add classical diagnostics without influencing the environment.
    Exceptions are re-raised; partial native progress is separate evidence.
    """
    if log_path is not None and Path(log_path).exists():
        raise FileExistsError(log_path)
    observation, info = env.reset(seed=seed, options=options)
    metadata = {**(metadata or {}), **env.environment_metadata(),
                'versions': {**dependency_versions(), 'gymnasium': gym.__version__},
                'initial_state': asdict(record_state(info['state'])), 'seed': info['seed'],
                'reset_mode': info['reset_mode']}
    log = new_log(metadata, record_state(info['state']))
    log['states'][0]['observation'] = observation.tolist()
    try:
        while True:
            action = policy(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            append_transition(log, env.last_transition)
            log['states'][-1]['observation'] = observation.tolist()
            tr = log['transitions'][-1]
            tr.update(action_requested=info['action_requested'], action_commanded=info['action_commanded'],
                      reward=reward, reward_version=info['reward_version'],
                      reward_scale=info['reward_scale'], reward_terms=dict(info['reward_terms']),
                      gym_terminated=terminated, gym_truncated=truncated)
            if 'reward_factors' in info:
                tr['reward_factors'] = info['reward_factors']
            if 'request_index' in info:
                tr['request_index'] = info['request_index']
            if hasattr(policy, 'diagnostic_phase'):
                tr['diagnostic_phase'] = policy.diagnostic_phase
            if hasattr(policy, 'phase'):
                tr['control_phase'] = policy.phase
            if hasattr(policy, 'raw_acceleration'):
                tr['controller_u_requested'] = policy.raw_acceleration
            if terminated or truncated:
                finish_log(log, 'terminal' if terminated else 'horizon')
                log['gym_final_hold_metrics'] = info['hold_metrics']
                break
    except Exception as exc:
        finish_log(log, 'exception')
        log['completion']['exception'] = {'type': type(exc).__name__, 'message': str(exc)}
        log['completion']['observed_after_exception'] = (env.last_exception or {}).get('observed_after_exception')
        if on_failure is not None:
            try:
                on_failure(log)
            except Exception as save_error:
                exc.add_note(f'Could not save compressed failure evidence: {save_error}')
        if log_path is not None:
            try:
                save_log(log_path, log)
            except Exception as save_error:
                exc.add_note(f'Could not save failed Gymnasium rollout: {save_error}')
        raise
    log['gym_return'] = math.fsum(tr['reward'] for tr in log['transitions'])
    if log_path is not None:
        save_log(log_path, log)
    return log


class ClassicalPolicy:
    def __init__(self, config, nominal):
        self.controller = TrackingThenBalance(config, nominal)
        self.scale = config.max_acceleration
        self.phase = 'swing_up'

    def __call__(self, observation, info):
        self.raw_acceleration = self.controller(info['time'], record_state(info['state']))
        self.phase = self.controller.phase
        # Preserve this original command separately in the diagnostic log.
        # The environment sees the actual float32 command, including >1.
        return np.array([self.raw_acceleration/self.scale], dtype=np.float32)


def result_summary(log):
    return {'metrics': log['metrics'], 'evaluation': log['evaluation'],
            'completion': log['completion'], 'gym_return': log['gym_return']}


def benchmark(steps=5000, repetitions=3):
    """Measure only Env.step and explicit reset, with pre-generated actions."""
    records = []
    for repeat in range(repetitions):
        env = CartPoleEnv()
        actions = np.random.default_rng(700+repeat).uniform(-1., 1., size=(steps, 1)).astype(np.float32)
        env.reset(seed=900+repeat)
        resets = 0
        started = time.perf_counter()
        for action in actions:
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset()
                resets += 1
        elapsed = time.perf_counter()-started
        records.append({'transitions': steps, 'explicit_resets': resets, 'seconds': elapsed,
                        'transitions_per_second': steps/elapsed})
        env.close()
    return {'runs': records, 'median_transitions_per_second': statistics.median(r['transitions_per_second'] for r in records),
            'render_mode': None, 'detailed_logging': False, 'includes_explicit_reset_time': True,
            'action_generation_in_timing': False, 'training_measured': False}


def reward_examples(config):
    reward = RewardR0()
    cases = {'bottom_rest': State(), 'top_rest': State(pole_angle=math.pi),
             'top_omega_20': State(pole_angle=math.pi, pole_angular_velocity=20.),
             'top_omega_100': State(pole_angle=math.pi, pole_angular_velocity=100.),
             'near_boundary_rest': State(cart_position=.24)}
    return {name: {'state': asdict(state), 'reward': reward.evaluate(state, 0., .01, False, config)[0],
                   'terms': reward.evaluate(state, 0., .01, False, config)[1]} for name, state in cases.items()}


def main():
    parser = argparse.ArgumentParser(description='Gymnasium checker, random rollouts, fixed classical plan and R0 diagnostics; no training.')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    output = args.output_dir or ROOT/'results'/datetime.now(timezone.utc).strftime('gym_diagnostics_%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True, exist_ok=False)
    git = capture_source(output)
    (output/'RL_ENV_SPEC.md').write_bytes((ROOT/'docs/RL_ENV_SPEC.md').read_bytes())
    report = {'git': git, 'dependencies': {name: version(name) for name in
              ('gymnasium', 'cloudpickle', 'farama-notifications', 'typing_extensions', 'drake', 'numpy')}}
    with warnings.catch_warnings(record=True) as messages:
        checked = gym.make('cartpole.rl:CourseworkCartPole-v0').unwrapped
        check_env(checked)
        checked.close()
    report['check_env'] = {'result': 'OK', 'warnings': [str(w.message) for w in messages]}
    report['random'] = {}
    for index in range(5):
        env = CartPoleEnv(reset_mode='bottom' if index == 0 else 'random')
        action_seed = 1000+index
        env.action_space.seed(action_seed)
        log = record_episode(env, lambda obs, info: env.action_space.sample(), seed=index,
                             metadata={'experiment': 'gymnasium_random', 'case': f'random_{index}', 'git': git,
                                       'action_seed': action_seed}, log_path=output/f'random_{index}.json')
        report['random'][str(index)] = result_summary(log)
        env.close()
    plan_path = ROOT/'tests/fixtures/swing_up_nominal.npz'
    nominal = NominalTrajectory.load(plan_path)
    report['classical'] = {}
    for case, angle in [('bottom', 0.), ('plus_002', .02), ('minus_002', -.02)]:
        env = CartPoleEnv()
        policy = ClassicalPolicy(env.config, nominal)
        initial = State(pole_angle=angle)
        log = record_episode(env, policy, seed=0, options={'initial_state': initial},
                             metadata={'experiment': 'classical_swing_up', 'case': case, 'git': git,
                                       'controller': policy.controller.parameters,
                                       'nominal': {'path': str(plan_path), 'sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest()}},
                             log_path=output/f'classical_{case}.json')
        reference_controller = TrackingThenBalance(env.config, nominal)
        reference = run_episode(reference_controller, initial, timed_controller=True,
                                metadata={'required_hold_duration': 2., 'experiment': 'classical_swing_up',
                                          'case': case, 'git': git, 'interface': 'direct CartPoleEpisode',
                                          'controller': reference_controller.parameters,
                                          'nominal': log['metadata']['nominal']},
                                log_path=output/f'direct_{case}.json')
        common = [(a, b) for a, b in zip(log['states'], reference['states']) if abs(a['time']-b['time']) <= 1e-12]
        comparison = {key: max(abs(a[key]-b[key]) for a, b in common) for key in PHYSICAL_FIELDS}
        report['classical'][case] = {**result_summary(log), 'direct_success': reference['evaluation']['success_episode'],
                                    'max_difference_from_unquantized_direct': comparison, 'matched_timestamps': len(common),
                                    'normalization_max_abs_error': max(abs(tr['u_requested']-tr['controller_u_requested']) for tr in log['transitions'])}
        env.close()
    report['reward_examples'] = reward_examples(CartPoleEnv().config)
    report['early_termination_probe'] = {}
    gamma = math.exp(-.01/5.)
    for name, action in [('wait', 0.), ('exit', 1.)]:
        env = CartPoleEnv()
        log = record_episode(env, lambda obs, info: np.array([action], dtype=np.float32), seed=0,
                             options={'initial_state': State(cart_position=.24)},
                             metadata={'experiment': 'gymnasium_reward_probe', 'case': name, 'git': git},
                             log_path=output/f'reward_{name}.json')
        report['early_termination_probe'][name] = {**result_summary(log),
            'discounted_return': math.fsum(gamma**i*tr['reward'] for i, tr in enumerate(log['transitions']))}
        env.close()
    report['early_termination_probe']['discount_gamma_for_diagnostic_only'] = gamma
    report['benchmark'] = benchmark()
    write_json(output/'summary.json', report)
    print(output.resolve())
    print(json.dumps({key: report[key] for key in ('check_env', 'classical', 'early_termination_probe', 'benchmark')}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
