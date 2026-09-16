"""Versioned S2 environment and request mapping. No learner or second filter."""
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from cartpole.rl.pilot_config import make_pilot_config, make_pilot_env
from cartpole.safety import SafetyLimits, VERSION

CONTRACT_ID = 'cartpole-filtered-request-v1'
RESET_LOW = [-.02, -.02, -.05, -.05]
RESET_HIGH = [.02, .02, .05, .05]
ROOT = Path(__file__).resolve().parents[2]
# Source identity is intentionally separate from the mathematical protocol hash.
SOURCES = ('cartpole/rl/training_contract.py', 'cartpole/rl/env.py',
           'cartpole/rl/pilot_config.py', 'cartpole/rl/reward_pilot.py',
           'cartpole/safety.py', 'cartpole/simulator/constrained.py',
           'cartpole/simulator/pydrake/simulator.py', 'cartpole/simulator/pydrake/system.py',
           'cartpole/experiment_config.py', 'cartpole/common/interface.py',
           'cartpole/experiments/lqr_hold.py', 'cartpole/common/metrics.py',
           'cartpole/common/episode_log.py', 'cartpole/experiments/gym_diagnostics.py',
           'cartpole/experiments/safety_metrics.py', 'cartpole/rl/training_diagnostics.py',
           'cartpole/experiments/cloud_learning.py', 'cartpole/experiments/cloud_runner.py',
           'cartpole/experiments/cloud_checkpoint.py', 'cartpole/experiments/cloud_evaluation.py',
           'cartpole/experiments/cloud_config.py', 'cartpole/experiments/cloud_checks.py',
           'cartpole/experiments/filtered_sac.py', 'cartpole/rl/methods.py',
           'cartpole/experiments/method_config.py')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def specification(contract_id=CONTRACT_ID):
    if contract_id != CONTRACT_ID:
        from cartpole.rl.methods import parse_contract, GRID
        algorithm, mode = parse_contract(contract_id)
        spec = specification()
        spec.update(id=contract_id, environment_api='CourseworkCartPole-request-v2', algorithm=algorithm)
        spec['filter']['enabled'] = mode == 'on'
        if mode == 'off':
            spec['reset']['admission'] = 'working'
        if algorithm == 'DQN':
            spec['action'] = dict(space='Discrete(9)', grid=GRID.tolist(), replay='integer request index',
                mapping='u_requested=grid[index]; float64 normalized adapter before common physical filter',
                evaluation='deterministic argmax index; identical adapter', filtered_dtype='binary64; no requantization')
        if algorithm == 'DDPG':
            spec['action'].update(exploration='independent PCG64 Gaussian sigma=0.1 in normalized units; then clip',
                                  replay='post-noise post-clip requested buffer_action', entropy='none')
        return spec
    old = make_pilot_config()
    return dict(id=CONTRACT_ID, environment_api='CourseworkCartPole-filtered-v1',
                filter=dict(enabled=True, version=VERSION, limits=asdict(SafetyLimits())),
                physical_config=old['physical_config'], reward_version='R1-v0', reward=old['reward'],
                gamma=old['gamma'], timestep=old['timestep'], horizon=old['horizon'],
                integrator=old['integrator'], observation=old['observation'], observation_dtype='float32',
                action=dict(space='Box(-1,1,(1,),float32)', replay='requested buffer_action',
                            mapping='SB3 scale/unscale; u_requested=4*float(received_action[0])',
                            evaluation='predict deterministic -> scale -> unscale',
                            filtered_dtype='binary64; direct to Drake', entropy='requests'),
                reset=dict(distribution='lower-box-v1', low=RESET_LOW[:], high=RESET_HIGH[:],
                           fields=['cart_position', 'cart_velocity', 'pole_angle', 'pole_angular_velocity'],
                           admission='filter_inner', attempts=1, rejection_sampling=False),
                seed_scheme=dict(entropy=['training_seed', 20260915, 2], generator='PCG64',
                                 spawn_order=['env_reset', 'action_space', 'ddpg_noise_reserved']),
                replay=dict(n_steps=1, n_envs=1, optimize_memory_usage=False,
                            handle_timeout_termination=True),
                native_errors=dict(physical=[2, 3], fatal=[1, 4, 5, 6], unknown='fatal'),
                horizon_semantics='external truncation; bootstrap terminal_observation; failure wins',
                refusal='dt=0; applied=None; reward=-5; terminated; store once; no bootstrap')


def identity(contract_id=CONTRACT_ID):
    spec = specification(contract_id)
    return dict(id=contract_id, sha256=canonical_hash(spec), specification=spec,
                sources={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES})


def validate_identity(saved):
    if not isinstance(saved, dict) or saved != identity(saved.get('id')):
        raise ValueError('training contract/source identity mismatch')


def make_training_env(contract_id=CONTRACT_ID):
    from cartpole.rl.env import CartPoleEnv
    spec = specification(contract_id)
    env = CartPoleEnv(reward_version=spec['reward_version'], horizon=spec['horizon'],
                      safety_filter=spec['filter']['enabled'], reset_admission=spec['reset']['admission'])
    env.training_contract = identity(contract_id)
    if spec.get('algorithm') == 'DQN':
        from cartpole.rl.methods import DiscreteRequests
        env = DiscreteRequests(env)
    return env


def environment_factory(config):
    """Old configs remain off; an explicit unknown contract never falls back."""
    if 'training_contract' not in config:
        if config.get('id', '').startswith('sac_filtered'):
            raise ValueError('filtered config requires training_contract')
        return make_pilot_env
    validate_identity(config['training_contract'])
    return lambda: make_training_env(config['training_contract']['id'])


def assert_environment(env, saved):
    validate_identity(saved)
    base = env.unwrapped
    if getattr(base, 'training_contract', None) != saved:
        raise ValueError('environment training contract mismatch (off checkpoints cannot resume filtered runs)')
    spec = saved['specification']
    meta = base.environment_metadata()
    if (base.safety_filter != spec['filter']['enabled'] or base.reset_admission != spec['reset']['admission']
            or meta['config'] != spec['physical_config'] or meta['reward'] != spec['reward']
            or meta['control_interval'] != spec['timestep'] or base.reset_mode != 'random'
            or meta['horizon'] != spec['horizon']
            or {'type': 'RungeKutta3Integrator', **meta['integrator']} != spec['integrator']
            or meta.get('safety_filter') != (spec['filter']['limits'] if base.safety_filter else None)):
        raise ValueError('effective environment differs from training contract')
    if (base.action_space.shape != (1,) or base.action_space.dtype != np.float32
            or not np.array_equal(base.action_space.low, [-1.])
            or not np.array_equal(base.action_space.high, [1.])
            or base.observation_space.shape != (5,) or base.observation_space.dtype != np.float32):
        raise ValueError('environment observation/action spaces differ from training contract')
    if spec.get('algorithm') == 'DQN':
        import gymnasium as gym
        from cartpole.rl.methods import DiscreteRequests
        wrapped = env
        while isinstance(wrapped, gym.Wrapper) and not isinstance(wrapped, DiscreteRequests):
            wrapped = wrapped.env
        if not isinstance(wrapped, DiscreteRequests) or not isinstance(env.action_space, gym.spaces.Discrete) or env.action_space.n != 9:
            raise ValueError('DQN requires the versioned Discrete(9) request adapter')
    elif env.action_space != base.action_space:
        raise ValueError('unexpected continuous action wrapper')


def seed_streams(training_seed):
    if isinstance(training_seed, bool) or not isinstance(training_seed, int) or not 0 <= training_seed < 2**32:
        raise ValueError('training seed must be an integer in [0, 2**32)')
    children = np.random.SeedSequence([training_seed, 20260915, 2]).spawn(3)
    return {name: dict(seed=int(child.generate_state(1, dtype=np.uint32)[0]),
                       spawn_key=list(child.spawn_key)) for name, child in zip(
                           specification()['seed_scheme']['spawn_order'], children)}


def initialize_streams(model):
    """After SB3 setup; first collection consumes this seed only once."""
    streams = seed_streams(model.seed)
    base = model.get_env().envs[0].unwrapped
    base.np_random = np.random.default_rng(streams['env_reset']['seed'])
    base._np_random_seed = streams['env_reset']['seed']
    model.get_env().seed(streams['env_reset']['seed'])
    model.action_space.seed(streams['action_space']['seed'])
    model.training_seed_streams = copy.deepcopy(streams)
    return streams


def evaluation_action(model, observation, *, filtered=False):
    action, _ = model.predict(observation, deterministic=True)
    from gymnasium.spaces import Discrete
    if filtered and not isinstance(model.action_space, Discrete):
        action = model.policy.unscale_action(model.policy.scale_action(action))
    if not np.isfinite(action).all():
        raise FloatingPointError('non-finite deterministic evaluation action')
    if isinstance(model.action_space, Discrete) and np.asarray(action).ndim == 0:
        return int(action)
    return action
