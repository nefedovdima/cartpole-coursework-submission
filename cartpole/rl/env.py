"""Five-observation Gymnasium API for the original constrained Drake model.

No rendering, disk writes, autoreset, policy or trajectory history in step.
The separate diagnostic recorder uses the same Transition as other methods.
"""
import copy
from dataclasses import asdict, replace
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from cartpole.common import Error, State
from cartpole.common.episode_log import state_record
from cartpole.common.metrics import UprightThresholds
from cartpole.experiment_config import CONTROL_INTERVAL, make_experiment_config
from cartpole.experiments.lqr_hold import IntegratorSettings
from cartpole.rl.reward import RewardR0
from cartpole.rl.reward_pilot import RewardR1
from cartpole.simulator.constrained import CartPoleEpisode
from cartpole.simulator.pydrake.simulator import _finite_scalar
from cartpole.safety import SafetyLimits, assess_cart_state


PHYSICAL_FIELDS = ('cart_position', 'cart_velocity', 'pole_angle', 'pole_angular_velocity')


def observation_from_state(state, config):
    values = [float(getattr(state, name)) for name in (*PHYSICAL_FIELDS, 'cart_acceleration')]
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError('non-finite simulator state')
    x, v, theta, omega, _ = values
    with np.errstate(over='ignore', invalid='ignore'):
        observation = np.array([x/config.max_position, v/config.max_velocity,
                                math.sin(theta), math.cos(theta),
                                omega/math.sqrt(config.gravity/config.pole_length)], dtype=np.float32)
    if not np.isfinite(observation).all():
        raise FloatingPointError('state observation cannot be represented as finite float32')
    return observation


class CartPoleEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, *, reset_mode='random', horizon=10., render_mode=None,
                 reward_version='R0-v0', safety_filter=False, reset_admission='working'):
        if not isinstance(safety_filter, bool):
            raise TypeError('safety_filter must be bool')
        self.safety_filter = safety_filter
        if reset_admission not in ('working', 'filter_inner'):
            raise ValueError('unknown reset_admission')
        if reset_admission == 'filter_inner' and not safety_filter:
            raise ValueError('filter_inner admission requires the safety filter')
        self.reset_admission = reset_admission
        if reset_mode not in ('random', 'bottom'):
            raise ValueError('reset_mode must be random or bottom')
        horizon = _finite_scalar(horizon, 'horizon')
        if horizon <= 0:
            raise ValueError('horizon must be positive')
        if render_mode is not None:
            raise ValueError('only render_mode=None is supported; replay saved logs separately')
        self.reset_mode, self.horizon, self.render_mode = reset_mode, horizon, render_mode
        self._config = make_experiment_config()
        rewards = {'R0-v0': RewardR0, 'R1-v0': RewardR1}
        if not isinstance(reward_version, str) or reward_version not in rewards:
            raise ValueError('reward_version must be R0-v0 or R1-v0')
        self._reward = rewards[reward_version]()
        self._thresholds = UprightThresholds()
        self.action_space = spaces.Box(-1., 1., shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(np.array([-np.inf, -np.inf, -1., -1., -np.inf], dtype=np.float32),
                                            np.array([np.inf, np.inf, 1., 1., np.inf], dtype=np.float32))
        self._episode = None
        self._done = True
        self._last_transition = None
        self._last_exception = None

    @property
    def config(self):
        return replace(self._config)

    @property
    def last_transition(self):
        tr = self._last_transition
        return None if tr is None else replace(tr, state=replace(tr.state))

    @property
    def last_exception(self):
        return copy.deepcopy(self._last_exception)

    def environment_metadata(self):
        return {'environment': (self.training_contract['specification']['environment_api']
                                if hasattr(self, 'training_contract') else 'CourseworkCartPole-v0'),
                'gymnasium': gym.__version__,
                'config': asdict(self._config), 'control_interval': CONTROL_INTERVAL,
                'horizon': self.horizon, 'default_reset_mode': self.reset_mode,
                'integrator': copy.deepcopy(self._integrator) if self._episode is not None else asdict(IntegratorSettings()),
                'reward': self._reward.metadata(), 'upright_thresholds': asdict(self._thresholds),
                'required_hold_duration': 2.,
                'observation': '[x/X,v/V,sin(theta),cos(theta),omega/sqrt(g/l)] float32',
                'action': 'Box(-1,1,(1,),float32); u_requested=U*received_action; limiter in CartPoleEpisode',
                **({'safety_filter': asdict(self._episode.safety_limits if self._episode is not None else SafetyLimits())}
                   if self.safety_filter else {}),
                **({'reset_admission': self.reset_admission} if self.reset_admission != 'working' else {}),
                **({'training_contract': copy.deepcopy(self.training_contract)}
                   if hasattr(self, 'training_contract') else {})}

    def reset(self, *, seed=None, options=None):
        options = {} if options is None else options
        if not isinstance(options, dict):
            raise TypeError('options must be a dict')
        if set(options)-{'initial_state', 'reset_mode'}:
            raise ValueError('unknown reset options')
        if 'initial_state' in options and 'reset_mode' in options:
            raise ValueError('choose initial_state or reset_mode, not both')
        mode = options.get('reset_mode', self.reset_mode)
        if mode not in ('random', 'bottom'):
            raise ValueError('reset_mode must be random or bottom')
        explicit = options.get('initial_state')
        if 'initial_state' in options:
            if isinstance(explicit, dict):
                if set(explicit) != set(PHYSICAL_FIELDS):
                    raise ValueError('initial_state mapping must contain exactly the four physical fields')
                explicit = State(**explicit)
            if not isinstance(explicit, State):
                raise TypeError('initial_state must be State or a mapping of four physical components')
            explicit = State(**{key: _finite_scalar(getattr(explicit, key), key) for key in PHYSICAL_FIELDS})
            observation_from_state(explicit, self._config)
            mode = 'explicit'
        # Build a replacement episode independently; even a failed reseeded
        # reset must not partially replace the existing episode or RNG state.
        old_rng, old_seed = self._np_random, self._np_random_seed
        old_rng_state = copy.deepcopy(old_rng.bit_generator.state) if old_rng is not None else None
        try:
            super().reset(seed=seed)
            if explicit is not None:
                initial = explicit
            elif mode == 'bottom':
                initial = State.home()
            else:
                if self.reset_admission == 'filter_inner':
                    from cartpole.rl.training_contract import RESET_LOW, RESET_HIGH
                    samples = self.np_random.uniform(RESET_LOW, RESET_HIGH)
                else:
                    samples = self.np_random.uniform([-.02, -.02, -.05, -.05], [.02, .02, .05, .05])
                initial = State(**dict(zip(PHYSICAL_FIELDS, samples)))
            if self.reset_admission == 'filter_inner':
                candidate = {key: _finite_scalar(getattr(initial, key), key) for key in PHYSICAL_FIELDS}
                admission = assess_cart_state(x=candidate['cart_position'], v=candidate['cart_velocity'],
                                              limits=SafetyLimits())
                if not admission.admissible:
                    raise ValueError(f'reset admission rejected: candidate={candidate}; '
                                     f'reasons={admission.reasons}; attempts=1; rejected=1')
            episode = CartPoleEpisode(safety_filter=self.safety_filter)
            state = episode.reset(initial)
            integrator = IntegratorSettings().apply(episode.backend)
            observation = observation_from_state(state, self._config)
            actual_seed = self.np_random_seed
        except Exception:
            self._np_random, self._np_random_seed = old_rng, old_seed
            if old_rng is not None:
                old_rng.bit_generator.state = old_rng_state
            raise
        self._episode, self._integrator = episode, integrator
        self._done, self._time, self._step_count = False, 0., 0
        self._last_transition = self._last_exception = None
        self._state = replace(state)
        self._hold_start = self._first_hold_start = self._hold_confirmation = None
        self._hold_current = self._hold_longest = 0.
        self._update_hold(state, 0.)
        initial_record = state_record(state, 0., 0)
        if hasattr(self, 'training_contract'):
            from cartpole.rl.training_contract import canonical_hash
            history = copy.deepcopy(getattr(self, 'training_resets', dict(count=0, chain=None, first=[], last=[])))
            record = dict(index=history['count'], state=initial_record, seed=actual_seed, mode=mode)
            history['count'] += 1
            history['chain'] = canonical_hash(dict(previous=history['chain'], reset=record))
            if len(history['first']) < 16:
                history['first'].append(record)
            history['last'] = (history['last']+[record])[-16:]
            self.training_resets = history
        return observation, {'state': initial_record, 'initial_state': dict(initial_record),
                             'time': 0., 'reset_mode': mode, 'seed': actual_seed,
                             'integrator': copy.deepcopy(integrator),
                             'simulator_error': int(state.error), 'completion_reason': 'running',
                             **({'reset_admission': dict(mode='filter_inner', attempts=1, rejected=0,
                                                       distribution='lower-box-v1' if mode == 'random' else mode)}
                                if self.reset_admission == 'filter_inner' else {}),
                             'hold_metrics': self._hold_metrics(False)}

    def _update_hold(self, state, stamp):
        if self._thresholds.contains(state):
            if self._hold_start is None:
                self._hold_start = stamp
            self._hold_current = stamp-self._hold_start
            self._hold_longest = max(self._hold_longest, self._hold_current)
            if self._first_hold_start is None and self._hold_current >= 2.-1e-12:
                self._first_hold_start, self._hold_confirmation = self._hold_start, stamp
        else:
            self._hold_start, self._hold_current = None, 0.

    def _hold_metrics(self, full_horizon):
        return {'in_upright_region': self._hold_start is not None,
                'first_hold_start': self._first_hold_start, 'first_hold_confirmation': self._hold_confirmation,
                'hold_final': self._hold_current, 'hold_longest': self._hold_longest,
                'success_episode': bool(full_horizon and self._hold_current >= 2.-1e-12)}

    def _capture_exception(self, exc):
        self._done, self._last_transition = True, None
        try:
            stamp = self._episode.backend.timestamp()
            observed = state_record(self._episode.backend.get_state(), stamp, None)
            observed = {k: v if not isinstance(v, float) or math.isfinite(v) else str(v) for k, v in observed.items()}
        except Exception as diagnostic_error:
            observed = {'unavailable': str(diagnostic_error)}
        self._last_exception = {'type': type(exc).__name__, 'message': str(exc),
                                'observed_after_exception': observed}

    def step(self, action):
        if self._done or self._episode is None:
            raise RuntimeError('reset is required before step, including after termination or truncation')
        if not isinstance(action, np.ndarray):
            raise TypeError('action must be a real NumPy array of shape (1,)')
        if action.shape != (1,):
            raise ValueError('action must have shape (1,)')
        if action.dtype.kind != 'f':
            raise TypeError('action array must have a real floating dtype')
        requested_action = _finite_scalar(action[0], 'action')
        requested_u = requested_action*self._config.max_acceleration
        if not math.isfinite(requested_u):
            raise ValueError('scaled action must be finite')
        if hasattr(self, 'training_contract') and abs(requested_action) > 1.:
            raise ValueError('training action outside declared Box')
        deadline = min((self._step_count+1)*CONTROL_INTERVAL, self.horizon)
        try:
            # Catch corrupt state before passing it to Drake's integrator.
            before = self._episode.backend.get_state()
            observation_from_state(before, self._config)
            tr = self._episode.step(requested_u, deadline-self._time)
            observation = observation_from_state(tr.state, self._config)
            if (hasattr(self, 'training_contract') and tr.simulator_error not in
                    (Error.NO_ERROR, Error.X_OVERFLOW, Error.V_OVERFLOW)):
                raise RuntimeError(f'unexpected native error in filtered simulation: {int(tr.simulator_error)}')
            if not all(math.isfinite(v) for v in (tr.timestamp, tr.dt_actual)):
                raise FloatingPointError('non-finite transition time')
            if not tr.terminated and not math.isclose(tr.timestamp, deadline, rel_tol=0., abs_tol=1e-12):
                raise RuntimeError('backend did not reach the requested deadline')
            terminated = bool(tr.terminated)
            truncated = bool(not terminated and deadline == self.horizon)
            reward, terms, scale = self._reward.evaluate(tr.state, tr.applied_acceleration,
                                                        tr.dt_actual, terminated, self._config)
            factors = (self._reward.factors(tr.state, tr.applied_acceleration)
                       if isinstance(self._reward, RewardR1) and tr.dt_actual > 0 else None)
        except Exception as exc:
            self._capture_exception(exc)
            raise
        self._step_count += 1
        self._state, self._time, self._last_transition = replace(tr.state), tr.timestamp, tr
        self._done = terminated or truncated
        self._update_hold(tr.state, tr.timestamp)
        info = {'state': state_record(tr.state, tr.timestamp, self._step_count), 'time': tr.timestamp,
                'dt_requested': tr.dt_requested, 'dt_actual': tr.dt_actual,
                'action_requested': requested_action, 'action_commanded': (tr.u_commanded/self._config.max_acceleration
                                                                         if tr.u_commanded is not None else None),
                'u_requested': tr.u_requested, 'u_commanded': tr.u_commanded,
                'applied_acceleration': tr.applied_acceleration, 'action_clipped': tr.action_clipped,
                'simulator_error': int(tr.simulator_error), 'stop_reasons': [r.value for r in tr.stop_reasons],
                'boundary_events': [dict(reason=e.reason.value, side=e.side, time=e.time) for e in tr.boundary_events],
                'completion_reason': 'terminal' if terminated else 'horizon' if truncated else 'running',
                'reward_version': self._reward.version, 'reward_terms': terms, 'reward_scale': scale,
                'hold_metrics': self._hold_metrics(truncated)}
        if isinstance(self._reward, RewardR1):
            # Multipliers are diagnostics, not additive reward terms. A zero
            # transition has no applied control and no ordinary reward.
            info['reward_factors'] = factors
        if tr.safety_decision is not None:
            info['safety_filter'] = asdict(tr.safety_decision)
        if hasattr(self, 'training_contract'):
            from cartpole.experiments.safety_metrics import interval_evidence
            info['training_before_state'] = state_record(before, tr.timestamp-tr.dt_actual, self._step_count-1)
            info['cart_interval'] = interval_evidence(info['training_before_state'], info['state'], info)
        return observation, reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        self._episode, self._last_transition, self._done = None, None, True
