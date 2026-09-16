"""Observational E6b diagnostics. No policy sampling or alternative success rule."""
from dataclasses import asdict
import math

import numpy as np
import torch

from cartpole.common import State
from cartpole.common.metrics import UprightThresholds
from cartpole.experiment_config import make_experiment_config
from cartpole.rl.env import observation_from_state


def diagnostic_states():
    """Fixed before training; admissible SI states, not a new reset distribution."""
    specifications = [
        ('bottom', 0., 0., 0., 0.),
        ('swing_positive', .05, .1, .8, 3.),
        ('swing_negative', -.05, -.1, -.8, -3.),
        ('upright_slow_positive', 0., 0., math.pi-.05, .1),
        ('upright_slow_negative', 0., 0., math.pi+.05, -.1),
        ('fast_rotation', 0., 0., math.pi, 20.),
    ]
    config = make_experiment_config()
    rows = []
    for name, x, v, theta, omega in specifications:
        state = State(cart_position=x, cart_velocity=v, pole_angle=theta,
                      pole_angular_velocity=omega)
        rows.append(dict(id=name, state=asdict(state),
                         observation=observation_from_state(state, config).tolist()))
    return rows


def fixed_normal_samples():
    # A local generator: never np.random or Torch's training stream.
    return np.random.default_rng(60602).standard_normal((4096, 1))


def conditional_actions(model, states, normal_samples):
    """Read actual mu/log_std heads; transform fixed epsilon without sampling actor.

    use_sde=False has a learned log_std linear head, not constant log_std_init.
    This function does not call predict (which switches policy.training).
    """
    observations = torch.as_tensor(np.array([r['observation'] for r in states],
                                           dtype=np.float32), device=model.device)
    with torch.no_grad():
        mean, log_std, extra = model.actor.get_action_dist_params(observations)
        if extra or model.use_sde:
            raise ValueError('conditional probes require the fixed non-gSDE SAC')
        alpha = model.log_ent_coef.exp().item()
        deterministic = torch.tanh(mean).cpu().numpy()
        mean, log_std = mean.cpu().numpy(), log_std.cpu().numpy()
    if not all(np.isfinite(v).all() for v in (mean, log_std, deterministic, alpha)):
        raise FloatingPointError('non-finite conditional policy distribution')
    epsilon = np.asarray(normal_samples, dtype=np.float64)
    if epsilon.ndim != 2 or epsilon.shape[1] != 1 or not np.isfinite(epsilon).all():
        raise ValueError('fixed epsilon must be finite with shape (N, 1)')
    rows = []
    for i, state in enumerate(states):
        sigma = math.exp(float(log_std[i, 0]))
        samples = np.tanh(float(mean[i, 0])+sigma*epsilon[:, 0])
        rows.append(dict(id=state['id'], pre_tanh_mean=float(mean[i, 0]),
                         pre_tanh_log_std=float(log_std[i, 0]), pre_tanh_std=sigma,
                         deterministic_action=float(deterministic[i, 0]),
                         action_mean=float(samples.mean()), action_std=float(samples.std()),
                         action_quantiles=dict(zip(('p01', 'p05', 'p25', 'p50', 'p75', 'p95', 'p99'),
                                                   np.quantile(samples, [.01, .05, .25, .5, .75, .95, .99]).tolist())),
                         saturation_fraction=float(np.mean(np.abs(samples) >= .95))))
    return dict(transitions=model.num_timesteps, updates=model._n_updates,
                alpha=alpha, samples_per_state=len(epsilon), states=rows)


def quantiles(values):
    return dict(zip(('minimum', 'p25', 'p50', 'p75', 'p90', 'p99', 'maximum'),
                    np.quantile(values, [0., .25, .5, .75, .9, .99, 1.]).tolist()))


class ExperienceBlocks:
    """Disjoint endpoint counts; duration/confirmation come from Env hold_metrics.

    Continuous duration is the *whole current episode run observed in a block*,
    including its prefix before that block. Counts are never duplicated across
    blocks. Confirmation is counted once where first observed; completed episode
    outcomes are counted in their completion block. No reset of hold at a block.
    """
    def __init__(self, block_size=5000):
        if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 1:
            raise ValueError('block_size must be a positive integer')
        self.block_size, self.total = block_size, 0
        self.rows, self.episodes = [], []
        self.confirmed_in_episode = False
        self._reset_block()

    def _reset_block(self):
        self.counts = dict(top=0, top_omega_le2=0, top_omega_le05=0, upright=0,
                           physical_terminations=0, timeouts=0, completed_episodes=0,
                           first_confirmations=0, completed_with_confirmed_hold=0,
                           successful_completed_episodes=0, zero_transitions=0,
                           shortened_transitions=0)
        self.rewards, self.ordinary = [], []
        self.seconds, self.longest_current_run = 0., 0.

    def observe(self, info):
        state = State(**{key: info['state'][key] for key in
                         ('cart_position', 'cart_velocity', 'pole_angle', 'pole_angular_velocity')})
        hold = info['hold_metrics']
        inside = UprightThresholds().contains(state)
        if inside != hold['in_upright_region']:
            raise ValueError('Env hold criterion differs from shared thresholds')
        if info['reward_version'] != 'R1-v0':
            raise ValueError('experience diagnostics require R1-v0')
        dt = info['dt_actual']
        alignment = info['reward_terms']['alignment']
        reward = math.fsum(info['reward_terms'].values())
        if not all(math.isfinite(v) for v in (dt, alignment, reward, hold['hold_final'])) or dt < 0:
            raise FloatingPointError('non-finite training diagnostics')
        self.total += 1
        top = (1-math.cos(state.pole_angle))/2 >= .9
        for key, value in [('top', top), ('top_omega_le2', top and abs(state.pole_angular_velocity) <= 2),
                           ('top_omega_le05', top and abs(state.pole_angular_velocity) <= .5),
                           ('upright', inside), ('zero_transitions', dt == 0),
                           ('shortened_transitions', 0 < dt < .01-1e-12)]:
            self.counts[key] += int(value)
        self.seconds += dt
        self.longest_current_run = max(self.longest_current_run, hold['hold_final'])
        self.rewards.append(reward)
        self.ordinary.append(alignment)
        if hold['first_hold_confirmation'] is not None and not self.confirmed_in_episode:
            self.counts['first_confirmations'] += 1
            self.confirmed_in_episode = True
        if info['completion_reason'] != 'running':
            if info['completion_reason'] not in ('terminal', 'horizon'):
                raise ValueError('unexpected training completion reason')
            self.counts['completed_episodes'] += 1
            self.counts['physical_terminations'] += info['completion_reason'] == 'terminal'
            self.counts['timeouts'] += info['completion_reason'] == 'horizon'
            self.counts['completed_with_confirmed_hold'] += hold['first_hold_confirmation'] is not None
            self.counts['successful_completed_episodes'] += hold['success_episode']
            self.episodes.append(dict(transitions=self.total, time=info['time'],
                                      completion_reason=info['completion_reason'],
                                      stop_reasons=info['stop_reasons'], hold=dict(hold)))
            self.confirmed_in_episode = False

    def flush(self, *, partial=False):
        count = len(self.rewards)
        if not count or (not partial and count != self.block_size):
            return None
        row = dict(first_transition=self.total-count+1, last_transition=self.total,
                   transitions=count, complete_block=count == self.block_size,
                   physical_seconds=self.seconds, counts=self.counts.copy(),
                   fractions={key: self.counts[key]/count for key in
                              ('top', 'top_omega_le2', 'top_omega_le05', 'upright', 'physical_terminations')},
                   max_contiguous_hold_observed=self.longest_current_run,
                   ordinary_reward_quantiles=quantiles(self.ordinary),
                   task_reward_quantiles=quantiles(self.rewards),
                   reward_ge_point5=int(np.sum(np.asarray(self.rewards) >= .5)),
                   reward_ge_point8=int(np.sum(np.asarray(self.rewards) >= .8)))
        self.rows.append(row)
        self._reset_block()
        return row
