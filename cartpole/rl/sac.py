"""Fixed CPU pilot settings and observations of the installed SB3 SAC update.

The optimizer/target/update algorithm is SB3's unmodified SAC.train. Temporary
observers read its actual sampled batch, actions and target critics without
drawing extra random samples or changing gradients.
"""
import math
import time

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer

from cartpole.rl.pilot_config import make_pilot_config


def configure_torch():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if torch.version.cuda is not None:
        raise RuntimeError('E6a requires a CPU-only PyTorch build')


def sac_settings():
    """All SAC constructor options, with names for non-JSON Python classes."""
    return dict(policy='MlpPolicy', device='cpu', seed=0, learning_rate=3e-4,
                buffer_size=100000, learning_starts=2000, batch_size=256,
                train_freq=1, gradient_steps=1, tau=.005,
                gamma=make_pilot_config()['gamma'], ent_coef='auto_0.1',
                target_entropy='auto', n_steps=1, action_noise=None,
                use_sde=False, sde_sample_freq=-1, use_sde_at_warmup=False,
                optimize_memory_usage=False, replay_buffer_class='ReplayBuffer',
                replay_buffer_kwargs={'handle_timeout_termination': True},
                target_update_interval=1, stats_window_size=100,
                tensorboard_log=None, verbose=0, _init_setup_model=True,
                policy_kwargs={'net_arch': {'pi': [64, 64], 'qf': [64, 64]},
                               'activation_fn': 'ReLU', 'optimizer_class': 'Adam',
                               'optimizer_kwargs': {'eps': 1e-8},
                               'n_critics': 2, 'share_features_extractor': False,
                               'log_std_init': -3., 'use_expln': False,
                               'clip_mean': 2., 'normalize_images': True})


def constructor_settings(settings=None):
    import copy
    args = copy.deepcopy(sac_settings() if settings is None else settings)
    args['replay_buffer_class'] = ReplayBuffer
    args['policy_kwargs']['activation_fn'] = torch.nn.ReLU
    args['policy_kwargs']['optimizer_class'] = torch.optim.Adam
    return args


def assert_pilot_contract(env):
    actual, expected = env.unwrapped.environment_metadata(), make_pilot_config()
    if env.unwrapped._episode is not None:
        integrator = env.unwrapped._episode.backend.simulator.get_mutable_integrator()
        actual['integrator'] = dict(type=type(integrator).__name__,
                                    target_accuracy=integrator.get_target_accuracy(),
                                    maximum_step_size=integrator.get_maximum_step_size(),
                                    fixed_step_mode=integrator.get_fixed_step_mode())
    for actual_key, expected_key in [('reward', 'reward'), ('config', 'physical_config'),
                                     ('integrator', 'integrator'), ('horizon', 'horizon'),
                                     ('control_interval', 'timestep')]:
        if actual[actual_key] != expected[expected_key]:
            raise ValueError(f'pilot contract mismatch: {actual_key}')
    if env.unwrapped.reset_mode != expected['reset_mode']:
        raise ValueError('pilot must use the random lower reset distribution')
    if env.observation_space.shape != (5,) or env.action_space.shape != (1,):
        raise ValueError('pilot space shape mismatch')
    if not (np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1)):
        raise ValueError('pilot action scale mismatch')


def finite_parameters(model):
    return all(torch.isfinite(p).all().item() for p in model.policy.parameters())


def tensor_stats(value):
    value = value.detach()
    if not torch.isfinite(value).all():
        raise FloatingPointError('non-finite SAC batch/target diagnostic')
    return dict(mean=value.mean().item(), std=value.std(unbiased=False).item(),
                minimum=value.min().item(), maximum=value.max().item())


class ObservedSAC(SAC):
    """SAC with sparse observational telemetry; standard one-step replay."""
    def __init__(self, *args, **kwargs):
        self.diagnostics = []
        self.update_seconds = 0.
        self.after_update = None
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params()+['after_update']

    def train(self, gradient_steps, batch_size=64):
        started = time.perf_counter()
        observe = self._n_updates % 100 == 0
        captured, calls, targets = {}, [], []
        sample = self.replay_buffer.sample
        action_log_prob = self.actor.action_log_prob
        handles = []
        if observe:
            if gradient_steps != 1:
                raise ValueError('pilot telemetry expects one update per transition')
            def observe_sample(*args, **kwargs):
                data = sample(*args, **kwargs)
                captured['batch'] = data
                return data
            def observe_action(*args, **kwargs):
                result = action_log_prob(*args, **kwargs)
                calls.append(tuple(v.detach().clone() for v in result))
                return result
            self.replay_buffer.sample = observe_sample
            self.actor.action_log_prob = observe_action
            handles.append(self.critic_target.register_forward_hook(
                lambda module, args, result: targets.append(tuple(v.detach().clone() for v in result))))
        try:
            super().train(gradient_steps, batch_size)
        finally:
            if observe:
                del self.replay_buffer.sample
                del self.actor.action_log_prob
                for handle in handles:
                    handle.remove()
        self.update_seconds += time.perf_counter()-started
        values = self.logger.name_to_value
        losses = {key: float(values['train/'+key]) for key in
                  ('actor_loss', 'critic_loss', 'ent_coef', 'ent_coef_loss')}
        if not all(math.isfinite(v) for v in losses.values()):
            raise FloatingPointError('non-finite SAC loss or entropy coefficient')
        if observe:
            if len(calls) != 2 or len(targets) != 1:
                raise RuntimeError('installed SAC update differs from the audited telemetry path')
            if not finite_parameters(self):
                raise FloatingPointError('non-finite SAC network parameter')
            batch = captured['batch']
            min_q = torch.cat(targets[0], dim=1).min(dim=1, keepdim=True).values
            entropy_term = -losses['ent_coef']*calls[1][1].reshape(-1, 1)
            discounts = batch.discounts if batch.discounts is not None else self.gamma
            target = batch.rewards+(1-batch.dones)*discounts*(min_q+entropy_term)
            self.diagnostics.append(dict(transitions=self.num_timesteps, updates=self._n_updates,
                                         **losses, action_sample=tensor_stats(calls[0][0]),
                                         log_probability=tensor_stats(calls[0][1]),
                                         task_reward=tensor_stats(batch.rewards),
                                         target_q_without_entropy=tensor_stats(min_q),
                                         entropy_bonus=tensor_stats(entropy_term),
                                         critic_target=tensor_stats(target),
                                         bootstrap_fraction=float((1-batch.dones).mean()),
                                         parameter_check_finite=True))
        if self.after_update is not None:
            self.after_update(self)
