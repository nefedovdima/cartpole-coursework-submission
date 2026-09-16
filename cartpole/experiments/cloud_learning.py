"""Thin observations around the installed SAC/TQC, never replacement updates."""
from contextlib import contextmanager
import copy
import hashlib
import importlib
import inspect
import math
import random
import time
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3 import SAC, DDPG, DQN
from gymnasium.spaces import Discrete
from stable_baselines3.common.buffers import ReplayBuffer
from sb3_contrib import TQC

from cartpole.experiments.cloud_config import constructor


def configure_device(device):
    if device == 'auto':
        raise ValueError('device must be explicit')
    requested = torch.device(device)
    if requested.type not in ('cpu', 'cuda'):
        raise ValueError('only CPU and CUDA are supported')
    if requested.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; CPU fallback is forbidden')
        index = 0 if requested.index is None else requested.index
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError('requested CUDA device does not exist')
        requested = torch.device('cuda', index)
        torch.cuda.set_device(requested)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if torch.get_default_dtype() != torch.float32:
        raise ValueError('float32 default dtype required')
    # Full fp32 reference, explicit and identical for both algorithm families.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return requested


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda'] and torch.cuda.is_available():
        if len(state['torch_cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA RNG device count differs; continuation requires the same visible devices')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['torch_cuda']])


def env_state(model):
    vec = model.get_env()
    if vec is None:
        return None
    base = vec.envs[0].unwrapped
    return copy.deepcopy(dict(state=getattr(base, '_state', None), time=getattr(base, '_time', None),
                              steps=getattr(base, '_step_count', None), rng=base.np_random.bit_generator.state,
                              seed=base.np_random_seed, action_rng=model.action_space.np_random.bit_generator.state,
                              exploration_noise=model.action_noise.state() if hasattr(model.action_noise, 'state') else None,
                              pending_seeds=vec._seeds,
                              **({'training_resets': base.training_resets} if hasattr(base, 'training_resets') else {})))


@contextmanager
def isolated_policy(model):
    rng = rng_state()
    modes = [(m, m.training) for m in model.policy.modules()]
    parameters = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
    environment = env_state(model)
    try:
        model.policy.set_training_mode(False)
        yield
    finally:
        changed = any(not torch.equal(parameters[k], v) for k, v in model.policy.state_dict().items())
        if changed:
            model.policy.load_state_dict(parameters)
        for module, mode in modes:
            module.training = mode
        restore_rng(rng)
        if changed or env_state(model) != environment:
            raise RuntimeError('evaluation/diagnostics changed model statistics or the training environment')


class AuditedReplayBuffer(ReplayBuffer):
    """SB3 storage/sampling plus write sequence witnesses for circular replay."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.n_envs != 1 or self.optimize_memory_usage or not self.handle_timeout_termination:
            raise ValueError('E6d requires one env, explicit next observations and timeout masking')
        self.total_added = 0
        self.slot_sequence = np.full(self.buffer_size, -1, dtype=np.int64)
        self.training_diagnostics = None

    def add(self, obs, next_obs, action, reward, done, infos):
        if not all(np.isfinite(v).all() for v in (obs, next_obs, action, reward)):
            raise FloatingPointError('non-finite transition before replay storage')
        pos = self.pos
        super().add(obs, next_obs, action, reward, done, infos)
        self.slot_sequence[pos] = self.total_added
        self.total_added += 1
        if getattr(self, 'training_diagnostics', None) is not None:
            self.training_diagnostics.observe(infos[0], np.asarray(action).reshape(-1)[0], reward[0], self.total_added-1)

    def audit(self, num_timesteps):
        n, cap = self.total_added, self.buffer_size
        if n != num_timesteps or self.size() != min(n, cap) or self.pos != n % cap or self.full != (n >= cap):
            raise AssertionError('replay counters/ring position disagree with collected transitions')
        occupied = self.slot_sequence if self.full else self.slot_sequence[:self.pos]
        if not np.array_equal(np.sort(occupied), np.arange(max(0, n-cap), n)):
            raise AssertionError('circular replay lost or duplicated a retained transition')
        if n and self.slot_sequence[(self.pos-1) % cap] != n-1:
            raise AssertionError('the most recently collected transition was not stored')
        if getattr(self, 'training_diagnostics', None) is not None:
            valid = slice(None) if self.full else slice(0, self.pos)
            if (self.training_diagnostics.counts['stored'] != n
                    or not all(np.isfinite(getattr(self, key)[valid]).all() for key in
                               ('observations', 'next_observations', 'actions', 'rewards'))
                    or (not np.isin(self.actions[valid], np.arange(9)).all() if isinstance(self.action_space, Discrete)
                        else np.any(np.abs(self.actions[valid]) > 1.))
                    or not np.isin(self.dones[valid], (0., 1.)).all()
                    or not np.isin(self.timeouts[valid], (0., 1.)).all()
                    or np.any(self.timeouts[valid] > self.dones[valid])):
                raise ValueError('filtered replay contains invalid actions, timeouts or diagnostics')
        return dict(total_collected=n, size=self.size(), capacity=cap, pos=self.pos, full=self.full,
                    oldest_retained_sequence=max(0, n-cap), newest_retained_sequence=n-1 if n else None)


def finite_model(model):
    tensors = list(model.policy.state_dict().values())
    if getattr(model, 'log_ent_coef', None) is not None:
        tensors.append(model.log_ent_coef)
    optimizers = [getattr(getattr(model, key, None), 'optimizer', None) for key in ('actor', 'critic', 'policy')]
    optimizers.append(getattr(model, 'ent_coef_optimizer', None))
    for optimizer in (o for o in optimizers if o is not None):
        for state in optimizer.state.values():
            tensors.extend(v for v in state.values() if torch.is_tensor(v))
    if not all(torch.isfinite(t).all().item() for t in tensors):
        raise FloatingPointError('non-finite model/optimizer/entropy state')


def policy_fingerprint(model):
    """Content identity for reports, independent of ZIP metadata and RNG."""
    h = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        h.update(f'{name}:{tensor.dtype}:{tuple(tensor.shape)}'.encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def stats(tensor):
    tensor = tensor.detach()
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError('non-finite actual critic target or critic prediction')
    return dict(shape=list(tensor.shape), device=str(tensor.device), dtype=str(tensor.dtype),
                minimum=tensor.min().item(), maximum=tensor.max().item(), mean=tensor.mean().item())


class UpdateAudit:
    """Observe the real loss arguments, separately for MSE and quantile Huber.

    No actor-call assumptions, alternate targets or extra stochastic sampling.
    CUDA events are resolved in batches, never synchronized on each transition.
    """
    def __init__(self, *args, **kwargs):
        self.telemetry_interval = 1000
        self.telemetry = []
        self.after_update = None
        self.update_host_seconds = self.update_device_seconds = 0.
        self._cuda_events = []
        self.profile_cuda = True  # preserve historical paths; S3a defaults to False
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params()+['after_update', '_cuda_events']

    def flush_cuda_timing(self):
        if self._cuda_events:
            torch.cuda.synchronize(self.device)
            self.update_device_seconds += sum(a.elapsed_time(b)/1000 for a, b in self._cuda_events)
            self._cuda_events.clear()

    def train(self, gradient_steps, batch_size=64):
        if gradient_steps != 1:
            raise ValueError('E6d uses exactly one update per collected transition')
        observe = self._n_updates % self.telemetry_interval == 0
        captures = []
        module_name = ('sb3_contrib.tqc.tqc' if isinstance(self, TQC) else 'stable_baselines3.td3.td3'
                       if isinstance(self, DDPG) else 'stable_baselines3.dqn.dqn' if isinstance(self, DQN)
                       else 'stable_baselines3.sac.sac')
        module = importlib.import_module(module_name)
        owner = module if isinstance(self, TQC) else module.F
        name = 'quantile_huber_loss' if isinstance(self, TQC) else 'smooth_l1_loss' if isinstance(self, DQN) else 'mse_loss'
        original = getattr(owner, name)
        def loss(prediction, target, *args, **kwargs):
            captures.append((prediction.detach().clone(), target.detach().clone()))
            return original(prediction, target, *args, **kwargs)
        start = time.perf_counter()
        pair = None
        if self.device.type == 'cuda' and self.profile_cuda:
            pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)); pair[0].record()
        with patch.object(owner, name, loss) if observe else _no_observer():
            super().train(gradient_steps, batch_size)
        if pair:
            pair[1].record(); self._cuda_events.append(pair)
        self.update_host_seconds += time.perf_counter()-start
        loss_keys = ('loss',) if isinstance(self, DQN) else ('actor_loss', 'critic_loss') if isinstance(self, DDPG) else (
            'actor_loss', 'critic_loss', 'ent_coef', 'ent_coef_loss')
        losses = {key: float(self.logger.name_to_value['train/'+key]) for key in loss_keys}
        if not all(math.isfinite(v) for v in losses.values()):
            raise FloatingPointError('non-finite training loss/alpha')
        if observe:
            if not captures:
                raise RuntimeError('installed algorithm bypassed the audited critic-loss path')
            expected = [batch_size, 1, self.critic.n_quantiles*self.critic.n_critics
                        -self.top_quantiles_to_drop_per_net*self.critic.n_critics] if isinstance(self, TQC) else [batch_size, 1]
            if any(list(t.shape) != expected or t.device != self.device for _, t in captures):
                raise RuntimeError('actual critic target shape/device differs from the E6d contract')
            finite_model(self)
            self.telemetry.append(dict(transitions=self.num_timesteps, updates=self._n_updates,
                                        algorithm=next(a for a, cls in (('TQC', TQC), ('DDPG', DDPG), ('DQN', DQN), ('SAC', SAC)) if isinstance(self, cls)), **losses,
                                        actual_critic_targets=[stats(t) for _, t in captures],
                                        actual_critic_predictions=[stats(p) for p, _ in captures],
                                        target_semantics='actual library Bellman target, not task reward', finite=True))
        if len(self._cuda_events) >= 1000:
            self.flush_cuda_timing()
        if self.after_update:
            self.after_update(self)


@contextmanager
def _no_observer():
    yield


class RequestStorage:
    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
        if getattr(self, 'training_contract', None) is not None:
            expected = self.policy.unscale_action(buffer_action)
            received = np.asarray([[i['action_requested']] for i in infos])
            if (not np.isfinite(buffer_action).all() or np.any(np.abs(buffer_action) > 1)
                    or not np.array_equal(received, expected)
                    or any(i['u_requested'] != 4*float(a[0]) for i, a in zip(infos, expected))):
                raise ValueError('collector requested buffer_action / Env mapping mismatch')
        return super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)


class CloudSAC(RequestStorage, UpdateAudit, SAC):
    pass


class CloudTQC(RequestStorage, UpdateAudit, TQC):
    pass


class CloudDDPG(RequestStorage, UpdateAudit, DDPG):
    pass


class CloudDQN(UpdateAudit, DQN):
    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
        from cartpole.rl.methods import GRID
        for index, info in zip(np.asarray(buffer_action).reshape(-1), infos):
            if not isinstance(index, np.integer) or not 0 <= index < len(GRID) or info['request_index'] != index or info['u_requested'] != GRID[index]:
                raise ValueError('DQN integer request/replay/physical mapping mismatch')
        return super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)


def algorithm_class(config):
    registry = {'SAC': CloudSAC, 'TQC': CloudTQC}
    if config.get('training_contract', {}).get('id', '').startswith('cartpole-request-v2-'):
        from cartpole.rl.methods import parse_contract
        algorithm, _ = parse_contract(config['training_contract']['id'])
        if algorithm != config.get('algorithm'):
            raise ValueError('algorithm/contract mismatch')
        return dict(SAC=CloudSAC, TQC=CloudTQC, DDPG=CloudDDPG, DQN=CloudDQN)[algorithm]
    if config.get('algorithm') not in registry:
        raise ValueError(f'unknown algorithm: {config.get("algorithm")}')
    if 'training_contract' in config and config['algorithm'] != 'SAC':
        raise ValueError('filtered training currently supports SAC only')
    return registry[config['algorithm']]


def create_model(config, env):
    if 'training_contract' in config:
        from cartpole.rl.training_contract import assert_environment
        assert_environment(env, config['training_contract'])
    expected_device = configure_device(config['settings']['device'])
    model = algorithm_class(config)(env=env, **constructor(config, AuditedReplayBuffer))
    model.telemetry_interval = config['telemetry_interval']
    if 'training_contract' in config:
        from cartpole.rl.training_contract import initialize_streams
        from cartpole.rl.training_diagnostics import TrainingDiagnostics
        model.training_contract = copy.deepcopy(config['training_contract'])
        model.replay_buffer.training_diagnostics = TrainingDiagnostics()
        model.profile_cuda = config['protocol']['profile_cuda']
        initialize_streams(model)
    if model.device != expected_device or any(p.device != expected_device for p in model.policy.parameters()):
        raise RuntimeError('model device differs from explicit request')
    return model


def implementation_hashes(config=None):
    calls = [('SAC.train', SAC.train), ('TQC.train', TQC.train), ('ReplayBuffer._get_samples', ReplayBuffer._get_samples)]
    if config is not None and 'training_contract' in config:
        if config['algorithm'] in ('DDPG', 'DQN'):
            calls += [('DDPG.train', DDPG.train), ('DQN.train', DQN.train), ('DQN.predict', DQN.predict), ('DQN._on_step', DQN._on_step)]
        from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
        from stable_baselines3.common.policies import BasePolicy
        from stable_baselines3.common.vec_env import DummyVecEnv
        calls += [(f'OffPolicyAlgorithm.{name}', getattr(OffPolicyAlgorithm, name)) for name in
                  ('_sample_action', 'collect_rollouts', '_store_transition', 'learn')]
        calls += [('BasePolicy.scale_action', BasePolicy.scale_action),
                  ('BasePolicy.unscale_action', BasePolicy.unscale_action),
                  ('DummyVecEnv.step_wait', DummyVecEnv.step_wait), ('ReplayBuffer.add', ReplayBuffer.add)]
    return {name: hashlib.sha256(inspect.getsource(call).encode()).hexdigest() for name, call in calls}
