"""Complete, checksummed recovery generations with atomic best/last pointers."""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid

import numpy as np
import torch

from cartpole.experiments.cloud_io import atomic_json, manifest, sha256, sync_dir, verify_manifest
from cartpole.experiments.cloud_learning import (
    algorithm_class, configure_device, env_state, finite_model, implementation_hashes, isolated_policy, policy_fingerprint, restore_rng, rng_state,
)
from cartpole.experiments.sac_pilot import fixed_observations
from cartpole.rl.training_contract import validate_identity, assert_environment, evaluation_action


def resolve_checkpoint(run, label='last'):
    run = Path(run)
    pointer = json.loads((run/(label+'.json')).read_text())
    path = run/'checkpoints'/pointer['generation']
    if not path.resolve().is_relative_to((run/'checkpoints').resolve()):
        raise ValueError('unsafe checkpoint pointer')
    verify_manifest(path)
    if sha256(path/'manifest.json') != pointer['manifest_sha256']:
        raise ValueError('checkpoint pointer/manifest mismatch')
    return path


def publish_pointer(run, label, checkpoint):
    atomic_json(Path(run)/(label+'.json'), dict(generation=Path(checkpoint).name,
                manifest_sha256=sha256(Path(checkpoint)/'manifest.json')))


def save_checkpoint(model, run, config, state, cases, *, min_free_bytes=256*1024**2):
    if 'training_contract' in config:
        validate_identity(config['training_contract'])
        assert_environment(model.get_env().envs[0], config['training_contract'])
        if model.training_contract != config['training_contract']:
            raise ValueError('model training contract mismatch')
    run = Path(run); parent = run/'checkpoints'; parent.mkdir(exist_ok=True)
    finite_model(model)
    replay = model.replay_buffer.audit(model.num_timesteps)
    if shutil.disk_usage(parent).free < min_free_bytes:
        raise OSError('insufficient disk for an atomic recovery checkpoint; previous generation retained')
    model.flush_cuda_timing()
    generation = f't{model.num_timesteps:09d}_u{model._n_updates:09d}_{uuid.uuid4().hex[:10]}'
    temporary = Path(tempfile.mkdtemp(prefix='.incomplete-', dir=parent))
    try:
        model.save(temporary/'model.zip')
        model.save_replay_buffer(temporary/'replay.pkl')
        with isolated_policy(model):
            observations = fixed_observations(cases)
            actions, _ = model.predict(observations, deterministic=True)
            received = evaluation_action(model, observations, filtered='training_contract' in config)
        atomic_json(temporary/'probes.json', dict(observations=observations.tolist(), actions=actions.tolist(),
                    training_seed=config['settings']['seed'], transitions=model.num_timesteps,
                    updates=model._n_updates, device=str(model.device),
                    **({'env_actions': received.tolist()} if 'training_contract' in config else {})))
        torch.save(dict(rng=rng_state(), environment=env_state(model), run_state=state), temporary/'runtime.pt')
        metadata = dict(schema=1, config=config, transitions=model.num_timesteps, updates=model._n_updates,
                        replay=replay, seed=config['settings']['seed'], device=str(model.device),
                        implementation=implementation_hashes(config), phase=state['phase'],
                        cumulative_seconds=state['cumulative_seconds'],
                        policy_sha256=policy_fingerprint(model),
                        recovery_accounting=state.get('recovery_accounting', {}),
                        preparation_version='E6d-r1',
                        source_manifest_sha256=sha256(run/'source_manifest.json'),
                        saved_at_epoch=time.time(), complete=True,
                        evaluation_status='separate report; recovery point alone is not selection eligible')
        atomic_json(temporary/'metadata.json', metadata)
        atomic_json(temporary/'manifest.json', manifest(temporary))
        for file in temporary.iterdir():
            with file.open('rb') as stream:
                os.fsync(stream.fileno())
        sync_dir(temporary)
        destination = parent/generation
        os.rename(temporary, destination); sync_dir(parent)
        publish_pointer(run, 'last', destination)
        return destination
    finally:
        # Incomplete files are not published. No previous generation is removed.
        if temporary.exists():
            shutil.rmtree(temporary)


def prune_checkpoints(run, keep):
    run = Path(run)
    protected = set()
    pin = run/'pinned_evaluation.json'
    if pin.exists():
        protected.add(json.loads(pin.read_text())['generation'])
    for label in ('best', 'last'):
        if (run/(label+'.json')).exists():
            protected.add(resolve_checkpoint(run, label).name)
    generations = []
    for directory in (run/'checkpoints').iterdir():
        if directory.name.startswith('.') or not directory.is_dir():
            continue
        metadata = json.loads((directory/'metadata.json').read_text())
        if metadata['transitions'] in (100000, 300000):
            protected.add(directory.name)
        generations.append((metadata['saved_at_epoch'], directory))
    candidates = [p for _, p in sorted(generations, reverse=True) if p.name not in protected]
    for directory in candidates[keep:]:
        shutil.rmtree(directory)


def load_checkpoint(checkpoint, *, device, env=None, expected_config=None):
    requested = configure_device(device)
    checkpoint = Path(checkpoint)
    verify_manifest(checkpoint)
    data = json.loads((checkpoint/'metadata.json').read_text())
    if not data['complete'] or data['implementation'] != implementation_hashes(data['config']):
        raise ValueError('incomplete checkpoint or incompatible algorithm implementation')
    config = data['config']
    saved_contract = config.get('training_contract')
    env_contract = getattr(env.unwrapped, 'training_contract', None) if env is not None else None
    if saved_contract is not None:
        validate_identity(saved_contract)
        from cartpole.experiments.cloud_config import validate_config
        from cartpole.experiments.cloud_checks import check_stack
        validate_config(config)
        check_stack(config)
        if env is not None:
            assert_environment(env, saved_contract)
    elif env_contract is not None:
        raise ValueError('off checkpoint cannot resume filtered environment')
    if expected_config is not None and config != expected_config:
        raise ValueError('checkpoint configuration mismatch')
    model = algorithm_class(config).load(checkpoint/'model.zip', env=env, device=device, force_reset=True)
    if model.device != requested or any(p.device != requested for p in model.policy.parameters()):
        raise RuntimeError('loaded model device differs from explicit request; fallback is forbidden')
    model.after_update = None; model._cuda_events = []
    if saved_contract is not None and getattr(model, 'training_contract', None) != saved_contract:
        raise ValueError('model/checkpoint training contract mismatch')
    model.load_replay_buffer(checkpoint/'replay.pkl')
    model.replay_buffer.device = model.device
    model.replay_buffer.audit(data['transitions'])
    if (model.num_timesteps, model._n_updates, model.seed) != (data['transitions'], data['updates'], data['seed']):
        raise ValueError('model/replay/metadata counters disagree')
    finite_model(model)
    # Only trusted, own-run artifacts: runtime contains Python/NumPy RNG objects.
    runtime = torch.load(checkpoint/'runtime.pt', map_location='cpu', weights_only=False)
    if saved_contract is not None:
        if runtime['run_state']['phase'] != data['phase'] or data['phase'] not in (
                'untrained', 'warmup', 'post_update', 'collected_pending_update'):
            raise ValueError('checkpoint phase mismatch')
        expected_updates = max(0, model.num_timesteps-model.learning_starts)
        if data['phase'] == 'collected_pending_update':
            expected_updates -= 1
        if model._n_updates != expected_updates:
            raise ValueError('checkpoint pending update/counters mismatch')
        rb = model.replay_buffer
        if (rb.training_diagnostics is None or rb.training_diagnostics.counts['stored'] != rb.total_added
                or not rb.handle_timeout_termination or rb.optimize_memory_usage):
            raise ValueError('checkpoint replay/diagnostics/timeout contract mismatch')
    return model, data, runtime


def prepare_resume_environment(model, runtime):
    """Restart physical episode; keep future reset/action RNG streams and counts.

    We do not claim to restore Drake adaptive integrator history. The unfinished
    episode is explicitly abandoned, never counted as successful or a failure.
    Its last transition already has its correct next observation in replay.
    """
    saved = runtime['environment']; base = model.get_env().envs[0].unwrapped
    if saved is not None:
        base.np_random.bit_generator.state = copy.deepcopy(saved['rng'])
        base._np_random_seed = saved['seed']
        model.action_space.np_random.bit_generator.state = copy.deepcopy(saved['action_rng'])
        if saved.get('exploration_noise') is not None:
            if not hasattr(model.action_noise, 'restore'):
                raise ValueError('missing independent exploration noise')
            model.action_noise.restore(saved['exploration_noise'])
        if 'training_resets' in saved:
            base.training_resets = copy.deepcopy(saved['training_resets'])
    model.get_env()._seeds = [None]
    model._last_obs = None
    model._last_episode_starts = None
    restore_rng(runtime['rng'])
    return dict(physical_episode='fresh reset continuing saved environment RNG',
                abandoned_episode_state=repr(saved['state']) if saved else None,
                bitwise_continuation_guaranteed=False)
