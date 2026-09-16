"""CPU-verifiable preflight and fresh-process model checks; no provider access."""
import copy
import importlib.metadata as packages
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import tempfile
import time

import numpy as np
import torch
from stable_baselines3.common.logger import configure

from cartpole.experiments.cloud_checkpoint import load_checkpoint, resolve_checkpoint
from cartpole.experiments.cloud_config import VERSIONS, load_config, validate_config
from cartpole.experiments.cloud_evaluation import evaluate, validation_cases
from cartpole.experiments.cloud_io import ROOT, atomic_json, manifest, new_output, sha256, verify_manifest
from cartpole.experiments.cloud_learning import configure_device, finite_model, isolated_policy
from cartpole.rl.pilot_config import make_pilot_env
from cartpole.rl.training_contract import environment_factory, evaluation_action


def check_stack(config=None):
    if platform.system() != 'Linux' or platform.machine() != 'x86_64' or platform.python_version() != '3.12.3':
        raise RuntimeError('E6d reviewed platform is Linux x86_64 / Python 3.12.3')
    actual = {name: packages.version(name) for name in VERSIONS}
    for name, expected in VERSIONS.items():
        if actual[name].split('+')[0] != expected:
            raise RuntimeError(f'{name}: expected {expected}, installed {actual[name]}')
    if torch.version.cuda not in (None, '12.6'):
        raise RuntimeError('this bundle reviews CPU and cu126 only; choose a reviewed compatible wheel')
    reference = json.loads((ROOT/'configs/cloud/reference_metrics.json').read_text())
    versioned = set()
    if config is not None and 'training_contract' in config:
        from cartpole.rl.training_contract import validate_identity
        validate_identity(config['training_contract'])
        # These wrappers/loggers evolved since E6d; S3a verifies their own hashes.
        # The remaining E6d checks (including physics, reward and classic) stay intact.
        versioned = {'cartpole/common/episode_log.py', 'cartpole/experiments/gym_diagnostics.py',
                     'cartpole/experiments/lqr_hold.py', 'cartpole/rl/env.py',
                     'cartpole/rl/pilot_config.py', 'cartpole/simulator/constrained.py'}
    changed = [name for name, digest in reference['baseline_dependencies'].items()
               if name not in versioned and (not (ROOT/name).is_file() or sha256(ROOT/name) != digest)]
    if changed:
        raise ValueError(f'frozen physics/reward/classical dependencies changed: {changed}')
    return actual


def hardware_report(device, output):
    from cartpole.experiments.cloud_host import memory_report, check_gpu
    requested = configure_device(device)
    memory = memory_report()
    disk = shutil.disk_usage(output)
    with tempfile.NamedTemporaryFile(dir=output) as probe:
        probe.write(b'write preflight'); probe.flush(); os.fsync(probe.fileno())
    report = dict(platform=platform.platform(), python=platform.python_version(),
                  machine=platform.machine(), cpu_count=os.cpu_count(), cpu_affinity=sorted(os.sched_getaffinity(0)),
                  ram_available_bytes=memory['host_available_bytes'], ram_total_bytes=memory['host_total_bytes'],
                  memory=memory, effective_ram_available_bytes=memory['effective_available_bytes'],
                  disk_free_bytes=disk.free, disk_total_bytes=disk.total, result_write_test=True,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  requested_device=device, actual_device=str(requested), torch_cuda_version=torch.version.cuda,
                  cuda_available=torch.cuda.is_available(), torch_threads=torch.get_num_threads(),
                  interop_threads=torch.get_num_interop_threads(),
                  thread_environment={k: os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    if requested.type == 'cuda':
        report['pre_download_gpu_policy'] = check_gpu()
        properties = torch.cuda.get_device_properties(requested)
        free, total = torch.cuda.mem_get_info(requested)
        report['gpu'] = dict(name=properties.name, compute_capability=list(torch.cuda.get_device_capability(requested)),
                             arch_list=torch.cuda.get_arch_list(), free_bytes=free, total_bytes=total)
        report['nvidia_smi'] = subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total',
                                                        '--format=csv,noheader'],text=True).strip()
        if free < 1024**3:
            raise RuntimeError('preflight needs at least 1 GiB free VRAM; use the measured peak plus headroom')
    if memory['effective_available_bytes'] < 2*1024**3 or disk.free < 4*1024**3:
        raise RuntimeError('preflight requires at least 2 GiB available RAM and 4 GiB free disk after installation')
    return report


def preflight(name, seed, device, output=None, session=None):
    from cartpole.experiments.cloud_runner import run_training
    out = new_output(output, 'cloud_preflight_'+name+'_'+device.replace(':','_'))
    versions = check_stack(); hardware = hardware_report(device, out)
    cfg = load_config(name, seed, device, purpose='smoke')
    start = time.perf_counter()
    run, summary = run_training(cfg, output=out/'physical_smoke', max_transitions=12, max_seconds=60.,
                                evaluate_enabled=False, session=session)
    point = resolve_checkpoint(run)
    model, data, _ = load_checkpoint(point, device=device)
    with isolated_policy(model):
        batch = model.replay_buffer.sample(4)
        tensors = {key: str(value.device) for key, value in zip(batch._fields, batch) if torch.is_tensor(value)}
    if any(value != str(model.device) for value in tensors.values()) or not model.telemetry:
        raise AssertionError('training tensors/actual target telemetry are not on the requested device')
    if summary['gradient_updates'] != 4 or not summary['replay']['size']:
        raise AssertionError('preflight did not execute real updates on Drake transitions')
    report = dict(versions=versions, hardware=hardware, reward_version=cfg['environment']['reward_version'],
                  physical_smoke=summary, training_tensor_devices=tensors,
                  actual_target=model.telemetry[-1]['actual_critic_targets'],
                  cuda_execution_tested=model.device.type == 'cuda',
                  gpu_checkpoint_to_cpu='pending on a CUDA machine', seconds=time.perf_counter()-start)
    if model.device.type == 'cuda':
        # Fresh interpreter, explicit CPU; no silent migration of this process.
        import sys
        destination = out/'gpu_to_cpu'
        timeout = min(120., session.status()['seconds_before_save_reserve']) if session else 120.
        if timeout <= 0:
            raise RuntimeError('session budget exhausted before GPU-to-CPU verification')
        subprocess.run([sys.executable,'-B','-m','cartpole.experiments.cloud','verify','--run-dir',str(run),
                        '--device','cpu','--training-seed',str(seed),'--labels','last','--skip-evaluation',
                        '--output-dir',str(destination)],check=True,timeout=timeout)
        report['gpu_checkpoint_to_cpu'] = json.loads((destination/'verification.json').read_text())
    report['pip_check'] = subprocess.check_output([os.sys.executable,'-m','pip','check'],text=True).strip()
    atomic_json(out/'preflight.json', report); atomic_json(out/'manifest.json', manifest(out))
    return out, report


def metrics_close(actual, expected, tolerance=1e-9, path='metrics'):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(path+' keys differ')
        for key in expected:
            metrics_close(actual[key], expected[key], tolerance, path+'.'+key)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(path+' length differs')
        for a, b in zip(actual, expected):
            metrics_close(a, b, tolerance, path)
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
            raise AssertionError(f'{path}: {actual} != {expected}')
    elif actual != expected:
        raise AssertionError(path+' differs')


def verify(run, device, seed, *, output=None, labels=('best','last'), skip_evaluation=False):
    start = time.perf_counter(); configure_device(device)
    run = Path(run); config = json.loads((run/'config.json').read_text()); validate_config(config)
    check_stack(config)
    if config['settings']['seed'] != seed:
        raise ValueError('explicit training seed differs from the saved run')
    out = new_output(output, config['id']+f'_seed{seed}_verify')
    if out.resolve().is_relative_to(run.resolve()):
        raise ValueError('verification output must be outside the immutable input run')
    checks = {}; evaluated = {}
    if (run/'manifest.json').exists():
        verify_manifest(run)
    for label in labels:
        point = resolve_checkpoint(run, label)
        model, data, _ = load_checkpoint(point, device=device)
        probes = json.loads((point/'probes.json').read_text())
        with isolated_policy(model):
            actions, _ = model.predict(np.asarray(probes['observations'], dtype=np.float32), deterministic=True)
        expected = np.asarray(probes['actions'], dtype=np.float32)
        same_device = str(model.device) == data['device']
        np.testing.assert_allclose(actions, expected, atol=0. if same_device else 1e-6,
                                   rtol=0. if same_device else 1e-5)
        if 'training_contract' in config:
            with isolated_policy(model):
                mapped = evaluation_action(model, np.asarray(probes['observations'], dtype=np.float32), filtered=True)
            np.testing.assert_allclose(mapped, probes['env_actions'], atol=0. if same_device else 1e-6,
                                       rtol=0. if same_device else 1e-5)
        row = dict(generation=point.name, model_sha256=sha256(point/'model.zip'),
                   training_seed=seed, actions_max_abs_difference=float(np.max(np.abs(actions-expected))),
                   loaded_device=str(model.device), saved_device=data['device'], replay=data['replay'],
                   evaluated_episodes=0, metrics_match=None)
        if not skip_evaluation:
            key = sha256(point/'model.zip')
            if key in evaluated:
                report = evaluated[key]; row['evaluation_reused_from_identical_model'] = True
            else:
                report = evaluate(model, validation_cases(), out, label, algorithm=config['algorithm'], seed=seed,
                                  details=False, diagnostic=config['purpose'] != 'experiment' or 'training_contract' in config,
                                  env_factory=environment_factory(config))
                evaluated[key] = report; row['evaluated_episodes'] = len(report['episodes'])
            expected_path = run/'evaluations'/(point.name+'.json')
            if expected_path.exists():
                saved = json.loads(expected_path.read_text())
                if saved['complete']:
                    metrics_close(report['episodes'], saved['episodes'], 1e-9 if same_device else 1e-5)
                    row['metrics_match'] = True
        if label == 'last':
            before = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
            n, updates = model.num_timesteps, model._n_updates
            model.set_logger(configure(folder=None, format_strings=[])); model.train(1, config['settings']['batch_size'])
            finite_model(model)
            if model.num_timesteps != n or model._n_updates != updates+1:
                raise AssertionError('verification update changed collection counters')
            if not any(not torch.equal(before[k], v) for k, v in model.policy.state_dict().items()):
                raise AssertionError('verification did not change network weights')
            model.save(out/'updated_copy.zip')
            row.update(one_update_on_copy=True, additional_collected_transitions=0)
        checks[label] = row
    result = dict(checks=checks, seconds=time.perf_counter()-start, pid=os.getpid(),
                  seed=seed, originals_unchanged=True, bitwise_resume_guaranteed=False)
    atomic_json(out/'verification.json', result); atomic_json(out/'manifest.json', manifest(out))
    return out, result


def smoke_proof(run, device):
    """Compare actual initial/final tensors, not ZIP bytes or update counters alone."""
    run=Path(run)
    initial=[p for p in (run/'checkpoints').iterdir() if p.is_dir() and not p.name.startswith('.')
             and json.loads((p/'metadata.json').read_text())['transitions'] == 0]
    if not initial:
        raise ValueError('initial smoke checkpoint unavailable')
    before,_,_=load_checkpoint(initial[0],device=device)
    after,data,_=load_checkpoint(resolve_checkpoint(run),device=device)
    delta=max((before.policy.state_dict()[key]-value).abs().max().item()
              for key,value in after.policy.state_dict().items())
    if delta <= 0 or after._n_updates <= 0:
        raise AssertionError('smoke did not change weights through real updates')
    finite_model(after)
    proof=dict(transitions=after.num_timesteps,updates=after._n_updates,
               replay=after.replay_buffer.audit(after.num_timesteps),parameters_changed=True,
               max_parameter_difference=delta,finite_parameters_optimizer_entropy=True,
               actual_target_records=len(after.telemetry),initial_generation=initial[0].name,
               final_generation=resolve_checkpoint(run).name,scientific_result=False)
    atomic_json(run/'smoke_contract.json',proof)
    return proof
