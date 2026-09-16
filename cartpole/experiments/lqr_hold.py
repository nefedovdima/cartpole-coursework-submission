"""Local upright LQR holding only; run with python -m cartpole.experiments.lqr_hold."""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata as packages
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from cartpole.common import State
from cartpole.common.episode_log import append_transition, finish_log, new_log, save_log
from cartpole.common.metrics import UprightThresholds
from cartpole.control import BalanceLQRControl
from cartpole.experiment_config import CONTROL_INTERVAL, make_experiment_config
from cartpole.simulator.constrained import CartPoleEpisode
from cartpole.simulator.pydrake.simulator import _finite_scalar


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class IntegratorSettings:
    target_accuracy: float = 1e-6
    maximum_step_size: float = .01
    fixed_step_mode: bool = False

    def apply(self, backend):
        integrator = backend.simulator.get_mutable_integrator()
        integrator.set_target_accuracy(self.target_accuracy)
        integrator.set_maximum_step_size(self.maximum_step_size)
        integrator.set_fixed_step_mode(self.fixed_step_mode)
        return {'type': type(integrator).__name__,
                'target_accuracy': integrator.get_target_accuracy(),
                'maximum_step_size': integrator.get_maximum_step_size(),
                'fixed_step_mode': integrator.get_fixed_step_mode()}


def dependency_versions():
    return {'python': platform.python_version(),
            **{name: packages.version(name) for name in ('drake', 'numpy', 'matplotlib', 'Pillow')}}


def capture_source(output):
    """Snapshot tracked and untracked relevant source, never a repository reset."""
    output = Path(output)
    paths = sorted([*ROOT.glob('cartpole/**/*.py'), *ROOT.glob('tests/*.py'),
                    *ROOT.glob('tests/fixtures/*'),
                    *ROOT.glob('requirements*.txt'),
                    *(ROOT/name for name in ('pyproject.toml', 'poetry.lock'))])
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        destination = output/'source_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        hashes[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest = json.dumps(hashes, sort_keys=True, indent=2) + '\n'
    (output/'source_manifest.json').write_text(manifest)
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
    diff = subprocess.check_output(['git', 'diff', '--binary', 'HEAD', '--', 'cartpole', 'tests',
                                    'pyproject.toml', 'poetry.lock', 'requirements-coursework.txt'], cwd=ROOT)
    (output/'source.diff').write_bytes(diff)
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], cwd=ROOT)
    (output/'environment.freeze.txt').write_bytes(freeze)
    return {'sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'dirty': bool(status), 'status_porcelain': status,
            'source_manifest_sha256': hashlib.sha256(manifest.encode()).hexdigest(),
            'source_snapshot': 'source_snapshot', 'source_manifest': 'source_manifest.json',
            'tracked_source_diff': 'source.diff'}


def run_episode(controller, initial, *, horizon=10., dt=CONTROL_INTERVAL,
                integrator=IntegratorSettings(), metadata=None, log_path=None,
                timed_controller=False, safety_filter=False):
    """Run on an absolute control grid, stopping on terminal or the horizon.

    Only CartPoleEpisode executes transitions. The final grid deadline is
    clipped to the requested horizon, so a nonintegral horizon has one shorter
    last step, and no extra zero/full step is made after the deadline.
    """
    horizon, dt = _finite_scalar(horizon, 'horizon'), _finite_scalar(dt, 'dt')
    if horizon < 0 or dt <= 0:
        raise ValueError('horizon must be nonnegative and dt must be positive')
    if log_path is not None and Path(log_path).exists():
        raise FileExistsError(log_path)
    episode = CartPoleEpisode(safety_filter=safety_filter)
    state = episode.reset(initial)
    actual_integrator = integrator.apply(episode.backend)
    info = {**(metadata or {}), 'config': asdict(make_experiment_config()),
            'initial_state': asdict(state), 'control_interval': dt, 'horizon': horizon,
            'integrator': actual_integrator, 'versions': dependency_versions(),
            'upright_thresholds': asdict(UprightThresholds())}
    log = new_log(info, state)
    if episode.safety_limits is not None:
        info['safety_filter'] = asdict(episode.safety_limits)
    index = 0
    try:
        while episode.backend.timestamp() < horizon:
            deadline = min((index+1)*dt, horizon)
            stamp = episode.backend.timestamp()
            action = controller(stamp, state) if timed_controller else controller(state)
            transition = episode.step(action, deadline-stamp)
            append_transition(log, transition)
            if hasattr(controller, 'phase'):
                log['transitions'][-1]['control_phase'] = controller.phase
            state = transition.state
            if transition.terminated:
                finish_log(log, 'terminal')
                break
            if deadline == horizon:
                finish_log(log, 'horizon')
                break
            index += 1
        else:
            finish_log(log, 'horizon')
    except Exception as exc:
        finish_log(log, 'exception', exception=exc,
                   observed_state=episode.backend.get_state(), observed_time=episode.backend.timestamp())
        if log_path is not None:
            try:
                save_log(log_path, log)
            except Exception as log_error:
                exc.add_note(f'Could not save failure log: {log_error}')
        raise
    if log_path is not None:
        save_log(log_path, log)
    return log


def main():
    parser = argparse.ArgumentParser(description='Локальное удержание сверху через LQR, без подъёма снизу.')
    parser.add_argument('--output-dir', type=Path, help='Новый каталог; существующий не перезаписывается.')
    args = parser.parse_args()
    output = args.output_dir or ROOT/'results'/datetime.now(timezone.utc).strftime('lqr_hold_%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True, exist_ok=False)
    git = capture_source(output)
    controller = BalanceLQRControl(make_experiment_config())
    controller_info = {'name': 'BalanceLQRControl', 'Q': controller.Q.tolist(),
                       'R': controller.R.tolist(), 'K': controller.K.tolist(),
                       'feedback': 'u=-K@[x,atan2(sin(theta-pi),cos(theta-pi)),v,omega]'}
    summary = {}
    for name, theta in [('pi_plus', math.pi+.02), ('pi_minus', math.pi-.02),
                        ('minus_pi_plus', -math.pi+.02)]:
        log = run_episode(controller, State(pole_angle=theta), log_path=output/f'{name}.json',
                          metadata={'experiment': 'local_upright_lqr_hold', 'case': name,
                                    'git': git, 'controller': controller_info})
        summary[name] = {**log['metrics'], 'completion': log['completion'], 'log': f'{name}.json'}
    (output/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
    print(output.resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
