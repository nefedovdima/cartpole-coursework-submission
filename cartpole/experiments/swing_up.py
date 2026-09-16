"""Bounded collocation search on the exact bottom start; fixed-plan validation."""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from pydrake.solvers import SnoptSolver
from pydrake.trajectories import PiecewisePolynomial

from cartpole.common import State
from cartpole.common.episode_log import load_log
from cartpole.common.metrics import UprightThresholds
from cartpole.control.swing_up import NominalTrajectory, TrackingThenBalance
from cartpole.control.trajectory import make_trajectory_program
from cartpole.experiment_config import make_experiment_config
from cartpole.experiments.lqr_hold import ROOT, IntegratorSettings, capture_source, run_episode


MAX_VARIANTS = 30
SEARCH_SECONDS = 1200.
VARIANT_SECONDS = 45.


@dataclass(frozen=True)
class Candidate:
    nodes: int = 81
    max_duration: float = 5.
    guess_duration: float = 3.
    guess_oscillations: float = 1.
    guess_amplitude: float = 0.
    tracking_scale: float = 1.


def write_json(path, payload):
    with Path(path).open('x') as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def candidate_list():
    # Deterministic proposal order. Validation starts never select parameters.
    return [Candidate(nodes=n, guess_duration=duration, guess_oscillations=cycles, guess_amplitude=amplitude)
            for n in (81, 101, 61)
            for duration, cycles, amplitude in ((3., 1., 0.), (3., 1., 1.5), (4., 2., 2.),
                                                (3., 2., -2.), (4., 1., -1.5), (5., 2., 2.),
                                                (3., 1., -2.), (4., 2., -2.), (5., 3., 2.), (5., 3., -2.))]


def solve_candidate(candidate, output):
    config = make_experiment_config()
    program, plant = make_trajectory_program(config, State(), candidate.nodes, candidate.max_duration)
    s = np.linspace(0., 1., candidate.nodes)
    times = candidate.guess_duration*s
    angle = math.pi*(3*s*s-2*s*s*s) + candidate.guess_amplitude*np.sin(2*math.pi*candidate.guess_oscillations*s)*np.sin(math.pi*s)
    states = np.zeros((4, candidate.nodes))
    states[1] = angle
    states[3] = np.gradient(angle, times)
    states[:, 0] = 0.
    states[:, -1] = [0., math.pi, 0., 0.]
    program.SetInitialTrajectory(PiecewisePolynomial.FirstOrderHold(times, np.zeros((1, candidate.nodes))),
                                 PiecewisePolynomial.FirstOrderHold(times, states))
    solver = SnoptSolver()
    options = {'Major iterations limit': 1000, 'Major feasibility tolerance': 1e-7,
               'Major optimality tolerance': 1e-5, 'Print file': str((output/'snopt.out').resolve())}
    for key, value in options.items():
        program.prog().SetSolverOption(solver.solver_id(), key, value)
    write_json(output/'optimization.json', {'candidate': asdict(candidate), 'solver': 'SNOPT', 'options': options,
              'config': asdict(config), 'minimum_time_step': .001, 'maximum_time_step': .1,
              'equal_time_intervals': True, 'initial_state': asdict(State()),
              'final_state_bounds': [[-.75*config.max_position, math.pi, 0., 0.], [.75*config.max_position, math.pi, 0., 0.]],
              'cost': 'integral(u^2 + time) + final_x^2', 'worker_wall_time_limit_seconds': VARIANT_SECONDS})
    started = time.monotonic()
    result = solver.Solve(program.prog())
    info = {'success': result.is_success(), 'solution_result': str(result.get_solution_result()),
            'solver_id': result.get_solver_id().name(), 'solve_seconds': time.monotonic()-started,
            'snopt_info': int(result.get_solver_details().info), 'optimal_cost': None}
    cost = float(result.get_optimal_cost())
    if math.isfinite(cost):
        info['optimal_cost'] = cost
    write_json(output/'solver.json', info)
    # Failed finite iterates are evidence, never executable plans.
    np.save(output/'solver_iterate.npy', result.get_x_val(), allow_pickle=False)
    if not result.is_success():
        return None
    trajectory = NominalTrajectory(program.ReconstructStateTrajectory(result), program.ReconstructInputTrajectory(result))
    trajectory.save(output/'nominal.npz')
    return trajectory


def evaluate_plan(trajectory, candidate, output, cases, git, nominal_path):
    summary = {}
    for name, theta in cases:
        controller = TrackingThenBalance(make_experiment_config(), trajectory, candidate.tracking_scale)
        log = run_episode(controller, State(pole_angle=theta), timed_controller=True,
                          log_path=output/f'{name}.json', metadata={
                              'experiment': 'classical_swing_up', 'method': 'direct_collocation_tvlqr_then_lqr',
                              'case': name, 'git': git, 'candidate': asdict(candidate),
                              'nominal': {'path': str(nominal_path.resolve()),
                                          'sha256': hashlib.sha256(nominal_path.read_bytes()).hexdigest()},
                              'controller': controller.parameters, 'required_hold_duration': 2.})
        summary[name] = {'metrics': log['metrics'], 'evaluation': log['evaluation'], 'completion': log['completion']}
    return summary


def worker(output):
    request = json.loads((output/'request.json').read_text())
    candidate = Candidate(**request['candidate'])
    trajectory = solve_candidate(candidate, output)
    if trajectory is None:
        write_json(output/'outcome.json', {'status': 'solver_failure'})
        return
    summary = evaluate_plan(trajectory, candidate, output, [('bottom', 0.)], request['git'], output/'nominal.npz')
    write_json(output/'outcome.json', {'status': 'evaluated', **summary['bottom']})


def rank(outcome):
    if outcome.get('status') != 'evaluated':
        return (-1.,)
    metric, evaluation = outcome['metrics'], outcome['evaluation']
    return (float(evaluation['success_episode']), evaluation['hold_final'],
            evaluation['hold_longest'], metric['duration'], -abs(metric['final_angle_error']))


def search(output):
    git = capture_source(output)
    git['artifact_root'] = str(output.resolve())
    planned = candidate_list()
    write_json(output/'search_plan.json', {
        'max_variants': MAX_VARIANTS, 'max_search_seconds': SEARCH_SECONDS,
        'per_variant_wall_seconds': VARIANT_SECONDS, 'stops_on_first_bottom_success': True,
        'initial_state': asdict(State()), 'horizon': 10., 'control_interval': .01,
        'config': asdict(make_experiment_config()), 'integrator': asdict(IntegratorSettings()),
        'thresholds': asdict(UprightThresholds()), 'required_final_contiguous_hold': 2.,
        'criterion': 'full horizon, no terminal, final consecutive sampled hold >=2 s; no summing separate runs',
        'ranking': 'success, final hold, longest hold, duration, negative absolute final angle error',
        'candidates': [asdict(c) for c in planned], 'git': git})
    started = time.monotonic()
    records, best = [], None
    for index, candidate in enumerate(planned[:MAX_VARIANTS]):
        remaining = SEARCH_SECONDS-(time.monotonic()-started)
        if remaining <= 0:
            break
        folder = output/f'variant_{index:02d}'
        folder.mkdir()
        write_json(folder/'request.json', {'candidate': asdict(candidate), 'git': git})
        attempt = time.monotonic()
        with (folder/'worker.log').open('w') as stream:
            try:
                completed = subprocess.run([sys.executable, '-B', '-m', 'cartpole.experiments.swing_up', '--worker', str(folder.resolve())],
                                           stdout=stream, stderr=subprocess.STDOUT,
                                           timeout=min(VARIANT_SECONDS, remaining), cwd=ROOT)
                if completed.returncode:
                    write_json(folder/'outcome.json', {'status': 'worker_error', 'returncode': completed.returncode})
            except subprocess.TimeoutExpired:
                write_json(folder/'outcome.json', {'status': 'timeout'})
        outcome = json.loads((folder/'outcome.json').read_text())
        record = {'variant': folder.name, 'candidate': asdict(candidate),
                  'wall_seconds': time.monotonic()-attempt, 'outcome': outcome}
        records.append(record)
        if best is None or rank(outcome) > rank(best['outcome']):
            best = record
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if outcome.get('evaluation', {}).get('success_episode'):
            break
    report = {'variants_attempted': len(records), 'wall_seconds': time.monotonic()-started,
              'max_variants': MAX_VARIANTS, 'max_search_seconds': SEARCH_SECONDS,
              'records': records, 'selected_variant': best['variant'] if best else None,
              'stop_reason': 'bottom_success' if best and best['outcome'].get('evaluation', {}).get('success_episode')
              else 'budget_exhausted'}
    write_json(output/'search_report.json', report)
    return report, git


def main():
    parser = argparse.ArgumentParser(description='Ограниченный поиск классического подъёма и проверка фиксированного плана.')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--plan', type=Path, help='Сохранённый nominal.npz; запуск без нового поиска.')
    parser.add_argument('--candidate', type=Path, help='request.json с настройками выбранного варианта.')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    output = args.output_dir or ROOT/'results'/datetime.now(timezone.utc).strftime('swing_up_%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True, exist_ok=False)
    if args.plan:
        if args.candidate is None:
            parser.error('--plan requires --candidate')
        git = capture_source(output)
        candidate = Candidate(**json.loads(args.candidate.read_text())['candidate'])
        trajectory = NominalTrajectory.load(args.plan)
        # Copy the exact selected artifact, retaining its identifier bytewise.
        nominal_path = output/'nominal.npz'
        with nominal_path.open('xb') as stream:
            stream.write(args.plan.read_bytes())
        write_json(output/'request.json', {'candidate': asdict(candidate), 'source_plan': str(args.plan.resolve())})
    else:
        report, git = search(output)
        selected = output/report['selected_variant']
        if not (selected/'nominal.npz').exists():
            print('No feasible nominal plan; solver evidence saved at', output)
            return
        candidate = Candidate(**json.loads((selected/'request.json').read_text())['candidate'])
        nominal_path = selected/'nominal.npz'
        trajectory = NominalTrajectory.load(nominal_path)
    summary = evaluate_plan(trajectory, candidate, output,
                            [('bottom', 0.), ('plus_002', .02), ('minus_002', -.02)], git, nominal_path)
    write_json(output/'summary.json', summary)
    print(output.resolve(), flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
