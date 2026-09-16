"""LQR signs/periodicity, episode logging and playback of nonuniform time."""
import base64
from contextlib import redirect_stdout
import copy
from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cartpole.common import State
from cartpole.common.episode_log import load_log, save_log, validate_log
from cartpole.common.metrics import upright_angle_error
from cartpole.control import BalanceLQRControl
from cartpole.experiment_config import make_experiment_config
from cartpole.experiments.lqr_hold import ROOT, capture_source, run_episode
from cartpole.replay import export_replay
from cartpole.common.view import pole_geometry, replay_frame_times, sample_saved_log


class LQRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = make_experiment_config()
        cls.controller = BalanceLQRControl(cls.config)

    def test_ordinary_import_in_fresh_process_does_not_load_trajectory_optimizer(self):
        program = ('import sys; from cartpole.control import BalanceLQRControl; '
                   'from cartpole.control.lqr import BalanceLQRControl as Direct; '
                   'assert Direct is BalanceLQRControl; '
                   'assert "cartpole.control.trajectory" not in sys.modules')
        subprocess.run([sys.executable, '-B', '-c', program], cwd=ROOT, check=True, capture_output=True)

    def test_equilibria_and_equivalent_angles_produce_equal_commands(self):
        self.assertEqual(self.controller(State(pole_angle=math.pi)), 0.)
        for offset in (0., .02, -.02, .3):
            original = State(cart_position=.01, cart_velocity=-.02,
                             pole_angle=math.pi+offset, pole_angular_velocity=.04)
            expected = self.controller(original)
            for turns in (-2, -1, 1, 2):
                equivalent = copy.copy(original)
                equivalent.pole_angle += turns*2*math.pi
                self.assertAlmostEqual(self.controller(equivalent), expected, delta=1e-12)
        for theta in (-math.pi, 3*math.pi):
            self.assertAlmostEqual(self.controller(State(pole_angle=theta)), 0., delta=1e-12)

    def test_component_order_feedback_sign_and_closed_loop_stability(self):
        state = State(cart_position=.01, cart_velocity=-.03,
                      pole_angle=-math.pi+.02, pole_angular_velocity=.04)
        expected = -(self.controller.K @ np.array([.01, .02, -.03, .04]))[0]
        self.assertAlmostEqual(self.controller(state), expected, delta=1e-12)
        A = np.array([[0., 0., 1., 0.], [0., 0., 0., 1.],
                      [0., 0., 0., 0.], [0., self.config.gravity/self.config.pole_length, 0., 0.]])
        B = np.array([[0.], [0.], [1.], [1/self.config.pole_length]])
        self.assertTrue(np.all(np.real(np.linalg.eigvals(A-B @ self.controller.K)) < 0))
        self.assertLess(self.controller(State(pole_angle=math.pi+.02)), 0.)

    def test_controller_is_quiet_and_does_not_apply_the_working_limiter(self):
        output = io.StringIO()
        with redirect_stdout(output):
            command = self.controller(State(pole_angle=math.pi+.2))
        self.assertEqual(output.getvalue(), '')
        self.assertIsInstance(command, float)
        self.assertGreater(abs(command), 4.)
        log = run_episode(self.controller, State(pole_angle=math.pi+.2), horizon=.01)
        self.assertEqual(log['transitions'][0]['u_requested'], command)
        self.assertEqual(log['transitions'][0]['u_commanded'], -4.)
        self.assertTrue(log['transitions'][0]['action_clipped'])

    def test_ten_second_hold_and_equivalent_short_starts(self):
        log = run_episode(self.controller, State(pole_angle=math.pi+.02))
        validate_log(log)
        self.assertEqual(len(log['transitions']), 1000)
        self.assertEqual(len(log['states']), 1001)
        self.assertEqual(log['completion']['reason'], 'horizon')
        self.assertEqual(log['metrics']['duration'], 10.)
        self.assertLess(log['metrics']['max_abs_position'], .02)
        self.assertLess(log['metrics']['max_abs_velocity'], .05)
        self.assertLess(log['metrics']['max_abs_requested_acceleration'], 1.)
        self.assertLess(abs(log['metrics']['final_angle_error']), 1e-6)
        self.assertTrue(log['metrics']['all_samples_in_upright_region'])
        positive = run_episode(self.controller, State(pole_angle=math.pi+.02), horizon=2.)
        equivalent = run_episode(self.controller, State(pole_angle=-math.pi+.02), horizon=2.)
        negative = run_episode(self.controller, State(pole_angle=math.pi-.02), horizon=2.)
        for other, sign in ((equivalent, 1), (negative, -1)):
            self.assertEqual(other['completion']['reason'], 'horizon')
            for a, b in zip(positive['states'], other['states']):
                expected = [sign*a['cart_position'], sign*a['cart_velocity'],
                            sign*upright_angle_error(a['pole_angle']), sign*a['pole_angular_velocity']]
                actual = [b['cart_position'], b['cart_velocity'],
                          upright_angle_error(b['pole_angle']), b['pole_angular_velocity']]
                np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-11)


class LogTests(unittest.TestCase):
    def test_absolute_horizon_grid_and_short_final_step(self):
        for horizon, count in ((.035, 4), (.07, 7), (0., 0)):
            log = run_episode(lambda state: 1., State(), horizon=horizon)
            validate_log(log)
            self.assertEqual(log['completion']['reason'], 'horizon')
            self.assertFalse(log['completion']['terminated'])
            self.assertTrue(log['completion']['truncated'])
            self.assertEqual(len(log['transitions']), count)
            self.assertEqual(len(log['states']), count+1)
            self.assertEqual(log['states'][-1]['time'], horizon)
            self.assertAlmostEqual(sum(row['dt_actual'] for row in log['transitions']), horizon)
            for row in log['states']:
                self.assertAlmostEqual(row['cart_position'], .5*row['time']**2, delta=1e-12)
                self.assertAlmostEqual(row['cart_velocity'], row['time'], delta=1e-12)
            if horizon == .035:
                self.assertAlmostEqual(log['transitions'][-1]['dt_actual'], .005)

    def test_terminal_at_horizon_takes_priority_over_truncation(self):
        log = run_episode(lambda state: 0., State(cart_position=.24, cart_velocity=1.), horizon=.01)
        validate_log(log)
        self.assertEqual(log['completion']['reason'], 'terminal')
        self.assertEqual(log['completion']['stop_reasons'], ['position_limit'])
        self.assertFalse(log['completion']['truncated'])

    def test_short_and_zero_events_round_trip_without_fictitious_steps(self):
        cases = [(State(cart_position=.249, cart_velocity=.5), 4.),
                 (State(cart_position=.25, cart_velocity=.1), 0.)]
        with tempfile.TemporaryDirectory() as directory:
            for index, (initial, command) in enumerate(cases):
                path = Path(directory)/f'{index}.json'
                original = run_episode(lambda state: command, initial, log_path=path)
                saved = load_log(path)
                self.assertEqual(original, saved)
                self.assertEqual(len(saved['transitions']), 1)
                self.assertEqual(len(saved['states']), 2)
                tr = saved['transitions'][0]
                self.assertLess(tr['dt_actual'], .01)
                self.assertEqual(tr['simulator_error'], 0)
                self.assertEqual(tr['stop_reasons'], ['position_limit'])
                if index == 1:
                    self.assertEqual(tr['dt_actual'], 0.)
                    self.assertIsNone(tr['applied_acceleration'])
                    self.assertEqual(saved['states'][0]['time'], saved['states'][1]['time'])
                else:
                    self.assertAlmostEqual(tr['dt_actual'], .0019842509920029432, delta=1e-12)
                    self.assertEqual(tr['applied_acceleration'], 4.)

    def test_metadata_records_actual_config_integrator_initial_state_and_versions(self):
        initial = State(pole_angle=np.float32(math.pi+.02))
        log = run_episode(lambda state: 0., initial, horizon=.01)
        json.dumps(log, allow_nan=False)  # accepted numpy scalars must serialize
        metadata = log['metadata']
        self.assertEqual(metadata['config'], asdict(make_experiment_config()))
        self.assertEqual(metadata['initial_state'], asdict(initial))
        self.assertEqual(metadata['integrator'], {'type': 'RungeKutta3Integrator',
                         'target_accuracy': 1e-6, 'maximum_step_size': .01, 'fixed_step_mode': False})
        self.assertTrue(all(metadata['versions'].get(name) for name in ('python', 'drake', 'numpy', 'matplotlib', 'Pillow')))

    def test_source_snapshot_identifies_uncommitted_code(self):
        with tempfile.TemporaryDirectory() as directory:
            git = capture_source(directory)
            manifest = Path(directory)/git['source_manifest']
            self.assertEqual(hashlib.sha256(manifest.read_bytes()).hexdigest(), git['source_manifest_sha256'])
            entries = json.loads(manifest.read_text())
            for name in ('cartpole/control/lqr.py', 'cartpole/experiments/lqr_hold.py', 'cartpole/replay.py'):
                original = (ROOT/name).read_bytes()
                self.assertEqual(entries[name], hashlib.sha256(original).hexdigest())
                self.assertEqual((Path(directory)/'source_snapshot'/name).read_bytes(), original)
            self.assertEqual(git['dirty'], bool(git['status_porcelain']))
            self.assertEqual(len(git['sha']), 40)

    def test_corrupt_timing_action_and_terminal_history_are_rejected(self):
        original = run_episode(lambda state: 1., State(), horizon=.02)
        for field, value in [('dt_actual', .02), ('time_end', .015),
                             ('applied_acceleration', None), ('u_commanded', 2.), ('simulator_error', 2)]:
            damaged = copy.deepcopy(original)
            damaged['transitions'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_log(damaged)
        damaged = copy.deepcopy(original)
        damaged['transitions'][0].update(terminated=True, stop_reasons=['position_limit'])
        with self.assertRaises(ValueError):
            validate_log(damaged)

    def test_existing_log_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'episode.json'
            path.write_text('user result')
            controller = unittest.mock.Mock(return_value=0.)
            with self.assertRaises(FileExistsError):
                run_episode(controller, State(), log_path=path)
            controller.assert_not_called()
            self.assertEqual(path.read_text(), 'user result')

    def test_exception_is_recorded_and_reraised_without_fake_transition(self):
        failure = RuntimeError('controller failed')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'failure.json'
            def controller(state):
                raise failure
            with self.assertRaises(RuntimeError) as caught:
                run_episode(controller, State(), log_path=path)
            self.assertIs(caught.exception, failure)
            log = load_log(path)
            self.assertEqual(log['completion']['reason'], 'exception')
            self.assertEqual(log['completion']['exception']['message'], 'controller failed')
            self.assertEqual(len(log['transitions']), 0)
            self.assertEqual(log['completion']['observed_after_exception']['time'], 0.)
            with patch('cartpole.experiments.lqr_hold.save_log', side_effect=OSError('disk error')):
                with self.assertRaises(RuntimeError) as caught:
                    run_episode(controller, State(), log_path=Path(directory)/'other.json')
            self.assertIs(caught.exception, failure)
            self.assertIn('disk error', failure.__notes__[-1])


class ReplayTests(unittest.TestCase):
    def test_geometry_matches_down_up_and_horizontal_conventions(self):
        for theta, expected in ((0., (.1, -.18)), (math.pi, (.1, .18)), (math.pi/2, (.28, 0.))):
            x, y = pole_geometry(State(cart_position=.1, pole_angle=theta), .18)
            np.testing.assert_allclose([x[0], y[0], x[1], y[1]], [.1, 0., *expected], atol=1e-14)

    def test_sampling_uses_log_times_and_does_not_wrap_raw_angle(self):
        log = run_episode(lambda state: 1., State(pole_angle=3*math.pi+.02), horizon=.025)
        self.assertEqual(replay_frame_times(log), [0., .025])
        state, transition, completed = sample_saved_log(log, .015)
        expected = .5*(log['states'][1]['pole_angle']+log['states'][2]['pole_angle'])
        self.assertAlmostEqual(state.pole_angle, expected)
        self.assertGreater(state.pole_angle, 2*math.pi)
        self.assertEqual(transition['index'], 1)
        self.assertFalse(completed)
        last, _, completed = sample_saved_log(log, .025)
        self.assertTrue(completed)
        self.assertEqual(last.cart_position, log['states'][-1]['cart_position'])

    def test_replay_import_does_not_load_drake(self):
        code = 'import sys; import cartpole.replay; assert not any(k.startswith("pydrake") for k in sys.modules)'
        subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, check=True, capture_output=True)

    def test_mp4_preserves_short_last_interval_and_html_is_self_contained(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            log_path = directory/'short.json'
            run_episode(lambda state: 0., State(pole_angle=math.pi), horizon=.045, log_path=log_path)
            output = export_replay(log_path, directory/'replay')
            probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                                                       'frame=pts_time,pkt_duration_time:format=duration', '-of', 'json', str(output/'animation.mp4')]))
            np.testing.assert_allclose([float(frame['pts_time']) for frame in probe['frames']], [0., .04, .045], atol=1e-6)
            self.assertAlmostEqual(float(probe['format']['duration']), .085, delta=1e-6)
            self.assertAlmostEqual(float(probe['frames'][1]['pkt_duration_time']), .005, delta=1e-6)
            document = (output/'animation.html').read_text()
            self.assertIn('<video controls', document)
            self.assertNotIn('https://', document)
            video = re.search(r'src="data:video/mp4;base64,([^"]+)"', document).group(1)
            self.assertEqual(base64.b64decode(video), (output/'animation.mp4').read_bytes())
            with self.assertRaises(FileExistsError):
                export_replay(log_path, output)

    def test_zero_and_short_terminal_events_have_exact_final_replay_state(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for index, initial in enumerate((State(cart_position=.25, cart_velocity=.1),
                                             State(cart_position=.249, cart_velocity=.5))):
                path = directory/f'{index}.json'
                log = run_episode(lambda state: 4., initial, log_path=path)
                end = log['states'][-1]['time']
                state, transition, completed = sample_saved_log(log, end)
                self.assertTrue(completed)
                self.assertEqual(state.cart_position, log['states'][-1]['cart_position'])
                self.assertTrue(transition['terminated'])
                output = export_replay(path, directory/f'replay_{index}')
                probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                                                           'frame=pts_time:format=duration', '-of', 'json', str(output/'animation.mp4')]))
                self.assertAlmostEqual(float(probe['frames'][-1]['pts_time']), end, delta=1e-6)
                self.assertAlmostEqual(float(probe['format']['duration']), end+.04, delta=1e-6)
                self.assertEqual(len(probe['frames']), 1 if end == 0 else 2)


if __name__ == '__main__':
    unittest.main()
