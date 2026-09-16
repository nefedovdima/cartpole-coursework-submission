"""Interface and physical regressions for the original Drake backend.

Run from the repository root:
    .venv/bin/python -B -m unittest discover -s tests -v

The trajectory comparisons load the unmodified wrapper from the pinned audit
commit via read-only git show. They need that Git object, not ignored CSV files,
hardware, controllers, pytest or any installed RL package.
"""
import dataclasses as dc
import math
from pathlib import Path
import subprocess
import types
import unittest

import numpy as np

from cartpole.common import Config, Error, State
from cartpole.simulator.pydrake import CartPoleSimulator


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = 'e50dd1779695af12fa7d7156c5d2ba025dd8cd52'


class SimulatorInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.sim = CartPoleSimulator()
        self.config = Config(pole_length=.18)
        self.sim.reset(self.config)

    def snapshot(self):
        """Include the live objects, all telemetry, parameters and input port."""
        sim = self.sim
        state = sim.get_state()
        return dict(
            context_id=id(sim.context), simulator_id=id(sim.simulator),
            config_id=id(sim.config),
            config=dc.asdict(sim.get_config()) if sim.config is not None else None,
            parameters=sim.context.get_numeric_parameter(0).CopyToVector().tolist(),
            t=sim.timestamp(), info=sim.get_info(), q=state.as_array().tolist(),
            error=sim.error, state_error=state.error,
            acceleration=state.cart_acceleration, target=sim.get_target(),
            port=sim.system.get_input_port().Eval(sim.context).tolist()
            if sim.config is not None else None,
        )

    def assert_rejected_unchanged(self, exception, message, method, *args):
        before = self.snapshot()
        with self.assertRaisesRegex(exception, message):
            method(*args)
        self.assertEqual(self.snapshot(), before)

    def warm_up(self):
        self.sim.set_target(.75)
        self.sim.advance(.05)

    def assert_fault_is_latched(self, error):
        self.assertEqual(self.sim.error, error)
        self.assertEqual(self.sim.get_state().error, error)
        before = self.snapshot()
        with self.assertLogs('simulator', level='WARNING'):
            self.sim.set_target(-.5)
            self.sim.advance(.01)
            self.sim.advance(0)
        self.assertEqual(self.snapshot(), before)

    def test_need_reset_is_visible_and_blocks_valid_commands(self):
        self.sim = CartPoleSimulator()
        self.assertEqual(self.sim.get_state().error, Error.NEED_RESET)
        self.assertEqual(self.sim.get_state().cart_acceleration, 0.)
        self.assertEqual(self.sim.get_target(), 0.)
        self.assertEqual(self.sim.get_info(), {'time': 0.})
        self.assert_fault_is_latched(Error.NEED_RESET)

    def test_reset_preserves_order_and_clears_telemetry(self):
        self.warm_up()
        initial = State(cart_position=.12, cart_velocity=-.34,
                        pole_angle=1.23 + 4 * math.pi, pole_angular_velocity=4.56,
                        error=Error.X_OVERFLOW, cart_acceleration=99.)
        initial_before = dc.asdict(initial)
        self.sim.reset_to(self.config, initial)
        actual = self.sim.get_state()
        np.testing.assert_array_equal(actual.as_array(), [.12, 1.23 + 4 * math.pi, -.34, 4.56])
        self.assertEqual(actual.as_array().shape, (4,))
        self.assertEqual(actual.error, Error.NO_ERROR)
        self.assertEqual(actual.cart_acceleration, 0.)
        self.assertEqual(self.sim.timestamp(), 0.)
        self.assertEqual(self.sim.get_target(), 0.)
        self.assertEqual(self.sim.system.get_input_port().Eval(self.sim.context)[0], 0.)
        self.assertEqual(dc.asdict(initial), initial_before)

    def test_reset_after_each_overflow_starts_a_fresh_episode(self):
        cases = [
            (State(cart_position=.269, cart_velocity=1.), .5, .01, Error.X_OVERFLOW),
            (State(cart_velocity=2.49), 2., .01, Error.V_OVERFLOW),
            (State.home(), 6., None, Error.A_OVERFLOW),
        ]
        for initial, target, dt, error in cases:
            with self.subTest(error=error):
                self.sim.reset_to(self.config, initial)
                self.sim.set_target(target)
                if dt is not None:
                    self.sim.advance(dt)
                self.assertEqual(self.sim.get_state().error, error)
                self.sim.reset(self.config)
                np.testing.assert_array_equal(self.sim.get_state().as_array(), np.zeros(4))
                self.assertEqual(self.sim.get_state().error, Error.NO_ERROR)
                self.assertEqual(self.sim.get_state().cart_acceleration, 0.)
                self.assertEqual(self.sim.get_target(), 0.)
                self.assertEqual(self.sim.timestamp(), 0.)
                self.sim.set_target(1.)
                self.sim.advance(.02)
                self.assertAlmostEqual(self.sim.get_state().cart_velocity, .02)
                self.assertEqual(self.sim.get_state().cart_acceleration, 1.)

    def test_pending_command_is_not_last_applied_acceleration(self):
        self.sim.set_target(1.)
        self.assertEqual(self.sim.get_state().cart_acceleration, 0.)
        self.sim.advance(.02)
        self.assertEqual(self.sim.get_state().cart_acceleration, 1.)
        self.sim.set_target(-2.)
        self.assertEqual(self.sim.get_target(), -2.)
        before = self.snapshot()
        self.sim.advance(0.)
        self.assertEqual(self.snapshot(), before)
        # A positive delta too small to change representable time is not a step.
        self.sim.advance(np.nextafter(0., 1.))
        self.assertEqual(self.snapshot(), before)
        self.sim.advance(.01)
        self.assertEqual(self.sim.get_state().cart_acceleration, -2.)
        self.assertAlmostEqual(self.sim.get_state().cart_velocity, 0.)

    def test_numeric_scalars_and_one_element_arrays_share_the_same_input(self):
        for target in [1, -.5, np.float32(.25), np.float64(-.75), np.int64(2),
                       np.array([1.25]), np.array([-2], dtype=np.int64)]:
            with self.subTest(target=target):
                self.sim.reset(self.config)
                expected = float(target[0]) if isinstance(target, np.ndarray) else float(target)
                self.sim.set_target(target)
                self.assertIsInstance(self.sim.get_target(), float)
                self.assertEqual(self.sim.get_target(), expected)
                self.assertEqual(self.sim.system.get_input_port().Eval(self.sim.context).shape, (1,))
                self.assertEqual(self.sim.system.get_input_port().Eval(self.sim.context)[0], expected)
                self.sim.advance(.01)
                self.assertAlmostEqual(self.sim.get_state().cart_velocity, expected * .01)
                self.assertEqual(self.sim.get_state().cart_acceleration, expected)

    def test_acceleration_overflow_clamps_port_but_never_executes_command(self):
        for target in [6., -6., np.array([6.]), np.array([-6.])]:
            with self.subTest(target=target):
                self.sim.reset(self.config)
                self.warm_up()
                state_before = self.sim.get_state().as_array()
                time_before = self.sim.timestamp()
                self.sim.set_target(target)
                expected = 5. if np.asarray(target).item() > 0 else -5.
                self.assertEqual(self.sim.get_target(), expected)
                self.assertEqual(self.sim.system.get_input_port().Eval(self.sim.context)[0], expected)
                self.assert_fault_is_latched(Error.A_OVERFLOW)
                np.testing.assert_array_equal(self.sim.get_state().as_array(), state_before)
                self.assertEqual(self.sim.timestamp(), time_before)
                self.assertEqual(self.sim.get_state().cart_acceleration, .75)

    def test_exact_hard_acceleration_boundary_is_allowed(self):
        for target in [-5., 5.]:
            with self.subTest(target=target):
                self.sim.reset(self.config)
                self.sim.set_target(target)
                self.sim.advance(.01)
                self.assertEqual(self.sim.get_state().error, Error.NO_ERROR)
                self.assertEqual(self.sim.get_target(), target)
                self.assertEqual(self.sim.get_state().cart_acceleration, target)
                self.assertAlmostEqual(self.sim.get_state().cart_velocity, .01 * target)

    def test_position_overflow_reports_executed_acceleration_and_freezes(self):
        for sign in [-1, 1]:
            with self.subTest(sign=sign):
                self.sim.reset_to(self.config, State(cart_position=sign*.269, cart_velocity=sign*1.))
                self.sim.set_target(sign*.5)
                self.sim.advance(.01)
                self.assertGreater(abs(self.sim.get_state().cart_position), .27)
                self.assertEqual(self.sim.get_state().cart_acceleration, sign*.5)
                self.assert_fault_is_latched(Error.X_OVERFLOW)

    def test_velocity_overflow_reports_executed_acceleration_and_freezes(self):
        for sign in [-1, 1]:
            with self.subTest(sign=sign):
                self.sim.reset_to(self.config, State(cart_velocity=sign*2.49))
                self.sim.set_target(sign*2.)
                self.sim.advance(.01)
                self.assertAlmostEqual(self.sim.get_state().cart_velocity, sign*2.51)
                self.assertEqual(self.sim.get_state().cart_acceleration, sign*2.)
                self.assert_fault_is_latched(Error.V_OVERFLOW)

    def test_invalid_actions_are_atomic(self):
        self.warm_up()
        cases = [
            (ValueError, float('nan')), (ValueError, float('inf')),
            (ValueError, -float('inf')), (ValueError, np.array([np.nan])),
            (ValueError, np.array([np.inf])), (ValueError, np.array([-np.inf])),
            (ValueError, np.array([])), (ValueError, np.array([1., 2.])),
            (ValueError, np.array([[1.]])), (ValueError, np.array(1.)),
            (ValueError, 10**1000), (TypeError, '1'), (TypeError, None),
            (TypeError, True), (TypeError, np.bool_(False)),
            (TypeError, 1+0j), (TypeError, [1.]), (TypeError, (1.,)),
            (TypeError, np.array(['1'])), (TypeError, np.array([True])),
            (TypeError, np.array([1+0j])),
        ]
        for index, (exception, value) in enumerate(cases):
            with self.subTest(case=index):
                self.assert_rejected_unchanged(exception, 'target', self.sim.set_target, value)

    def test_invalid_deltas_are_atomic_and_do_not_reach_drake(self):
        self.warm_up()
        for index, value in enumerate([-1., -.01, np.nan, np.inf, -np.inf, 10**1000]):
            with self.subTest(case=index):
                self.assert_rejected_unchanged(ValueError, 'delta', self.sim.advance, value)
        for value in ['.01', None, True, 1+0j, np.array([.01]), np.array(.01)]:
            with self.subTest(value=value):
                self.assert_rejected_unchanged(TypeError, 'delta', self.sim.advance, value)
        self.sim.advance(np.float64(.01))
        self.assertAlmostEqual(self.sim.timestamp(), .06)

    def test_finite_delta_cannot_overflow_absolute_time(self):
        self.sim.context.SetTime(np.finfo(float).max)
        self.assert_rejected_unchanged(ValueError, 'time', self.sim.advance, np.finfo(float).max)

    def test_invalid_arguments_are_rejected_even_before_reset_and_after_fault(self):
        for uninitialized in [True, False]:
            with self.subTest(uninitialized=uninitialized):
                self.sim = CartPoleSimulator()
                if not uninitialized:
                    self.sim.reset(self.config)
                    self.warm_up()
                    self.sim.set_target(6.)
                self.assert_rejected_unchanged(ValueError, 'target', self.sim.set_target, np.nan)
                self.assert_rejected_unchanged(ValueError, 'delta', self.sim.advance, -.01)
                self.assert_rejected_unchanged(ValueError, 'pole_angle', self.sim.reset_to,
                                               self.config, State(pole_angle=np.nan))

    def test_invalid_initial_components_are_atomic(self):
        self.warm_up()
        for field in ['cart_position', 'pole_angle', 'cart_velocity', 'pole_angular_velocity']:
            for value in [np.nan, np.inf, -np.inf]:
                with self.subTest(field=field, value=value):
                    self.assert_rejected_unchanged(ValueError, field, self.sim.reset_to,
                                                   self.config, State(**{field: value}))
            for value in ['0', True, 0j, np.array([0.])]:
                with self.subTest(field=field, value=value):
                    self.assert_rejected_unchanged(TypeError, field, self.sim.reset_to,
                                                   self.config, State(**{field: value}))

    def test_initial_position_and_velocity_must_fit_hard_limits(self):
        self.warm_up()
        for field, bound in [('cart_position', .27), ('cart_velocity', 2.5)]:
            for sign in [-1, 1]:
                with self.subTest(field=field, sign=sign):
                    value = sign * np.nextafter(bound, np.inf)
                    self.assert_rejected_unchanged(ValueError, field, self.sim.reset_to,
                                                   self.config, State(**{field: value}))
                    self.sim.reset_to(self.config, State(**{field: sign*bound}))
                    self.assertEqual(getattr(self.sim.get_state(), field), sign*bound)
                    self.assertEqual(self.sim.get_state().error, Error.NO_ERROR)

    def test_used_config_parameters_must_be_positive_finite_scalars(self):
        self.warm_up()
        for field in ['gravity', 'pole_length', 'hard_max_position',
                      'hard_max_velocity', 'hard_max_acceleration']:
            for value in [0., -1., np.nan, np.inf, -np.inf]:
                with self.subTest(field=field, value=value):
                    self.assert_rejected_unchanged(ValueError, field, self.sim.reset,
                                                   dc.replace(self.config, **{field: value}))
            for value in ['1', True, 1j, np.array([1.])]:
                with self.subTest(field=field, value=value):
                    self.assert_rejected_unchanged(TypeError, field, self.sim.reset,
                                                   dc.replace(self.config, **{field: value}))

    def test_invalid_reset_types_do_not_replace_live_objects(self):
        self.warm_up()
        self.assert_rejected_unchanged(TypeError, 'config', self.sim.reset, None)
        self.assert_rejected_unchanged(TypeError, 'state', self.sim.reset_to, self.config, np.zeros(4))

    def test_config_snapshot_keeps_validated_limits_and_drake_parameters_consistent(self):
        self.config.gravity = -1.
        self.config.pole_length = .3
        self.config.hard_max_acceleration = .1
        exported = self.sim.get_config()
        exported.hard_max_position = 0.
        self.assertEqual(self.sim.get_config().pole_length, .18)
        self.assertEqual(self.sim.get_config().hard_max_position, .27)
        np.testing.assert_array_equal(self.sim.context.get_numeric_parameter(0).CopyToVector(), [9.8, .18])
        self.sim.set_target(1.)
        self.sim.advance(.01)
        self.assertEqual(self.sim.get_state().error, Error.NO_ERROR)
        self.assertEqual(self.sim.get_state().cart_acceleration, 1.)

    def test_custom_config_values_are_used_without_changing_working_limits(self):
        config = Config(pole_length=np.float64(.21), gravity=np.float64(9.7),
                        hard_max_position=.4, hard_max_velocity=3., hard_max_acceleration=6.)
        self.sim.reset_to(config, State(cart_position=.3, cart_velocity=2.6))
        np.testing.assert_array_equal(self.sim.context.get_numeric_parameter(0).CopyToVector(), [9.7, .21])
        self.sim.set_target(5.5)
        self.sim.advance(.001)
        self.assertEqual(self.sim.get_state().error, Error.NO_ERROR)
        self.assertEqual(self.sim.get_state().cart_acceleration, 5.5)

    def test_get_info_and_timestamp_agree_without_extra_steps(self):
        for delta in [0., .01, .003, .017]:
            self.sim.advance(delta)
            self.assertEqual(self.sim.get_info(), {'time': self.sim.timestamp()})
        self.assertAlmostEqual(self.sim.timestamp(), .03)
        self.sim.set_target(6.)
        self.assertEqual(self.sim.get_info(), {'time': self.sim.timestamp()})


class SimulatorPhysicsRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = subprocess.check_output(
            ['git', 'show', f'{BASELINE_COMMIT}:cartpole/simulator/pydrake/simulator.py'],
            cwd=ROOT, text=True,
        )
        module = types.ModuleType('audit_baseline_simulator')
        exec(compile(source, f'{BASELINE_COMMIT}:simulator.py', 'exec'), module.__dict__)
        cls.baseline = module.CartPoleSimulator

    @staticmethod
    def trajectory(simulator_class, config, initial, actions, dt, accuracy):
        sim = simulator_class()
        sim.reset_to(config, initial)
        integrator = sim.simulator.get_mutable_integrator()
        integrator.set_target_accuracy(accuracy)
        integrator.set_maximum_step_size(.1)
        integrator.set_fixed_step_mode(False)
        rows = [[sim.timestamp(), *sim.get_state().as_array()]]
        for target in actions:
            sim.set_target(target)
            sim.advance(dt)
            if sim.error:
                raise AssertionError(f'unexpected physical fault: {sim.error}')
            rows.append([sim.timestamp(), *sim.get_state().as_array()])
        return np.array(rows)

    def assert_matches_baseline(self, initial, actions, dt, accuracy=1e-4, length=.18):
        config = Config(pole_length=length)
        old = self.trajectory(self.baseline, config, initial, actions, dt, accuracy)
        new = self.trajectory(CartPoleSimulator, config, initial, actions, dt, accuracy)
        # Physical states/times should agree; telemetry intentionally differs.
        np.testing.assert_array_equal(new, old)
        return new

    def test_free_oscillations_match_original_at_every_sample(self):
        data = self.assert_matches_baseline(State(pole_angle=.2), [0.]*500, .01)
        self.assertEqual(data.shape, (501, 5))
        self.assertAlmostEqual(data[-1, 0], 5.)
        np.testing.assert_array_equal(data[:, [1, 3]], np.zeros((501, 2)))
        np.testing.assert_allclose(data[-1, [2, 4]], [.1246237177599799, 1.150401299685667],
                                   rtol=0, atol=1e-8)
        # Check conserved energy to the measured default-integrator accuracy.
        energy = .5*.18**2*data[:, 4]**2 + 9.8*.18*(1-np.cos(data[:, 2]))
        self.assertLess(np.max(np.abs(energy-energy[0]))/energy[0], .002)

    def test_constant_acceleration_matches_original_and_analytic_cart_motion(self):
        for target in [-1., 1.]:
            with self.subTest(target=target):
                data = self.assert_matches_baseline(State.home(), [target]*20, .01)
                np.testing.assert_allclose(data[:, 1], .5*target*data[:, 0]**2, rtol=0, atol=1e-12)
                np.testing.assert_allclose(data[:, 3], target*data[:, 0], rtol=0, atol=1e-12)
                self.assertLess(data[-1, 2]*target, 0.)

    def test_alternating_commands_match_original_with_same_stricter_integrator(self):
        self.assert_matches_baseline(State(pole_angle=.2), [1., -.5, .25, -.75]*50,
                                     .01, accuracy=1e-6)

    def test_default_length_equilibria_and_full_rotation_match_original(self):
        for angle, omega in [(0., 0.), (math.pi, 0.), (6., 20.)]:
            with self.subTest(angle=angle, omega=omega):
                data = self.assert_matches_baseline(State(pole_angle=angle, pole_angular_velocity=omega),
                                                     [0.]*10, .01, length=.3)
                if omega:
                    self.assertGreater(data[-1, 2], 2*math.pi)
                else:
                    np.testing.assert_allclose(data[:, 2], angle, rtol=0, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
