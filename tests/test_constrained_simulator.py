"""Working-region events, lifecycle and Drake equivalence (no RL packages)."""
import dataclasses as dc
import math
import unittest
from unittest.mock import patch

import numpy as np

from cartpole.common import Config, Error, State
from cartpole.experiment_config import CONTROL_INTERVAL, make_experiment_config
from cartpole.simulator.constrained import (
    CartPoleEpisode, POSITION_TOLERANCE, StopReason, TIME_TOLERANCE,
)
from cartpole.simulator.pydrake import CartPoleSimulator


def configure_integrator(backend, accuracy=1e-6):
    integrator = backend.simulator.get_mutable_integrator()
    integrator.set_target_accuracy(accuracy)
    integrator.set_maximum_step_size(.1)
    integrator.set_fixed_step_mode(False)


class ExperimentConfigTests(unittest.TestCase):
    def test_explicit_shared_values_and_independent_instances(self):
        config = make_experiment_config()
        self.assertEqual(
            (config.gravity, config.pole_length, config.max_position,
             config.max_velocity, config.max_acceleration, config.hard_max_position,
             config.hard_max_velocity, config.hard_max_acceleration, CONTROL_INTERVAL),
            (9.8, .18, .25, 2., 4., .27, 2.5, 5., .01),
        )
        config.max_acceleration = 99.
        self.assertEqual(make_experiment_config().max_acceleration, 4.)
        self.assertEqual(Config().max_acceleration, 3.5)
        self.assertEqual(Config().pole_length, .3)


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.episode = CartPoleEpisode()
        self.episode.reset()

    def snapshot(self):
        owner = self.episode
        backend = owner.backend
        return (
            id(backend.context), id(backend.simulator), dc.asdict(backend.get_config()),
            dc.asdict(backend.get_state()), backend.timestamp(), backend.get_target(),
            backend.system.get_input_port().Eval(backend.context).tolist(),
            owner.terminated, owner.stop_reasons, owner.u_requested, owner.u_commanded,
        )

    def assert_atomic_rejection(self, exception, method, *args):
        before = self.snapshot()
        with self.assertRaises(exception):
            method(*args)
        self.assertEqual(self.snapshot(), before)

    def test_step_requires_initial_reset(self):
        episode = CartPoleEpisode()
        with self.assertRaisesRegex(RuntimeError, 'reset'):
            episode.step(0.)
        self.assertEqual(episode.backend.get_state().error, Error.NEED_RESET)

    def test_reset_validates_working_region_strictly_and_is_atomic(self):
        self.episode.step(1.)
        for name, limit in [('cart_position', .25), ('cart_velocity', 2.)]:
            for sign in (-1, 1):
                for value in [sign*np.nextafter(limit, np.inf), sign*(limit+.01)]:
                    with self.subTest(name=name, value=value):
                        self.assert_atomic_rejection(ValueError, self.episode.reset,
                                                     State(**{name: value}))
        for name in ['cart_position', 'cart_velocity', 'pole_angle', 'pole_angular_velocity']:
            for value in [np.nan, np.inf, -np.inf]:
                with self.subTest(name=name, value=value):
                    self.assert_atomic_rejection(ValueError, self.episode.reset,
                                                 State(**{name: value}))
        self.assert_atomic_rejection(TypeError, self.episode.reset, np.zeros(4))
        self.assert_atomic_rejection(TypeError, self.episode.reset, State(cart_position=True))

    def test_reset_accepts_exact_boundaries_and_preserves_all_components(self):
        for sign in (-1, 1):
            initial = State(cart_position=sign*.25, cart_velocity=sign*2.,
                            pole_angle=8., pole_angular_velocity=-.3)
            before = dc.asdict(initial)
            actual = self.episode.reset(initial)
            self.assertEqual(dc.asdict(actual), before)
            self.assertEqual(dc.asdict(initial), before)
            self.assertEqual(self.episode.backend.timestamp(), 0.)
            self.assertFalse(self.episode.terminated)

    def test_limiter_records_request_and_command_without_native_overflow(self):
        for action in [-9., 9., -4., 4., 0, np.float32(.5), np.array([-6.]),
                       np.array([6.]), -1e308, 1e308]:
            with self.subTest(action=action):
                self.episode.reset()
                requested = np.asarray(action).item()
                expected = max(-4., min(requested, 4.))
                result = self.episode.step(action)
                self.assertEqual(result.u_requested, requested)
                self.assertEqual(result.u_commanded, expected)
                self.assertEqual(self.episode.u_requested, requested)
                self.assertEqual(self.episode.u_commanded, expected)
                self.assertEqual(result.action_clipped, requested != expected)
                self.assertEqual(result.applied_acceleration, expected)
                self.assertEqual(result.state.cart_acceleration, expected)
                self.assertEqual(self.episode.backend.get_target(), expected)
                self.assertEqual(self.episode.backend.system.get_input_port().Eval(
                    self.episode.backend.context)[0], expected)
                self.assertEqual(result.simulator_error, Error.NO_ERROR)
                self.assertFalse(result.terminated)
                self.assertEqual(result.dt_actual, CONTROL_INTERVAL)
                self.assertAlmostEqual(result.state.cart_velocity, expected*.01)

    def test_invalid_actions_and_durations_do_not_change_either_layer(self):
        self.episode.step(.5)
        for value in [np.nan, np.inf, -np.inf, np.array([np.nan]),
                      np.array([]), np.array([1., 2.]), np.array([[1.]]), np.array(1.)]:
            with self.subTest(action=value):
                self.assert_atomic_rejection(ValueError, self.episode.step, value)
        for value in [None, True, '1', 1j, [1.], (1.,)]:
            with self.subTest(action=value):
                self.assert_atomic_rejection(TypeError, self.episode.step, value)
        for delta in [-.01, np.nan, np.inf, -np.inf]:
            with self.subTest(delta=delta):
                self.assert_atomic_rejection(ValueError, self.episode.step, 1., delta)
        for delta in [True, '0.01', np.array([.01])]:
            with self.subTest(delta=delta):
                self.assert_atomic_rejection(TypeError, self.episode.step, 1., delta)

    def test_zero_duration_records_command_but_no_new_applied_acceleration(self):
        previous = self.episode.step(1.)
        result = self.episode.step(-2., 0.)
        self.assertEqual(result.timestamp, previous.timestamp)
        self.assertEqual(result.dt_actual, 0.)
        self.assertIsNone(result.applied_acceleration)
        self.assertEqual(dc.asdict(result.state), dc.asdict(previous.state))
        self.assertEqual(result.u_commanded, -2.)
        self.assertEqual(self.episode.backend.get_target(), -2.)
        self.assertFalse(result.terminated)

    def test_post_terminal_guard_and_reset_clear_both_layers(self):
        self.episode.reset(State(cart_position=.249, cart_velocity=1.))
        result = self.episode.step(0.)
        self.assertTrue(result.terminated)
        self.assert_atomic_rejection(RuntimeError, self.episode.step, -4.)
        self.assert_atomic_rejection(RuntimeError, self.episode.step, 0., 0.)
        self.assert_atomic_rejection(ValueError, self.episode.reset, State(cart_position=.26))
        state = self.episode.reset(State(pole_angle=.2))
        self.assertFalse(self.episode.terminated)
        self.assertEqual(self.episode.stop_reasons, ())
        self.assertEqual((self.episode.u_requested, self.episode.u_commanded), (0., 0.))
        self.assertEqual(state.cart_acceleration, 0.)
        self.assertEqual(state.error, Error.NO_ERROR)
        self.assertEqual(self.episode.backend.timestamp(), 0.)
        self.assertFalse(self.episode.step(1.).terminated)

    def test_native_fault_before_step_is_returned_without_movement(self):
        # Normally this owner prevents reaching hard limits. A native fault
        # must still be observable rather than overwritten with a working one.
        for error in [Error.NEED_RESET, Error.X_OVERFLOW, Error.V_OVERFLOW, Error.A_OVERFLOW]:
            with self.subTest(error=error):
                self.episode.reset()
                self.episode.step(.5)
                if error == Error.A_OVERFLOW:
                    self.episode.backend.set_target(6.)
                else:
                    self.episode.backend.error = error
                before = dc.asdict(self.episode.backend.get_state())
                target = self.episode.backend.get_target()
                result = self.episode.step(1.)
                self.assertEqual(result.stop_reasons, (StopReason.SIMULATOR_ERROR,))
                self.assertEqual(result.simulator_error, error)
                self.assertEqual(dc.asdict(result.state), before)
                self.assertEqual(result.dt_actual, 0.)
                self.assertIsNone(result.applied_acceleration)
                self.assertEqual(self.episode.backend.get_target(), target)
                self.assert_atomic_rejection(RuntimeError, self.episode.step, 0.)

    def test_native_fault_after_partial_transition_preserves_actual_time(self):
        self.episode.reset(State(cart_position=.24, cart_velocity=1.))
        original_advance = self.episode.backend.advance

        def advance_with_fault(delta):
            original_advance(delta/2)
            self.episode.backend.error = Error.V_OVERFLOW

        with patch.object(self.episode.backend, 'advance', side_effect=advance_with_fault):
            result = self.episode.step(0.)
        self.assertEqual(result.dt_actual, .005)
        self.assertEqual(result.simulator_error, Error.V_OVERFLOW)
        self.assertEqual(result.stop_reasons, (StopReason.SIMULATOR_ERROR,))
        self.assertEqual(result.boundary_events, ())  # planned position event at .01 was not reached
        self.assertEqual(result.applied_acceleration, 0.)

    def test_integrator_exception_is_reraised_and_latches_episode(self):
        failure = RuntimeError('integrator diagnostic')
        original_advance = self.episode.backend.advance

        def failing_advance(delta):
            original_advance(delta/2)
            raise failure

        with patch.object(self.episode.backend, 'advance', side_effect=failing_advance):
            with self.assertRaises(RuntimeError) as caught:
                self.episode.step(1.)
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.episode.backend.timestamp(), .005)
        self.assertTrue(self.episode.terminated)
        self.assertEqual(self.episode.stop_reasons, (StopReason.SIMULATOR_EXCEPTION,))
        self.assert_atomic_rejection(RuntimeError, self.episode.step, 0.)
        self.episode.reset()
        self.assertFalse(self.episode.step(0.).terminated)


class BoundaryTests(unittest.TestCase):
    def run_step(self, x, v, u, delta):
        self.initial = State(cart_position=x, cart_velocity=v,
                             pole_angle=.2, pole_angular_velocity=.3)
        self.episode = CartPoleEpisode()
        self.episode.reset(self.initial)
        configure_integrator(self.episode.backend)
        return self.episode.step(u, delta)

    def assert_event(self, result, time, reasons, side):
        self.assertTrue(result.terminated)
        self.assertEqual(result.stop_reasons, reasons)
        self.assertEqual(result.simulator_error, Error.NO_ERROR)
        self.assertAlmostEqual(result.dt_actual, time, delta=1e-12)
        self.assertEqual(result.timestamp, result.dt_actual)
        self.assertTrue(all(event.side == side for event in result.boundary_events))
        for event in result.boundary_events:
            self.assertAlmostEqual(event.time, time, delta=1e-12)
        expected_x = self.initial.cart_position + self.initial.cart_velocity*time + .5*result.u_commanded*time**2
        expected_v = self.initial.cart_velocity + result.u_commanded*time
        self.assertAlmostEqual(result.state.cart_position, expected_x, delta=1e-12)
        self.assertAlmostEqual(result.state.cart_velocity, expected_v, delta=1e-12)
        # The pendulum at a shortened event step must come from the very same
        # Drake trajectory, not a separate cart/pendulum integrator.
        direct = CartPoleSimulator()
        direct.reset_to(make_experiment_config(), self.initial)
        configure_integrator(direct)
        direct.set_target(result.u_commanded)
        if result.dt_actual:
            direct.advance(result.dt_actual)
        np.testing.assert_array_equal(result.state.as_array(), direct.get_state().as_array())

    def test_position_crossing_at_constant_velocity_both_directions(self):
        for side in (-1, 1):
            result = self.run_step(side*.245, side*1., 0., .01)
            self.assert_event(result, .005, (StopReason.POSITION_LIMIT,), side)

    def test_accelerated_position_crossing_inside_control_interval(self):
        expected_time = 2*.001 / (.5 + math.sqrt(.5**2 + 2*4*.001))
        for side in (-1, 1):
            result = self.run_step(side*.249, side*.5, side*4., .01)
            self.assert_event(result, expected_time, (StopReason.POSITION_LIMIT,), side)
            self.assertAlmostEqual(result.state.cart_position, side*.25, delta=1e-12)

    def test_velocity_crossing_both_directions(self):
        for side in (-1, 1):
            result = self.run_step(0., side*1.99, side*4., .01)
            self.assert_event(result, .0025, (StopReason.VELOCITY_LIMIT,), side)

    def test_exit_and_return_inside_interval_is_detected(self):
        expected_time = (.3 - math.sqrt(.3**2 - 4*.01)) / 2
        for side in (-1, 1):
            # x(.3)=x(0)=.24*side; the inner extremum is .2625*side.
            result = self.run_step(side*.24, side*.3, -side*2., .3)
            self.assert_event(result, expected_time, (StopReason.POSITION_LIMIT,), side)

    def test_touch_and_return_inside_is_allowed(self):
        for side in (-1, 1):
            result = self.run_step(side*.125, side*1., -side*4., .5)
            self.assertFalse(result.terminated)
            self.assertEqual(result.boundary_events, ())
            self.assertEqual(result.dt_actual, .5)
            self.assertAlmostEqual(result.state.cart_position, side*.125, delta=1e-12)
            self.assertAlmostEqual(result.state.cart_velocity, -side*1., delta=1e-12)

    def test_touch_at_end_and_following_inward_step_are_allowed(self):
        for side in (-1, 1):
            result = self.run_step(side*.125, side*1., -side*4., .25)
            self.assertFalse(result.terminated)
            self.assertAlmostEqual(result.state.cart_position, side*.25, delta=1e-12)
            self.assertAlmostEqual(result.state.cart_velocity, 0., delta=1e-12)
            following = self.episode.step(-side*4.)
            self.assertFalse(following.terminated)
            self.assertLess(abs(following.state.cart_position), .25)

    def test_zero_time_event_keeps_previous_interval_acceleration(self):
        for side in (-1, 1):
            previous = self.run_step(side*.125, side*1., -side*4., .25)
            result = self.episode.step(side*4.)
            self.assertTrue(result.terminated)
            self.assertEqual(result.stop_reasons, (StopReason.POSITION_LIMIT,))
            self.assertEqual(result.dt_actual, 0.)
            self.assertEqual(result.timestamp, previous.timestamp)
            self.assertEqual(dc.asdict(result.state), dc.asdict(previous.state))
            self.assertEqual(result.state.cart_acceleration, -side*4.)
            self.assertIsNone(result.applied_acceleration)
            self.assertEqual(result.u_commanded, side*4.)

    def test_position_boundary_outward_motion_or_acceleration_stops_at_zero(self):
        for side in (-1, 1):
            for v, u in [(side*.1, -side*4.), (0., side*1.)]:
                result = self.run_step(side*.25, v, u, .01)
                self.assert_event(result, 0., (StopReason.POSITION_LIMIT,), side)
                self.assertIsNone(result.applied_acceleration)
                self.assertEqual(result.state.cart_acceleration, 0.)
                self.assertEqual(self.episode.backend.get_target(), u)

    def test_velocity_boundary_outward_acceleration_stops_at_zero(self):
        for side in (-1, 1):
            result = self.run_step(0., side*2., side*1., .01)
            self.assert_event(result, 0., (StopReason.VELOCITY_LIMIT,), side)

    def test_boundary_inward_motion_and_stationary_position_are_allowed(self):
        for side in (-1, 1):
            for x, v, u in [(side*.25, -side*.1, 0.), (side*.25, 0., -side*1.),
                            (side*.25, 0., 0.), (0., side*2., -side*1.), (0., side*2., 0.)]:
                with self.subTest(side=side, x=x, v=v, u=u):
                    result = self.run_step(x, v, u, .01)
                    self.assertFalse(result.terminated)
                    self.assertEqual(result.dt_actual, .01)

    def test_start_inward_then_return_outward_to_same_boundary(self):
        for side in (-1, 1):
            result = self.run_step(side*.25, -side*.2, side*2., .3)
            self.assert_event(result, .2, (StopReason.POSITION_LIMIT,), side)

    def test_outward_crossings_exactly_at_end_are_terminal(self):
        for side in (-1, 1):
            position = self.run_step(side*.24, side*1., 0., .01)
            self.assert_event(position, .01, (StopReason.POSITION_LIMIT,), side)
            velocity = self.run_step(0., side*1.96, side*4., .01)
            self.assert_event(velocity, .01, (StopReason.VELOCITY_LIMIT,), side)

    def test_simultaneous_position_and_velocity_events_preserve_both_reasons(self):
        for side in (-1, 1):
            result = self.run_step(side*.2302, side*1.96, side*4., .02)
            self.assert_event(result, .01,
                              (StopReason.POSITION_LIMIT, StopReason.VELOCITY_LIMIT), side)
            self.assertEqual(len(result.boundary_events), 2)

    def test_simultaneous_zero_time_events_and_zero_requested_duration(self):
        for side in (-1, 1):
            for delta in (0., .01):
                result = self.run_step(side*.25, side*2., side*1., delta)
                self.assert_event(result, 0.,
                                  (StopReason.POSITION_LIMIT, StopReason.VELOCITY_LIMIT), side)
                self.assertIsNone(result.applied_acceleration)

    def test_first_event_wins_when_later_event_is_also_in_requested_interval(self):
        for side in (-1, 1):
            velocity = self.run_step(side*.24, side*1.99, side*4., .01)
            self.assert_event(velocity, .0025, (StopReason.VELOCITY_LIMIT,), side)
            position = self.run_step(side*.249, side*1.99, side*4., .01)
            expected_time = 2*.001 / (1.99 + math.sqrt(1.99**2 + 8*.001))
            self.assert_event(position, expected_time, (StopReason.POSITION_LIMIT,), side)

    def test_geometric_tolerance_distinguishes_touch_from_resolved_excursion(self):
        for offset in (0., POSITION_TOLERANCE/2, POSITION_TOLERANCE*10):
            result = self.run_step(.125+offset, 1., -4., .5)
            self.assertEqual(result.terminated, offset > POSITION_TOLERANCE)
            if result.terminated:
                expected_time = (1 - math.sqrt(8*offset))/4
                self.assertAlmostEqual(result.dt_actual, expected_time, delta=1e-10)

    def test_endpoint_time_tolerance_does_not_pull_in_later_events(self):
        near = self.run_step(.24, 1., 0., .01-TIME_TOLERANCE/2)
        self.assertTrue(near.terminated)
        self.assertEqual(near.dt_actual, near.dt_requested)
        later = self.run_step(.24, 1., 0., .01-TIME_TOLERANCE*10)
        self.assertFalse(later.terminated)

    def test_almost_zero_acceleration_keeps_constant_velocity_event_time(self):
        for side in (-1, 1):
            for acceleration in (1e-200, -1e-200):
                result = self.run_step(side*.245, side*1., side*acceleration, .01)
                self.assert_event(result, .005, (StopReason.POSITION_LIMIT,), side)

    def test_native_error_at_boundary_preserves_native_and_working_reasons(self):
        episode = CartPoleEpisode()
        episode.reset(State(cart_position=.249, cart_velocity=1.))
        original_advance = episode.backend.advance

        def advance_with_fault(delta):
            original_advance(delta)
            episode.backend.error = Error.X_OVERFLOW

        with patch.object(episode.backend, 'advance', side_effect=advance_with_fault):
            result = episode.step(0.)
        self.assertEqual(result.stop_reasons,
                         (StopReason.SIMULATOR_ERROR, StopReason.POSITION_LIMIT))
        self.assertEqual(result.simulator_error, Error.X_OVERFLOW)


class DirectDrakeRegressionTests(unittest.TestCase):
    def test_allowed_transitions_match_direct_drake_at_same_integrator_settings(self):
        for accuracy in (1e-4, 1e-6):
            with self.subTest(accuracy=accuracy):
                backend = CartPoleSimulator()
                episode = CartPoleEpisode(backend)
                initial = State(pole_angle=.2, pole_angular_velocity=.1)
                episode.reset(initial)
                direct = CartPoleSimulator()
                direct.reset_to(make_experiment_config(), initial)
                configure_integrator(backend, accuracy)
                configure_integrator(direct, accuracy)
                for action in [0.]*50 + [1., -.5, .25, -.75]*20:
                    result = episode.step(action)
                    direct.set_target(action)
                    direct.advance(CONTROL_INTERVAL)
                    self.assertFalse(result.terminated)
                    self.assertFalse(result.action_clipped)
                    self.assertEqual(result.timestamp, direct.timestamp())
                    self.assertEqual(dc.asdict(result.state), dc.asdict(direct.get_state()))


if __name__ == '__main__':
    unittest.main()
