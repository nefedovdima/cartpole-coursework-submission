"""Mathematical tests of fixed R1; physical transition checks live separately."""
from dataclasses import FrozenInstanceError, replace
import json
import math
import unittest
from unittest.mock import patch

from cartpole.common import State
from cartpole.experiment_config import make_experiment_config
from cartpole.rl.pilot_config import make_pilot_config, make_pilot_env
from cartpole.rl.reward import RewardR0
from cartpole.rl.reward_pilot import RewardR1


class PilotRewardTests(unittest.TestCase):
    def setUp(self):
        self.reward = RewardR1()
        self.config = make_experiment_config()
        self.top = State(pole_angle=math.pi)

    def value(self, state=None, u=0., dt=.01, terminated=False):
        return self.reward.evaluate(self.top if state is None else state,
                                    u, dt, terminated, self.config)

    def test_fixed_version_and_independent_metadata(self):
        with self.assertRaises(TypeError):
            RewardR1(failure=100.)
        with self.assertRaises(FrozenInstanceError):
            self.reward.version = 'silent-change'
        metadata = self.reward.metadata()
        metadata['weights']['failure'] = 100.
        self.assertEqual(self.reward.metadata()['weights']['failure'], 5.)
        self.assertEqual(self.reward.version, 'R1-v0')

    def test_known_equilibria_and_interpretable_scales(self):
        self.assertEqual(self.value()[0], 1.)
        self.assertEqual(self.value(State.home())[0], 0.)
        self.assertEqual(self.value(State(cart_position=.24))[0], 0.)
        for field, value in (('cart_position', .25), ('cart_velocity', 2.),
                             ('pole_angular_velocity', 2.)):
            with self.subTest(field=field):
                self.assertAlmostEqual(self.value(replace(self.top, **{field: value}))[0], .5)
        self.assertAlmostEqual(self.value(u=4.)[0], 1/1.01)
        self.assertAlmostEqual(self.value(replace(self.top, pole_angular_velocity=.5))[0], 1/1.0625)
        self.assertAlmostEqual(self.value(replace(self.top, pole_angular_velocity=20.))[0], 1/101)

    def test_periodicity_and_even_cost_factors(self):
        state = State(cart_position=.1, cart_velocity=.3, pole_angle=1.2,
                      pole_angular_velocity=2.3)
        base = self.value(state, u=1.5)[0]
        for turns in (-5, -1, 1, 5):
            self.assertAlmostEqual(self.value(replace(state, pole_angle=1.2+turns*2*math.pi), u=1.5)[0], base)
        mirror = State(cart_position=-.1, cart_velocity=-.3, pole_angle=-1.2,
                       pole_angular_velocity=-2.3)
        self.assertEqual(self.value(mirror, u=-1.5)[0], base)

    def test_short_duration_scales_product_but_not_failure(self):
        state = replace(self.top, cart_position=.1, pole_angular_velocity=.7)
        full, _, _ = self.value(state, u=2.)
        short, terms, scale = self.value(state, u=2., dt=.0025, terminated=True)
        self.assertEqual(scale, .25)
        self.assertAlmostEqual(short, .25*full-5.)
        self.assertEqual(terms['failure'], -5.)
        self.assertEqual(set(terms), {'alignment', 'failure'})
        self.assertEqual(short, math.fsum(terms.values()))

    def test_zero_transition_has_no_invented_input(self):
        with patch.object(RewardR1, 'factors', side_effect=AssertionError('no input on zero dt')):
            self.assertEqual(self.value(u=None, dt=0.), (0., {'alignment': 0., 'failure': 0.}, 0.))
            self.assertEqual(self.value(u=None, dt=0., terminated=True),
                             (-5., {'alignment': 0., 'failure': -5.}, 0.))
        for u, dt in ((0., 0.), (None, .01)):
            with self.subTest(u=u, dt=dt), self.assertRaises(ValueError):
                self.value(u=u, dt=dt)

    def test_finite_extremes_do_not_overflow_or_leave_bounds(self):
        for magnitude in (1e-300, 1., 1e20, 1e100, 1e308):
            for sign in (-1, 1):
                state = State(cart_position=sign*magnitude, cart_velocity=sign*magnitude,
                              pole_angle=sign*magnitude, pole_angular_velocity=sign*magnitude)
                with self.subTest(magnitude=magnitude, sign=sign):
                    factors = self.reward.factors(state, sign*magnitude)
                    self.assertTrue(all(math.isfinite(x) and 0 <= x <= 1 for x in factors.values()))
                    value, terms, _ = self.value(state, u=sign*magnitude)
                    self.assertTrue(math.isfinite(value))
                    self.assertTrue(0 <= value <= 1)
                    self.assertEqual(value, terms['alignment'])

    def test_factors_decrease_monotonically_with_absolute_cost_arguments(self):
        for field, factor in (('cart_position', 'position'), ('cart_velocity', 'velocity'),
                              ('pole_angular_velocity', 'angular_velocity')):
            values = [self.reward.factors(replace(self.top, **{field: x}), 0.)[factor]
                      for x in (0., .1, 1., 10., 1e308)]
            self.assertEqual(values, sorted(values, reverse=True))
            self.assertEqual(values[0], 1.)
        effort = [self.reward.factors(self.top, x)['acceleration'] for x in (0., 1., 4., 40., 1e308)]
        self.assertEqual(effort, sorted(effort, reverse=True))

    def test_nonfinite_physical_states_inputs_and_durations_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            for field in ('cart_position', 'cart_velocity', 'pole_angle', 'pole_angular_velocity'):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.value(replace(self.top, **{field: value}))
                with self.subTest(field=field, dt=0.), self.assertRaises(ValueError):
                    self.value(replace(self.top, **{field: value}), u=None, dt=0.)
            with self.assertRaises(ValueError):
                self.value(u=value)
            with self.assertRaises(ValueError):
                self.value(dt=value)
        for dt in (-.01, .010001):
            with self.assertRaises(ValueError):
                self.value(dt=dt)

    def test_fixed_normalizations_reject_silent_config_changes(self):
        for field, value in (('max_position', .3), ('max_velocity', 3.), ('max_acceleration', 5.)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'R1-v0 requires'):
                self.reward.evaluate(self.top, 0., .01, False, replace(self.config, **{field: value}))

    def test_immediate_failure_bound_and_safe_bottom_continuation(self):
        for dt in (0., .0025, .01):
            reward, _, _ = self.value(u=None if dt == 0 else 0., dt=dt, terminated=True)
            self.assertLessEqual(reward, -4.)
        # Analytic bottom equilibrium has x/v/theta/omega=u=0 at every step.
        self.assertEqual(self.value(State.home())[0], 0.)

    def test_R0_characteristic_values_stay_reproducible(self):
        legacy = RewardR0()
        self.assertEqual(legacy.version, 'R0-v0')
        value, _, _ = legacy.evaluate(State(cart_position=.24), 0., .01, False, self.config)
        self.assertAlmostEqual(value, -.09216)
        value, _, _ = legacy.evaluate(replace(self.top, pole_angular_velocity=20.), 0., .01, False, self.config)
        self.assertAlmostEqual(value, .926530612244898)

    def test_pilot_config_is_json_fresh_and_matches_fixed_reward(self):
        config = make_pilot_config()
        self.assertEqual(json.loads(json.dumps(config, allow_nan=False)), config)
        self.assertEqual(config['reward'], self.reward.metadata())
        self.assertEqual(config['gamma'], math.exp(-.01/5.))
        self.assertEqual((config['timestep'], config['horizon'], config['reset_mode']), (.01, 10., 'random'))
        self.assertEqual(config['integrator'], {'type': 'RungeKutta3Integrator', 'target_accuracy': 1e-6,
                                               'maximum_step_size': .01, 'fixed_step_mode': False})
        config['reward']['weights']['failure'] = 99.
        config['physical_config']['max_position'] = 99.
        self.assertEqual(make_pilot_config()['reward']['weights']['failure'], 5.)
        self.assertEqual(make_pilot_config()['physical_config']['max_position'], .25)

    def test_pilot_factory_selects_reward_without_starting_training(self):
        with patch('cartpole.rl.env.CartPoleEnv') as env:
            self.assertIs(make_pilot_env(), env.return_value)
            env.assert_called_once_with(reward_version='R1-v0', reset_mode='random', horizon=10.)


if __name__ == '__main__':
    unittest.main()
