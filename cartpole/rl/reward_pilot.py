"""Fixed R1-v0 candidate: bounded alignment, with no survival constant.

The product of the five factors lies in [0, 1] for every finite physical
state, even outside the working region. A failure contributes exactly -5;
the environment, rather than this stateless evaluator, prevents a second
transition after failure. R0-v0 remains a separate, unchanged evaluator.
"""
from dataclasses import dataclass
import math
from numbers import Real
from typing import ClassVar

from cartpole.experiment_config import CONTROL_INTERVAL


def _finite_real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a real numeric scalar')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def _q(value, scale):
    """1/(1+(value/scale)^2), without overflowing an intermediate square."""
    return (scale/math.hypot(scale, value))**2


@dataclass(frozen=True)
class RewardR1:
    """A fixed version, not a constructor for an unrecorded weight search."""

    version: ClassVar[str] = 'R1-v0'

    def metadata(self):
        return {
            'version': self.version,
            'reference_dt': CONTROL_INTERVAL,
            'evaluation_state': 'end of actual transition',
            'formula': 'dt_actual/.01*h*q(x/.25)*q(v/2)*q(omega/2)*q(.1*u_applied/4)-5*terminated',
            'height_formula': 'h=(1-cos(theta))/2',
            'factor_formula': 'q(z)=1/(1+z^2)',
            'normalizations': {'position_m': .25, 'velocity_m_per_s': 2.,
                               'angular_velocity_rad_per_s': 2.,
                               'acceleration_m_per_s2': 4.},
            'weights': {'height': 1., 'position': 1., 'velocity': 1.,
                        'angular_velocity': 1., 'acceleration': .1, 'failure': 5.},
            'running_reward_bounds': [0., 1.],
            'survival_constant': 0.,
            'duration_tolerance_seconds': 1e-12,
        }

    def factors(self, state, applied_u):
        """Return the five dimensionless factors before duration scaling.

        Angular speed uses its own 2 rad/s scale, not sqrt(g/l). The effort
        coefficient .1 is equivalent to an acceleration scale of 40 m/s².
        Diagnostics receive factors separately; reward_terms only contains
        the product and the failure penalty, avoiding cancellation.
        """
        try:
            x, v, theta, omega = (
                _finite_real(getattr(state, name), name)
                for name in ('cart_position', 'cart_velocity', 'pole_angle',
                             'pole_angular_velocity'))
        except AttributeError as exc:
            raise TypeError('state must expose all four physical components') from exc
        u = _finite_real(applied_u, 'applied_acceleration')
        return {'height': (1-math.cos(theta))/2,
                'position': _q(x, .25),
                'velocity': _q(v, 2.),
                'angular_velocity': _q(omega, 2.),
                'acceleration': _q(u, 40.)}

    def evaluate(self, state, applied_acceleration, dt_actual, terminated, config):
        """Return (reward, additive terms, actual-duration scale).

        The fixed physical normalizations are checked against the supplied
        experiment config, preventing a different reward under this version
        identifier. The duration tolerance matches R0 and floating timestamps;
        alpha is the actual quotient and is not silently clipped to one.
        """
        dt_actual = _finite_real(dt_actual, 'dt_actual')
        if not 0 <= dt_actual <= CONTROL_INTERVAL+1e-12:
            raise ValueError('R1 expects an actual duration within one control interval')
        for name, expected in (('max_position', .25), ('max_velocity', 2.),
                               ('max_acceleration', 4.)):
            actual = _finite_real(getattr(config, name), name)
            if actual != expected:
                raise ValueError(f'R1-v0 requires {name}={expected}')
        scale = dt_actual/CONTROL_INTERVAL
        if dt_actual > 0:
            if applied_acceleration is None:
                raise ValueError('a positive transition requires finite applied acceleration')
            alignment = scale*math.prod(self.factors(state, applied_acceleration).values())
        else:
            if applied_acceleration is not None:
                raise ValueError('a zero transition has no applied acceleration')
            # Validate the physical state without inventing an applied input.
            try:
                for name in ('cart_position', 'cart_velocity', 'pole_angle',
                             'pole_angular_velocity'):
                    _finite_real(getattr(state, name), name)
            except AttributeError as exc:
                raise TypeError('state must expose all four physical components') from exc
            alignment = 0.
        terms = {'alignment': alignment, 'failure': -5. if terminated else 0.}
        reward = math.fsum(terms.values())
        if not math.isfinite(reward):
            raise FloatingPointError('non-finite R1 reward')
        return reward, terms, scale
