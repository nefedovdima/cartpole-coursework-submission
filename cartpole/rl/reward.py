"""R0-v0: endpoint quadrature, applied input, one unscaled failure penalty."""
from dataclasses import dataclass, asdict
import math

from cartpole.experiment_config import CONTROL_INTERVAL


@dataclass(frozen=True)
class RewardR0:
    version: str = 'R0-v0'
    height: float = 1.
    position: float = .1
    velocity: float = .01
    angular_velocity: float = .01
    acceleration: float = .001
    failure: float = 5.

    def metadata(self):
        return {**asdict(self), 'reference_dt': CONTROL_INTERVAL,
                'evaluation_state': 'end of actual transition',
                'formula': 'dt_actual/.01*(h-.1*(x/X)^2-.01*(v/V)^2-.01*(omega/sqrt(g/l))^2-.001*(u_applied/U)^2)-5*terminated'}

    def evaluate(self, state, applied_acceleration, dt_actual, terminated, config):
        if not math.isfinite(dt_actual) or not 0 <= dt_actual <= CONTROL_INTERVAL+1e-12:
            raise ValueError('R0 expects an actual duration within one control interval')
        scale = dt_actual/CONTROL_INTERVAL
        terms = dict.fromkeys(('height', 'position', 'velocity', 'angular_velocity', 'acceleration'), 0.)
        if dt_actual > 0:
            if applied_acceleration is None or not math.isfinite(applied_acceleration):
                raise ValueError('a positive transition requires finite applied acceleration')
            terms.update(
                height=scale*self.height*(1-math.cos(state.pole_angle))/2,
                position=-scale*self.position*(state.cart_position/config.max_position)**2,
                velocity=-scale*self.velocity*(state.cart_velocity/config.max_velocity)**2,
                angular_velocity=-scale*self.angular_velocity*(state.pole_angular_velocity/math.sqrt(config.gravity/config.pole_length))**2,
                acceleration=-scale*self.acceleration*(applied_acceleration/config.max_acceleration)**2,
            )
        elif applied_acceleration is not None:
            raise ValueError('a zero transition has no applied acceleration')
        terms['failure'] = -self.failure if terminated else 0.
        reward = math.fsum(terms.values())
        if not math.isfinite(reward):
            raise FloatingPointError('non-finite R0 reward')
        return float(reward), terms, scale
