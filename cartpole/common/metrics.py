"""Shared periodic angle and explicitly defined local upright region."""
from dataclasses import dataclass
import math


def upright_angle_error(theta: float) -> float:
    delta = theta - math.pi
    return math.atan2(math.sin(delta), math.cos(delta))


@dataclass(frozen=True)
class UprightThresholds:
    angle: float = math.pi / 18  # 10 degrees
    angular_velocity: float = .5  # rad/s
    position: float = .20  # m
    velocity: float = .20  # m/s

    def contains(self, state) -> bool:
        return (abs(upright_angle_error(state.pole_angle)) <= self.angle
                and abs(state.pole_angular_velocity) <= self.angular_velocity
                and abs(state.cart_position) <= self.position
                and abs(state.cart_velocity) <= self.velocity)


def evaluate_hold(log, required_duration=2.):
    """Contiguous runs of qualifying samples, never sums of disjoint visits.

    An interval starts/ends at its first/last qualifying sample. Confirmation
    is the first recorded time at least required_duration after that start.
    No claim is made about the pendulum between observations. The 1 ps time
    tolerance only resolves floating-point subtraction at an exact deadline.
    """
    from cartpole.common import State
    if not math.isfinite(required_duration) or required_duration <= 0:
        raise ValueError('required_duration must be positive and finite')
    thresholds = UprightThresholds(**log['metadata']['upright_thresholds'])
    intervals, start, last = [], None, None
    first_start = confirmation = None
    previous_time = -math.inf
    for row in log['states']:
        stamp = row['time']
        if not math.isfinite(stamp) or stamp < previous_time:
            raise ValueError('hold samples must have finite nondecreasing times')
        previous_time = stamp
        state = State(**{key: row[key] for key in ('cart_position', 'cart_velocity',
                                                  'pole_angle', 'pole_angular_velocity')})
        if thresholds.contains(state):
            if start is None:
                start = stamp
            last = stamp
            if first_start is None and last-start >= required_duration-1e-12:
                first_start, confirmation = start, stamp
        elif start is not None:
            intervals.append({'start': start, 'end': last, 'duration': last-start})
            start = last = None
    final = 0. if start is None else last-start
    if start is not None:
        intervals.append({'start': start, 'end': last, 'duration': final})
    phases = [row.get('control_phase') for row in log['transitions']]
    switches = [dict(time=log['transitions'][i]['time_start'],
                     from_phase=phases[i-1], to_phase=phase)
                for i, phase in enumerate(phases) if i and phase != phases[i-1]]
    full = (log['completion']['reason'] == 'horizon'
            and math.isclose(log['completion']['time'], log['metadata']['horizon'], abs_tol=1e-12, rel_tol=0.))
    return {'required_hold_duration': required_duration,
            'criterion': 'consecutive recorded states; first/last qualifying timestamps; time tolerance 1e-12 s',
            'success_episode': full and final >= required_duration-1e-12,
            'first_hold_start': first_start, 'first_hold_confirmation': confirmation,
            'hold_final': final, 'hold_longest': max((r['duration'] for r in intervals), default=0.),
            'hold_intervals': intervals, 'mode_switch_count': len(switches), 'mode_switches': switches}
