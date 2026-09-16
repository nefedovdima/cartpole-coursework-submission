"""Shared working-limit supervision of the original Drake simulator.

Only cart boundary times are computed analytically. All state transitions,
including the pendulum, are integrated by the supplied CartPoleSimulator.
See docs/PROJECT_CONTEXT.md for the boundary and numerical-tolerance contract.
"""
from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from cartpole.common import Error, State
from cartpole.experiment_config import CONTROL_INTERVAL, make_experiment_config
from cartpole.simulator.pydrake import CartPoleSimulator
from cartpole.simulator.pydrake.simulator import _finite_scalar
from cartpole.safety import SafetyDecision, SafetyLimits, filter_cart_command


# Absolute geometric/time resolution, in m, m/s and s respectively. These are
# analysis tolerances; returned Drake states are never projected onto a bound.
POSITION_TOLERANCE = 1e-12
VELOCITY_TOLERANCE = 1e-12
TIME_TOLERANCE = 1e-12


class StopReason(str, Enum):
    POSITION_LIMIT = 'position_limit'
    VELOCITY_LIMIT = 'velocity_limit'
    SIMULATOR_ERROR = 'simulator_error'
    SIMULATOR_EXCEPTION = 'simulator_exception'
    SAFETY_FILTER_REFUSAL = 'safety_filter_refusal'


@dataclass(frozen=True)
class BoundaryEvent:
    reason: StopReason
    side: int  # -1: lower boundary; +1: upper boundary
    time: float  # seconds from the beginning of this requested transition


@dataclass(frozen=True)
class Transition:
    state: State
    timestamp: float
    dt_requested: float
    dt_actual: float
    u_requested: float
    u_commanded: float | None  # None on filter refusal: no command was submitted
    action_clipped: bool
    applied_acceleration: float | None  # None if no time elapsed
    stop_reasons: tuple[StopReason, ...]
    boundary_events: tuple[BoundaryEvent, ...]
    safety_decision: SafetyDecision | None = None

    @property
    def terminated(self) -> bool:
        return bool(self.stop_reasons)

    @property
    def simulator_error(self) -> Error:
        return self.state.error


def _snap_boundary(value, limit, tolerance):
    if abs(abs(value) - limit) <= tolerance:
        return math.copysign(limit, value)
    return value


def _position_exit_time(x, v, u, limit, side):
    """First outward root of side*x(t)=limit, excluding a tangent maximum.

    In side coordinates, distance >= 0 is the distance to this boundary.
    For acceleration toward the interior, the internal maximum decides
    whether there is an exit even when the end of the interval is inside.
    """
    distance = limit - side*x
    velocity, acceleration = side*v, side*u
    if distance < 0:  # already outside beyond the analysis tolerance
        return 0.0
    if acceleration == 0:
        return distance / velocity if velocity > 0 else None
    if acceleration < 0:
        if velocity <= 0:
            return None
        turning_time = -velocity / acceleration
        peak_excess = .5*velocity*turning_time - distance
        if peak_excess <= POSITION_TOLERANCE:
            return None  # tangency or an excursion below geometric resolution
        # Scale the discriminant to avoid subtracting large quadratic roots.
        contact_speed = math.sqrt(2*abs(acceleration))*math.sqrt(distance)
        root_speed = velocity * math.sqrt(max(0.0, 1-contact_speed/velocity)) * math.sqrt(1+contact_speed/velocity)
    else:
        root_speed = math.hypot(velocity, math.sqrt(2*acceleration)*math.sqrt(distance))
        if velocity < 0:
            return (root_speed - velocity) / acceleration
    # Rationalized outward root avoids cancellation for a nearby boundary or
    # almost-zero acceleration. At rest on an outward-accelerating boundary,
    # the denominator and distance are both zero: departure is immediate.
    denominator = velocity + root_speed
    return 2*distance / denominator if denominator else 0.0


def _boundary_events(state, u, delta, config):
    x = _snap_boundary(state.cart_position, config.max_position, POSITION_TOLERANCE)
    v = _snap_boundary(state.cart_velocity, config.max_velocity, VELOCITY_TOLERANCE)
    candidates = []

    def add(reason, side, time):
        if time is not None and 0 <= time <= delta + TIME_TOLERANCE:
            # A residual velocity at a numerically integrated tangent can
            # produce a root a fraction of a picosecond after the start.
            # Resolve it as t=0 before calling Drake, not as a fictitious
            # positive transition with a newly applied acceleration.
            time = 0.0 if time <= TIME_TOLERANCE else min(time, delta)
            candidates.append(BoundaryEvent(reason, side, time))

    for side in (-1, 1):
        add(StopReason.POSITION_LIMIT, side,
            _position_exit_time(x, v, u, config.max_position, side))
    for side in (-1, 1):
        distance = config.max_velocity - side*v
        if distance < 0:
            add(StopReason.VELOCITY_LIMIT, side, 0.0)
        elif side*u > 0:
            add(StopReason.VELOCITY_LIMIT, side, distance / (side*u))
    if not candidates:
        return ()
    first = min(event.time for event in candidates)
    return tuple(event for event in candidates if event.time - first <= TIME_TOLERANCE)


class CartPoleEpisode:
    """Execute limited transitions, with an explicit terminal latch.

    reset(state=None) uses the shared experiment configuration. step(action,
    delta=CONTROL_INTERVAL) accepts the E1 action shapes and a finite,
    nonnegative duration. Longer intervals are useful for boundary diagnostics;
    normal experiments use the default 0.01 s. The backend must be exclusively
    advanced/reset through this owner during an episode. Its integrator may be
    configured after reset; this layer does not change integrator settings.

    safety_filter=True opts into the fixed .24 m cart filter before set_target.
    It uses the raw state and physical request for this exact hold duration.
    Refusal returns a zero-time terminal Transition with no submitted/applied
    command; reset still validates the original working region, not viability.
    The default False retains the original boundary-only execution path.
    """

    def __init__(self, backend: CartPoleSimulator | None = None, *, safety_filter=False):
        if not isinstance(safety_filter, bool):
            raise TypeError('safety_filter must be bool')
        self.backend = backend if backend is not None else CartPoleSimulator()
        self.safety_limits = SafetyLimits() if safety_filter else None
        self._config = make_experiment_config()
        self._initialized = False
        self.terminated = False
        self.stop_reasons = ()
        self.u_requested = 0.0
        self.u_commanded = 0.0

    def reset(self, state: State | None = None) -> State:
        """Reject an invalid working-region start before changing either layer."""
        initial = State.home() if state is None else state
        if not isinstance(initial, State):
            raise TypeError('state must be a State')
        x = _finite_scalar(initial.cart_position, 'cart_position')
        v = _finite_scalar(initial.cart_velocity, 'cart_velocity')
        if abs(x) > self._config.max_position:
            raise ValueError('initial cart_position exceeds working max_position')
        if abs(v) > self._config.max_velocity:
            raise ValueError('initial cart_velocity exceeds working max_velocity')
        # E1 validates every remaining component before replacing its context.
        self.backend.reset_to(self._config, initial)
        self._initialized = True
        self.terminated = False
        self.stop_reasons = ()
        self.u_requested = self.u_commanded = 0.0
        return self.backend.get_state()

    def step(self, action, delta=CONTROL_INTERVAL) -> Transition:
        """Stop at the first outward boundary, including t=0 and t=delta.

        All simultaneous boundary reasons (within TIME_TOLERANCE) are kept.
        Native errors are returned unchanged, with SIMULATOR_ERROR first in
        stop_reasons. A backend exception latches termination and is re-raised
        unchanged; no successful Transition is manufactured for that call.
        """
        if not self._initialized or self.terminated:
            raise RuntimeError('reset is required before another transition')
        if isinstance(action, np.ndarray):
            if action.shape != (1,):
                raise ValueError('action array must have shape (1,)')
            action = action[0]
        requested = _finite_scalar(action, 'action')
        delta = _finite_scalar(delta, 'delta')
        if delta < 0:
            raise ValueError('delta must be nonnegative')
        start_time = self.backend.timestamp()
        if not math.isfinite(start_time + delta):
            raise ValueError('delta would make simulator time non-finite')
        commanded = max(-self._config.max_acceleration,
                        min(requested, self._config.max_acceleration))
        initial = self.backend.get_state()
        decision = None
        # Filter the original physical request using raw binary64 telemetry and
        # the actual upcoming hold duration. No observation/action-space roundtrip.
        # Native errors retain priority. A zero-time diagnostic has no hold to certify.
        if self.safety_limits is not None and not initial.error and delta > 0:
            decision = filter_cart_command(x=initial.cart_position, v=initial.cart_velocity,
                                           proposed_u=requested, dt=delta, limits=self.safety_limits)
            if not decision.feasible:
                self.terminated = True
                self.stop_reasons = (StopReason.SAFETY_FILTER_REFUSAL,)
                self.u_requested, self.u_commanded = requested, None
                return Transition(initial, start_time, delta, 0., requested, None,
                                  False, None,
                                  self.stop_reasons, (), decision)
            commanded = decision.u_filtered
        events = () if initial.error else _boundary_events(initial, commanded, delta, self._config)
        duration = min(event.time for event in events) if events else delta
        self.u_requested, self.u_commanded = requested, commanded

        if not initial.error:
            try:
                self.backend.set_target(commanded)
                if duration > 0:
                    self.backend.advance(duration)
            except Exception:
                self.terminated = True
                self.stop_reasons = (StopReason.SIMULATOR_EXCEPTION,)
                raise
        state = self.backend.get_state()
        timestamp = self.backend.timestamp()
        actual = timestamp - start_time
        # An early native failure can end the transition before a predicted
        # working event. Report only boundaries actually reached (within the
        # time resolution), never a later planned event as an observed cause.
        events = tuple(event for event in events if event.time <= actual + TIME_TOLERANCE)
        reasons = tuple(dict.fromkeys(event.reason for event in events))
        if state.error:
            reasons = (StopReason.SIMULATOR_ERROR, *reasons)
        self.stop_reasons = reasons
        self.terminated = bool(reasons)
        return Transition(
            state=state, timestamp=timestamp, dt_requested=delta, dt_actual=actual,
            u_requested=requested, u_commanded=commanded,
            action_clipped=abs(requested) > self._config.max_acceleration,
            applied_acceleration=state.cart_acceleration if actual > 0 else None,
            stop_reasons=reasons, boundary_events=events, safety_decision=decision,
        )
