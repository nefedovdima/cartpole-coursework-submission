"""Optional cart-only acceleration filter. No simulator, policy or NumPy import.

The original experiment entrypoints do NOT call this module. See
docs/CART_SAFETY.md for the sampled-data proof and numerical limitations.
Returned u_filtered is a command, not evidence of physical execution.
"""
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
import math
from numbers import Real


VERSION = 'cart-safety-v1'


def _real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a real scalar (not an array, bool or string)')
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f'{name} must be representable as a finite float') from exc
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result


def _q(value):
    return Fraction.from_float(float(value))


def _display(value):
    """An unrepresentably large negative margin is None, never JSON Infinity."""
    try:
        return float(value)
    except OverflowError:
        return None


@dataclass(frozen=True)
class SafetyLimits:
    position_limit: float = .24
    velocity_limit: float = 2.
    acceleration_limit: float = 4.
    position_reserve: float = 1e-6
    velocity_reserve: float = 1e-6
    max_dt: float = .01
    min_dt: float = 1e-9
    dt_roundoff: float = 1e-12
    # Bounded departure from the ideal projection, only at a geometric/speed
    # endpoint when intervention is required. No state is ever rounded/clipped.
    intervention_inset: float = 1e-8

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = _real(getattr(self, name), name)
            object.__setattr__(self, name, value)
            if value < 0:
                raise ValueError(f'{name} must be nonnegative')
        if min(self.position_limit, self.velocity_limit, self.acceleration_limit,
               self.min_dt, self.max_dt) <= 0 or self.min_dt > self.max_dt:
            raise ValueError('limits and dt range must be positive and ordered')
        if self.position_reserve >= self.position_limit or self.velocity_reserve >= self.velocity_limit:
            raise ValueError('reserves must be smaller than the corresponding limits')
        if ((self.position_reserve and self.inner_position == self.position_limit)
                or (self.velocity_reserve and self.inner_velocity == self.velocity_limit)):
            raise ValueError('reserve is below float resolution')
        a, h, b, w = map(_q, (self.acceleration_limit, self.max_dt,
                             self.inner_position, self.inner_velocity))
        if a*h > w or a*h*h > 2*b:
            raise ValueError('sampled viability proof requires A*max_dt<=V and A*max_dt**2<=2*L internally')
        if self.intervention_inset > self.acceleration_limit:
            raise ValueError('intervention_inset must not exceed acceleration_limit')

    @property
    def inner_position(self):
        return self.position_limit-self.position_reserve

    @property
    def inner_velocity(self):
        return self.velocity_limit-self.velocity_reserve


@dataclass(frozen=True)
class CartMargins:
    position: float | None
    velocity: float | None
    right_stopping: float | None
    left_stopping: float | None


@dataclass(frozen=True)
class StateAssessment:
    admissible: bool
    viable_at_requested_limits: bool
    reasons: tuple[str, ...]
    inner: CartMargins
    requested: CartMargins


@dataclass(frozen=True)
class CartPrediction:
    x_end: float
    v_end: float
    x_min: float
    x_max: float
    max_abs_velocity: float
    position_margin_interval: float
    velocity_margin_interval: float
    acceleration_margin: float
    terminal: CartMargins


@dataclass(frozen=True)
class SafetyDecision:
    u_proposed: float
    u_filtered: float | None
    feasible: bool
    intervened: bool | None
    correction: float | None
    correction_abs: float | None
    reason: str
    proposal_violations: tuple[str, ...]
    state: StateAssessment
    # Endpoints are inward-rounded, individually certified float commands.
    feasible_interval: tuple[float, float] | None
    prediction: CartPrediction | None
    numerical_inset: float = 0.
    endpoint_rounding_ulps: tuple[int, int] = (0, 0)


def _margins(x, v, b, w, a):
    return (b-abs(x), w-abs(v), b-x-max(v, 0)**2/(2*a),
            b+x-max(-v, 0)**2/(2*a))


def _margin_record(values):
    return CartMargins(*map(_display, values))


def assess_cart_state(*, x, v, limits=SafetyLimits()):
    """Strict, exact-rational membership for the supplied binary floats.

    A failure of the inner set may still be viable for the requested .24/2
    limits. These cases are distinguished; neither is repaired by clipping x.
    """
    if not isinstance(limits, SafetyLimits):
        raise TypeError('limits must be SafetyLimits')
    x, v = _q(_real(x, 'x')), _q(_real(v, 'v'))
    a = _q(limits.acceleration_limit)
    inner = _margins(x, v, _q(limits.inner_position), _q(limits.inner_velocity), a)
    requested = _margins(x, v, _q(limits.position_limit), _q(limits.velocity_limit), a)
    names = ('position_outside_inner', 'velocity_outside_inner',
             'right_stopping_distance', 'left_stopping_distance')
    return StateAssessment(all(m >= 0 for m in inner), all(m >= 0 for m in requested),
                           tuple(name for name, m in zip(names, inner) if m < 0),
                           _margin_record(inner), _margin_record(requested))


def _motion(x, v, u, h):
    end_v = v+u*h
    end_x = x+v*h+u*h*h/2
    positions = [x, end_x]
    if u:
        turn = -v/u
        if 0 < turn < h:
            positions.append(x+v*turn+u*turn*turn/2)
    return end_x, end_v, min(positions), max(positions), max(abs(v), abs(end_v))


def _violations(x, v, u, h, b, w, a):
    end_x, end_v, low, high, speed = _motion(x, v, u, h)
    checks = (('acceleration_limit', a-abs(u)), ('position_left_interval', low+b),
              ('position_right_interval', b-high), ('velocity_interval', w-speed),
              ('future_right_stopping', b-end_x-max(end_v, 0)**2/(2*a)),
              ('future_left_stopping', b+end_x-max(-end_v, 0)**2/(2*a)))
    return tuple(name for name, margin in checks if margin < 0)


def _prediction(x, v, u, h, b, w, a):
    end_x, end_v, low, high, speed = _motion(x, v, u, h)
    return CartPrediction(*map(float, (end_x, end_v, low, high, speed,
                          min(b-high, low+b), w-speed, a-abs(u))),
                          _margin_record(_margins(end_x, end_v, b, w, a)))


def _upper_bounds(x, v, h, b, w, a):
    """Decimal analytic roots. Exact certification below distrusts their rounding."""
    d = b-x
    if d == 0 and v == 0:
        position = Decimal(0)
    elif v > 0 and 2*d <= v*h:
        position = -v*v/(2*d)  # admitted state excludes d=0,v>0
    else:
        position = 2*(d-v*h)/(h*h)
    c = d-v*h/2
    if c < 0:
        future = 2*(d-v*h)/(h*h)  # root has end velocity < 0
    else:
        root = ((a*h)**2+8*a*c).sqrt()
        if v >= 0 and root+a*h+2*v != 0:
            # Rationalize w_end-v to retain a nearly-zero acceleration root.
            future = 2*(2*a*(d-v*h)-v*v)/(h*(root+a*h+2*v))
        else:
            end_velocity = 4*a*c/(root+a*h) if c else Decimal(0)
            future = (end_velocity-v)/h
    return (a, (w-v)/h, position, future)


def _interval(x, v, h, b, w, a):
    with localcontext() as context:
        context.prec = 80
        dx, dv, dh, db, dw, da = [Decimal(q.numerator)/Decimal(q.denominator)
                                for q in (x, v, h, b, w, a)]
        upper = _upper_bounds(dx, dv, dh, db, dw, da)
        mirror = _upper_bounds(-dx, -dv, dh, db, dw, da)
        return (float(-min(mirror)), float(min(upper)),
                min(mirror[1:]) <= da, min(upper[1:]) <= da)


def filter_cart_command(*, x, v, proposed_u, dt=.01, limits=SafetyLimits()):
    """Project a physical acceleration onto the interval of viable ZOH inputs.

    Pure function; keyword x/v avoids state-order mistakes. Invalid numbers,
    shapes/types or dt raise ValueError/TypeError before any work is executed.
    An inadmissible state or numerical inability to certify a float action
    returns feasible=False, u_filtered=None: no disguised emergency command.
    Nominal dt is [min_dt,max_dt], with dt_roundoff allowed above max_dt for
    absolute-grid subtraction. The actual supplied dt is used unchanged and
    must still satisfy the exact sampled viability condition. Zero is invalid.
    """
    if not isinstance(limits, SafetyLimits):
        raise TypeError('limits must be SafetyLimits')
    x, v, proposed, delta = (_real(value, name) for value, name in
                             ((x, 'x'), (v, 'v'), (proposed_u, 'proposed_u'), (dt, 'dt')))
    if delta < limits.min_dt or delta-limits.max_dt > limits.dt_roundoff:
        raise ValueError(f'dt must be in [{limits.min_dt}, {limits.max_dt}] seconds (roundoff allowance {limits.dt_roundoff})')
    actual_h, actual_a = _q(delta), _q(limits.acceleration_limit)
    if actual_a*actual_h > _q(limits.inner_velocity) or actual_a*actual_h**2 > 2*_q(limits.inner_position):
        raise ValueError('actual dt exceeds the proved sampled viability bound')
    state = assess_cart_state(x=x, v=v, limits=limits)
    if not state.admissible:
        return SafetyDecision(proposed, None, False, None, None, None,
                              'inadmissible_state', (), state, None, None)
    x, v, h, b, w, a = map(_q, (x, v, delta, limits.inner_position,
                               limits.inner_velocity, limits.acceleration_limit))
    failed = _violations(x, v, _q(proposed), h, b, w, a)
    low, high, inset_low, inset_high = _interval(x, v, h, b, w, a)

    def certified(u):
        return not _violations(x, v, _q(u), h, b, w, a)

    def endpoint(u, toward):
        for ulps in range(65):
            if certified(u):
                return u, ulps
            if u == toward:
                break
            u = math.nextafter(u, toward)
        return None, 0

    if low <= high:
        lo, lo_ulps = endpoint(low, high)
        hi, hi_ulps = endpoint(high, low)
    else:
        lo = hi = None
    if lo is None or hi is None or lo > hi:
        # A mathematically admitted state has an exact-real feasible action
        # under the dt condition. Empty/uncertifiable float interval is NOT
        # evidence that an arbitrary braking command is safe.
        if not failed:
            return SafetyDecision(proposed, proposed, True, False, 0., 0.,
                                  'unchanged', (), state, None,
                                  _prediction(x, v, _q(proposed), h, b, w, a))
        return SafetyDecision(proposed, None, False, None, None, None,
                              'no_certified_float_action', failed, state, None, None)
    interval = (lo, hi)
    inset = 0.
    if not failed:
        selected = proposed  # even a certified boundary input is left unchanged
    else:
        projected = min(hi, max(lo, proposed))
        # Do not halve a narrow interval on every braking step: that would
        # repeatedly halve the stopping margin until float noise consumes it.
        # The opposite endpoint is certified too (often full braking +/-A).
        inset = min(limits.intervention_inset, hi-lo)
        if proposed > hi and inset_high:
            selected = max(lo, hi-inset)
        elif proposed < lo and inset_low:
            selected = min(hi, lo+inset)
        else:
            selected = projected
        inset = abs(selected-projected)
    if not certified(selected):
        return SafetyDecision(proposed, None, False, None, None, None,
                              'no_certified_float_action', failed, state, interval, None)
    correction = selected-proposed
    return SafetyDecision(proposed, selected, True, selected != proposed,
                          correction, abs(correction), 'projected' if failed else 'unchanged',
                          failed, state, interval, _prediction(x, v, _q(selected), h, b, w, a),
                          inset, (lo_ulps, hi_ulps))
