from pydrake.systems.analysis import Simulator

from cartpole.common import CartPoleBase, Config, Error, State
from cartpole.simulator.pydrake.system import CartPoleSystem

import logging
import math
from dataclasses import replace
from numbers import Real
import numpy

log = logging.getLogger('simulator')


def _finite_scalar(value, name):
    """Accept real numeric scalars, without coercing strings, bools or arrays."""
    if isinstance(value, (bool, numpy.bool_)) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a real numeric scalar')
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f'{name} must be finite and representable as a float') from exc
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def clamp(value, limit):
    if value > limit:
        return limit, True

    if value < -limit:
        return -limit, True

    return value, False

class CartPoleSimulator(CartPoleBase):
    '''
    Description:
        Сlass implements a physical simulation of the cart-pole device.
        A pole is attached by an joint to a cart, which moves along guide axis.
        The pendulum is initially at rest state. The goal is to maintain it in
        upright pose by increasing and reducing the cart's velocity.
    Source:
        This environment is some variation of the cart-pole problem
        described by Barto, Sutton, and Anderson
    Initial state:
        A pole is at starting position 0 with no velocity and acceleration.
    Technical details:
        Each instance owns a Drake continuous-time system and simulator.
        The input is cart acceleration, held constant during advance().
    '''

    def __init__(self):
        self.system = CartPoleSystem()
        self.context = self.system.CreateDefaultContext()
        self.simulator = Simulator(self.system, self.context)
        self.config = None
        self.error = Error.NEED_RESET # Formally, we need reset env to reset error.
        self.target_acceleration = 0
        self._last_applied_acceleration = 0.0
        
    def reset_to(self, config, state):
        """Validate a new episode before replacing any live simulation objects."""
        if not isinstance(config, Config):
            raise TypeError('config must be a Config')
        if not isinstance(state, State):
            raise TypeError('state must be a State')

        parameters = {}
        for name in ('gravity', 'pole_length', 'hard_max_position',
                     'hard_max_velocity', 'hard_max_acceleration'):
            value = _finite_scalar(getattr(config, name), name)
            if value <= 0:
                raise ValueError(f'{name} must be positive')
            parameters[name] = value
        # Own a validated snapshot; changes to the caller's Config must not
        # change limits behind the already-created Drake numeric parameters.
        config = replace(config, **parameters)
        q = numpy.array([
            _finite_scalar(getattr(state, name), name)
            for name in ('cart_position', 'pole_angle', 'cart_velocity',
                         'pole_angular_velocity')
        ])
        if abs(q[0]) > config.hard_max_position:
            raise ValueError('initial cart_position exceeds hard_max_position')
        if abs(q[2]) > config.hard_max_velocity:
            raise ValueError('initial cart_velocity exceeds hard_max_velocity')

        context = self.system.CreateContext(config, q)
        simulator = Simulator(self.system, context)
        self.system.get_input_port().FixValue(context, numpy.array([0.0]))

        self.config = config
        self.context = context
        self.simulator = simulator

        self.error = Error.NO_ERROR
        self.target_acceleration = 0.0
        self._last_applied_acceleration = 0.0

    def reset(self, config):
        self.reset_to(config, State.home())
        
    def get_state(self):
        """Return physical state and the latched error.

        cart_acceleration is the input of the last time-advancing interval,
        including one ending in X/V_OVERFLOW. It is zero after reset and is
        unchanged by new commands, zero-time calls and blocked advances.
        It is historical telemetry, not the pending command or braking.
        """
        state = State.from_array(
            self.context.get_continuous_state_vector()
        )
        state.error = self.error
        state.cart_acceleration = self._last_applied_acceleration
        return state

    def get_target(self):
        return self.target_acceleration

    def get_config(self):
        assert self.config
        return replace(self.config)

    def set_target(self, target):
        """Set a real scalar or ndarray of shape (1,); overflow latches a fault.

        The bounded command is stored in both get_target() and Drake's port.
        An overflowing command does not execute a transition: advance() stays
        blocked until reset. Invalid arguments raise even while faulted.
        """
        if isinstance(target, numpy.ndarray):
            if target.shape != (1,):
                raise ValueError('target array must have shape (1,)')
            target = target[0]
        target = _finite_scalar(target, 'target')
        if self.error:
            log.warning('set target, error=%s', self.error)
            return

        config = self.get_config()

        bounded_target, clamped = clamp(target, config.hard_max_acceleration)
        self.system.get_input_port().FixValue(self.context, numpy.array([bounded_target]))
        self.target_acceleration = bounded_target

        if clamped:
            self.error = Error.A_OVERFLOW
            return

        log.debug('set acc=%.2f', target)

    def validate(self):
        state = self.get_state()
        config = self.get_config()

        if abs(state.cart_position) > config.hard_max_position:
            self.error = Error.X_OVERFLOW
            return False

        if abs(state.cart_velocity) > config.hard_max_velocity:
            self.error = Error.V_OVERFLOW
            return False

        return True

    def advance(self, delta):
        delta = _finite_scalar(delta, 'delta')
        if delta < 0:
            raise ValueError('delta must be nonnegative')
        start_time = self.context.get_time()
        end_time = start_time + delta
        if not math.isfinite(end_time):
            raise ValueError('delta would make simulator time non-finite')
        if self.error:
            log.warning('advance, error=%i', self.error)
            return

        applied = float(self.system.get_input_port().Eval(self.context)[0])
        self.simulator.AdvanceTo(end_time)
        if self.context.get_time() > start_time:
            self._last_applied_acceleration = applied

        if not self.validate():
            return

    def timestamp(self):
        return self.context.get_time()

    def get_info(self):
        return {
            'time': self.context.get_time(),
        }

    def close(self):
        pass
