from pydrake.systems.controllers import LinearQuadraticRegulator
from pydrake.systems.controllers import FiniteHorizonLinearQuadraticRegulator
from pydrake.systems.controllers import FiniteHorizonLinearQuadraticRegulatorOptions
from pydrake.systems.primitives import Linearize

from cartpole.common import Config, Error, State
from cartpole.common.metrics import upright_angle_error
from cartpole.simulator.pydrake.system import CartPoleSystem

import math
import numpy

class BalanceLQRControl:
    def __init__(self, config):
        state = State(
            cart_position = 0,
            cart_velocity = 0,
            pole_angle = math.pi,
            pole_angular_velocity = 0
        )
        
        self.q0 = state.as_array()
        self.u0 = numpy.array([0])
        
        system = CartPoleSystem()
        context = system.CreateContext(config, self.q0)
        system.get_input_port().FixValue(context, self.u0)

        # Q = numpy.diag([1, 13, 1, 4])
        # R = numpy.diag([0.18])
        self.Q = numpy.diag([10, 1, 10, 1])
        self.R = numpy.diag([0.5])

        linearized = Linearize(system, context)
        self.K, _ = LinearQuadraticRegulator(linearized.A(), linearized.B(), self.Q, self.R)
        
    def __call__(self, state):
        q = state.as_array()
        error = q - self.q0
        # State.as_array() is [x, theta, v, omega]. Feedback is u=-K*error;
        # only the angular error is periodic. CartPoleEpisode limits u.
        error[1] = upright_angle_error(state.pole_angle)
        u = -self.K @ error
        return float(u[0])
    
class ReverseBalanceLQRControl:
    def __init__(self, config):
        state = State(
            cart_position = 0,
            cart_velocity = 0,
            pole_angle = 0,
            pole_angular_velocity = 0
        )
        
        self.q0 = state.as_array()
        self.u0 = numpy.array([0])
        
        system = CartPoleSystem()
        context = system.CreateContext(config, self.q0)
        system.get_input_port().FixValue(context, self.u0)

        # Q = numpy.diag([1, 13, 1, 4])
        # R = numpy.diag([0.18])
        Q = numpy.diag([30, 1, 5, 1])
        R = numpy.diag([0.8])

        linearized = Linearize(system, context)
        self.K, _ = LinearQuadraticRegulator(linearized.A(), linearized.B(), Q, R)
        
    def __call__(self, state):
        q = state.as_array()
        error = q - self.q0
        u = -self.K @ error
        return u[0]


class TrajectoryLQRControl:
    def __init__(self, config, trajectory, *, Q=None, R=None, Qf=None):
        Q = numpy.eye(4) if Q is None else Q
        R = numpy.eye(1) if R is None else R

        options = FiniteHorizonLinearQuadraticRegulatorOptions()
        options.x0 = trajectory.states
        options.u0 = trajectory.targets
        options.Qf = Q if Qf is None else Qf
        options.simulator_config.accuracy = 1e-6
        options.simulator_config.max_step_size = .01

        system = CartPoleSystem()
        context = system.CreateContext(config, State().as_array())
        
        self.trajectory = trajectory
        self.regulator = FiniteHorizonLinearQuadraticRegulator(
            system,
            context,
            t0=options.u0.start_time(),
            tf=options.u0.end_time(),
            Q=Q,
            R=R,
            options=options
        )

    def __call__(self, stamp, state):
        q = state.as_array_4x1()
        q0 = self.trajectory.states.value(stamp)
        u0 = self.trajectory.targets.value(stamp)

        error = q - q0
        K = self.regulator.K.value(stamp)
        # Drake's affine term accounts for the reconstructed nominal's
        # collocation defect. The physical input remains a scalar acceleration.
        u = u0 - K @ error - self.regulator.k0.value(stamp)

        return float(u[0, 0])
