"""Execute one saved collocation plan with TVLQR, then periodic upright LQR."""
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from pydrake.systems.controllers import LinearQuadraticRegulator
from pydrake.systems.primitives import Linearize
from pydrake.trajectories import PiecewisePolynomial

from cartpole.common import State
from cartpole.control.lqr import BalanceLQRControl, TrajectoryLQRControl
from cartpole.simulator.pydrake.system import CartPoleSystem


@dataclass
class NominalTrajectory:
    states: PiecewisePolynomial
    targets: PiecewisePolynomial

    @property
    def duration(self):
        return self.targets.end_time()-self.targets.start_time()

    def save(self, path):
        times = np.array(self.states.get_segment_times())
        with Path(path).open('xb') as stream:
            np.savez(stream, times=times, states=self.states.vector_values(times),
                     derivatives=self.states.derivative().vector_values(times),
                     inputs=self.targets.vector_values(times))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            times, states, derivatives, inputs = (data[key] for key in ('times', 'states', 'derivatives', 'inputs'))
            if (times.ndim != 1 or len(times) < 2 or times[0] != 0 or np.any(np.diff(times) <= 0)
                    or states.shape != (4, len(times)) or derivatives.shape != states.shape
                    or inputs.shape != (1, len(times)) or not all(np.isfinite(a).all() for a in (times, states, derivatives, inputs))):
                raise ValueError('invalid nominal trajectory arrays')
            return cls(PiecewisePolynomial.CubicHermite(times, states, derivatives),
                       PiecewisePolynomial.FirstOrderHold(times, inputs))


class TrackingThenBalance:
    """One-way time-based handoff at the first control sample t >= plan end.

    This finite-horizon plan is not restarted after a failed capture. The
    upright LQR keeps running until the experiment horizon or episode failure.
    Raw angles remain unwrapped in tracking, preserving the planned winding.
    """
    def __init__(self, config, trajectory, tracking_scale=1.):
        if not math.isfinite(tracking_scale) or tracking_scale <= 0:
            raise ValueError('tracking_scale must be positive and finite')
        self.trajectory = trajectory
        self.balance = BalanceLQRControl(config)
        Q, R = self.balance.Q*tracking_scale, self.balance.R
        plant = CartPoleSystem()
        context = plant.CreateContext(config, State(pole_angle=math.pi).as_array())
        plant.get_input_port().FixValue(context, [0.])
        linear = Linearize(plant, context)
        _, Qf = LinearQuadraticRegulator(linear.A(), linear.B(), Q, R)
        self.tracking = TrajectoryLQRControl(config, trajectory, Q=Q, R=R, Qf=Qf)
        self.phase = 'swing_up'
        self.parameters = {'tracking_Q': Q.tolist(), 'tracking_R': R.tolist(), 'tracking_Qf': Qf.tolist(),
                           'balance_Q': self.balance.Q.tolist(), 'balance_R': self.balance.R.tolist(),
                           'balance_K': self.balance.K.tolist(), 'planned_handoff_time': trajectory.targets.end_time(),
                           'tracking_law': 'u=u0-K@(q-q0)-k0; raw angle error',
                           'handoff': 'first control timestamp >= planned end; one-way; no replanning',
                           'riccati_integrator': {'accuracy': 1e-6, 'max_step_size': .01, 'integration_scheme': 'runge_kutta3'}}

    def __call__(self, timestamp, state):
        if self.phase == 'lqr' or timestamp >= self.trajectory.targets.end_time():
            self.phase = 'lqr'
            return self.balance(state)
        return self.tracking(timestamp, state)
