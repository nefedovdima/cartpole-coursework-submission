from cartpole.control.lqr import BalanceLQRControl, TrajectoryLQRControl


def __getattr__(name):
    # Local stabilization does not need the optional legacy trajectory solver.
    # Keep its explicit import available without importing it for every LQR.
    if name == 'Trajectory':
        from cartpole.control.trajectory import Trajectory
        return Trajectory
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
