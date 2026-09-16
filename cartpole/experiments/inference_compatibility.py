"""Numerical inference compatibility, separate from file/weight identity.

Continuous requests live in [-1, 1]. Across devices the absolute bound is
2e-6 (about 17 float32 eps at unit scale), with no relative allowance near
saturation. It covers the observed 2**-23 backend rounding and matches the
existing safety_pilot inference bound. It is not a trajectory/safety tolerance.
DQN decisions are integer indices in Discrete(9), compared exactly.
"""
import numpy as np

CONTINUOUS_ATOL = 2e-6
PROTOCOL = 'inference-compatibility-v1'


def observations_for_inference(values):
    obs = np.asarray(values)
    if obs.dtype.kind not in 'fiu' or obs.ndim != 2 or obs.shape[1] != 5 or not len(obs):
        raise ValueError('probes require a nonempty numeric (N,5) observation array')
    if not np.isfinite(obs).all():
        raise ValueError('nonfinite saved probe observations')
    with np.errstate(over='ignore'):
        obs = obs.astype(np.float32)
    if not np.isfinite(obs).all():
        raise ValueError('probe observations overflow float32')
    return obs


def compare_actions(actual, saved, *, algorithm, count, saved_device, loaded_device):
    if algorithm not in ('SAC', 'TQC', 'DDPG', 'DQN'):
        raise ValueError('unknown probe algorithm')
    discrete = algorithm == 'DQN'
    shape = (count,) if discrete else (count, 1)
    arrays = [np.asarray(x) for x in (actual, saved)]
    for a in arrays:
        if a.shape != shape or a.dtype.kind not in ('iu' if discrete else 'fiu'):
            raise ValueError('probe action shape/type differs from the action contract')
        if not np.isfinite(a).all():
            raise ValueError('nonfinite probe actions')
        if (np.any(a < 0) or np.any(a >= 9)) if discrete else np.any(np.abs(a) > 1):
            raise ValueError('probe actions outside the action contract')
    error = float(np.max(np.abs(arrays[0].astype(np.float64)-arrays[1].astype(np.float64))))
    exact = discrete or saved_device == loaded_device
    tolerance = 0. if exact else CONTINUOUS_ATOL
    if error > tolerance:
        raise ValueError(f'inference incompatible: {algorithm}, {saved_device} -> {loaded_device}, '
                         f'max_abs_difference={error:.17g}, absolute_tolerance={tolerance:g}')
    return dict(protocol=PROTOCOL, algorithm=algorithm, saved_device=saved_device,
                loaded_device=loaded_device, comparisons=count, exact_required=exact,
                max_abs_difference=error, atol=tolerance, rtol=0., finite=True,
                units='Discrete(9) index' if discrete else 'normalized request [-1,1]')
