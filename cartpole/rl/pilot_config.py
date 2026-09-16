"""Fixed E5.1 technical-pilot proposal; this module does not start training."""
from dataclasses import asdict
import math

from cartpole.experiment_config import CONTROL_INTERVAL, make_experiment_config
from cartpole.rl.reward_pilot import RewardR1


def make_pilot_config():
    """Return a fresh JSON-friendly record of the selected physical/RL contract."""
    return {
        'id': 'sac-technical-pilot-r1-v0',
        'status': 'specified; no trained policy',
        'reward_version': 'R1-v0',
        'reward': RewardR1().metadata(),
        'gamma': math.exp(-CONTROL_INTERVAL/5.),
        'discount_time_constant_seconds': 5.,
        'timestep': CONTROL_INTERVAL,
        'horizon': 10.,
        'physical_config': asdict(make_experiment_config()),
        'integrator': {'type': 'RungeKutta3Integrator', 'target_accuracy': 1e-6,
                       'maximum_step_size': .01, 'fixed_step_mode': False},
        'reset_mode': 'random',
        'reset_distribution': {
            'cart_position': [-.02, .02], 'cart_velocity': [-.02, .02],
            'pole_angle': [-.05, .05], 'pole_angular_velocity': [-.05, .05],
            'sampling': 'independent uniform; local Gymnasium seeded generator'},
        'termination': 'working boundary or native error; no bootstrap',
        'truncation': 'external 10-second experience limit; bootstrap from final observation',
        'discount_semantics': 'constant gamma per control transition; short transitions occur only at terminal/time limit',
        'observation': '[x/.25,v/2,sin(theta),cos(theta),omega/sqrt(9.8/.18)]',
        'action': 'Box(-1,1,(1,),float32); u_requested=4*action; CartPoleEpisode limiter',
        'training_started': False,
    }


def make_pilot_env(*, safety_filter=False):
    """Construct only the chosen environment; importing this module trains nothing."""
    from cartpole.rl.env import CartPoleEnv
    config = make_pilot_config()
    if not isinstance(safety_filter, bool):
        raise TypeError('safety_filter must be bool')
    return CartPoleEnv(reward_version=config['reward_version'],
                       reset_mode=config['reset_mode'], horizon=config['horizon'],
                       **({'safety_filter': True} if safety_filter else {}))
