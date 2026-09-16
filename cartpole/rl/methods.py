"""Version 2 request adapters; all learning updates remain in SB3/contrib."""
import copy
import gymnasium as gym
import numpy as np
from stable_baselines3.common.noise import ActionNoise

ALGORITHMS = ('SAC', 'TQC', 'DDPG', 'DQN')
GRID = np.arange(-4., 5., dtype=np.float64)


def contract_id(algorithm, mode='on'):
    if algorithm not in ALGORITHMS or mode not in ('on', 'off'):
        raise ValueError('unknown method/filter mode')
    return f'cartpole-request-v2-{algorithm}-{mode}'


def parse_contract(value):
    for algorithm in ALGORITHMS:
        for mode in ('on', 'off'):
            if value == contract_id(algorithm, mode):
                return algorithm, mode
    raise ValueError('unknown version 2 request contract')


class DiscreteRequests(gym.Wrapper):
    """Only the request is discrete. The inner filter output stays binary64."""
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(len(GRID))

    def environment_metadata(self):
        return self.env.environment_metadata()

    @property
    def last_transition(self):
        return self.env.last_transition

    @property
    def last_exception(self):
        return self.env.last_exception

    def step(self, action):
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise TypeError('DQN request must be an integer index')
        if not self.action_space.contains(action):
            raise ValueError('DQN request index must be in [0,8]')
        result = self.env.step(np.array([GRID[int(action)]/4.], dtype=np.float64))
        result[4]['request_index'] = int(action)
        return result


class GeneratorGaussianNoise(ActionNoise):
    """Independent PCG64 stream; pickled with model and explicitly in runtime.

    Episode reset does not rewind the stream. Evaluation never draws noise.
    Sigma is in normalized request units, before SB3 clips to [-1,1].
    """
    def __init__(self, seed, sigma=.1):
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)
        self.draws = 0

    def __call__(self):
        self.draws += 1
        return self.rng.normal(0., self.sigma, size=1).astype(np.float32)

    def state(self):
        return copy.deepcopy(dict(rng=self.rng.bit_generator.state, draws=self.draws, sigma=self.sigma))

    def restore(self, state):
        if state['sigma'] != self.sigma:
            raise ValueError('exploration sigma mismatch')
        self.rng.bit_generator.state = copy.deepcopy(state['rng'])
        self.draws = state['draws']
