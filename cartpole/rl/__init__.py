"""Gymnasium interface; registration adds no TimeLimit or autoreset wrapper."""
from gymnasium.envs.registration import register, registry

ENV_ID = 'CourseworkCartPole-v0'
if ENV_ID not in registry:
    register(id=ENV_ID, entry_point='cartpole.rl.env:CartPoleEnv')

from cartpole.rl.env import CartPoleEnv

__all__ = ['CartPoleEnv', 'ENV_ID']
