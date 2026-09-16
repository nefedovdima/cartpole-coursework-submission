"""Shared simulation settings for new classical and RL experiments.

Working limits and length follow the active ACTOR_CONFIG in
scripts/debug_session_runner.py. Gravity and hard limits are made explicit
from its inherited Config defaults. These are simulation parameters, not
identified hardware parameters. The historical Config defaults stay intact.
"""
from cartpole.common import Config


CONTROL_INTERVAL = 0.01  # seconds; acceleration is held during this interval


def make_experiment_config() -> Config:
    """Return an independent configuration for each experiment/episode owner."""
    return Config(
        gravity=9.8,
        pole_length=0.18,
        max_position=0.25,
        max_velocity=2.0,
        max_acceleration=4.0,
        hard_max_position=0.27,
        hard_max_velocity=2.5,
        hard_max_acceleration=5.0,
    )
