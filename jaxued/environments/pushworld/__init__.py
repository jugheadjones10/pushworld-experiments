"""
PushWorld Environment for JAX-based reinforcement learning.

This module provides a JAX implementation of the PushWorld puzzle environment,
compatible with the UnderspecifiedEnv interface for use with PLR.
"""

from .env import Actions, EnvParams, EnvState, Observation, PushWorld
from .env_editor import PushWorldEditor
from .env_solved import PushWorldSolved
from .level import (
    DATA_PATH,
    GRID_SIZE,
    MAX_PIXELS,
    Benchmark,
    Level,
    load_puzzle_from_file,
    prefabs,
    registered_benchmarks,
)
from .renderer import PushWorldRenderer
from .util import make_level_generator, make_level_mutator, make_level_mutator_minimax
