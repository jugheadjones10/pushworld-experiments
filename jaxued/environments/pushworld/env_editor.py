"""
PushWorld Editor Environment for adversarial level generation.

This environment allows an adversary (PLR) to construct PushWorld levels
by placing objects on the grid step by step.
"""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces

from jaxued.environments import UnderspecifiedEnv

from .env import NUM_CHANNELS, PushWorld
from .level import MAX_PIXELS, Level


@struct.dataclass
class EditorEnvState:
    level: Level
    time: int
    terminal: bool


@struct.dataclass
class EditorObservation:
    image: chex.Array  # Current level visualization
    action_mask: chex.Array  # Valid actions
    time: int
    random_z: chex.Array  # Random noise for adversary


@struct.dataclass
class EditorEnvParams:
    pass


class PushWorldEditor(UnderspecifiedEnv):
    """Editor environment for constructing PushWorld levels.

    The adversary constructs levels in phases:
    - Phase 0: Place goal G1
    - Phase 1: Place movable M1
    - Phase 2: Place agent
    - Phase 3+: Toggle walls

    Action space: Discrete(width * height) - selects grid cell
    """

    def __init__(
        self,
        env: PushWorld,
        random_z_dimensions: int = 16,
        zero_out_random_z: bool = False,
    ):
        super().__init__()
        self._env = env
        self.random_z_dimensions = random_z_dimensions
        self.zero_out_random_z = zero_out_random_z

    @property
    def default_params(self) -> EditorEnvParams:
        return EditorEnvParams()

    @property
    def num_actions(self) -> int:
        return self._env.grid_size * self._env.grid_size

    def step_env(
        self,
        rng: chex.PRNGKey,
        state: EditorEnvState,
        action: int,
        params: EditorEnvParams,
    ) -> Tuple[EditorObservation, EditorEnvState, float, bool, dict]:
        """Execute one editing step."""
        rng, rng_obs = jax.random.split(rng)

        # Don't edit if terminal
        new_level = jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(state.terminal, x, y),
            state.level,
            self._edit_level(rng, state, action, params),
        )

        # Update state
        state = state.replace(level=new_level, time=state.time + 1)
        done = self.is_terminal(state, params)
        state = state.replace(terminal=done)

        return self.get_obs(rng_obs, state), state, 0.0, done, {}

    def reset_env_to_level(
        self,
        rng: chex.PRNGKey,
        level: Level,
        params: EditorEnvParams,
    ) -> Tuple[EditorObservation, EditorEnvState]:
        """Reset editor to a starting level."""
        state = self.init_state_from_level(level)
        return self.get_obs(rng, state), state

    def action_space(self, params: EditorEnvParams) -> spaces.Discrete:
        return spaces.Discrete(self.num_actions)

    def init_state_from_level(self, level: Level) -> EditorEnvState:
        return EditorEnvState(
            level=level,
            time=jnp.array(0, dtype=jnp.int32),
            terminal=False,
        )

    def get_obs(self, rng: chex.PRNGKey, state: EditorEnvState) -> EditorObservation:
        """Get observation for the adversary."""
        max_w = self._env.grid_size

        # Build action mask based on current phase
        # Phase 0: Place G1 - any non-wall cell
        # Phase 1: Place M1 - any non-occupied cell
        # Phase 2: Place agent - any non-occupied cell
        # Phase 3+: Toggle walls - any cell except occupied ones

        level = state.level

        def get_occupied_mask():
            """Get mask of cells occupied by objects."""
            mask = jnp.zeros(self.num_actions, dtype=jnp.bool_)

            def mark_coords(m, coords):
                for i in range(MAX_PIXELS):
                    x, y = coords[i, 0], coords[i, 1]
                    valid = (x >= 0) & (y >= 0)
                    idx = y * max_w + x
                    m = jax.lax.cond(
                        valid, lambda mm: mm.at[idx].set(True), lambda mm: mm, m
                    )
                return m

            mask = mark_coords(mask, level.agent_pos)
            mask = mark_coords(mask, level.m1_pos)
            mask = mark_coords(mask, level.m2_pos)
            mask = mark_coords(mask, level.g1_pos)
            mask = mark_coords(mask, level.g2_pos)
            return mask

        occupied = get_occupied_mask()
        wall_flat = level.wall_map.flatten()

        # Different masks for different phases
        action_mask = jax.lax.switch(
            state.time.clip(0, 3),
            [
                lambda: ~wall_flat,  # Phase 0: Place G1
                lambda: ~(wall_flat | occupied),  # Phase 1: Place M1
                lambda: ~(wall_flat | occupied),  # Phase 2: Place agent
                lambda: ~occupied,  # Phase 3+: Toggle walls
            ],
        )

        # Create image observation
        image = self._create_image(state)

        # Random noise
        if self.zero_out_random_z:
            random_z = jnp.zeros(self.random_z_dimensions, dtype=jnp.float32)
        else:
            random_z = jax.random.uniform(
                rng, (self.random_z_dimensions,), dtype=jnp.float32
            )

        return EditorObservation(
            image=image,
            action_mask=action_mask,
            time=state.time,
            random_z=random_z,
        )

    def _create_image(self, state: EditorEnvState) -> chex.Array:
        """Create image representation of current level state."""
        level = state.level
        grid_size = self._env.grid_size

        # Create multi-channel observation
        obs = jnp.zeros((grid_size, grid_size, NUM_CHANNELS), dtype=jnp.float32)

        def place_object(obs, coords, channel):
            def place_pixel(o, coord):
                x, y = coord
                valid = (x >= 0) & (y >= 0) & (x < grid_size) & (y < grid_size)
                o = jax.lax.cond(
                    valid, lambda oo: oo.at[y, x, channel].set(1.0), lambda oo: oo, o
                )
                return o, None

            obs, _ = jax.lax.scan(place_pixel, obs, coords)
            return obs

        # Place objects based on current phase
        # Always show walls
        obs = obs.at[:, :, 7].set(level.wall_map.astype(jnp.float32))

        # Show G1 after phase 0
        obs = jax.lax.cond(
            state.time > 0, lambda o: place_object(o, level.g1_pos, 5), lambda o: o, obs
        )

        # Show M1 after phase 1
        obs = jax.lax.cond(
            state.time > 1, lambda o: place_object(o, level.m1_pos, 1), lambda o: o, obs
        )

        # Show agent after phase 2
        obs = jax.lax.cond(
            state.time > 2,
            lambda o: place_object(o, level.agent_pos, 0),
            lambda o: o,
            obs,
        )

        return obs

    def is_terminal(self, state: EditorEnvState, params: EditorEnvParams) -> bool:
        """Editor never terminates - external control needed."""
        return False

    def _edit_level(
        self,
        rng: chex.PRNGKey,
        state: EditorEnvState,
        action: int,
        params: EditorEnvParams,
    ) -> Level:
        """Apply edit action to level."""
        max_w = self._env.grid_size
        x = action % max_w
        y = action // max_w
        level = state.level

        def place_g1():
            """Phase 0: Place goal G1."""
            new_g1_pos = jnp.full((MAX_PIXELS, 2), -1, dtype=jnp.int32)
            new_g1_pos = new_g1_pos.at[0].set(jnp.array([x, y]))
            # Clear wall at this position
            new_wall_map = level.wall_map.at[y, x].set(False)
            return level.replace(g1_pos=new_g1_pos, wall_map=new_wall_map)

        def place_m1():
            """Phase 1: Place movable M1."""
            new_m1_pos = jnp.full((MAX_PIXELS, 2), -1, dtype=jnp.int32)
            new_m1_pos = new_m1_pos.at[0].set(jnp.array([x, y]))
            new_wall_map = level.wall_map.at[y, x].set(False)
            return level.replace(m1_pos=new_m1_pos, wall_map=new_wall_map)

        def place_agent():
            """Phase 2: Place agent."""
            new_agent_pos = jnp.full((MAX_PIXELS, 2), -1, dtype=jnp.int32)
            new_agent_pos = new_agent_pos.at[0].set(jnp.array([x, y]))
            new_wall_map = level.wall_map.at[y, x].set(False)
            return level.replace(agent_pos=new_agent_pos, wall_map=new_wall_map)

        def toggle_wall():
            """Phase 3+: Toggle wall at position."""

            # Check if position is occupied
            def is_occupied(coords):
                for i in range(MAX_PIXELS):
                    cx, cy = coords[i, 0], coords[i, 1]
                    if (cx == x) & (cy == y):
                        return True
                return False

            occupied = (
                is_occupied(level.agent_pos)
                | is_occupied(level.m1_pos)
                | is_occupied(level.g1_pos)
            )

            # Only toggle if not occupied
            new_val = jax.lax.select(
                occupied, level.wall_map[y, x], ~level.wall_map[y, x]
            )
            return level.replace(wall_map=level.wall_map.at[y, x].set(new_val))

        edit_phase = state.time.clip(0, 3)
        return jax.lax.switch(
            edit_phase, [place_g1, place_m1, place_agent, toggle_wall]
        )
