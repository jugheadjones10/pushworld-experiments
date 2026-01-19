"""
PushWorld Renderer for visual logging.

Renders PushWorld states/levels as RGB images, compatible with JAX JIT.
"""

from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np

from .env import (
    CHANNEL_AGENT,
    CHANNEL_G1,
    CHANNEL_G2,
    CHANNEL_M1,
    CHANNEL_M2,
    CHANNEL_M3,
    CHANNEL_M4,
    CHANNEL_WALL,
    GRID_SIZE,
    EnvParams,
    EnvState,
    Observation,
    PushWorld,
)
from .level import Level

# Color palette matching the original PushWorld video rendering
# From xminigrid's COLOR_MAP:
#   0: White (background)
#   1: Green (agent)
#   2: Blue (movable without goal)
#   3: Red (movable with goal - m1, m2 when g2 exists)
#   4: Black (wall)
#   5: Light Red (empty goal)
#   6: Red (filled goal)
COLORS = {
    "empty": np.array([255, 255, 255]),  # White background
    "wall": np.array([0, 0, 0]),  # Black walls
    "agent": np.array([0, 255, 0]),  # Green agent
    "m1": np.array([255, 0, 0]),  # Red (has goal g1)
    "m2": np.array([0, 0, 255]),  # Blue (default, changes to red if g2 exists)
    "m3": np.array([0, 0, 255]),  # Blue movable
    "m4": np.array([0, 0, 255]),  # Blue movable
    "g1": np.array([255, 127, 127]),  # Light red (empty goal)
    "g2": np.array([255, 127, 127]),  # Light red (empty goal)
    "grid_line": np.array([200, 200, 200]),  # Light grey grid lines
}


class PushWorldRenderer:
    """Renderer for PushWorld environment states.

    Args:
        env: The PushWorld environment instance.
        tile_size: Number of pixels per grid tile.
        render_grid_lines: Whether to render grid lines.
    """

    def __init__(
        self,
        env: PushWorld,
        tile_size: int = 32,
        render_grid_lines: bool = True,
    ):
        self.env = env
        self.tile_size = tile_size
        self.render_grid_lines = render_grid_lines
        self._atlas = jnp.array(_make_tile_atlas(tile_size, render_grid_lines))

    @partial(jax.jit, static_argnums=(0,))
    def render_level(self, level: Level, env_params: EnvParams) -> chex.Array:
        """Render a Level as an RGB image."""
        # Create state from level
        state = self.env.init_state_from_level(level)
        return self.render_state(state, env_params)

    @partial(jax.jit, static_argnums=(0,))
    def render_state(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Render an EnvState as an RGB image.

        Color logic (matching original PushWorld):
        - Background: White
        - Walls: Black
        - Agent: Green
        - M1: Red (always has goal g1)
        - M2: Red if g2 exists, Blue otherwise
        - M3, M4: Blue
        - Goals (empty): Light red
        - Filled goal (movable on goal): Red
        """
        tile_size = self.tile_size
        grid_size = self.env.grid_size
        width_px = grid_size * tile_size
        height_px = grid_size * tile_size

        # Check if g2 exists (has any valid pixels)
        g2_exists = jnp.any(state.g2_pos[:, 0] >= 0)

        # Tile indices:
        # 0=empty(white), 1=wall(black), 2=agent(green),
        # 3=movable_goal(red), 4=movable(blue), 5=goal_empty(light red)
        cells = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)

        # Place walls (black)
        cells = jnp.where(state.wall_map, 1, cells)

        # Place goals (light red for empty)
        cells = self._place_pixels(cells, state.g1_pos, 5)
        cells = self._place_pixels(cells, state.g2_pos, 5)

        # Place movables - m3, m4 always blue
        cells = self._place_pixels(cells, state.m4_pos, 4)
        cells = self._place_pixels(cells, state.m3_pos, 4)

        # m2: red if g2 exists, blue otherwise
        m2_color = jax.lax.select(g2_exists, 3, 4)
        cells = self._place_pixels_dynamic(cells, state.m2_pos, m2_color)

        # m1: always red (has goal g1)
        cells = self._place_pixels(cells, state.m1_pos, 3)

        # Place agent (green, highest priority)
        cells = self._place_pixels(cells, state.agent_pos, 2)

        # Render using atlas
        img = (
            self._atlas[cells].transpose(0, 2, 1, 3, 4).reshape(height_px, width_px, 3)
        )

        return img

    @partial(jax.jit, static_argnums=(0,))
    def render_observation(self, obs: Observation, env_params: EnvParams) -> chex.Array:
        """Render an Observation as an RGB image.

        Matches the original PushWorld color scheme.
        """
        tile_size = self.tile_size
        grid_size = self.env.grid_size
        width_px = grid_size * tile_size
        height_px = grid_size * tile_size

        # Extract channels
        agent = obs.image[:, :, CHANNEL_AGENT]
        m1 = obs.image[:, :, CHANNEL_M1]
        m2 = obs.image[:, :, CHANNEL_M2]
        m3 = obs.image[:, :, CHANNEL_M3]
        m4 = obs.image[:, :, CHANNEL_M4]
        g1 = obs.image[:, :, CHANNEL_G1]
        g2 = obs.image[:, :, CHANNEL_G2]
        walls = obs.image[:, :, CHANNEL_WALL]

        # Check if g2 exists
        g2_exists = jnp.any(g2 > 0)

        # Tile indices:
        # 0=empty(white), 1=wall(black), 2=agent(green),
        # 3=movable_goal(red), 4=movable(blue), 5=goal_empty(light red)
        cells = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)

        # Layer walls (black)
        cells = jnp.where(walls > 0, 1, cells)

        # Layer goals (light red for empty)
        cells = jnp.where(g1 > 0, 5, cells)
        cells = jnp.where(g2 > 0, 5, cells)

        # Layer movables - m3, m4 always blue
        cells = jnp.where(m4 > 0, 4, cells)
        cells = jnp.where(m3 > 0, 4, cells)

        # m2: red if g2 exists, blue otherwise
        m2_color = jax.lax.select(g2_exists, 3, 4)
        cells = jnp.where(m2 > 0, m2_color, cells)

        # m1: always red (has goal g1)
        cells = jnp.where(m1 > 0, 3, cells)

        # Layer agent (green, highest priority)
        cells = jnp.where(agent > 0, 2, cells)

        # Render using atlas
        img = (
            self._atlas[cells].transpose(0, 2, 1, 3, 4).reshape(height_px, width_px, 3)
        )

        return img

    def _place_pixels(
        self,
        cells: chex.Array,
        coords: chex.Array,
        value: int,
    ) -> chex.Array:
        """Place object pixels on cell grid with static value."""

        def place_one(cells, coord):
            x, y = coord
            valid = (x >= 0) & (y >= 0) & (x < GRID_SIZE) & (y < GRID_SIZE)
            cells = jax.lax.cond(
                valid, lambda c: c.at[y, x].set(value), lambda c: c, cells
            )
            return cells, None

        cells, _ = jax.lax.scan(place_one, cells, coords)
        return cells

    def _place_pixels_dynamic(
        self,
        cells: chex.Array,
        coords: chex.Array,
        value: chex.Array,
    ) -> chex.Array:
        """Place object pixels on cell grid with dynamic value."""

        def place_one(cells, coord):
            x, y = coord
            valid = (x >= 0) & (y >= 0) & (x < GRID_SIZE) & (y < GRID_SIZE)
            cells = jax.lax.cond(
                valid, lambda c: c.at[y, x].set(value), lambda c: c, cells
            )
            return cells, None

        cells, _ = jax.lax.scan(place_one, cells, coords)
        return cells


def _make_tile_atlas(tile_size: int, render_grid_lines: bool = True) -> np.ndarray:
    """Create tile atlas with all tile types.

    Tile indices (matching original PushWorld color scheme):
    0 = empty (white background)
    1 = wall (black)
    2 = agent (green)
    3 = movable_goal (red - m1, or m2 when g2 exists)
    4 = movable (blue - m3, m4, or m2 when no g2)
    5 = goal_empty (light red)
    """
    atlas = np.empty((6, tile_size, tile_size, 3), dtype=np.uint8)

    def fill_solid(color):
        tile = np.tile(color, (tile_size, tile_size, 1))
        if render_grid_lines:
            tile = add_border(tile, COLORS["grid_line"])
        return tile

    def add_border(tile, color, width=1):
        tile = tile.copy()
        tile[:width, :] = color
        tile[-width:, :] = color
        tile[:, :width] = color
        tile[:, -width:] = color
        return tile

    # Build atlas with simple solid colors (matching original PushWorld)
    atlas[0] = fill_solid(COLORS["empty"])  # White background
    atlas[1] = fill_solid(COLORS["wall"])  # Black wall
    atlas[2] = fill_solid(COLORS["agent"])  # Green agent
    atlas[3] = fill_solid(COLORS["m1"])  # Red movable (has goal)
    atlas[4] = fill_solid(COLORS["m3"])  # Blue movable (no goal)
    atlas[5] = fill_solid(COLORS["g1"])  # Light red goal

    return atlas


def render_to_rgb(state_or_obs, env: PushWorld, tile_size: int = 32) -> np.ndarray:
    """Convenience function to render state or observation to RGB.

    Non-JIT version for debugging.
    """
    renderer = PushWorldRenderer(env, tile_size)

    if isinstance(state_or_obs, EnvState):
        return np.array(renderer.render_state(state_or_obs, env.default_params))
    elif isinstance(state_or_obs, Observation):
        return np.array(renderer.render_observation(state_or_obs, env.default_params))
    else:
        raise TypeError(f"Expected EnvState or Observation, got {type(state_or_obs)}")
