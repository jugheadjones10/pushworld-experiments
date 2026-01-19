"""
PushWorld Environment implementation for the "All" variant.

This is a JAX-based implementation of the PushWorld puzzle environment,
compatible with the UnderspecifiedEnv interface for use with PLR.

Key features:
- 10x10 grid
- Multi-pixel polyomino objects
- 4 movable objects (m1-m4), 2 goals (g1-g2)
- Wave-front collision detection for pushing chains
- Full observation (no partial/egocentric view)
"""

from enum import IntEnum
from typing import Tuple

import chex
import jax
import jax.lax as lax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces

from jaxued.environments import UnderspecifiedEnv

from .level import GRID_SIZE, Level


class Actions(IntEnum):
    """PushWorld uses 4 cardinal movement actions."""

    up = 0
    right = 1
    down = 2
    left = 3


# Direction vectors for each action
DISPLACEMENTS = jnp.array(
    [
        (0, -1),  # UP
        (1, 0),  # RIGHT
        (0, 1),  # DOWN
        (-1, 0),  # LEFT
    ],
    dtype=jnp.int32,
)


# Channel indices for observation
CHANNEL_AGENT = 0
CHANNEL_M1 = 1
CHANNEL_M2 = 2
CHANNEL_M3 = 3
CHANNEL_M4 = 4
CHANNEL_G1 = 5
CHANNEL_G2 = 6
CHANNEL_WALL = 7
NUM_CHANNELS = 8


# Rewards
SUCCESS_REWARD = 10.0
STEP_REWARD = -0.01
GOAL_PROGRESS_REWARD = 1.0


@struct.dataclass
class EnvState:
    """State of the PushWorld environment."""

    # Object positions - each is (MAX_PIXELS, 2) with -1 padding
    agent_pos: chex.Array  # Current agent position
    m1_pos: chex.Array  # Movable 1 position
    m2_pos: chex.Array  # Movable 2 position
    m3_pos: chex.Array  # Movable 3 position
    m4_pos: chex.Array  # Movable 4 position
    # Goal positions (fixed for the level)
    g1_pos: chex.Array
    g2_pos: chex.Array
    # Wall map
    wall_map: chex.Array
    # Episode tracking
    time: int
    terminal: bool


@struct.dataclass
class Observation:
    """Observation for PushWorld - full grid view with 8 channels."""

    image: chex.Array  # Shape: (GRID_SIZE, GRID_SIZE, NUM_CHANNELS)


@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = 100


class PushWorld(UnderspecifiedEnv):
    """PushWorld environment implementation.

    This is the "All" variant with:
    - 10x10 grid
    - Multi-pixel objects (polyominoes)
    - 4 movable objects, 2 goals
    - Wave-front push mechanics

    Args:
        penalize_time: If True, adds a small negative reward per step.
        reward_shaping: If True, gives intermediate rewards for goal progress.
    """

    def __init__(
        self,
        penalize_time: bool = True,
        reward_shaping: bool = False,
    ):
        super().__init__()
        self.grid_size = GRID_SIZE
        self.penalize_time = penalize_time
        self.reward_shaping = reward_shaping

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[Observation, EnvState, float, bool, dict]:
        """Perform single timestep state transition."""
        # Get observation before action (for reward shaping)
        prev_obs = self.get_obs(state) if self.reward_shaping else None

        # Execute action
        state = self._step_agent(state, action)

        # Update time
        state = state.replace(time=state.time + 1)

        # Get new observation
        obs = self.get_obs(state)

        # Check goal condition
        goal_reached = self._check_goal(obs)
        state = state.replace(terminal=goal_reached)

        # Check termination
        done = self.is_terminal(state, params)

        # Compute reward
        reward = self._compute_reward(prev_obs, obs, goal_reached, state, params)

        return obs, state, reward, done, {}

    def reset_env_to_level(
        self,
        rng: chex.PRNGKey,
        level: Level,
        params: EnvParams,
    ) -> Tuple[Observation, EnvState]:
        """Reset environment to a specific level."""
        state = self.init_state_from_level(level)
        return self.get_obs(state), state

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(Actions))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(
            low=0,
            high=1,
            shape=(self.grid_size, self.grid_size, NUM_CHANNELS),
            dtype=jnp.float32,
        )

    # =========================================================================
    # Internal methods
    # =========================================================================

    def init_state_from_level(self, level: Level) -> EnvState:
        """Initialize environment state from a Level."""
        return EnvState(
            agent_pos=level.agent_pos.astype(jnp.int32),
            m1_pos=level.m1_pos.astype(jnp.int32),
            m2_pos=level.m2_pos.astype(jnp.int32),
            m3_pos=level.m3_pos.astype(jnp.int32),
            m4_pos=level.m4_pos.astype(jnp.int32),
            g1_pos=level.g1_pos.astype(jnp.int32),
            g2_pos=level.g2_pos.astype(jnp.int32),
            wall_map=level.wall_map.astype(jnp.bool_),
            time=0,
            terminal=False,
        )

    def get_obs(self, state: EnvState) -> Observation:
        """Get full observation as multi-channel grid."""
        obs = jnp.zeros(
            (self.grid_size, self.grid_size, NUM_CHANNELS), dtype=jnp.float32
        )

        def place_object(obs, coords, channel):
            """Place object pixels on the observation grid."""

            def place_pixel(obs, coord):
                x, y = coord
                valid = (
                    (x >= 0) & (y >= 0) & (x < self.grid_size) & (y < self.grid_size)
                )
                obs = jax.lax.cond(
                    valid, lambda o: o.at[y, x, channel].set(1.0), lambda o: o, obs
                )
                return obs, None

            obs, _ = jax.lax.scan(place_pixel, obs, coords)
            return obs

        # Place all objects
        obs = place_object(obs, state.agent_pos, CHANNEL_AGENT)
        obs = place_object(obs, state.m1_pos, CHANNEL_M1)
        obs = place_object(obs, state.m2_pos, CHANNEL_M2)
        obs = place_object(obs, state.m3_pos, CHANNEL_M3)
        obs = place_object(obs, state.m4_pos, CHANNEL_M4)
        obs = place_object(obs, state.g1_pos, CHANNEL_G1)
        obs = place_object(obs, state.g2_pos, CHANNEL_G2)

        # Place walls from wall_map
        obs = obs.at[:, :, CHANNEL_WALL].set(state.wall_map.astype(jnp.float32))

        return Observation(image=obs)

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        done_steps = state.time >= params.max_steps_in_episode
        return jnp.logical_or(done_steps, state.terminal)

    def _step_agent(self, state: EnvState, action: int) -> EnvState:
        """Execute action and return new state using wave-front push mechanics."""
        displacement = DISPLACEMENTS[action]
        return self._move_with_push(state, displacement)

    def _move_with_push(self, state: EnvState, displacement: chex.Array) -> EnvState:
        """Execute movement with wave-front collision detection for chain pushes."""
        # Stack all movable object coordinates
        coords = jnp.stack(
            [
                state.agent_pos,
                state.m1_pos,
                state.m2_pos,
                state.m3_pos,
                state.m4_pos,
            ],
            axis=0,
        )  # Shape: (5, MAX_PIXELS, 2)

        N = coords.shape[0]

        # Compute displaced positions for all objects
        all_disp, all_valid = jax.vmap(
            lambda c: self._masked_displacement(c, displacement)
        )(coords)
        # all_disp: (5, MAX_PIXELS, 2)
        # all_valid: (5, MAX_PIXELS)

        # Initialize wavefront with only agent
        frontier = jnp.array([True, False, False, False, False])
        pushed = jnp.zeros((N,), dtype=jnp.bool_).at[0].set(True)
        broken = jnp.array(False)

        def cond_fn(carry):
            frontier, pushed, broken = carry
            return frontier.any()

        def body_fn(carry):
            frontier, pushed, broken = carry

            # Check for wall/OOB collisions for objects in frontier
            blocked = self._check_blocked(all_disp, all_valid, state.wall_map)
            blocked_any = jnp.any(blocked & frontier)
            broken = broken | blocked_any

            # Compute collision graph between displaced and stationary objects
            coll_mat = self._compute_collisions(all_disp, all_valid, coords)

            # Find neighbors hit by frontier objects
            frontier_mat = frontier[:, None]
            neighbors = jnp.any(coll_mat * frontier_mat, axis=0)

            # New frontier: hit neighbors we haven't pushed yet
            new_frontier = neighbors & (~pushed)
            pushed = pushed | new_frontier

            # If blocked, clear frontier
            frontier = jnp.where(blocked_any, jnp.zeros_like(frontier), new_frontier)

            return frontier, pushed, broken

        # Run wavefront propagation
        _, final_pushed, final_broken = lax.while_loop(
            cond_fn, body_fn, (frontier, pushed, broken)
        )

        # Agent moved successfully if it was pushed and we didn't break
        moved_ok = final_pushed[0] & (~final_broken)

        def do_push(st):
            # Apply displacement to all pushed objects
            should_move = final_pushed[:, None, None]  # (5, 1, 1)
            moved = jnp.where(should_move, all_disp, coords)

            return st.replace(
                agent_pos=moved[0],
                m1_pos=moved[1],
                m2_pos=moved[2],
                m3_pos=moved[3],
                m4_pos=moved[4],
            )

        return lax.cond(moved_ok, do_push, lambda st: st, state)

    def _masked_displacement(
        self,
        coords: chex.Array,  # (MAX_PIXELS, 2)
        displacement: chex.Array,  # (2,)
    ) -> Tuple[chex.Array, chex.Array]:
        """Compute displaced coordinates, respecting -1 padding."""
        valid = (coords[:, 0] >= 0) & (coords[:, 1] >= 0)
        disp_all = coords + displacement
        disp = jnp.where(valid[:, None], disp_all, coords)
        return disp, valid

    def _check_blocked(
        self,
        all_disp: chex.Array,  # (N, MAX_PIXELS, 2)
        all_valid: chex.Array,  # (N, MAX_PIXELS)
        wall_map: chex.Array,  # (GRID_SIZE, GRID_SIZE)
    ) -> chex.Array:
        """Check which objects hit walls or go out of bounds."""
        xs = all_disp[..., 0]
        ys = all_disp[..., 1]

        # Clamp to valid indices for wall lookup
        xs_clamped = jnp.clip(xs, 0, self.grid_size - 1)
        ys_clamped = jnp.clip(ys, 0, self.grid_size - 1)

        # Check walls
        raw_vals = wall_map[ys_clamped, xs_clamped]
        wall_vals = jnp.where(all_valid, raw_vals, False)
        hit_wall = jnp.any(wall_vals, axis=1)

        # Check out of bounds
        in_bounds = (
            (xs >= 0) & (xs < self.grid_size) & (ys >= 0) & (ys < self.grid_size)
        )
        valid_in_bounds = jnp.where(all_valid, in_bounds, True)
        oob = ~jnp.all(valid_in_bounds, axis=1)

        return hit_wall | oob

    def _compute_collisions(
        self,
        all_disp: chex.Array,  # (N, MAX_PIXELS, 2)
        all_valid: chex.Array,  # (N, MAX_PIXELS)
        coords: chex.Array,  # (N, MAX_PIXELS, 2)
    ) -> chex.Array:
        """Compute collision matrix between displaced and stationary objects."""
        # Compare all pixel combinations
        # all_disp[i, p_i] vs coords[j, p_j]
        eq = all_disp[:, :, None, None, :] == coords[None, None, :, :, :]
        # eq: (N, MAX_PIXELS, N, MAX_PIXELS, 2)

        pixel_eq = jnp.all(eq, axis=-1)  # Both x and y match

        # Mask invalid pixels
        valid_i = all_valid[:, :, None, None]
        valid_j = all_valid[None, None, :, :]
        valid_collision = pixel_eq & valid_i & valid_j

        # Reduce to object-level collision matrix
        coll_mat = jnp.any(valid_collision, axis=(1, 3))  # (N, N)

        return coll_mat

    def _check_goal(self, obs: Observation) -> chex.Array:
        """Check if all goals are satisfied (movable covers goal exactly)."""
        # M1 should cover G1, M2 should cover G2
        m1_channel = obs.image[:, :, CHANNEL_M1]
        g1_channel = obs.image[:, :, CHANNEL_G1]
        m2_channel = obs.image[:, :, CHANNEL_M2]
        g2_channel = obs.image[:, :, CHANNEL_G2]

        # Check if goal exists
        g1_exists = jnp.any(g1_channel > 0)
        g2_exists = jnp.any(g2_channel > 0)

        # Check if movable covers goal exactly
        m1_on_g1 = jnp.all(m1_channel == g1_channel)
        m2_on_g2 = jnp.all(m2_channel == g2_channel)

        # Goal is reached if all existing goals are covered
        g1_satisfied = (~g1_exists) | (g1_exists & m1_on_g1)
        g2_satisfied = (~g2_exists) | (g2_exists & m2_on_g2)

        # At least one goal must exist and be satisfied
        has_any_goal = g1_exists | g2_exists
        all_goals_satisfied = g1_satisfied & g2_satisfied

        return has_any_goal & all_goals_satisfied

    def _num_goals_reached(self, obs: Observation) -> chex.Array:
        """Count how many goals are currently satisfied."""
        m1_channel = obs.image[:, :, CHANNEL_M1]
        g1_channel = obs.image[:, :, CHANNEL_G1]
        m2_channel = obs.image[:, :, CHANNEL_M2]
        g2_channel = obs.image[:, :, CHANNEL_G2]

        g1_exists = jnp.any(g1_channel > 0)
        g2_exists = jnp.any(g2_channel > 0)

        m1_on_g1 = jnp.all(m1_channel == g1_channel)
        m2_on_g2 = jnp.all(m2_channel == g2_channel)

        count = jnp.array(0)
        count = count + jax.lax.select(g1_exists & m1_on_g1, 1, 0)
        count = count + jax.lax.select(g2_exists & m2_on_g2, 1, 0)

        return count

    def _compute_reward(
        self,
        prev_obs: Observation,
        obs: Observation,
        goal_reached: chex.Array,
        state: EnvState,
        params: EnvParams,
    ) -> chex.Array:
        """Compute reward for the transition."""
        # Base reward for reaching goal
        reward = jax.lax.select(goal_reached, SUCCESS_REWARD, 0.0)

        # Reward shaping for goal progress
        if self.reward_shaping and prev_obs is not None:
            prev_goals = self._num_goals_reached(prev_obs)
            curr_goals = self._num_goals_reached(obs)
            progress = (curr_goals - prev_goals).astype(jnp.float32)
            reward = reward + progress * GOAL_PROGRESS_REWARD

        # Time penalty
        if self.penalize_time:
            reward = reward + STEP_REWARD

        return reward.astype(jnp.float32)
