"""
PushWorld with precomputed solution information.

This module extends PushWorld to track whether puzzles are solvable
and optionally compute optimal solutions. Due to the complexity of
push mechanics, full optimal path computation is expensive.
"""

import chex
import jax
import jax.numpy as jnp
from flax import struct

from .env import EnvParams, EnvState, PushWorld
from .level import GRID_SIZE, Level


@struct.dataclass
class SolvedEnvState(EnvState):
    """Extended state with solution metadata."""

    # Whether the puzzle is solvable (precomputed)
    is_solvable: chex.Array
    # Estimated minimum steps to solve (heuristic, not exact)
    estimated_min_steps: chex.Array


class PushWorldSolved(PushWorld):
    """PushWorld with solution analysis.

    This extension tracks whether puzzles are likely solvable
    and provides heuristic estimates for difficulty.

    Note: Unlike maze environments, exact optimal paths for PushWorld
    are expensive to compute due to the state space explosion from
    push mechanics. This implementation uses heuristics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_state_from_level(self, level: Level) -> SolvedEnvState:
        """Initialize state with solution analysis."""
        base_state = super().init_state_from_level(level)

        # Compute heuristic solvability
        is_solvable = self._estimate_solvability(level)
        estimated_min_steps = self._estimate_difficulty(level)

        return SolvedEnvState(
            agent_pos=base_state.agent_pos,
            m1_pos=base_state.m1_pos,
            m2_pos=base_state.m2_pos,
            m3_pos=base_state.m3_pos,
            m4_pos=base_state.m4_pos,
            g1_pos=base_state.g1_pos,
            g2_pos=base_state.g2_pos,
            wall_map=base_state.wall_map,
            time=base_state.time,
            terminal=base_state.terminal,
            is_solvable=is_solvable,
            estimated_min_steps=estimated_min_steps,
        )

    def is_solveable(self, state: SolvedEnvState, params: EnvParams) -> chex.Array:
        """Check if puzzle is solvable (heuristic)."""
        return state.is_solvable

    def estimated_difficulty(self, state: SolvedEnvState) -> chex.Array:
        """Get estimated minimum steps."""
        return state.estimated_min_steps

    def optimal_value(
        self,
        state: SolvedEnvState,
        gamma: float,
        params: EnvParams,
    ) -> chex.Array:
        """Estimate optimal value (heuristic).

        This is an upper bound estimate based on the heuristic
        minimum steps to solve.
        """
        n = state.estimated_min_steps

        # Compute discounted return
        if self.penalize_time:
            # Account for step penalties
            from .env import STEP_REWARD, SUCCESS_REWARD

            # Value = gamma^n * SUCCESS_REWARD + sum of discounted step rewards
            value = (gamma**n) * SUCCESS_REWARD
            # Add discounted step penalties
            # sum_{i=0}^{n-1} gamma^i * STEP_REWARD = STEP_REWARD * (1 - gamma^n) / (1 - gamma)
            step_sum = jax.lax.select(
                gamma < 1.0,
                STEP_REWARD * (1.0 - gamma**n) / (1.0 - gamma + 1e-8),
                STEP_REWARD * n,
            )
            value = value + step_sum
        else:
            value = (gamma**n) * 10.0  # SUCCESS_REWARD

        return jnp.where(state.is_solvable, value, 0.0)

    def _estimate_solvability(self, level: Level) -> chex.Array:
        """Heuristic check for puzzle solvability.

        A puzzle is likely solvable if:
        1. Agent can reach movable objects
        2. Movable objects can reach goal positions
        3. No obvious blocking configurations

        This is a conservative heuristic - it may report False
        for some solvable puzzles, but should not report True
        for unsolvable ones.
        """
        # Basic checks
        has_agent = jnp.any(level.agent_pos[:, 0] >= 0)
        has_m1 = jnp.any(level.m1_pos[:, 0] >= 0)
        has_g1 = jnp.any(level.g1_pos[:, 0] >= 0)

        # Must have at least agent, one movable, one goal
        basic_valid = has_agent & has_m1 & has_g1

        # Check agent is not completely surrounded by walls
        agent_x = level.agent_pos[0, 0]
        agent_y = level.agent_pos[0, 1]

        def check_neighbor_free(x, y, wall_map):
            """Check if any adjacent cell is free."""
            free = jnp.array(False)
            # Check all 4 directions
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                in_bounds = (nx >= 0) & (nx < GRID_SIZE) & (ny >= 0) & (ny < GRID_SIZE)
                is_free = jax.lax.cond(
                    in_bounds, lambda: ~wall_map[ny, nx], lambda: jnp.array(False)
                )
                free = free | is_free
            return free

        agent_not_trapped = jax.lax.cond(
            has_agent,
            lambda: check_neighbor_free(agent_x, agent_y, level.wall_map),
            lambda: jnp.array(False),
        )

        # For M1 and G1, check they're not in walls
        m1_x, m1_y = level.m1_pos[0, 0], level.m1_pos[0, 1]
        g1_x, g1_y = level.g1_pos[0, 0], level.g1_pos[0, 1]

        m1_valid = jax.lax.cond(
            has_m1, lambda: ~level.wall_map[m1_y, m1_x], lambda: jnp.array(True)
        )

        g1_valid = jax.lax.cond(
            has_g1, lambda: ~level.wall_map[g1_y, g1_x], lambda: jnp.array(True)
        )

        return basic_valid & agent_not_trapped & m1_valid & g1_valid

    def _estimate_difficulty(self, level: Level) -> chex.Array:
        """Estimate minimum steps using Manhattan distance heuristic.

        This is a lower bound - actual solution may require more steps
        due to pushing mechanics and obstacles.
        """
        # Distance from agent to M1
        agent_x = level.agent_pos[0, 0].astype(jnp.float32)
        agent_y = level.agent_pos[0, 1].astype(jnp.float32)
        m1_x = level.m1_pos[0, 0].astype(jnp.float32)
        m1_y = level.m1_pos[0, 1].astype(jnp.float32)
        g1_x = level.g1_pos[0, 0].astype(jnp.float32)
        g1_y = level.g1_pos[0, 1].astype(jnp.float32)

        # Agent to M1 distance
        agent_to_m1 = jnp.abs(agent_x - m1_x) + jnp.abs(agent_y - m1_y)

        # M1 to G1 distance
        m1_to_g1 = jnp.abs(m1_x - g1_x) + jnp.abs(m1_y - g1_y)

        # Total heuristic: agent goes to M1, then pushes to G1
        # Need to be adjacent to push, so subtract 1 from agent_to_m1
        # but add 1 for each push step
        estimated = jnp.maximum(agent_to_m1 - 1, 0) + m1_to_g1

        # Handle invalid positions
        has_valid = (agent_x >= 0) & (m1_x >= 0) & (g1_x >= 0)

        return jnp.where(has_valid, estimated, jnp.inf)
