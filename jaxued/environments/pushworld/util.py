"""
Utility functions for PushWorld level generation and mutation.

These functions are used by PLR (Prioritized Level Replay) for
generating and mutating levels during training.
"""

from enum import IntEnum
from typing import Callable

import chex
import jax
import jax.numpy as jnp

from .level import GRID_SIZE, MAX_PIXELS, Level


def make_level_generator(
    height: int = GRID_SIZE,
    width: int = GRID_SIZE,
    n_walls: int = 10,
    n_movables: int = 1,
) -> Callable[[chex.PRNGKey], Level]:
    """Create a function that generates random PushWorld levels.

    Args:
        height: Grid height (default: 10)
        width: Grid width (default: 10)
        n_walls: Number of wall cells to place
        n_movables: Number of movable objects (1-4)

    Returns:
        Function that takes a PRNGKey and returns a Level
    """

    def sample(rng: chex.PRNGKey) -> Level:
        max_w, max_h = width, height
        all_pos = jnp.arange(max_w * max_h, dtype=jnp.int32)

        rng, rng_wall, rng_agent, rng_m1, rng_m2, rng_g1, rng_g2 = jax.random.split(
            rng, 7
        )

        # Initialize empty wall map
        wall_map = jnp.zeros((max_h, max_w), dtype=jnp.bool_)
        occupied_mask = jnp.zeros(max_w * max_h, dtype=jnp.bool_)

        # Place walls
        wall_choices = jax.random.choice(
            rng_wall, max_w * max_h, shape=(n_walls,), replace=False
        )
        occupied_mask = occupied_mask.at[wall_choices].set(True)
        for i in range(n_walls):
            idx = wall_choices[i]
            y, x = idx // max_w, idx % max_w
            wall_map = wall_map.at[y, x].set(True)

        def place_object(rng, occupied, num_pixels=1):
            """Place an object with num_pixels pixels, returning coordinates."""
            coords = jnp.full((MAX_PIXELS, 2), -1, dtype=jnp.int32)

            # Place first pixel
            available = ~occupied
            first_idx = jax.random.choice(rng, all_pos, p=available.astype(jnp.float32))
            occupied = occupied.at[first_idx].set(True)
            x, y = first_idx % max_w, first_idx // max_w
            coords = coords.at[0].set(jnp.array([x, y]))

            return coords, occupied

        # Place agent
        agent_pos, occupied_mask = place_object(rng_agent, occupied_mask)

        # Place movables and goals
        m1_pos, occupied_mask = place_object(rng_m1, occupied_mask)
        g1_pos, occupied_mask = place_object(rng_g1, occupied_mask)

        # Optional second movable/goal pair
        def place_second_pair(rng, occupied):
            m2_pos, occupied = place_object(rng, occupied)
            g2_pos, occupied = place_object(jax.random.split(rng)[1], occupied)
            return m2_pos, g2_pos, occupied

        def skip_second_pair(rng, occupied):
            empty = jnp.full((MAX_PIXELS, 2), -1, dtype=jnp.int32)
            return empty, empty, occupied

        m2_pos, g2_pos, occupied_mask = jax.lax.cond(
            n_movables >= 2, place_second_pair, skip_second_pair, rng_m2, occupied_mask
        )

        # Empty positions for m3, m4 (not used in basic generation)
        empty_coords = jnp.full((MAX_PIXELS, 2), -1, dtype=jnp.int32)

        return Level(
            agent_pos=agent_pos,
            m1_pos=m1_pos,
            m2_pos=m2_pos,
            m3_pos=empty_coords,
            m4_pos=empty_coords,
            g1_pos=g1_pos,
            g2_pos=g2_pos,
            wall_map=wall_map,
            width=width,
            height=height,
        )

    return sample


class Mutations(IntEnum):
    """Types of mutations for level editing."""

    NO_OP = 0
    FLIP_WALL = 1
    MOVE_AGENT = 2
    MOVE_M1 = 3
    MOVE_G1 = 4


def make_level_mutator(
    max_num_edits: int,
) -> Callable[[chex.PRNGKey, Level, int], Level]:
    """Create a function that mutates PushWorld levels.

    Args:
        max_num_edits: Maximum number of edits to apply

    Returns:
        Function that takes (rng, level, num_edits) and returns mutated Level
    """

    def mutate(rng: chex.PRNGKey, level: Level, num_edits: int = 1) -> Level:
        # Use constant grid size to avoid JAX tracing issues
        max_w, max_h = GRID_SIZE, GRID_SIZE
        all_pos = jnp.arange(max_w * max_h, dtype=jnp.int32)

        def flip_wall(rng, lvl):
            """Toggle a random wall cell."""
            # Don't flip walls where objects are
            wall_mask = jnp.ones((max_h * max_w,), dtype=jnp.bool_)

            # Mark object positions as invalid
            def mark_invalid(mask, coords):
                for i in range(MAX_PIXELS):
                    x, y = coords[i, 0], coords[i, 1]
                    idx = y * max_w + x
                    valid = (x >= 0) & (y >= 0)
                    mask = jax.lax.cond(
                        valid, lambda m: m.at[idx].set(False), lambda m: m, mask
                    )
                return mask

            wall_mask = mark_invalid(wall_mask, lvl.agent_pos)
            wall_mask = mark_invalid(wall_mask, lvl.m1_pos)
            wall_mask = mark_invalid(wall_mask, lvl.m2_pos)
            wall_mask = mark_invalid(wall_mask, lvl.g1_pos)
            wall_mask = mark_invalid(wall_mask, lvl.g2_pos)

            flip_idx = jax.random.choice(
                rng, max_h * max_w, p=wall_mask.astype(jnp.float32)
            )
            flip_y, flip_x = flip_idx // max_w, flip_idx % max_w
            new_val = ~lvl.wall_map[flip_y, flip_x]
            new_wall_map = lvl.wall_map.at[flip_y, flip_x].set(new_val)

            return lvl.replace(wall_map=new_wall_map)

        def move_object(rng, lvl, coords_attr):
            """Move an object to a random valid position."""
            coords = getattr(lvl, coords_attr)

            # Find valid positions (not walls, not other objects)
            valid_mask = ~lvl.wall_map.flatten()

            # Pick new position
            new_idx = jax.random.choice(rng, all_pos, p=valid_mask.astype(jnp.float32))
            new_x, new_y = new_idx % max_w, new_idx // max_w

            # Update first pixel, keep rest invalid
            new_coords = jnp.full((MAX_PIXELS, 2), -1, dtype=jnp.int32)
            new_coords = new_coords.at[0].set(jnp.array([new_x, new_y]))

            return lvl.replace(**{coords_attr: new_coords})

        def apply_mutation(carry, step):
            rng, lvl = carry
            mutation_rng, mutation_type = step

            rng, apply_rng = jax.random.split(rng)

            def do_flip_wall(_):
                return flip_wall(mutation_rng, lvl)

            def do_move_agent(_):
                return move_object(mutation_rng, lvl, "agent_pos")

            def do_move_m1(_):
                return move_object(mutation_rng, lvl, "m1_pos")

            def do_move_g1(_):
                return move_object(mutation_rng, lvl, "g1_pos")

            def do_nothing(_):
                return lvl

            new_lvl = jax.lax.switch(
                mutation_type,
                [
                    do_nothing,  # NO_OP
                    do_flip_wall,  # FLIP_WALL
                    do_move_agent,  # MOVE_AGENT
                    do_move_m1,  # MOVE_M1
                    do_move_g1,  # MOVE_G1
                ],
                None,
            )

            return (rng, new_lvl), None

        # Generate mutation types and apply
        rng, *mrngs = jax.random.split(rng, max_num_edits + 1)
        mutation_types = jax.random.choice(rng, len(Mutations), shape=(max_num_edits,))
        # Mask out mutations beyond num_edits
        mutation_types = jnp.where(
            jnp.arange(max_num_edits) < num_edits, mutation_types, Mutations.NO_OP
        )

        (_, new_level), _ = jax.lax.scan(
            apply_mutation, (rng, level), (jnp.array(mrngs), mutation_types)
        )

        return new_level

    return mutate


def make_level_mutator_minimax(
    max_num_edits: int,
) -> Callable[[chex.PRNGKey, Level, int], Level]:
    """Create a mutation function inspired by the minimax approach.

    This is a simplified version focused on wall flipping and goal moving.
    """

    class MinimaxMutations(IntEnum):
        NO_OP = 0
        FLIP_WALL = 1
        MOVE_G1 = 2

    def flip_wall(rng, level):
        # Use constant grid size to avoid JAX tracing issues
        max_w, max_h = GRID_SIZE, GRID_SIZE

        # Create mask excluding object positions
        wall_mask = jnp.ones((max_h * max_w,), dtype=jnp.bool_)

        def mark_occupied(mask, coords):
            for i in range(MAX_PIXELS):
                x, y = coords[i, 0], coords[i, 1]
                valid = (x >= 0) & (y >= 0)
                idx = y * max_w + x
                mask = jax.lax.cond(
                    valid, lambda m: m.at[idx].set(False), lambda m: m, mask
                )
            return mask

        wall_mask = mark_occupied(wall_mask, level.agent_pos)
        wall_mask = mark_occupied(wall_mask, level.m1_pos)
        wall_mask = mark_occupied(wall_mask, level.g1_pos)
        wall_mask = mark_occupied(wall_mask, level.m2_pos)
        wall_mask = mark_occupied(wall_mask, level.g2_pos)

        flip_idx = jax.random.choice(
            rng, max_h * max_w, p=wall_mask.astype(jnp.float32)
        )
        flip_y, flip_x = flip_idx // max_w, flip_idx % max_w

        flip_val = ~level.wall_map[flip_y, flip_x]
        new_wall_map = level.wall_map.at[flip_y, flip_x].set(flip_val)

        return level.replace(wall_map=new_wall_map)

    def move_g1(rng, level):
        # Use constant grid size to avoid JAX tracing issues
        max_w, max_h = GRID_SIZE, GRID_SIZE

        # Valid positions: not walls, not occupied by agent/m1
        valid_mask = ~level.wall_map.flatten()

        def mark_occupied(mask, coords):
            for i in range(MAX_PIXELS):
                x, y = coords[i, 0], coords[i, 1]
                valid = (x >= 0) & (y >= 0)
                idx = y * max_w + x
                mask = jax.lax.cond(
                    valid, lambda m: m.at[idx].set(False), lambda m: m, mask
                )
            return mask

        valid_mask = mark_occupied(valid_mask, level.agent_pos)
        valid_mask = mark_occupied(valid_mask, level.m1_pos)

        new_idx = jax.random.choice(
            rng, max_h * max_w, p=valid_mask.astype(jnp.float32)
        )
        new_y, new_x = new_idx // max_w, new_idx % max_w

        new_g1_pos = jnp.full((MAX_PIXELS, 2), -1, dtype=jnp.int32)
        new_g1_pos = new_g1_pos.at[0].set(jnp.array([new_x, new_y]))

        # Clear wall at new goal position
        new_wall_map = level.wall_map.at[new_y, new_x].set(False)

        return level.replace(g1_pos=new_g1_pos, wall_map=new_wall_map)

    def mutate(rng: chex.PRNGKey, level: Level, n: int = 1) -> Level:
        def _mutate(carry, step):
            lvl = carry
            mutation_rng, mutation = step

            def _apply(rng, lvl):
                rng, flip_rng, move_rng = jax.random.split(rng, 3)

                is_flip = jnp.equal(mutation, MinimaxMutations.FLIP_WALL.value)
                flipped = flip_wall(flip_rng, lvl)
                next_lvl = jax.tree_util.tree_map(
                    lambda x, y: jax.lax.select(is_flip, x, y), flipped, lvl
                )

                is_move = jnp.equal(mutation, MinimaxMutations.MOVE_G1.value)
                moved = move_g1(move_rng, lvl)
                next_lvl = jax.tree_util.tree_map(
                    lambda x, y: jax.lax.select(is_move, x, y), moved, next_lvl
                )

                return next_lvl

            return jax.lax.cond(
                mutation != MinimaxMutations.NO_OP.value,
                _apply,
                lambda *_: lvl,
                mutation_rng,
                lvl,
            ), None

        rng, nrng, *mrngs = jax.random.split(rng, max_num_edits + 2)
        mutations = jax.random.choice(
            nrng, len(MinimaxMutations), shape=(max_num_edits,)
        )
        mutations = jnp.where(
            jnp.arange(max_num_edits) < n, mutations, MinimaxMutations.NO_OP.value
        )

        new_level, _ = jax.lax.scan(_mutate, level, (jnp.array(mrngs), mutations))

        return new_level

    return mutate
