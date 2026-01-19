"""
PushWorld Level representation for the "All" variant.

This module defines the Level class which represents a PushWorld puzzle.
The "All" variant supports:
- 10x10 grid
- Multi-pixel agent and objects (polyominoes up to 3 pixels each)
- 4 movable objects (m1, m2, m3, m4)
- 2 goal positions (g1, g2)
- Walls
"""

import bz2
import os
import pickle
import urllib.request
from collections import defaultdict
from typing import Literal

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

# Constants for the "All" variant
MAX_PIXELS = 3  # Max pixels per object (polyomino)
MAX_WALLS = 80  # Max wall positions
GRID_SIZE = 10  # Grid size for "All" variant

# Hugging Face download settings
HF_REPO_ID = os.environ.get("PUSHWORLD_HF_REPO_ID", "feynmaniac/pushworld")
DATA_PATH = os.environ.get("PUSHWORLD_DATA", os.path.expanduser("~/.pushworld"))

NAME2HFFILENAME = {
    "level0_transformed_all": "pushworld_level0_transformed_all.pkl",
    "level0_transformed_base": "pushworld_level0_transformed_base.pkl",
    "level0_mini": "pushworld_level0_mini.pkl",
}


def _download_from_hf(repo_id: str, filename: str) -> None:
    """Download a benchmark file from Hugging Face."""
    dataset_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
    save_path = os.path.join(DATA_PATH, filename)
    print(f"Downloading benchmark: {dataset_url} -> {save_path}")
    urllib.request.urlretrieve(dataset_url, save_path)
    if not os.path.exists(save_path):
        raise IOError(f"Failed to download benchmark from {dataset_url}")


def registered_benchmarks() -> tuple:
    """Return list of available benchmark names."""
    return tuple(NAME2HFFILENAME.keys())


@struct.dataclass
class Level:
    """Represents a PushWorld level/puzzle.

    All coordinate arrays use -1 as padding for unused pixels.
    Coordinates are stored as (x, y) pairs flattened: [x0, y0, x1, y1, ...]
    """

    # Agent position (up to 3 pixels) - shape: (MAX_PIXELS, 2)
    agent_pos: chex.Array
    # Movable objects (up to 3 pixels each) - shape: (MAX_PIXELS, 2)
    m1_pos: chex.Array
    m2_pos: chex.Array
    m3_pos: chex.Array
    m4_pos: chex.Array
    # Goal positions (up to 3 pixels each) - shape: (MAX_PIXELS, 2)
    g1_pos: chex.Array
    g2_pos: chex.Array
    # Wall map as 2D boolean array - shape: (GRID_SIZE, GRID_SIZE)
    wall_map: chex.Array
    # Grid dimensions
    width: int = GRID_SIZE
    height: int = GRID_SIZE

    def is_well_formatted(self):
        """Check if the level is well-formatted."""
        wall_map_is_binary = jnp.all((self.wall_map == 0) | (self.wall_map == 1))

        # Check agent has at least one valid pixel
        agent_valid = jnp.any(self.agent_pos[:, 0] >= 0)

        # Check at least one goal exists
        g1_exists = jnp.any(self.g1_pos[:, 0] >= 0)
        g2_exists = jnp.any(self.g2_pos[:, 0] >= 0)
        has_goal = g1_exists | g2_exists

        # Check corresponding movable exists for each goal
        m1_exists = jnp.any(self.m1_pos[:, 0] >= 0)
        m2_exists = jnp.any(self.m2_pos[:, 0] >= 0)
        goals_have_movables = (~g1_exists | m1_exists) & (~g2_exists | m2_exists)

        return wall_map_is_binary & agent_valid & has_goal & goals_have_movables

    @classmethod
    def from_str(cls, level_str: str):
        """Parse a PushWorld puzzle from string format (.pwp format).

        Format uses space-separated tokens:
        - '.' = empty
        - 'W' = wall
        - 'A' = agent
        - 'M1', 'M2', 'M3', 'M4' = movable objects
        - 'G1', 'G2' = goals
        """
        level_str = level_str.strip()
        lines = [line.strip() for line in level_str.split("\n") if line.strip()]

        original_height = len(lines)
        original_width = len(lines[0].split()) if lines else 0

        # Calculate padding to center in GRID_SIZE x GRID_SIZE
        x_padding_total = GRID_SIZE - original_width
        y_padding_total = GRID_SIZE - original_height
        x_offset = max(0, x_padding_total // 2)
        y_offset = max(0, y_padding_total // 2)

        # Collect object pixels
        obj_pixels = defaultdict(set)
        for line_idx, line in enumerate(lines):
            y = line_idx + y_offset
            tokens = line.split()
            for x_idx, token in enumerate(tokens):
                x = x_idx + x_offset
                token_upper = token.upper()
                if token_upper != ".":
                    obj_pixels[token_upper].add((x, y))

        # Add walls in padded areas
        walls = set()
        # Top padding
        for y in range(y_offset):
            for x in range(GRID_SIZE):
                walls.add((x, y))
        # Bottom padding
        for y in range(y_offset + original_height, GRID_SIZE):
            for x in range(GRID_SIZE):
                walls.add((x, y))
        # Left padding
        for x in range(x_offset):
            for y in range(y_offset, y_offset + original_height):
                walls.add((x, y))
        # Right padding
        for x in range(x_offset + original_width, GRID_SIZE):
            for y in range(y_offset, y_offset + original_height):
                walls.add((x, y))

        # Merge with existing walls
        if "W" in obj_pixels:
            obj_pixels["W"].update(walls)
        else:
            obj_pixels["W"] = walls

        def coords_to_array(coords_set, max_objects=MAX_PIXELS):
            """Convert coordinate set to fixed-size array with -1 padding."""
            coords_list = sorted(list(coords_set))
            result = np.full((max_objects, 2), -1, dtype=np.int32)
            for i, (x, y) in enumerate(coords_list[:max_objects]):
                result[i] = [x, y]
            return result

        # Create wall map
        wall_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        for x, y in obj_pixels.get("W", set()):
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                wall_map[y, x] = True

        return cls(
            agent_pos=jnp.array(coords_to_array(obj_pixels.get("A", set()))),
            m1_pos=jnp.array(coords_to_array(obj_pixels.get("M1", set()))),
            m2_pos=jnp.array(coords_to_array(obj_pixels.get("M2", set()))),
            m3_pos=jnp.array(coords_to_array(obj_pixels.get("M3", set()))),
            m4_pos=jnp.array(coords_to_array(obj_pixels.get("M4", set()))),
            g1_pos=jnp.array(coords_to_array(obj_pixels.get("G1", set()))),
            g2_pos=jnp.array(coords_to_array(obj_pixels.get("G2", set()))),
            wall_map=jnp.array(wall_map),
            width=GRID_SIZE,
            height=GRID_SIZE,
        )

    def to_str(self) -> str:
        """Convert level back to string format."""
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]

        def place_object(coords, label):
            for i in range(coords.shape[0]):
                x, y = int(coords[i, 0]), int(coords[i, 1])
                if x >= 0 and y >= 0:
                    grid[y][x] = label

        # Place walls first
        for y in range(self.height):
            for x in range(self.width):
                if self.wall_map[y, x]:
                    grid[y][x] = "W"

        # Place objects (order matters for overlapping display)
        place_object(self.g1_pos, "G1")
        place_object(self.g2_pos, "G2")
        place_object(self.m1_pos, "M1")
        place_object(self.m2_pos, "M2")
        place_object(self.m3_pos, "M3")
        place_object(self.m4_pos, "M4")
        place_object(self.agent_pos, "A")

        return "\n".join(["  ".join(row) for row in grid])

    @classmethod
    def from_puzzle_array(cls, puzzle_array: chex.Array):
        """Convert from the encoded puzzle array format used in benchmarks.

        Array format (202 elements total):
        - a: 6 elements (3 pixels * 2 coords)
        - m1: 6 elements
        - m2: 6 elements
        - m3: 6 elements
        - m4: 6 elements
        - g1: 6 elements
        - g2: 6 elements
        - w: 160 elements (80 pixels * 2 coords)
        """
        idx = 0

        def extract_coords(arr, start, num_pixels):
            coords = arr[start : start + num_pixels * 2].reshape(num_pixels, 2)
            return coords

        agent_pos = extract_coords(puzzle_array, idx, MAX_PIXELS)
        idx += MAX_PIXELS * 2
        m1_pos = extract_coords(puzzle_array, idx, MAX_PIXELS)
        idx += MAX_PIXELS * 2
        m2_pos = extract_coords(puzzle_array, idx, MAX_PIXELS)
        idx += MAX_PIXELS * 2
        m3_pos = extract_coords(puzzle_array, idx, MAX_PIXELS)
        idx += MAX_PIXELS * 2
        m4_pos = extract_coords(puzzle_array, idx, MAX_PIXELS)
        idx += MAX_PIXELS * 2
        g1_pos = extract_coords(puzzle_array, idx, MAX_PIXELS)
        idx += MAX_PIXELS * 2
        g2_pos = extract_coords(puzzle_array, idx, MAX_PIXELS)
        idx += MAX_PIXELS * 2
        wall_coords = extract_coords(puzzle_array, idx, MAX_WALLS)

        # Convert wall coordinates to wall map
        wall_map = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.bool_)

        def add_wall(carry, coord):
            wall_map = carry
            x, y = coord
            valid = (x >= 0) & (y >= 0) & (x < GRID_SIZE) & (y < GRID_SIZE)
            wall_map = jax.lax.cond(
                valid, lambda wm: wm.at[y, x].set(True), lambda wm: wm, wall_map
            )
            return wall_map, None

        wall_map, _ = jax.lax.scan(add_wall, wall_map, wall_coords)

        return cls(
            agent_pos=agent_pos.astype(jnp.int32),
            m1_pos=m1_pos.astype(jnp.int32),
            m2_pos=m2_pos.astype(jnp.int32),
            m3_pos=m3_pos.astype(jnp.int32),
            m4_pos=m4_pos.astype(jnp.int32),
            g1_pos=g1_pos.astype(jnp.int32),
            g2_pos=g2_pos.astype(jnp.int32),
            wall_map=wall_map,
            width=GRID_SIZE,
            height=GRID_SIZE,
        )

    @classmethod
    def stack(cls, levels):
        """Stack multiple levels into a batched Level."""
        return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *levels)

    @classmethod
    def load_prefabs(cls, ids):
        """Load prefab puzzles by their string IDs."""
        return cls.stack([cls.from_str(prefabs[id]) for id in ids])


@struct.dataclass
class Benchmark:
    """Stores train and test puzzles for JIT-compatible sampling."""

    train_puzzles: chex.Array  # Shape: (num_train, 202)
    test_puzzles: chex.Array  # Shape: (num_test, 202)

    def num_train_puzzles(self) -> int:
        return self.train_puzzles.shape[0]

    def num_test_puzzles(self) -> int:
        return self.test_puzzles.shape[0]

    def get_puzzle(
        self, puzzle_id: chex.Array, puzzle_type: Literal["train", "test"] = "train"
    ) -> Level:
        """Get a specific puzzle by ID."""
        puzzle_array = jax.lax.cond(
            puzzle_type == "train",
            lambda: jax.lax.dynamic_index_in_dim(
                self.train_puzzles, puzzle_id, keepdims=False
            ),
            lambda: jax.lax.dynamic_index_in_dim(
                self.test_puzzles, puzzle_id, keepdims=False
            ),
        )
        return Level.from_puzzle_array(puzzle_array)

    def sample_puzzle(
        self, key: chex.PRNGKey, puzzle_type: Literal["train", "test"] = "train"
    ) -> Level:
        """Sample a random puzzle."""
        key, subkey = jax.random.split(key)
        num_puzzles = jax.lax.cond(
            puzzle_type == "train",
            lambda: self.num_train_puzzles(),
            lambda: self.num_test_puzzles(),
        )
        puzzle_id = jax.random.randint(subkey, shape=(), minval=0, maxval=num_puzzles)
        return self.get_puzzle(puzzle_id, puzzle_type)

    def get_all_train_puzzles(self) -> Level:
        """Get all training puzzles as a batched Level."""
        return jax.vmap(lambda i: self.get_puzzle(i, "train"))(
            jnp.arange(self.num_train_puzzles())
        )

    def get_all_test_puzzles(self) -> Level:
        """Get all test puzzles as a batched Level."""
        return jax.vmap(lambda i: self.get_puzzle(i, "test"))(
            jnp.arange(self.num_test_puzzles())
        )

    @classmethod
    def load_from_path(cls, path: str):
        """Load benchmark from a bz2-compressed pickle file.
        
        Handles JAX version incompatibility by using a custom unpickler
        that converts old JAX arrays to numpy arrays.
        """
        import io
        
        class JaxCompatUnpickler(pickle.Unpickler):
            """Custom unpickler that handles JAX version incompatibility."""
            
            def find_class(self, module, name):
                # Redirect JAX array reconstruction to numpy
                if module.startswith("jax") and "array" in name.lower():
                    return np.array
                # Handle old jax._src.core types
                if module == "jax._src.core" and name == "ShapedArray":
                    # Return a dummy that will be replaced
                    return lambda *args, **kwargs: None
                if module == "jax._src.array" and name == "_reconstruct_array":
                    # Return numpy array constructor instead
                    return lambda *args, **kwargs: np.array(args[0]) if args else np.array([])
                return super().find_class(module, name)
        
        with bz2.open(path, "rb") as f:
            try:
                # First try normal loading
                data = pickle.load(f)
            except (TypeError, AttributeError) as e:
                # If JAX version incompatibility, try with custom unpickler
                f.seek(0)
                try:
                    data = JaxCompatUnpickler(f).load()
                except Exception:
                    # Last resort: re-download and try again
                    f.close()
                    os.remove(path)
                    raise RuntimeError(
                        f"Failed to load benchmark from {path}. "
                        f"The file may be corrupted or incompatible. "
                        f"Please re-run to download a fresh copy. Original error: {e}"
                    )
        
        # Convert to numpy first, then to jax arrays
        train = np.asarray(data["train"]) if hasattr(data["train"], '__array__') else data["train"]
        test = np.asarray(data["test"]) if hasattr(data["test"], '__array__') else data["test"]
        
        return cls(
            train_puzzles=jnp.array(train),
            test_puzzles=jnp.array(test),
        )

    @classmethod
    def load(cls, name: str, force_redownload: bool = False):
        """Load a benchmark by name, auto-downloading from HuggingFace if needed.
        
        Args:
            name: One of 'level0_transformed_all', 'level0_transformed_base', 'level0_mini'
            force_redownload: If True, delete cached file and re-download
        
        Returns:
            Benchmark instance with train and test puzzles
            
        Example:
            >>> benchmark = Benchmark.load("level0_transformed_all")
            >>> print(f"Loaded {benchmark.num_train_puzzles()} training puzzles")
        """
        if name not in NAME2HFFILENAME:
            raise RuntimeError(
                f"Unknown benchmark '{name}'. Available: {registered_benchmarks()}"
            )

        os.makedirs(DATA_PATH, exist_ok=True)
        path = os.path.join(DATA_PATH, NAME2HFFILENAME[name])
        numpy_path = path.replace(".pkl", "_numpy.pkl")

        # Check for numpy-converted cache first
        if os.path.exists(numpy_path) and not force_redownload:
            print(f"Loading from numpy cache: {numpy_path}")
            with bz2.open(numpy_path, "rb") as f:
                data = pickle.load(f)
            return cls(
                train_puzzles=jnp.array(data["train"]),
                test_puzzles=jnp.array(data["test"]),
            )

        if force_redownload and os.path.exists(path):
            os.remove(path)
        if force_redownload and os.path.exists(numpy_path):
            os.remove(numpy_path)

        if not os.path.exists(path):
            _download_from_hf(HF_REPO_ID, NAME2HFFILENAME[name])

        # Load and convert to numpy cache for future use
        benchmark = cls.load_from_path(path)
        
        # Save numpy version for faster future loads
        try:
            numpy_data = {
                "train": np.asarray(benchmark.train_puzzles),
                "test": np.asarray(benchmark.test_puzzles),
            }
            with bz2.open(numpy_path, "wb") as f:
                pickle.dump(numpy_data, f)
            print(f"Saved numpy cache: {numpy_path}")
        except Exception as e:
            print(f"Warning: Could not save numpy cache: {e}")
        
        return benchmark

    @classmethod
    def from_prefabs(cls, train_ids: list, test_ids: list):
        """Create a benchmark from prefab puzzle IDs."""

        def encode_level(level: Level) -> chex.Array:
            """Encode a Level back to the array format."""

            def coords_to_flat(coords, max_pixels):
                flat = coords.flatten()
                return flat

            # Get wall coordinates from wall_map
            wall_coords = []
            for y in range(level.height):
                for x in range(level.width):
                    if level.wall_map[y, x]:
                        wall_coords.append([x, y])

            # Pad walls to MAX_WALLS
            while len(wall_coords) < MAX_WALLS:
                wall_coords.append([-1, -1])
            wall_coords = wall_coords[:MAX_WALLS]

            return jnp.concatenate(
                [
                    level.agent_pos.flatten(),
                    level.m1_pos.flatten(),
                    level.m2_pos.flatten(),
                    level.m3_pos.flatten(),
                    level.m4_pos.flatten(),
                    level.g1_pos.flatten(),
                    level.g2_pos.flatten(),
                    jnp.array(wall_coords).flatten(),
                ]
            )

        train_levels = [Level.from_str(prefabs[id]) for id in train_ids]
        test_levels = [Level.from_str(prefabs[id]) for id in test_ids]

        train_puzzles = jnp.stack([encode_level(lvl) for lvl in train_levels])
        test_puzzles = jnp.stack([encode_level(lvl) for lvl in test_levels])

        return cls(train_puzzles=train_puzzles, test_puzzles=test_puzzles)


# ============================================================================
# Prefab puzzles for testing
# ============================================================================

# Simple puzzle: agent pushes M1 to G1
TrivialPush = """
.  .  .  .  .
.  A  M1  G1  .
.  .  .  .  .
.  .  .  .  .
.  .  .  .  .
"""

# Agent needs to go around wall to push
SimplePush = """
.  .  .  .  .  .
.  A  .  .  .  .
.  W  W  W  .  .
.  .  M1  .  .  .
.  .  G1  .  .  .
.  .  .  .  .  .
"""

# Two goals puzzle
TwoGoals = """
.  .  .  .  .  .
.  A  .  .  .  .
.  M1  .  M2  .  .
.  G1  .  G2  .  .
.  .  .  .  .  .
.  .  .  .  .  .
"""

# Chain push: agent pushes M1 into M2
ChainPush = """
.  .  .  .  .  .
.  A  M1  M2  .  .
.  .  .  G1  .  .
.  .  .  .  .  .
.  .  .  .  .  .
.  .  .  .  .  .
"""

# Multi-pixel agent
MultiPixelAgent = """
.  .  .  .  .  .  .
.  A  A  .  .  .  .
.  A  .  .  .  .  .
.  .  M1  M1  .  .  .
.  .  .  G1  G1  .  .
.  .  .  .  .  .  .
.  .  .  .  .  .  .
"""

# L-shaped movable
LShapedPush = """
.  .  .  .  .  .  .
.  A  .  .  .  .  .
.  M1  M1  .  .  .  .
.  M1  .  .  .  .  .
.  .  .  G1  G1  .  .
.  .  .  G1  .  .  .
.  .  .  .  .  .  .
"""

# Wall maze puzzle
WallMaze = """
.  .  .  .  .  .  .  .
.  A  .  W  .  .  .  .
.  .  .  W  .  .  .  .
.  .  .  W  W  W  .  .
.  .  M1  .  .  .  .  .
.  .  .  .  .  G1  .  .
.  .  .  .  .  .  .  .
.  .  .  .  .  .  .  .
"""

# Complex puzzle with multiple objects
ComplexPuzzle = """
.  W  .  .  .  .  .  .
.  .  .  .  M1  .  .  .
.  .  .  M1  M1  .  .  .
.  .  W  .  A  A  A  .
.  W  .  .  .  .  .  .
.  .  .  .  .  .  M2  .
.  W  .  .  .  .  .  .
.  .  .  .  .  W  .  G1
.  M3  M3  .  .  .  G1  G1
"""

# ============================================================================
# Puzzles from PushWorld benchmark (level0_transformed/all/test)
# ============================================================================

# Test puzzle 131 - L-shaped polyominoes
Test131 = """
.  W  .  W  A  .
.  .  .  .  A  A
.  .  G1  G1  G1  .
.  .  M1  M1  M1  .
.  .  .  W  .  .
.  .  .  M2  M2  M2
"""

# Test puzzle 22 - Multiple objects spread out
Test22 = """
.  .  .  .  .  .  .  .
.  .  .  .  W  .  .  A
.  .  .  W  .  .  A  A
.  .  .  .  .  .  .  W
.  .  .  .  .  .  .  .
.  .  .  .  .  .  G1  G1
.  .  .  M1  M1  .  .  G1
.  .  .  .  M1  .  M2  .
.  W  .  .  .  .  M2  .
"""

# Test puzzle 23 - Vertical M1 pushing to G1
Test23 = """
M2  M2  .  .  .  .  .  .  .
.  M2  .  M1  .  G1  W  .  .
.  .  .  M1  .  G1  .  .  W
.  .  W  M1  .  G1  .  .  .
.  .  W  .  .  .  W  .  .
.  .  .  .  .  .  .  A  A
"""

# Test puzzle 36 - Two goals with M3
Test36 = """
.  .  G1  G1  G1  .  .  .  .
M3  .  .  G2  M2  A  .  .  .
M3  M3  .  .  .  A  .  .  .
.  .  .  .  .  .  .  .  .
.  .  .  .  .  .  .  .  .
.  W  .  M1  M1  M1  .  W  .
.  .  .  .  .  .  .  .  .
.  .  .  W  .  .  .  .  .
"""

# Test puzzle 127 - Multiple movables with walls
Test127 = """
.  W  .  .  .  .  .  .
.  .  .  .  M1  .  .  .
.  .  .  M1  M1  .  .  .
.  .  W  .  A  A  A  .
.  W  .  .  .  .  .  .
.  .  .  .  .  .  M2  .
.  W  .  .  .  .  .  .
.  .  .  .  .  W  .  G1
.  M3  M3  .  .  .  G1  G1
"""


prefabs = {
    # Simple test puzzles
    "TrivialPush": TrivialPush.strip(),
    "SimplePush": SimplePush.strip(),
    "TwoGoals": TwoGoals.strip(),
    "ChainPush": ChainPush.strip(),
    "MultiPixelAgent": MultiPixelAgent.strip(),
    "LShapedPush": LShapedPush.strip(),
    "WallMaze": WallMaze.strip(),
    "ComplexPuzzle": ComplexPuzzle.strip(),
    # Benchmark puzzles
    "Test131": Test131.strip(),
    "Test22": Test22.strip(),
    "Test23": Test23.strip(),
    "Test36": Test36.strip(),
    "Test127": Test127.strip(),
}


def load_puzzle_from_file(path: str) -> Level:
    """Load a puzzle from a .pwp file.

    Args:
        path: Path to the .pwp puzzle file

    Returns:
        Level instance
    """
    with open(path, "r") as f:
        content = f.read()
    return Level.from_str(content)
