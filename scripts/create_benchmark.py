#!/usr/bin/env python3
"""
Create benchmark pickle files from .pwp puzzle directories.

This script creates numpy-based pickle files that are compatible across
JAX versions (avoiding the named_shape issue with JAX arrays).

Usage:
    python create_benchmark.py --train-dir /path/to/train --test-dir /path/to/test --output benchmark.pkl
    
    # Create level0_transformed_all benchmark:
    python create_benchmark.py \
        --train-dir ~/pushworld/benchmark/puzzles/level0_transformed/all/train \
        --test-dir ~/pushworld/benchmark/puzzles/level0_transformed/all/test \
        --output pushworld_level0_transformed_all.pkl \
        --name level0_transformed_all

Output format:
    The pickle file contains a dict with:
    - "train": numpy array of shape (num_train, 202) 
    - "test": numpy array of shape (num_test, 202)
    - "name": benchmark name (optional)
    - "version": format version
    
    Each puzzle is encoded as 202 values:
    - agent_pos: 6 values (3 pixels * 2 coords)
    - m1_pos: 6 values
    - m2_pos: 6 values  
    - m3_pos: 6 values
    - m4_pos: 6 values
    - g1_pos: 6 values
    - g2_pos: 6 values
    - walls: 160 values (80 wall positions * 2 coords)
"""

import argparse
import bz2
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from jaxued.environments.pushworld.level import Level, MAX_WALLS, MAX_PIXELS


def load_puzzle_from_file(path: str) -> Level:
    """Load a puzzle from a .pwp file."""
    with open(path, "r") as f:
        content = f.read()
    return Level.from_str(content)


def encode_level(level: Level) -> np.ndarray:
    """Encode a Level to array format for storage.
    
    Returns numpy array of shape (202,) containing:
    - agent position (6 values)
    - m1-m4 positions (24 values)
    - g1-g2 positions (12 values)
    - wall positions (160 values)
    """
    # Get wall coordinates from wall_map
    wall_coords = []
    wall_map = np.asarray(level.wall_map)
    for y in range(wall_map.shape[0]):
        for x in range(wall_map.shape[1]):
            if wall_map[y, x]:
                wall_coords.append([x, y])
    
    # Pad walls to MAX_WALLS with -1 (invalid)
    while len(wall_coords) < MAX_WALLS:
        wall_coords.append([-1, -1])
    wall_coords = wall_coords[:MAX_WALLS]
    
    return np.concatenate([
        np.asarray(level.agent_pos).flatten(),
        np.asarray(level.m1_pos).flatten(),
        np.asarray(level.m2_pos).flatten(),
        np.asarray(level.m3_pos).flatten(),
        np.asarray(level.m4_pos).flatten(),
        np.asarray(level.g1_pos).flatten(),
        np.asarray(level.g2_pos).flatten(),
        np.array(wall_coords).flatten(),
    ]).astype(np.int32)


def load_puzzles_from_dir(directory: str, max_puzzles: int = None) -> np.ndarray:
    """Load all .pwp files from a directory and encode them."""
    files = sorted(glob.glob(os.path.join(directory, "*.pwp")))
    if max_puzzles is not None:
        files = files[:max_puzzles]
    
    if not files:
        raise ValueError(f"No .pwp files found in {directory}")
    
    puzzles = []
    print(f"Loading {len(files)} puzzles from {directory}...")
    
    for f in tqdm(files, desc="Loading"):
        try:
            level = load_puzzle_from_file(f)
            encoded = encode_level(level)
            puzzles.append(encoded)
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
    
    return np.stack(puzzles)


def create_benchmark(
    train_dir: str,
    test_dir: str,
    output_path: str,
    name: str = None,
    max_train: int = None,
    max_test: int = None,
):
    """Create a benchmark pickle file from puzzle directories."""
    print(f"Creating benchmark: {name or output_path}")
    print(f"  Train dir: {train_dir}")
    print(f"  Test dir: {test_dir}")
    
    # Load puzzles
    train_puzzles = load_puzzles_from_dir(train_dir, max_train)
    test_puzzles = load_puzzles_from_dir(test_dir, max_test)
    
    print(f"\nLoaded {train_puzzles.shape[0]} train puzzles")
    print(f"Loaded {test_puzzles.shape[0]} test puzzles")
    print(f"Puzzle encoding shape: {train_puzzles.shape[1]}")
    
    # Create data dict with numpy arrays (NOT JAX arrays!)
    data = {
        "train": train_puzzles,  # numpy array
        "test": test_puzzles,    # numpy array
        "version": 2,            # Version 2 = numpy format
    }
    if name:
        data["name"] = name
    
    # Save as bz2-compressed pickle
    print(f"\nSaving to {output_path}...")
    with bz2.open(output_path, "wb") as f:
        pickle.dump(data, f)
    
    # Verify the file
    file_size = os.path.getsize(output_path)
    print(f"Saved! File size: {file_size / 1024 / 1024:.2f} MB")
    
    # Verify it can be loaded
    print("Verifying file can be loaded...")
    with bz2.open(output_path, "rb") as f:
        loaded = pickle.load(f)
    assert loaded["train"].shape == train_puzzles.shape
    assert loaded["test"].shape == test_puzzles.shape
    print("✓ Verification passed!")
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Create benchmark pickle files from .pwp puzzle directories"
    )
    parser.add_argument(
        "--train-dir", required=True,
        help="Directory containing training .pwp files"
    )
    parser.add_argument(
        "--test-dir", required=True,
        help="Directory containing test .pwp files"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output pickle file path (will be bz2 compressed)"
    )
    parser.add_argument(
        "--name", default=None,
        help="Benchmark name to store in the file"
    )
    parser.add_argument(
        "--max-train", type=int, default=None,
        help="Maximum number of training puzzles to include"
    )
    parser.add_argument(
        "--max-test", type=int, default=None,
        help="Maximum number of test puzzles to include"
    )
    
    args = parser.parse_args()
    
    # Expand paths
    train_dir = os.path.expanduser(args.train_dir)
    test_dir = os.path.expanduser(args.test_dir)
    output_path = os.path.expanduser(args.output)
    
    # Validate directories exist
    if not os.path.isdir(train_dir):
        print(f"Error: Train directory not found: {train_dir}")
        sys.exit(1)
    if not os.path.isdir(test_dir):
        print(f"Error: Test directory not found: {test_dir}")
        sys.exit(1)
    
    create_benchmark(
        train_dir=train_dir,
        test_dir=test_dir,
        output_path=output_path,
        name=args.name,
        max_train=args.max_train,
        max_test=args.max_test,
    )


if __name__ == "__main__":
    main()
