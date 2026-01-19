"""
Interactive PushWorld Test - Play levels with arrow keys.

Controls (Level Select):
- Click on a level to play it
- Arrow keys or scroll to navigate
- Q or ESC: Quit

Controls (Playing):
- Arrow keys: Move agent (up/down/left/right)
- R: Reset current level
- N: Next level
- P: Previous level
- B or ESC: Back to level select
- Q: Quit
"""

import sys
from pathlib import Path

import jax
import numpy as np

# ============================================================================
# CONFIGURATION - Edit this to point to your puzzle directory
# ============================================================================

# Path to directory containing .pwp puzzle files
PUZZLE_DIR = "/Users/kimyoungjin/Projects/monkey/pushworld/benchmark/puzzles/level0_transformed/all/test"

# Alternatively, use prefab puzzles (set to True to use built-in prefabs instead)
USE_PREFABS = True

# Window settings
TILE_SIZE = 64  # Pixels per grid cell when playing
PREVIEW_TILE_SIZE = 8  # Pixels per grid cell in preview
PREVIEW_COLS = 4  # Number of columns in level select grid
WINDOW_TITLE = "PushWorld Interactive Test"

# ============================================================================

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from jaxued.environments.pushworld.env import Actions, PushWorld
from jaxued.environments.pushworld.level import (
    GRID_SIZE,
    Level,
    load_puzzle_from_file,
    prefabs,
)
from jaxued.environments.pushworld.renderer import PushWorldRenderer


def load_levels_from_directory(puzzle_dir: str) -> list[tuple[str, Level]]:
    """Load all .pwp files from a directory."""
    levels = []
    puzzle_path = Path(puzzle_dir)

    if not puzzle_path.exists():
        print(f"Warning: Puzzle directory not found: {puzzle_dir}")
        return levels

    pwp_files = sorted(puzzle_path.glob("*.pwp"))

    for pwp_file in pwp_files:
        try:
            level = load_puzzle_from_file(str(pwp_file))
            levels.append((pwp_file.stem, level))
        except Exception as e:
            print(f"Warning: Failed to load {pwp_file.name}: {e}")

    return levels


def load_prefab_levels() -> list[tuple[str, Level]]:
    """Load all prefab puzzles."""
    levels = []
    for name in prefabs.keys():
        level = Level.from_str(prefabs[name])
        levels.append((name, level))
    return levels


def main():
    """Main game loop."""
    try:
        import pygame
    except ImportError:
        print("pygame is required for interactive mode.")
        print("Install it with: pip install pygame")
        sys.exit(1)

    # Initialize pygame
    pygame.init()

    # Load levels
    if USE_PREFABS:
        levels = load_prefab_levels()
        print(f"Loaded {len(levels)} prefab puzzles")
    else:
        levels = load_levels_from_directory(PUZZLE_DIR)
        print(f"Loaded {len(levels)} puzzles from {PUZZLE_DIR}")

    if not levels:
        print("No levels found! Check PUZZLE_DIR or set USE_PREFABS = True")
        sys.exit(1)

    # Initialize environment and renderers
    env = PushWorld()
    play_renderer = PushWorldRenderer(env, tile_size=TILE_SIZE, render_grid_lines=True)
    preview_renderer = PushWorldRenderer(
        env, tile_size=PREVIEW_TILE_SIZE, render_grid_lines=False
    )

    # Calculate window sizes
    play_window_size = GRID_SIZE * TILE_SIZE
    preview_size = GRID_SIZE * PREVIEW_TILE_SIZE  # Size of each preview
    preview_padding = 10
    preview_cell_size = preview_size + preview_padding * 2

    # Calculate level select window size
    num_rows = (len(levels) + PREVIEW_COLS - 1) // PREVIEW_COLS
    select_width = PREVIEW_COLS * preview_cell_size + preview_padding * 2
    select_height = (
        num_rows * (preview_cell_size + 20) + 80
    )  # +20 for labels, +80 for header

    # Use the larger of the two for window size
    window_width = max(play_window_size, select_width)
    window_height = max(play_window_size + 60, select_height)

    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption(WINDOW_TITLE)

    # Fonts
    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 18)
    title_font = pygame.font.Font(None, 36)

    # Pre-render all level previews
    print("Generating level previews...")
    preview_surfaces = []
    for i, (name, level) in enumerate(levels):
        state = env.init_state_from_level(level)
        img = preview_renderer.render_state(state, env.default_params)
        img_np = np.array(img, dtype=np.uint8)
        surface = pygame.surfarray.make_surface(img_np.swapaxes(0, 1))
        preview_surfaces.append(surface)
    print("Done!")

    # Game state
    rng = jax.random.PRNGKey(0)
    current_level_idx = 0
    state = None
    obs = None
    steps = 0
    done = False
    mode = "select"  # "select" or "play"
    scroll_offset = 0

    def reset_level(level_idx: int):
        """Reset to a specific level."""
        nonlocal current_level_idx, state, obs, steps, done
        current_level_idx = level_idx % len(levels)
        name, level = levels[current_level_idx]
        obs, state = env.reset_env_to_level(rng, level, env.default_params)
        steps = 0
        done = False
        return name

    current_name = ""

    # Action mapping for play mode
    key_to_action = {
        pygame.K_UP: Actions.up,
        pygame.K_DOWN: Actions.down,
        pygame.K_LEFT: Actions.left,
        pygame.K_RIGHT: Actions.right,
    }

    clock = pygame.time.Clock()
    running = True

    print("\nLevel Select: Click on a level to play")
    print("Playing: Arrow keys to move, B to go back, R to reset")
    print()

    while running:
        mouse_pos = pygame.mouse.get_pos()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

                elif mode == "select":
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_DOWN:
                        scroll_offset = min(
                            scroll_offset + 50,
                            max(0, select_height - window_height + 100),
                        )
                    elif event.key == pygame.K_UP:
                        scroll_offset = max(0, scroll_offset - 50)

                elif mode == "play":
                    if event.key in (pygame.K_b, pygame.K_ESCAPE):
                        mode = "select"
                        print("Back to level select")

                    elif event.key == pygame.K_r:
                        current_name = reset_level(current_level_idx)
                        print(f"Reset: {current_name}")

                    elif event.key == pygame.K_n:
                        current_name = reset_level(current_level_idx + 1)
                        mode = "play"
                        print(f"Next level: {current_name}")

                    elif event.key == pygame.K_p:
                        current_name = reset_level(current_level_idx - 1)
                        mode = "play"
                        print(f"Previous level: {current_name}")

                    elif event.key in key_to_action and not done:
                        action = key_to_action[event.key]
                        obs, state, reward, done, info = env.step_env(
                            rng, state, action, env.default_params
                        )
                        steps += 1

                        if done:
                            if reward > 0:
                                print(
                                    f"  🎉 Level completed in {steps} steps! (reward: {reward:.2f})"
                                )
                            else:
                                print(f"  ❌ Level failed after {steps} steps")

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if mode == "select" and event.button == 1:  # Left click
                    # Check which level was clicked
                    mx, my = event.pos
                    my += scroll_offset  # Adjust for scroll

                    # Calculate centering offset (must match drawing code)
                    center_offset = (
                        window_width - PREVIEW_COLS * preview_cell_size
                    ) // 2

                    for i in range(len(levels)):
                        col = i % PREVIEW_COLS
                        row = i // PREVIEW_COLS
                        x = preview_padding + col * preview_cell_size + center_offset
                        y = 70 + row * (preview_cell_size + 20)

                        if x <= mx < x + preview_size and y <= my < y + preview_size:
                            current_name = reset_level(i)
                            mode = "play"
                            print(f"Playing: {current_name}")
                            break

                elif mode == "select" and event.button in (4, 5):  # Scroll wheel
                    if event.button == 4:  # Scroll up
                        scroll_offset = max(0, scroll_offset - 30)
                    else:  # Scroll down
                        scroll_offset = min(
                            scroll_offset + 30,
                            max(0, select_height - window_height + 100),
                        )

        # Clear screen
        screen.fill((30, 30, 40))

        if mode == "select":
            # Draw level select screen
            title = title_font.render("Select a Level", True, (255, 255, 255))
            screen.blit(
                title, (window_width // 2 - title.get_width() // 2, 15 - scroll_offset)
            )

            subtitle = small_font.render(
                f"{len(levels)} levels available - Click to play, Q to quit",
                True,
                (150, 150, 150),
            )
            screen.blit(
                subtitle,
                (window_width // 2 - subtitle.get_width() // 2, 45 - scroll_offset),
            )

            # Draw level previews
            for i, (name, level) in enumerate(levels):
                col = i % PREVIEW_COLS
                row = i // PREVIEW_COLS
                x = (
                    preview_padding
                    + col * preview_cell_size
                    + (window_width - PREVIEW_COLS * preview_cell_size) // 2
                )
                y = 70 + row * (preview_cell_size + 20) - scroll_offset

                # Skip if off screen
                if y + preview_cell_size < 0 or y > window_height:
                    continue

                # Draw preview background
                rect = pygame.Rect(x - 5, y - 5, preview_size + 10, preview_size + 10)

                # Highlight on hover
                if rect.collidepoint(mouse_pos[0], mouse_pos[1] + scroll_offset):
                    pygame.draw.rect(screen, (80, 80, 100), rect, border_radius=5)
                else:
                    pygame.draw.rect(screen, (50, 50, 60), rect, border_radius=5)

                # Draw preview
                screen.blit(preview_surfaces[i], (x, y))

                # Draw border
                pygame.draw.rect(
                    screen, (100, 100, 120), rect, width=2, border_radius=5
                )

                # Draw level name (truncate if too long)
                display_name = name[:12] + "..." if len(name) > 15 else name
                name_text = small_font.render(
                    f"{i+1}. {display_name}", True, (200, 200, 200)
                )
                text_x = x + preview_size // 2 - name_text.get_width() // 2
                screen.blit(name_text, (text_x, y + preview_size + 5))

        else:  # Play mode
            # Render current state
            if state is not None:
                img = play_renderer.render_state(state, env.default_params)
                img_np = np.array(img, dtype=np.uint8)
                surface = pygame.surfarray.make_surface(img_np.swapaxes(0, 1))

                # Center the game view
                game_x = (window_width - play_window_size) // 2
                screen.blit(surface, (game_x, 0))

            # Draw info bar
            info_y = play_window_size + 5

            level_text = font.render(
                f"Level {current_level_idx + 1}/{len(levels)}: {current_name}",
                True,
                (255, 255, 255),
            )
            screen.blit(level_text, (10, info_y))

            steps_text = font.render(f"Steps: {steps}", True, (200, 200, 200))
            screen.blit(steps_text, (window_width - 100, info_y))

            if done:
                status_text = font.render(
                    "DONE! R=Reset, N=Next, B=Back", True, (100, 255, 100)
                )
                screen.blit(status_text, (10, info_y + 25))
            else:
                hint_text = small_font.render(
                    "Arrows=Move | R=Reset | N/P=Next/Prev | B=Back | Q=Quit",
                    True,
                    (150, 150, 150),
                )
                screen.blit(hint_text, (10, info_y + 28))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
