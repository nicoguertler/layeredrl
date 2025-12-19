"""Simple 2D maze environment with velocity-controlled point mass."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Optional, Tuple, Dict, Any


class Maze2DEnv(gym.Env):
    """
    A simple 2D maze environment with a velocity-controlled point mass.

    The agent controls velocity directly. The environment includes:
    - Collision detection with walls
    - Configurable maze layouts
    - Pygame-based visualization for rendering and planning overlays

    Observation:
        Type: Dict with keys:
            - 'observation': Box(2) - current position [x, y]
            - 'achieved_goal': Box(2) - current position [x, y]
            - 'desired_goal': Box(2) - goal position [x, y]
        Each Box has:
            Min: [0, 0]
            Max: [maze_width, maze_height]

    Action:
        Type: Box(2)
        Num    Action                Min            Max
        0      x velocity            -max_velocity  max_velocity
        1      y velocity            -max_velocity  max_velocity

    Reward:
        Sparse reward of 1.0 when reaching the goal, 0.0 otherwise.
        Can be customized with a reward function.

    Episode Termination:
        - Agent reaches within goal_radius of the goal position
        - Episode length is greater than max_episode_steps
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        maze_layout: Optional[np.ndarray] = None,
        maze_size: Tuple[int, int] = (10, 10),
        cell_size: float = 1.0,
        start_pos: Optional[List[Tuple[float, float]]] = None,
        goal_pos: Optional[List[Tuple[float, float]]] = None,
        goal_radius: float = 0.3,
        max_velocity: float = 1.0,
        dt: float = 0.1,
        max_episode_steps: int = 400,
        dense_reward: bool = True,
        render_mode: Optional[str] = None,
        pixel_size: int = 600,
    ):
        """
        Initialize the Maze2D environment.

        Args:
            maze_layout: Binary array where 1 = wall, 0 = free space. If None, creates empty maze.
            maze_size: Size of the maze in cells (height, width) if maze_layout is None
            cell_size: Size of each cell in world coordinates
            start_pos: List of starting positions (x, y). If None, uses all empty cells.
            goal_pos: List of goal positions (x, y). If None, uses all empty cells.
            goal_radius: Distance threshold for reaching the goal
            max_velocity: Maximum velocity magnitude in each dimension
            dt: Time step for integration
            max_episode_steps: Maximum steps per episode
            dense_reward: If True, provide dense reward based on distance to goal
            render_mode: "human" or "rgb_array"
            pixel_size: Size of the rendering window in pixels
        """
        super().__init__()

        # Maze layout: 1 = wall, 0 = free space
        if maze_layout is None:
            self.maze_layout = np.zeros(maze_size, dtype=np.uint8)
        else:
            self.maze_layout = np.array(maze_layout, dtype=np.uint8)

        # length of diagonal of maze
        self.maze_diameter = np.linalg.norm(
            np.array(self.maze_layout.shape) * cell_size
        )

        self.maze_height, self.maze_width = self.maze_layout.shape
        self.cell_size = cell_size
        self.world_width = self.maze_width * cell_size
        self.world_height = self.maze_height * cell_size

        # Agent parameters
        self.max_velocity = max_velocity
        self.dt = dt
        self.agent_radius = 0.15  # Agent collision radius

        # Start and goal positions
        if start_pos is None:
            free_cells = np.argwhere(self.maze_layout == 0)
            self.start_pos_lst = np.array(
                [((x + 0.5) * cell_size, (y + 0.5) * cell_size) for y, x in free_cells],
                dtype=np.float32,
            )
        else:
            self.start_pos_lst = np.array(start_pos, dtype=np.float32)
        self.start_pos = self.start_pos_lst[0]  # Default to first start
        if goal_pos is None:
            free_cells = np.argwhere(self.maze_layout == 0)
            self.goal_pos_lst = np.array(
                [((x + 0.5) * cell_size, (y + 0.5) * cell_size) for y, x in free_cells],
                dtype=np.float32,
            )
        else:
            self.goal_pos_lst = np.array(goal_pos, dtype=np.float32)
        self.goal_pos = self.goal_pos_lst[0]  # Default to first goal
        self.goal_radius = goal_radius

        # Episode management
        self.max_episode_steps = max_episode_steps
        self.dense_reward = dense_reward
        self._step_count = 0

        # State: [x, y] - position only
        self.state = None

        # Gymnasium spaces - goal-conditioned observation space
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=np.array([0, 0], dtype=np.float32),
                    high=np.array(
                        [self.world_width, self.world_height], dtype=np.float32
                    ),
                    dtype=np.float32,
                ),
                "achieved_goal": spaces.Box(
                    low=np.array([0, 0], dtype=np.float32),
                    high=np.array(
                        [self.world_width, self.world_height], dtype=np.float32
                    ),
                    dtype=np.float32,
                ),
                "desired_goal": spaces.Box(
                    low=np.array([0, 0], dtype=np.float32),
                    high=np.array(
                        [self.world_width, self.world_height], dtype=np.float32
                    ),
                    dtype=np.float32,
                ),
            }
        )

        self.action_space = spaces.Box(
            low=-max_velocity, high=max_velocity, shape=(2,), dtype=np.float32
        )

        # Rendering
        self.render_mode = render_mode
        self.pixel_size = pixel_size
        self.window = None
        self.clock = None
        self._plans_to_render = []  # List of plans for visualization

        self.reset()

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        Get the current observation in goal-conditioned format.

        Returns:
            Dict with 'observation', 'achieved_goal', and 'desired_goal' keys
        """
        return {
            "observation": self.state.copy().astype(np.float32),
            "achieved_goal": self.state.copy().astype(np.float32),
            "desired_goal": self.goal_pos.copy().astype(np.float32),
        }

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Reset goal to random
        self.goal_pos = self.goal_pos_lst[
            self.np_random.integers(len(self.goal_pos_lst))
        ]

        # Reset state
        for _ in range(100):  # Try up to 100 times to find a valid start
            self.state = self.start_pos_lst[
                self.np_random.integers(len(self.start_pos_lst))
            ].copy()
            if np.linalg.norm(self.goal_pos - self.state) > self.goal_radius:
                break
        self._step_count = 0
        self._plans_to_render = []

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment."""
        # Action is velocity directly - clip to max velocity
        velocity = np.clip(action, -self.max_velocity, self.max_velocity)

        # Current position
        pos = self.state

        # Update position with velocity
        new_pos = pos + velocity * self.dt

        # Check collision with walls and boundaries
        new_pos = self._resolve_collision(pos, new_pos)

        # Update state
        self.state = new_pos

        # Check if goal is reached
        dist_to_goal = np.linalg.norm(new_pos - self.goal_pos)
        goal_reached = dist_to_goal < self.goal_radius

        # Reward
        if self.dense_reward:
            reward = -dist_to_goal / self.maze_diameter
        else:
            reward = 1.0 if goal_reached else 0.0

        # Truncation
        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps

        info = {
            "dist_to_goal": dist_to_goal,
            "is_success": goal_reached,
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, False, truncated, info

    def _resolve_collision(
        self, old_pos: np.ndarray, new_pos: np.ndarray
    ) -> np.ndarray:
        """
        Resolve collisions with walls and boundaries.

        Args:
            old_pos: Previous position
            new_pos: Desired new position

        Returns:
            Adjusted position after collision resolution
        """
        # Check boundary collisions
        new_pos = np.clip(
            new_pos,
            [self.agent_radius, self.agent_radius],
            [
                self.world_width - self.agent_radius,
                self.world_height - self.agent_radius,
            ],
        )

        # Check wall collisions
        if self._is_collision(new_pos):
            # Try moving only in x direction
            test_x = np.array([new_pos[0], old_pos[1]])
            if not self._is_collision(test_x):
                return test_x

            # Try moving only in y direction
            test_y = np.array([old_pos[0], new_pos[1]])
            if not self._is_collision(test_y):
                return test_y

            # Both directions blocked, stay at old position
            return old_pos

        return new_pos

    def _is_collision(self, pos: np.ndarray) -> bool:
        """Check if position collides with any wall."""
        # Get cell coordinates
        cell_x = int(pos[0] / self.cell_size)
        cell_y = int(pos[1] / self.cell_size)

        # Check if out of bounds
        if (
            cell_x < 0
            or cell_x >= self.maze_width
            or cell_y < 0
            or cell_y >= self.maze_height
        ):
            return True

        # Check nearby cells for walls
        check_radius = int(np.ceil(self.agent_radius / self.cell_size))
        for dy in range(-check_radius, check_radius + 1):
            for dx in range(-check_radius, check_radius + 1):
                cx = cell_x + dx
                cy = cell_y + dy

                if 0 <= cx < self.maze_width and 0 <= cy < self.maze_height:
                    if self.maze_layout[cy, cx] == 1:
                        # Find closest point on cell to agent
                        closest = np.clip(
                            pos,
                            [cx * self.cell_size, cy * self.cell_size],
                            [(cx + 1) * self.cell_size, (cy + 1) * self.cell_size],
                        )

                        dist = np.linalg.norm(pos - closest)
                        if dist < self.agent_radius:
                            return True

        return False

    def set_plans(self, plans: list):
        """
        Set plans to visualize in the render.

        Args:
            plans: List of plans, where each plan is a dict with:
                - 'trajectory': np.ndarray of shape (T, 2) with positions
                - 'color': tuple (r, g, b) for rendering
        """
        for plan in plans:
            # Check if plans start at current position
            dist = np.linalg.norm(plan["trajectory"][0] - self.state)
            if dist > 1.0e-2:
                self._plans_to_render = (
                    []
                )  # Clear plans if they don't start at current position
                return
        self._plans_to_render = plans

    def render(self):
        """Render the environment using Pygame."""
        if self.render_mode is None:
            return None

        try:
            import pygame
        except ImportError:
            raise ImportError(
                "pygame is not installed. Install it with: pip install pygame"
            )

        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.pixel_size, self.pixel_size)
                )
                pygame.display.set_caption("Maze2D Environment")
            else:  # rgb_array
                self.window = pygame.Surface((self.pixel_size, self.pixel_size))
            self.clock = pygame.time.Clock()

        # Scale factor from world to pixels
        scale = self.pixel_size / max(self.world_width, self.world_height)

        # Clear screen
        self.window.fill((255, 255, 255))

        # Draw maze walls
        for y in range(self.maze_height):
            for x in range(self.maze_width):
                if self.maze_layout[y, x] == 1:
                    rect = pygame.Rect(
                        int(x * self.cell_size * scale),
                        int(y * self.cell_size * scale),
                        int(self.cell_size * scale),
                        int(self.cell_size * scale),
                    )
                    pygame.draw.rect(self.window, (50, 50, 50), rect)

        # Draw plans/trajectories
        for plan in self._plans_to_render:
            trajectory = plan.get("trajectory", [])
            color = plan.get("color", (100, 100, 255))

            if len(trajectory) > 1:
                points = [(int(p[0] * scale), int(p[1] * scale)) for p in trajectory]
                temp_surface = pygame.Surface(
                    (self.pixel_size, self.pixel_size), pygame.SRCALPHA
                )
                rgba_color = (*color, 100)  # Semi-transparent
                pygame.draw.lines(temp_surface, rgba_color, False, points, 2)

                # Draw arrow at end
                if len(points) >= 2:
                    end = np.array(points[-1], dtype=float)
                    prev = np.array(points[-2], dtype=float)
                    direction = end - prev
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        arrow_size = 8
                        perp = np.array([-direction[1], direction[0]])
                        p1 = end - direction * arrow_size + perp * arrow_size * 0.5
                        p2 = end - direction * arrow_size - perp * arrow_size * 0.5
                        pygame.draw.polygon(temp_surface, rgba_color, [end, p1, p2])

                # Blit the temporary surface onto the main window
                self.window.blit(temp_surface, (0, 0))

        # Draw goal
        goal_pixel = (int(self.goal_pos[0] * scale), int(self.goal_pos[1] * scale))
        goal_radius_pixel = int(self.goal_radius * scale)
        pygame.draw.circle(self.window, (50, 200, 50), goal_pixel, goal_radius_pixel)
        pygame.draw.circle(self.window, (0, 150, 0), goal_pixel, goal_radius_pixel, 2)

        # Draw agent
        if self.state is not None:
            agent_pixel = (int(self.state[0] * scale), int(self.state[1] * scale))
            agent_radius_pixel = int(self.agent_radius * scale)
            pygame.draw.circle(
                self.window, (200, 50, 50), agent_pixel, agent_radius_pixel
            )
            pygame.draw.circle(
                self.window, (150, 0, 0), agent_pixel, agent_radius_pixel, 2
            )

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

    def close(self):
        """Clean up resources."""
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


def create_simple_maze(size: int = 10) -> np.ndarray:
    """Create a simple maze with some walls."""
    maze = np.zeros((size, size), dtype=np.uint8)

    # Add border walls
    maze[0, :] = 1
    maze[-1, :] = 1
    maze[:, 0] = 1
    maze[:, -1] = 1

    # Add some internal walls
    if size >= 10:
        # Horizontal wall
        maze[size // 3, 2 : size - 2] = 1
        maze[size // 3, size // 2] = 0  # Gap

        # Vertical wall
        maze[2 : size - 2, 2 * size // 3] = 1
        maze[size // 2, 2 * size // 3] = 0  # Gap

    return maze


def create_medium_maze() -> np.ndarray:
    """Create a medium complexity maze."""
    maze = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    return maze


def create_corridor_maze(width: int = 20, height: int = 5) -> np.ndarray:
    """Create a corridor maze."""
    maze = np.zeros((height, width), dtype=np.uint8)

    # Add border walls
    maze[0, :] = 1
    maze[-1, :] = 1
    maze[:, 0] = 1
    maze[:, -1] = 1

    return maze
