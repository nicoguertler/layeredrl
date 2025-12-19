import gymnasium as gym

from .log_rew_wrapper import LogRewWrapper
from .ant_flipped_wrapper import AntFlippedWrapper, AntNoWallFlippedWrapper
from .affine_rew_wrapper import AffineRewWrapper
from .maze2d import (
    Maze2DEnv,
    create_simple_maze,
    create_corridor_maze,
    create_medium_maze,
)


__all__ = [
    "LogRewWrapper",
    "AntFlippedWrapper",
    "AntNoWallFlippedWrapper",
    "AffineRewWrapper",
    "Maze2DEnv",
    "create_simple_maze",
    "create_corridor_maze",
    "create_medium_maze",
]

# Register environment with Gymnasium
gym.register(
    id="Maze2D-Simple-v0",
    entry_point="layeredrl.envs.maze2d:Maze2DEnv",
    max_episode_steps=500,
    kwargs={
        "maze_layout": create_simple_maze(10),
        "start_pos": [(1.5, 1.5)],
        "goal_pos": [(8.5, 8.5)],
    },
)

gym.register(
    id="Maze2D-Medium-v0",
    entry_point="layeredrl.envs.maze2d:Maze2DEnv",
    max_episode_steps=500,
    kwargs={
        "maze_layout": create_medium_maze(),
        "start_pos": [(1.5, 1.5), (6.5, 6.5), (6.5, 1.5), (1.5, 6.5)],
        "goal_pos": [(1.5, 1.5), (6.5, 6.5), (6.5, 1.5), (1.5, 6.5)],
    },
)

gym.register(
    id="Maze2D-Corridor-v0",
    entry_point="layeredrl.envs.maze2d:Maze2DEnv",
    max_episode_steps=1000,
    kwargs={
        "maze_layout": create_corridor_maze(20, 5),
        "start_pos": [(1.5, 2.5)],
        "goal_pos": [(18.5, 2.5)],
    },
)
