"""Test for Maze2D environment."""

import numpy as np
import gymnasium as gym
import pytest
import layeredrl.envs  # noqa: F401
from layeredrl.envs import Maze2DEnv, create_simple_maze


def test_basic_functionality():
    """Test basic environment functionality."""
    # Create environment
    env = Maze2DEnv(
        maze_layout=create_simple_maze(10),
        start_pos=[(1.5, 1.5)],
        goal_pos=[(8.5, 8.5)],
    )

    # Test reset
    obs, info = env.reset()

    # Check observation
    assert isinstance(obs, dict)
    assert "observation" in obs
    assert "achieved_goal" in obs
    assert "desired_goal" in obs

    assert isinstance(obs["observation"], np.ndarray)
    assert obs["observation"].shape == (2,)
    assert isinstance(obs["achieved_goal"], np.ndarray)
    assert obs["achieved_goal"].shape == (2,)
    assert isinstance(obs["desired_goal"], np.ndarray)
    assert obs["desired_goal"].shape == (2,)

    assert env.observation_space.contains(obs)

    # Test step
    action = env.action_space.sample()
    obs_new, reward, terminated, truncated, info = env.step(action)

    # Check observation after step
    assert isinstance(obs_new, dict)
    assert "observation" in obs_new
    assert "achieved_goal" in obs_new
    assert "desired_goal" in obs_new
    assert env.observation_space.contains(obs_new)

    # Assert reward is numeric
    assert isinstance(reward, (int, float, np.number))

    # Assert terminated is always False (environment only truncates, never terminates)
    assert isinstance(terminated, bool)
    assert terminated is False
    assert isinstance(truncated, bool)

    # Assert info dict has expected keys
    assert isinstance(info, dict)
    assert "dist_to_goal" in info
    assert "is_success" in info

    env.close()


def test_registered_envs():
    """Test registered Gymnasium environments."""
    registered_envs = ["Maze2D-Simple-v0", "Maze2D-Corridor-v0"]

    for env_name in registered_envs:
        # Test environment can be created
        env = gym.make(env_name)
        env.close()


def test_dense_vs_sparse_reward():
    """Test that dense and sparse reward modes work correctly."""
    maze = create_simple_maze(10)
    start = [(1.5, 1.5)]
    goal = [(8.5, 8.5)]

    # Test sparse reward
    env_sparse = Maze2DEnv(
        maze_layout=maze,
        start_pos=start,
        goal_pos=goal,
        dense_reward=False,
    )
    obs, _ = env_sparse.reset()
    action = env_sparse.action_space.sample()
    obs, reward, _, _, info = env_sparse.step(action)

    # Sparse reward should be 0.0 when not at goal, 1.0 when at goal
    if not info["is_success"]:
        assert reward == 0.0
    else:
        assert reward == 1.0
    env_sparse.close()

    # Test dense reward
    env_dense = Maze2DEnv(
        maze_layout=maze,
        start_pos=start,
        goal_pos=goal,
        dense_reward=True,
    )
    obs, _ = env_dense.reset()
    action = env_dense.action_space.sample()
    obs, reward, _, _, info = env_dense.step(action)

    # Dense reward should be negative distance to goal normalized by maze diameter
    assert reward <= 0.0, "Dense reward should be negative (distance-based)"
    assert reward >= -1.0, "Dense reward should be >= -1.0"
    env_dense.close()


def test_truncation():
    """Test that episodes truncate after max_episode_steps."""
    env = Maze2DEnv(
        maze_layout=create_simple_maze(10),
        start_pos=[(1.5, 1.5)],
        goal_pos=[(8.5, 8.5)],
        max_episode_steps=10,  # Very short episode
    )

    obs, info = env.reset()
    truncated = False

    for step in range(15):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Should never terminate, only truncate
        assert terminated is False

        if truncated:
            assert step == 9
            break

    assert truncated
    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
