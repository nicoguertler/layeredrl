"""Hierarchy with a single TD-MPC2 level."""

import argparse

import gymnasium as gym
import torch

from layeredrl.levels import TDMPC2Level
from layeredrl.hierarchies import Hierarchy
from layeredrl.collectors import Collector
from layeredrl.tdmpc2 import get_default_tdmpc2_config


def demo_tdmpc2_hierarchy(
    env_name,
    n_envs,
    n_steps=10000,
    render_mode=None,
    device=torch.device("cpu"),
):
    # Create environment
    env = gym.make_vec(
        id=env_name,
        num_envs=n_envs,
        vectorization_mode="async",
        render_mode=render_mode,
        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP},
    )

    # TD-MPC2 level
    tdmpc2_cfg = get_default_tdmpc2_config()
    tdmpc2_level = TDMPC2Level(
        tdmpc2_config=tdmpc2_cfg,
        device=device,
    )

    # Assemble hierarchy consisting of a single TD-MPC2 level
    hierarchy = Hierarchy(
        levels=[tdmpc2_level],
        env=env,
        device=device,
    )

    # Train
    hierarchy.train()
    collector = Collector(
        hierarchy=hierarchy,
        env=env,
        device=hierarchy.device,
    )
    collector.reset()
    stats = collector.collect(
        n_steps=n_steps,
        learn=True,
        verbose=True,
    )
    print("Training stats: \n", stats)

    # Test
    hierarchy.eval()
    collector.reset()
    stats = collector.collect(
        n_steps=2000,
        learn=False,
        verbose=True,
    )
    print("Test stats: \n", stats)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--env", type=str, default="InvertedPendulum-v5")
    argparser.add_argument("--n_envs", type=int, default=6)
    argparser.add_argument("--device", type=str, default="cuda:0")
    argparser.add_argument("--n_steps", type=int, default=3000)
    argparser.add_argument("--render-mode", type=str, default=None)
    args = argparser.parse_args()
    demo_tdmpc2_hierarchy(
        env_name=args.env,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        render_mode=args.render_mode,
        device=torch.device(args.device),
    )
