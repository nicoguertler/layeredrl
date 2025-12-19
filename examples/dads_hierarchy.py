"""Learn skills with DADS and plan over them to solve tasks.

This demo shows how to create a hierarchy consisting of planner level and a skill level
trained with DADS.
"""

import argparse
from functools import partial
from pathlib import Path

import gymnasium as gym
import torch

import layeredrl.envs  # noqa: F401
from layeredrl.levels import PlannerLevel, DADSLevel
from layeredrl.planners import CEMPlanner
from layeredrl.hierarchies import Hierarchy
from layeredrl.collectors import Collector
from layeredrl.predictors import get_default_predictor_factory


def demo_planning_with_learned_model(
    n_envs,
    n_steps=10000,
    clip=False,
    momentum=0.0,
    horizon=10,
    render_mode=None,
    device=torch.device("cpu"),
):
    def make_env(n_envs, render_mode):
        return gym.make_vec(
            id="Maze2D-Medium-v0",
            num_envs=n_envs,
            vectorization_mode="async",
            wrappers=[gym.wrappers.FlattenObservation],
            render_mode=render_mode,
            vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP},
        )

    train_env = make_env(n_envs, render_mode)
    test_env = make_env(n_envs, render_mode)

    predictor_factory = get_default_predictor_factory(
        train_env, sb_start_duration=16000
    )

    # Callback to visualize planned trajectories (optional)
    def plan_viz_callback(trajectory, context, cost, actions):
        """Visualize the planned trajectory in first test environment."""
        plans_data = [
            {"trajectory": plan, "color": (100, 100, 255)}
            for plan in trajectory.state[0, :, 0, 0].detach().cpu().numpy()
        ]
        test_env.call("set_plans", plans_data)

    cem_params = {
        "n_samples": 256,
        "n_iterations": 6,
        "elite_ratio": 0.05,
        "momentum": momentum,
        "clip": clip,
        "return_mode": "mean",
    }
    planner_factory = partial(
        CEMPlanner,
        use_icem=False,
        initial_sigma=0.5,
        cem_params=cem_params,
        use_value_func=False,
        use_term_prob=False,
        deterministic=True,
        rollout_callback=plan_viz_callback,
    )

    skill_space_dim = 2
    # Planner level
    planner_level = PlannerLevel(
        partial_planner=planner_factory,
        predictor_factory=predictor_factory,
        initial_guess=torch.zeros(skill_space_dim),
        horizon=horizon,
        alternate_with_noise=True,
        switch_random_prob=0.02,
        switch_planner_prob=0.02,
        resample_random_action_prob=0.1,
        no_planning_steps=5000,
        warm_up_steps=2000,
        buffer_size=100000,
        update_interval=10,
        n_updates=1,
        model_batch_size=1024,
        device=device,
    )

    # DADS level
    dads_level = DADSLevel(
        skill_space_dim=skill_space_dim,
        control_interval=1,
        reward_calc_interval=10,
        warm_up_steps=1000,
        update_interval=4,
        n_updates=1,
        device=device,
        log_interval=2000,
    )

    # Assemble hierarchy consisting of a planner level and a DADS level
    hierarchy = Hierarchy(
        levels=[planner_level, dads_level],
        env=train_env,
        device=device,
    )

    # Train
    hierarchy.train()
    collector = Collector(
        hierarchy=hierarchy,
        env=train_env,
        test_env=test_env,
        device=hierarchy.device,
    )
    collector.reset()
    stats = collector.collect(
        n_steps=n_steps,
        learn=True,
        test_interval=1600,
        n_test_steps=800,
        verbose=True,
    )
    print("Training stats: \n", stats)

    # Save final model
    save_path = Path("./dads_hierarchy_checkpoint")
    hierarchy.save(save_path)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--n_envs", type=int, default=2)
    argparser.add_argument("--no-clip", action="store_false", dest="clip", default=True)
    argparser.add_argument("--momentum", type=float, default=0.1)
    argparser.add_argument("--horizon", type=int, default=10)
    argparser.add_argument("--device", type=str, default="cuda:0")
    argparser.add_argument("--n_steps", type=int, default=100000)
    argparser.add_argument("--render-mode", type=str, default=None)
    args = argparser.parse_args()
    demo_planning_with_learned_model(
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        clip=args.clip,
        momentum=args.momentum,
        horizon=args.horizon,
        render_mode=args.render_mode,
        device=torch.device(args.device),
    )
