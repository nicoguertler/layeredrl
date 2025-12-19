"""Learn a model from experience and use it for planning.

This demo shows how to create a hierarchy with a single planner level that uses a learned
model for planning. The planner level is mostly intented to work with a skill level below it.
The TD-MPC2 level, on the other hand, is tuned for direct application to continous control
(see tdmpc2_hierarchy.py).
"""

import argparse
from functools import partial

import gymnasium as gym
import torch

from layeredrl.levels import PlannerLevel
from layeredrl.planners import CEMPlanner
from layeredrl.hierarchies import Hierarchy
from layeredrl.predictors import PredictorFactory, RewardPredictor
from layeredrl.nets import ProbFCDynamics, RewardNet, IdentityEncoder, ValueNet
from layeredrl.models import ProbabilisticEnsemble
from layeredrl.collectors import Collector


def demo_planning_with_learned_model(
    env_name,
    n_envs,
    n_steps=10000,
    clip=False,
    momentum=0.0,
    horizon=10,
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

    # Networks for the predictor (partial because correct dimensions are automatically set
    # during assembly of the hierarchy)
    partial_net = partial(ProbFCDynamics, device=device)
    partial_model = partial(ProbabilisticEnsemble, partial_net=partial_net)
    partial_val_func = partial(ValueNet)
    partial_rew_func = partial(RewardNet)
    partial_encoder = partial(IdentityEncoder)

    # Predictor factory
    partial_predictor = partial(RewardPredictor, n_steps=3)
    predictor_factory = PredictorFactory(
        partial_model=partial_model,
        partial_val_func=partial_val_func,
        partial_rew_func=partial_rew_func,
        partial_encoder=partial_encoder,
        partial_predictor=partial_predictor,
    )

    # Planner factory
    cem_params = {
        "n_samples": 128,
        "n_iterations": 6,
        "elite_ratio": 0.05,
        "momentum": momentum,
        "clip": clip,
        "return_mode": "mean",
    }
    initial_sigma = (env.single_action_space.high - env.single_action_space.low).max()
    initial_sigma = float(initial_sigma)
    planner_factory = partial(
        CEMPlanner,
        use_icem=False,
        initial_sigma=initial_sigma,
        cem_params=cem_params,
        use_value_func=True,
    )

    # Planner level
    planner_level = PlannerLevel(
        partial_planner=planner_factory,
        predictor_factory=predictor_factory,
        initial_guess=torch.zeros(env.action_space.shape[-1]),
        horizon=horizon,
        alternate_with_noise=False,
        no_planning_steps=6000,
        buffer_size=24000,
        n_updates=2,
        verbose=False,
        device=device,
    )

    # Assemble hierarchy consisting of a single planner level
    hierarchy = Hierarchy(
        levels=[planner_level],
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
    argparser.add_argument("--no-clip", action="store_false", dest="clip", default=True)
    argparser.add_argument("--momentum", type=float, default=0.5)
    argparser.add_argument("--horizon", type=int, default=3)
    argparser.add_argument("--device", type=str, default="cuda:0")
    argparser.add_argument("--n_steps", type=int, default=4000)
    argparser.add_argument("--render-mode", type=str, default=None)
    args = argparser.parse_args()
    demo_planning_with_learned_model(
        env_name=args.env,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        clip=args.clip,
        momentum=args.momentum,
        horizon=args.horizon,
        render_mode=args.render_mode,
        device=torch.device(args.device),
    )
