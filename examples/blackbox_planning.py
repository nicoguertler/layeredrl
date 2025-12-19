"""Demonstrate planning with iCEM and a fixed predictor."""

import argparse
from time import time
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from gymnasium.spaces.box import Box

from layeredrl.planners import CEMPlanner
from layeredrl.predictors import StaticPredictor
from layeredrl.models.probabilistic_ensemble import ProbabilisticEnsemble


class DynamicsNet(torch.nn.Module):
    """A simple dynamics net."""

    def forward(self, state, context, action):
        # state delta is just the action
        delta = 1.0 * action
        # add dimension for (single) mode
        delta = delta.unsqueeze(-2)
        new_std = 0.001 * torch.ones_like(delta)
        weights = torch.ones_like(delta[..., 0])
        return (delta, new_std), weights, torch.zeros_like(state[..., 0])


def reward_function(state, context, act):
    new_mean = state + act
    reward = -1.0 * torch.heaviside(
        1.0 - torch.norm(new_mean, dim=-1), torch.tensor(0.0, device=state.device)
    )
    reward += 1.0 * torch.heaviside(
        1.0
        - torch.norm(new_mean - torch.tensor([0.0, 2.0], device=state.device), dim=-1),
        torch.tensor(0.0, device=state.device),
    )
    return reward


def plot_trajectories(
    trajectories, actions, n_envs, n_models, n_particles_per_model, horizon
):
    state = trajectories["state"].cpu().numpy()
    reward = reward_function(
        trajectories["state"][..., :-1, :], None, actions[:, None, None, ...]
    )
    fig, ax = plt.subplots(2, n_envs, sharex="row", sharey="row")
    for k in range(n_envs):
        ax[0][k].set_xlabel(r"$s_0$")
        ax[0][k].set_ylabel(r"$s_1$")
        ax[0][k].set_aspect("equal", adjustable="box")
        circle = plt.Circle((0, 0), 1.0, color="r", fill=True, label="reward < 0")
        ax[0][k].add_artist(circle)
        circle = plt.Circle((0, 2), 1.0, color="g", fill=True, label="reward > 0")
        ax[0][k].add_artist(circle)
        for i in range(n_models):
            for j in range(n_particles_per_model):
                ax[0][k].plot(state[k, i, j, :, 0], state[k, i, j, :, 1], color=f"C{i}")
                # plot trajectory with color gradient along it
                ax[0][k].scatter(
                    state[k, i, j, :, 0],
                    state[k, i, j, :, 1],
                    c=range(horizon + 1),
                    cmap="viridis",
                )
    # plot rewards
    for k in range(n_envs):
        for i in range(n_models):
            for j in range(n_particles_per_model):
                ax[1][k].set_xlabel("time step")
                ax[1][k].set_ylabel("reward")
                ax[1][k].plot(reward[k, i, j, :], color=f"C{i}")
    fig.savefig("final_trajectories.png")


def plot_samples(
    trajectories,
    actions,
    n_models,
    n_particles_per_model,
    it,
    xrange=(-10, 10),
    yrange=(-10, 10),
):
    n_samples = trajectories["state"].shape[0]
    state = trajectories["state"].cpu().numpy()
    reward = reward_function(
        trajectories["state"][..., :-1, :], None, actions[:, None, None, ...]
    )
    # plot
    fig, ax = plt.subplots(2, 1, sharex="row", sharey="row")
    ax[0].set_xlim(xrange)
    ax[0].set_ylim(yrange)
    ax[0].set_aspect("equal", adjustable="box")
    ax[0].set_xlabel(r"$s_0$")
    ax[0].set_ylabel(r"$s_1$")
    # plot circle around origin
    circle = plt.Circle((0, 0), 1.0, color="r", fill=True)
    ax[0].add_artist(circle)
    # plot circle around (0, 2)
    circle = plt.Circle((0, 2), 1.0, color="g", fill=True)
    ax[0].add_artist(circle)
    for k in range(n_samples):
        # plot trajectories
        for i in range(n_models):
            for j in range(n_particles_per_model):
                ax[0].plot(
                    state[k, i, j, :, 0], state[k, i, j, :, 1], color=f"C{i}", alpha=0.2
                )
        # plot rewards
        for i in range(n_models):
            for j in range(n_particles_per_model):
                ax[1].plot(reward[k, i, j, :], color=f"C{i}")
    # plot mean trajectory
    ax[0].plot(
        trajectories["state"].mean(dim=[0, 1, 2])[:, 0].cpu(),
        trajectories["state"].mean(dim=[0, 1, 2])[:, 1].cpu(),
        color="k",
    )
    # plot mean reward
    ax[1].set_xlabel("time step")
    ax[1].set_ylabel("reward")
    ax[1].plot(reward.mean(dim=[0, 1, 2]).cpu(), color="k")
    # number of iteration is padded with zeros to the left to 2 digits
    fig.savefig(f"samples_it_{it:02d}.png")
    plt.close()


def demo_cem_planner(
    n_envs=2,
    momentum=0.0,
    horizon=10,
    no_iteration_plots=False,
    device=torch.device("cpu"),
):
    """Demonstrate the CEM planner."""
    torch.manual_seed(42)
    np.random.seed(42)

    s_dim = 2

    # create probabilistic ensemble model
    state_space = Box(low=-10, high=10, shape=(2,))
    context_space = Box(low=-1.0, high=1.0, shape=(1,))
    action_space = Box(low=-1.5, high=1.5, shape=(2,))
    net_factory = (
        lambda state_space, context_space, action_space, n_modes: DynamicsNet()
    )
    n_models = 1
    n_particles_per_model = 1
    model = ProbabilisticEnsemble(
        state_space,
        action_space,
        context_space,
        partial_net=net_factory,
        n_models=n_models,
        n_modes=1,
        n_particles_per_model=n_particles_per_model,
        device=device,
        # dynamics predict delta of state
        predict_delta=True,
    )

    def value_function(state, context, action):
        return torch.zeros(state.shape[:-1], device=device)

    predictor = StaticPredictor(
        model=model,
        val_func=value_function,
        rew_func=reward_function,
        encoder=lambda x: (x, torch.zeros((x.shape[0], 1), device=x.device)),
        latent_state_dim=2,
        context_dim=1,
    )
    # planner
    n_samples = 200
    planner = CEMPlanner(
        predictor=predictor,
        action_space=action_space,
        initial_sigma=1.0,
        horizon=horizon,
        deterministic=True,
        cem_params={
            "n_samples": n_samples,
            "n_iterations": 30,
            "elite_ratio": 0.05,
            "beta": 1.0,
            "clip": True,
            "momentum": momentum,
            "return_mode": "mean",
            "record_samples": not no_iteration_plots,
        },
        n_env_instances=n_envs,
        lower_bound=action_space.low,
        upper_bound=action_space.high,
        use_icem=True,
        device=device,
    )

    # plan
    s0 = torch.zeros((n_envs, s_dim), device=device) + torch.tensor(
        [0.0, -2.0], device=device
    )
    if n_envs > 1:
        # second environment starts at (-1, 0)
        s0[1, :] += torch.tensor([-1.0, 0.0], device=device)
    context = torch.zeros((n_envs, 1), device=device)
    planner.reset(initial_guess=torch.zeros((n_envs, horizon, s_dim)))
    active_instances = torch.ones((n_envs,), dtype=torch.bool, device=device)
    t0 = time()
    torch.manual_seed(42)
    np.random.seed(42)
    with torch.no_grad():
        actions, info = planner.plan(s0, active_instances, verbose=True)
    print(f"Planning took {time() - t0:.3f}s.")

    # plot samples
    if not no_iteration_plots:
        for it in range(planner.optimizer.n_iterations):
            sample_a = planner.optimizer.samples[it].view(
                n_envs, n_samples, horizon, s_dim
            )[0, ...]
            # expand to match number of action sequence samples
            trajectories = model.rollout(
                s0[0, None, ...].expand(n_samples, -1),
                context[0, None, ...].expand(n_samples, -1),
                sample_a,
            )
            plot_samples(trajectories, sample_a, n_models, n_particles_per_model, it)

    # do rollout with final actions
    final_trajectories = model.rollout(s0, context, actions)
    plot_trajectories(
        final_trajectories, actions, n_envs, n_models, n_particles_per_model, horizon
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--n_envs", type=int, default=2)
    argparser.add_argument("--momentum", type=float, default=0.1)
    argparser.add_argument("--horizon", type=int, default=10)
    argparser.add_argument("--no-iteration-plots", action="store_true")
    argparser.add_argument("--device", type=str, default="cpu")
    args = argparser.parse_args()
    current_directory = os.getcwd()
    answer = input(
        f"This demo will write image files to the current directory ({current_directory}).\nContinue? (y/n) "
    )
    if answer.lower() != "y":
        print("Exiting.")
        exit(0)
    demo_cem_planner(
        n_envs=args.n_envs,
        momentum=args.momentum,
        horizon=args.horizon,
        no_iteration_plots=args.no_iteration_plots,
        device=torch.device(args.device),
    )
