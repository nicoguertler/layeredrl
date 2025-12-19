import os

import matplotlib.pyplot as plt
from gymnasium.spaces import Box, Discrete
import torch
import torch.nn as nn

from layeredrl.models.probabilistic_ensemble import ProbabilisticEnsemble


class DriftDynamicsContinuous(nn.Module):
    def __init__(self):
        super().__init__()
        self.v = torch.randn(2)

    def forward(self, state, context, action):
        delta = self.v + action
        # add dimension for (single) mode
        delta = delta.unsqueeze(-2)
        weights = torch.ones_like(delta[..., 0])
        # return (state_mean, state_std), weights, termination probability
        return (
            (delta, 0.5 * torch.ones_like(delta)),
            weights,
            torch.zeros_like(state[..., 0]),
        )


class DriftDynamicsDiscrete(nn.Module):
    def __init__(self):
        super().__init__()
        self.v = torch.randn(2)
        self.dirs = torch.randn(4, 2)

    def forward(self, state, context, action):
        action_dir = self.dirs[action].squeeze(-2)
        delta = self.v + action_dir
        # add dimension for (single) mode
        delta = delta.unsqueeze(-2)
        weights = torch.ones_like(delta[..., 0])
        # return (state_mean, state_std), weights, termination probability
        return (
            (delta, 0.5 * torch.ones_like(delta)),
            weights,
            torch.zeros_like(state[..., 0]),
        )


def star_rollout_continous_actions(device=torch.device("cpu")):
    """Demonstrate rollout of an ensemble of probabilistic models.

    The action space is continuous.

    The ensemble consists of 5 models, each with 4 particles. Note that
    there are two envs (that the batch dimension is 2). The rollouts
    in the different env instances are shown in different plots. The
    different models are shown in different colors. The environments
    start in different states (one at the origin, the other at (-10, 0))."""

    state_space = Box(low=-1, high=1, shape=(2,))
    context_space = Box(low=-1.0, high=1.0, shape=(1,))
    action_space = Box(low=-1, high=1, shape=(2,))
    # create net factory
    net_factory = (
        lambda state_space, context_space, action_space, n_modes: DriftDynamicsContinuous()
    )
    # create ensemble model
    batch_size = 2
    n_models = 5
    n_particles_per_model = 4
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
    s0 = torch.zeros((batch_size, 2))
    context = torch.zeros((batch_size, 1))
    s0[0, :] = torch.tensor([-10.0, 0.0])
    H = 10
    actions = torch.zeros((batch_size, H, 2))
    # perform rollout
    trajectories = model.rollout(s0, context, actions)

    # plot
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].set_title("Ensemble rollout with continous action space")
    for k in range(batch_size):
        for i in range(n_models):
            for j in range(n_particles_per_model):
                ax[k].plot(
                    trajectories["state"][k, i, j, :, 0],
                    trajectories["state"][k, i, j, :, 1],
                    color=f"C{i}",
                )
    fig.savefig("ensemble_rollout_continuous.png")


def star_rollout_discrete_actions(device=torch.device("cpu")):
    """Demonstrate rollout of an ensemble of probabilistic models.

    The action space is discrete with four actions.

    The ensemble consists of 5 models, each with 4 particles. Note that
    there are two envs (that the batch dimension is 2). The rollouts
    in the different env instances are shown in different plots. The
    different models are shown in different colors. The environments
    start in different states (one at the origin, the other at (-10, 0))."""

    state_space = Box(low=-1, high=1, shape=(2,))
    context_space = Box(low=-1.0, high=1.0, shape=(1,))
    action_space = Discrete(4)
    # create net factory
    net_factory = (
        lambda state_space, context_space, action_space, n_modes: DriftDynamicsDiscrete()
    )
    # create ensemble model
    batch_size = 2
    n_models = 5
    n_particles_per_model = 4
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
    s0 = torch.zeros((batch_size, 2))
    context = torch.zeros((batch_size, 1))
    s0[0, :] = torch.tensor([-10.0, 0.0])
    H = 10
    # actions = torch.zeros((batch_size, H, 1), dtype=torch.long)
    actions = torch.randint(0, 4, (batch_size, H, 1))
    # perform rollout
    batch = model.rollout(s0, context, actions)

    # plot
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].set_title("Ensemble rollout with discrete action space")
    for k in range(batch_size):
        for i in range(n_models):
            for j in range(n_particles_per_model):
                ax[k].plot(
                    batch["state"][k, i, j, :, 0],
                    batch["state"][k, i, j, :, 1],
                    color=f"C{i}",
                )
    fig.savefig("ensemble_rollout_discrete.png")


if __name__ == "__main__":
    current_directory = os.getcwd()
    answer = input(
        f"This demo will write image files to the current directory ({current_directory}).\nContinue? (y/n) "
    )
    if answer.lower() != "y":
        print("Exiting.")
        exit(0)
    torch.manual_seed(1)
    star_rollout_continous_actions()
    star_rollout_discrete_actions()
