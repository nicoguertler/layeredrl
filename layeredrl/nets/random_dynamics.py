from typing import Union

from gymnasium.spaces import Box, Discrete
from scipy.stats import special_ortho_group
import torch
import torch.nn as nn


class RandomDynamics(nn.Module):
    """Random but fixed dynamics from random linear transformation.

    Note: Assumes only state deltas are to be predicted."""

    def __init__(
        self,
        state_space: Box,
        action_space: Union[Box, Discrete],
        std: float = 0.3,
        device: torch.device = torch.device("cpu"),
        n_modes: int = 1,
    ):
        """Initialize the model.

        Args:
            state_space: The state space.
            action_space: The action space.
            device: The device to use.
        """
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        assert isinstance(state_space, Box)
        assert isinstance(action_space, Box)
        self.std = std
        self.device = device
        self.n_modes = n_modes

        # generate matrix with orthonormal columns for mapping action to state (delta)
        action_dim = action_space.shape[0]
        state_dim = state_space.shape[0]
        self.A = nn.Parameter(
            torch.tensor(
                special_ortho_group.rvs(state_dim)[:, :action_dim], device=device
            ).float(),
            requires_grad=False,
        )
        self.transform = nn.Parameter(
            torch.eye(state_dim, device=device), requires_grad=False
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict the next state given the current state and action.

        Args:
            state: The current state.
            action: The action.
        Returns:
            Mean and standard deviation of next state and expected reward.
        """
        s_mean = torch.einsum("ij,jk,...k->...i", self.transform, self.A, action)
        # Add mode dimension
        s_mean = s_mean[..., None, :].expand(
            *s_mean.shape[:-1], self.n_modes, s_mean.shape[-1]
        )
        weights = torch.ones(s_mean.shape[:-1], device=self.device) / self.n_modes
        s_std = self.std * torch.ones_like(s_mean, device=self.device)
        return (
            (s_mean, s_std),
            weights,
            torch.zeros(s_std.shape[:-2], device=self.device),
        )
