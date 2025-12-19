from typing import Any

import torch
from torch import nn


class RewardNet(nn.Module):
    """A reward function network that takes in state, context, and action."""

    def __init__(
        self,
        state_dim: int,
        context_dim: int,
        action_dim: int,
        hidden_sizes: int = [128, 128],
        nonlinearity: Any = nn.ReLU,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize the network.

        Args:
            state_dim: The dimension of the state space.
            context_dim: The dimension of the context space.
            action_dim: The dimension of the action space.
            hidden_sizes: A list of hidden sizes for each layer.
            nonlinearity: The nonlinearity to use.
        """
        super().__init__()
        self.device = device

        layers = []
        for k in range(len(hidden_sizes)):
            if k == 0:
                input_dim = state_dim + context_dim + action_dim
                layers.append(nn.Linear(input_dim, hidden_sizes[k], device=device))
            else:
                layers.append(
                    nn.Linear(hidden_sizes[k - 1], hidden_sizes[k], device=device)
                )
            layers.append(nonlinearity())
        self.net = nn.Sequential(*layers)

        # output a single number, the reward
        self.final_layer = nn.Linear(hidden_sizes[-1], 1, device=device)

    def forward(
        self, state: torch.Tensor, context: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate state, context and action and feed them to the network."""
        element_lst = [state, context, action]
        total_input = torch.cat(element_lst, dim=-1)
        return self.final_layer(self.net(total_input)).squeeze(dim=-1)
