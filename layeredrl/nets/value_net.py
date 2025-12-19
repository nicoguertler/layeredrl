from typing import Any

import torch
from torch import nn


class ValueNet(nn.Module):
    """A value network that takes in state, context, and action (which is ignored)."""

    def __init__(
        self,
        state_dim: int,
        context_dim: int,
        hidden_sizes: int = [128, 128],
        nonlinearity: Any = nn.ReLU,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize the network.

        Args:
            state_dim: The dimension of the state space.
            context_dim: The dimension of the context space.
            hidden_sizes: A list of hidden sizes for each layer.
            nonlinearity: The nonlinearity to use.
        """
        super().__init__()
        self.device = device

        layers = []
        for k in range(len(hidden_sizes)):
            if k == 0:
                layers.append(
                    nn.Linear(state_dim + context_dim, hidden_sizes[k], device=device)
                )
            else:
                layers.append(
                    nn.Linear(hidden_sizes[k - 1], hidden_sizes[k], device=device)
                )
            layers.append(nonlinearity())
        self.net = nn.Sequential(*layers)

        # output a single number, the value
        self.final_layer = nn.Linear(hidden_sizes[-1], 1, device=device)
        # set weights of final layer to zero
        self.final_layer.weight.data.fill_(0)

    def forward(
        self, state: torch.Tensor, context: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate the state and context and feed it through the network.

        Ignores action."""
        element_lst = [state, context]
        total_input = torch.cat(element_lst, dim=-1)
        return self.final_layer(self.net(total_input)).squeeze(dim=-1)
