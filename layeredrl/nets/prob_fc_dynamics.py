from typing import Any, Union

from gymnasium.spaces import Box, Discrete
import torch
import torch.nn as nn


class ProbFCDynamics(nn.Module):
    """Dynamics prediction network using a fully connected NN and predicting mean and standard deviation of the
    next state."""

    def __init__(
        self,
        state_space: Box,
        context_space: Box,
        action_space: Union[Box, Discrete],
        n_modes: int,
        hidden_sizes: int = [128, 128],
        nonlinearity: Any = nn.ReLU,
        one_hot_action: bool = False,
        input_batch_norm: bool = False,
        ignore_context: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize the model.

        Args:
            state_space: The state space.
            context_space: The context space (containing static information).
            action_space: The action space.
            n_modes: The number of modes to predict (in a mixture model).
            hidden_size_lst: A list of hidden sizes for each layer.
            one_hot_action: Whether the action comes already one-hot encoded.
            input_batch_norm: Whether to use batch normalization on the input.
            ignore_context: Whether to ignore the context and not concatenate it to the state.
            device: The device to use.
        """
        super().__init__()
        self.state_space = state_space
        self.context_space = context_space
        self.action_space = action_space
        self.n_modes = n_modes
        self.discrete_action = isinstance(action_space, Discrete)
        self.hidden_size_lst = hidden_sizes
        self.one_hot_action = one_hot_action
        self.input_batch_norm = input_batch_norm
        self.ignore_context = ignore_context
        self.device = device

        action_rep_size = (
            action_space.n if self.discrete_action else action_space.shape[0]
        )
        if self.input_batch_norm:
            self.input_bn = nn.BatchNorm1d(state_space.shape[0], momentum=0.01)
        else:
            self.input_bn = None

        layers = []
        for k in range(len(hidden_sizes)):
            if k == 0:
                input_size = state_space.shape[0] + action_rep_size
                if not self.ignore_context:
                    input_size += context_space.shape[0]
                layers.append(
                    nn.Linear(
                        input_size,
                        hidden_sizes[k],
                        device=device,
                    )
                )
            else:
                layers.append(
                    nn.Linear(hidden_sizes[k - 1], hidden_sizes[k], device=device)
                )
            layers.append(nonlinearity())
        self.net = nn.Sequential(*layers)

        input_size = hidden_sizes[-1]
        # mean and std for each mode, mixture weights and termination probability
        output_size = state_space.shape[0] * 2 * self.n_modes + self.n_modes + 1
        self.final_layer = nn.Linear(input_size, output_size, device=device)

    def forward(
        self, state: torch.Tensor, context: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Predict the next state given the current state and action.

        Args:
            state: The current state.
            context: The context, i.e., information that is constant over timesteps.
            action: The action.
        Returns:
            Mean and standard deviation of next state, and termination probability.
        """
        if self.input_batch_norm:
            state = self.input_bn(state.view(-1, self.state_space.shape[0])).view(
                state.shape
            )
        if self.discrete_action and not self.one_hot_action:
            action = nn.functional.one_hot(
                action.squeeze(-1), self.action_space.n
            ).float()
        if self.ignore_context:
            x = torch.cat((state, action), dim=-1)
        else:
            x = torch.cat((state, context, action), dim=-1)
        x = self.net(x)
        logits = self.final_layer(x)
        s_mean = logits[..., 0 : self.state_space.shape[0] * self.n_modes].view(
            *logits.shape[:-1], self.n_modes, self.state_space.shape[0]
        )

        log_std = (
            logits[..., self.state_space.shape[0] * self.n_modes : -1 - self.n_modes]
            .clone()
            .view(*logits.shape[:-1], self.n_modes, self.state_space.shape[0])
        )
        log_std = torch.clamp(log_std, min=-10.0, max=10.0)
        s_std = torch.exp(log_std)
        weight_logits = logits[..., -1 - self.n_modes : -1]
        weights = torch.softmax(weight_logits, dim=-1)
        term_prob = torch.sigmoid(logits[..., -1].clamp(min=-20.0))
        # output is mean and std of next state, and termination probability
        return (s_mean, s_std), weights, term_prob
