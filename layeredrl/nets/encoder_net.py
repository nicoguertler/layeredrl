from typing import Tuple

import torch
from torch import nn

from ..utils.normalization import RunningBatchNorm
from .encoder import Encoder


class EncoderNet(Encoder):
    """A linear encoder mapping env obs to a latent state and a context variable."""

    def __init__(
        self,
        mapped_env_obs_shape: Tuple[int, ...],
        latent_state_dim: int,
        context_dim: int,
        standardize: bool = False,
        bn_momentum: float = 0.1,
        freeze_after: int = None,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize the network.

        Args:
            mapped_env_obs_dim: The dimension of the mapped environment observation space.
            latent_state_dim: The dimension of the latent state space.
            standardize: Whether to standardize the input.
            bn_momentum: The momentum for the batch norm.
            freeze_after: The number of batches after which to freeze the normalization.
            device: The device to use.
        """
        super().__init__()
        self._latent_state_dim = latent_state_dim
        self._context_dim = context_dim
        self.standardize = standardize
        self.device = device
        assert (
            len(mapped_env_obs_shape) == 1
        ), "Only 1D mapped observations are supported."

        if standardize:
            self.input_bn = RunningBatchNorm(
                num_features=mapped_env_obs_shape[0],
                momentum=bn_momentum,
                freeze_after=freeze_after,
                device=self.device,
                dtype=torch.float32,
            )
        else:
            self.input_bn = None

        def get_net(output_dim):
            return nn.Linear(
                mapped_env_obs_shape[0],
                output_dim,
                bias=False,
                device=device,
            )

        self.state_net = get_net(latent_state_dim)
        self.context_net = get_net(context_dim)
        self.mask = torch.nn.Parameter(
            torch.zeros(mapped_env_obs_shape[0], device=device, dtype=torch.bool),
            requires_grad=False,
        )

    def forward(
        self, mapped_env_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.standardize:
            mapped_env_obs = self.input_bn(mapped_env_obs)
        context = self.context_net(~(self.mask) * mapped_env_obs)
        state = self.state_net(self.mask * mapped_env_obs)
        return state, context

    @property
    def latent_state_dim(self) -> int:
        """Dimension of the latent state space."""
        return self._latent_state_dim

    @property
    def context_dim(self) -> int:
        """Dimension of the context space."""
        return self._context_dim
