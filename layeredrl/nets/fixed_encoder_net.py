from typing import List, Tuple

import torch

from .encoder import Encoder


class FixedEncoderNet(Encoder):
    """A fixed map to a latent space picking out some dimensions of the observation."""

    def __init__(
        self,
        mapped_env_obs_shape: Tuple[int, ...],
        latent_state_dims: List[int],
        context_dims: List[int],
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize the network.

        Args:
            mapped_env_obs_dim: The dimension of the mapped environment observation space.
            latent_state_dim: The dimension of the latent state space.
            context_dim: The dimension of the context space.
            latent_state_dims: Which dimensions of the mapped observation to use as the latent state.
            context_dims: Which dimensions of the mapped observation to use as the context.
            device: The device to use.
        """
        super().__init__()
        self.latent_state_dims = torch.tensor(latent_state_dims, device=device)
        self.context_dims = torch.tensor(context_dims, device=device)
        self._latent_state_dim = len(latent_state_dims)
        self._context_dim = len(context_dims)
        self.mapped_env_obs_shape = mapped_env_obs_shape
        self.device = device
        assert (
            len(mapped_env_obs_shape) == 1
        ), "Only 1D mapped observations are supported."

    def forward(
        self, mapped_env_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the latent state and context."""
        state = mapped_env_obs[..., self.latent_state_dims]
        context = mapped_env_obs[..., self.context_dims]
        return state, context

    def decode(self, latent_state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Decode the latent state."""
        mapped_env_obs = torch.zeros(
            latent_state.shape[:-1] + self.mapped_env_obs_shape, device=self.device
        )
        mapped_env_obs[..., self.latent_state_dims] = latent_state
        mapped_env_obs[..., self.context_dims] = context
        return mapped_env_obs

    @property
    def latent_state_dim(self) -> int:
        """Dimension of the latent state space."""
        return self._latent_state_dim

    @property
    def context_dim(self) -> int:
        """Dimension of the context space."""
        return self._context_dim
