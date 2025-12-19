from typing import Tuple

import torch

from .encoder import Encoder


class IdentityEncoder(Encoder):
    """Uses mapped environment observation directly as latent state."""

    def __init__(
        self,
        mapped_env_obs_shape: Tuple[int, ...],
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize the network.

        Args:
            mapped_env_obs_dim: The dimension of the mapped environment observation space.
            latent_state_dim: The dimension of the latent state space.
            device: The device to use.
        """
        super().__init__()
        self.device = device
        self._latent_state_dim = mapped_env_obs_shape[0]
        self._context_dim = 0

    def forward(
        self, mapped_env_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the latent state and context."""
        state = mapped_env_obs
        context = torch.zeros(mapped_env_obs.shape[:-1] + (0,), device=self.device)
        return state, context

    def decode(self, latent_state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Decode the latent state."""
        mapped_env_obs = latent_state
        return mapped_env_obs

    @property
    def latent_state_dim(self) -> int:
        """Dimension of the latent state space."""
        return self._latent_state_dim

    @property
    def context_dim(self) -> int:
        """Dimension of the context space."""
        return self._context_dim
