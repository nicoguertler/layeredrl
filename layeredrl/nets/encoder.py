from abc import ABC, abstractmethod

from torch import nn


class Encoder(ABC, nn.Module):

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def latent_state_dim(self) -> int:
        """Dimension of the latent state space."""
        pass

    @property
    @abstractmethod
    def context_dim(self) -> int:
        """Dimension of the context space."""
        pass
