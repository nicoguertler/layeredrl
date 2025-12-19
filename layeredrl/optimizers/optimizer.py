from abc import ABC, abstractmethod
from typing import Callable, Dict, Tuple

import torch


class Optimizer(ABC):
    def __init__(self, n_samples: int, device: torch.device = torch.device("cpu")):
        """Initialize the optimizer.

        Args:
            n_samples: The number of samples to use for optimization, for example in CEM.
                If the algorithm does not use samples, this should be 1. This is defined
                for all optimizers in order to have a consistent number of dimensions for
                the input to the cost function.
            device: The device to use.
        """
        self.n_samples = n_samples
        self.device = device

    @abstractmethod
    def reset(self, initial_x: torch.Tensor, **kwargs) -> None:
        """Reset the optimizer to the given initial guess.

        If this is not called, the optimizer will derive the initial
        guess from its internal state.

        Args:
            initial_x: The initial guess for the optimal x.
        """
        pass

    @abstractmethod
    def optimize(
        self, cost: Callable[[torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        """Optimize the given cost function (starting from the initial guess) and return the optimal x.

        Note that everything is assumed to have a batch dimension. That includes x and the cost.
        The cost function can vary across the batch dimension.

        Args:
            cost: A function that takes in a tensor x and returns a tensor with a scalar cost for
                each environment instance.
        Returns:
            The optimal x, and an info dict."""
        pass
