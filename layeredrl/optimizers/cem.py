from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .optimizer import Optimizer
from ..utils.distributions import sample_truncated_normal


class CEM(Optimizer):

    def __init__(
        self,
        n_iterations: int,
        n_samples: int,
        initial_sigma: torch.Tensor,
        elite_ratio: float = 0.2,
        lower_bound: Optional[torch.Tensor] = None,
        upper_bound: Optional[torch.Tensor] = None,
        clip: bool = False,
        momentum: float = 0.0,
        return_mode: str = "mean",
        record_samples: bool = False,
        *args,
        **kwargs,
    ):
        """Initialize the CEM optimizer.

        Args:
            n_iterations: The number of iterations to run CEM for.
            n_samples: The number of samples to use draw in each iteration of CEM.
            initial_sigma: The initial standard deviation of the samples. Shape: (batch_size, x_dim)
            elite_ratio: The ratio of samples to keep per iteration.
            lower_bound: The lower bound of the samples.
            upper_bound: The upper bound of the samples. Either both lower_bound
                and upper_bound must be None or neither.
            clip: Whether to clip the samples to the bounds.
            momentum: Momentum factor for updating the mean.
            return_mode: Whether to return the mean of the elite samples ("mean"), the
                best sample ("best"), or a random sample from the last Gaussian distribution
                ("random").
            record_samples: Whether to record the samples drawn during optimization.
        """
        super().__init__(*args, n_samples=n_samples, **kwargs)
        self.n_iterations = n_iterations
        self.initial_sigma = initial_sigma
        self.elite_ratio = elite_ratio
        self.lower_bound = (
            lower_bound.to(device=self.device) if lower_bound is not None else None
        )
        self.upper_bound = (
            upper_bound.to(device=self.device) if upper_bound is not None else None
        )
        self.clip = clip
        self.momentum = momentum
        assert return_mode in ["mean", "best", "random"]
        self.return_mode = return_mode
        self.record_samples = record_samples
        self.samples = []
        self.n_elites = int(self.n_samples * self.elite_ratio)
        assert self.n_elites > 1
        self.mu = None
        self.sigma = None
        self.truncated = lower_bound is not None and upper_bound is not None

    def reset(self, initial_x: torch.Tensor) -> None:
        """Reset the optimizer to the given initial guess.

        If this is not called, the optimizer will derive the initial
        guess from its internal state.

        Args:
            initial_x: The initial guess for the optimal x. Shape: (batch_size, dim)
        """
        self.mu = initial_x.to(self.device)
        self.batch_size = initial_x.shape[0]
        self.dim = initial_x.shape[1]
        self.samples = []

    def _sample(self) -> torch.Tensor:
        if self.truncated and not self.clip:
            # sample from truncated normal
            return sample_truncated_normal(
                self.mu,
                self.sigma,
                self.lower_bound,
                self.upper_bound,
                self.n_samples,
                self.device,
            )
        else:
            exp_mu = self.mu[:, None, :].expand(
                (self.batch_size, self.n_samples, self.dim)
            )
            exp_sigma = self.sigma[:, None, :].expand(
                (self.batch_size, self.n_samples, self.dim)
            )
            samples = exp_mu + exp_sigma * torch.randn(
                (self.batch_size, self.n_samples, self.dim), device=self.device
            )
            if self.clip and self.truncated:
                samples = torch.clamp(samples, self.lower_bound, self.upper_bound)
            return samples

    def optimize(
        self, cost: Callable[[Tensor], Tensor], verbose: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """Optimize the given cost function using the Cross Entropy Method and return the optimal x.

        Note that everything is assumed to have a batch dimension. That includes x and the cost.

        Args:
            cost: A function that takes in a tensor x and returns a tensor with a scalar cost for
                each particle in each environment instance. Output shape: (batch_size, n_samples)
        Returns:
            The optimal x. Shape: (batch_size, dim)
            An info dict containing the keys "mean" and "std" for the mean and standard deviation
            of the final distribution."""

        assert (
            self.mu is not None
        ), "Please call reset before calling optimize for the first time."

        best_cost = np.inf * torch.ones(
            self.batch_size, device=self.device, dtype=torch.float32
        )
        best_x = torch.zeros(
            (self.batch_size, self.dim), device=self.device, dtype=torch.float32
        )
        self.sigma = self.initial_sigma

        for i in range(self.n_iterations):
            x_samples = self._sample()
            if self.record_samples:
                self.samples.append(x_samples)
            costs = cost(x_samples)
            elite_costs, elite_indices = torch.topk(
                costs, self.n_elites, dim=1, largest=False, sorted=True
            )
            elite_samples = torch.gather(
                x_samples, 1, elite_indices[:, :, None].expand((-1, -1, self.dim))
            )

            self.mu = self.momentum * self.mu + (
                1.0 - self.momentum
            ) * elite_samples.mean(dim=1)
            self.sigma = self.momentum * self.sigma + (
                1.0 - self.momentum
            ) * elite_samples.std(dim=1)

            where_better = elite_costs[:, 0] < best_cost
            best_cost = torch.where(where_better, elite_costs[:, 0], best_cost)
            best_x = torch.where(
                where_better[:, None].expand((self.batch_size, self.dim)),
                elite_samples[:, 0, :],
                best_x,
            )

            if verbose:
                print(
                    f"CEM Iteration {i}: cost={elite_costs.mean():.3f}+-{elite_costs.std():.3f};"
                )

        info = {"mean": self.mu, "std": self.sigma}
        if self.return_mode == "mean":
            return self.mu, info
        elif self.return_mode == "best":
            return best_x, info
        else:
            return self._sample()[:, 0, :], info
