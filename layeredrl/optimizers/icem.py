import torch
import colorednoise as cn

from . import CEM


class ICEM(CEM):

    def __init__(self, beta: float = 1.0, action_dim: int = 2, *args, **kwargs):
        """Initialize the iCEM optimizer.

        (see https://proceedings.mlr.press/v155/pinneri21a/pinneri21a.pdf)
        """
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.action_dim = action_dim
        assert (
            not self.truncated or self.clip
        ), "This implementation of iCEM only supports clipping and no truncation."

    def _sample(self) -> torch.Tensor:
        assert (
            self.dim % self.action_dim == 0
        ), "The dimension of the optimized variable must be divisible by the action dimension."
        exp_mu = self.mu[:, None, :].expand((self.batch_size, self.n_samples, self.dim))
        exp_sigma = self.sigma[:, None, :].expand(
            (self.batch_size, self.n_samples, self.dim)
        )
        colored_noise = cn.powerlaw_psd_gaussian(
            self.beta,
            (
                self.batch_size,
                self.n_samples,
                self.action_dim,
                self.dim // self.action_dim,
            ),
        )
        colored_noise = torch.tensor(
            colored_noise, device=self.device, dtype=torch.float32
        )
        colored_noise = colored_noise.transpose(-2, -1).reshape(
            self.batch_size, self.n_samples, self.dim
        )
        samples = exp_mu + exp_sigma * colored_noise
        if self.clip and self.truncated:
            samples = torch.clamp(samples, self.lower_bound, self.upper_bound)
        return samples
