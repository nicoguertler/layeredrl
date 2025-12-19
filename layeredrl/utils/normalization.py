import torch
import torch.nn as nn
from torch.nn import Module


class RunningBatchNorm(Module):
    """Batch normalization using running estimates of mean and variance."""

    def __init__(
        self,
        num_features: int,
        eps=1.0e-5,
        momentum=0.1,
        device=None,
        dtype=None,
        track_mean=True,
        freeze_after=None,
    ):
        super().__init__()
        self._num_features = num_features
        self._eps = eps
        self._momentum = momentum
        self._track_mean = track_mean
        self._freeze_after = freeze_after
        self.mean = torch.nn.Parameter(
            torch.zeros(num_features, device=device, dtype=dtype),
            requires_grad=False,
        )
        self.var = torch.nn.Parameter(
            torch.ones(num_features, device=device, dtype=dtype),
            requires_grad=False,
        )
        # register 'initialized' buffer
        self.register_buffer("_initialized", torch.tensor(0, dtype=torch.uint8))
        self.register_buffer("_counter", torch.tensor(0, dtype=torch.int64))

    def forward(self, x: torch.Tensor):
        if self.training and (
            self._freeze_after is None or self._counter < self._freeze_after
        ):
            if not self._initialized:
                # use statistics of first batch for initialization
                if self._track_mean:
                    self.mean.data = x.mean(dim=0)
                self.var.data = x.var(dim=0)
                self._initialized = torch.tensor(1, dtype=torch.uint8)
            else:
                if self._track_mean:
                    self.mean.data = (
                        1 - self._momentum
                    ) * self.mean + self._momentum * x.mean(dim=0)
                self.var.data = (
                    1 - self._momentum
                ) * self.var + self._momentum * x.var(dim=0)
            self._counter += 1
        return (x - self.mean) / (self.var + self._eps).sqrt()


class Standardizer(Module):
    """Standardize a random vector (trained externally)."""

    def __init__(self, latent_state_dim: int, device: torch.device):
        super().__init__()
        self.mat = nn.Parameter(torch.eye(latent_state_dim, device=device))

    def get_symm_mat(self):
        return 0.5 * (self.mat.triu() + self.mat.triu().T)

    def forward(self, x):
        return x @ self.get_symm_mat()
