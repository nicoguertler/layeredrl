import math

import torch


def cdf_normal(value, loc, scale):
    """Compute the CDF of a normal distribution."""
    return 0.5 * (1 + torch.erf((value - loc) * scale.reciprocal() / math.sqrt(2)))


def sample_truncated_normal(loc, scale, lower_bound, upper_bound, n_samples, device):
    """Sample from a truncated normal distribution.

    Note that everything is assumed to have a batch dimension. Sampling is done via
    the inverse CDF method rather than rejection sampling."""
    batch_size = loc.shape[0]
    dim = loc.shape[1]
    exp_loc = loc[:, None, :].expand((batch_size, n_samples, dim))
    exp_scale = scale[:, None, :].expand((batch_size, n_samples, dim))
    a = cdf_normal(lower_bound, loc, scale)
    b = cdf_normal(upper_bound, loc, scale)
    diff = (b - a)[:, None, :].expand((batch_size, n_samples, dim))
    a = a[:, None, :].expand((batch_size, n_samples, dim))
    u = a + diff * torch.rand((batch_size, n_samples, dim), device=device)
    return exp_loc + exp_scale * torch.erfinv(2 * u - 1) * math.sqrt(2)


def get_normal_prob(
    x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Get the probability (density) of x under a Normal distribution with the given mean and standard deviation.

    Args:
        x: The value.
        mean: The mean of the Gaussian.
        std: The standard deviation of the Normal distribution.
    Returns:
        The probability (density) of x under the Normal distribution.
    """
    prefactor = 1.0 / torch.sqrt(
        2 * torch.tensor(torch.pi) * torch.prod(std, dim=-1) ** 2
    )
    exponent = -0.5 * (((x - mean) / std) ** 2).sum(dim=-1)
    return prefactor * torch.exp(exponent)


def get_normal_log_prob(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    beta: float = 0.0,
) -> torch.Tensor:
    """Get the log probability of x under a 1D Normal distribution with the given mean and standard deviation.

    Can multiply a scale invariance factor to the second term of the log probability to remove the gradient
    of the log probability with respect to the scale of x and mean.

    No sum over the last dimension is performed.

    Args:
        x: The value.
        mean: The mean of the Gaussian.
        std: The standard deviation of the Normal distribution.
        beta: The beta parameter for \beta-NLL.
    Returns:
        The log probability of x under the Normal distribution.
    """
    first_term = -0.5 * torch.log(2 * torch.pi * std**2)
    second_term = -0.5 * (((x - mean) / std) ** 2)

    log_prob = first_term + second_term

    if beta > 0.0:
        # See https://arxiv.org/abs/2203.09168
        log_prob = torch.pow(std, 2.0 * beta).detach() * log_prob

    return first_term + second_term
