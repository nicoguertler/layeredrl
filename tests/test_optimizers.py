from layeredrl.optimizers import CEM

import torch


def test_cem_optimizer():
    # Test the CEM optimizer on a simple function.
    torch.manual_seed(0)
    initial_sigma = 1.0 * torch.ones((2, 1))
    optimizer = CEM(
        n_iterations=5,
        initial_sigma=initial_sigma,
        n_samples=30,
        elite_ratio=0.2,
        return_mode="mean",
    )
    initial_x = torch.tensor([[-1.5], [1.5]])
    optimizer.reset(initial_x)

    def func(x):
        return torch.min(torch.squeeze((x - 1.0) ** 2), torch.squeeze((x + 1.0) ** 2))

    x_hat, _ = optimizer.optimize(func, verbose=True)
    print("x_hat: ", x_hat)
    assert torch.allclose(x_hat, torch.tensor([[-1.0], [1.0]]), atol=1e-2)

    # Test the CEM optimizer on a simple function with bounds.
    optimizer = CEM(
        n_iterations=5,
        initial_sigma=initial_sigma,
        n_samples=30,
        elite_ratio=0.2,
        return_mode="mean",
        lower_bound=torch.tensor([0.0]),
        upper_bound=torch.tensor([1.6]),
    )
    optimizer.reset(initial_x)
    x_hat, _ = optimizer.optimize(func, verbose=True)
    print("x_hat: ", x_hat)
    assert torch.allclose(x_hat, torch.tensor([[1.0], [1.0]]), atol=1e-2)


if __name__ == "__main__":
    test_cem_optimizer()
