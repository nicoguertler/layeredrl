import torch


class EnsembleDisagreementReward:
    def __call__(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Compute the ensemble disagreement reward for the given trajectory.

        Args:
            trajectory: The trajectory. state_epistemic_var shape:
                (..., horizon, state_dim)
        Returns:
            The auxiliary reward. Shape: (..., horizon)
        """
        # sum over state dimensions and drop initial time step (no disagreement there and
        # for consistency with the extrinsic reward)
        return trajectory.state_epistemic_var.sum(dim=-1)[..., 1:]
