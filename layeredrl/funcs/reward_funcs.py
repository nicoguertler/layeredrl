import torch
from torch.nn import Module


class AffineNegDistanceReward(Module):
    def __init__(self, scale: float = 1.0, offset: float = 0.0):
        super().__init__()
        self.scale = scale
        self.offset = offset

    def forward(
        self, achieved_goal: torch.Tensor, desired_goal: torch.Tensor
    ) -> torch.Tensor:
        return -self.scale * (
            torch.norm(achieved_goal - desired_goal, dim=-1) + self.offset
        )


class ShortestPathReward(Module):
    def __init__(self, threshold: float = 1.0, offset: float = 0.0):
        super().__init__()
        self.threshold = threshold
        self.offset = offset

    def forward(
        self, achieved_goal: torch.Tensor, desired_goal: torch.Tensor
    ) -> torch.Tensor:
        return (
            -(torch.norm(achieved_goal - desired_goal, dim=-1) > self.threshold).float()
            + self.offset
        )
