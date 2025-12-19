from typing import SupportsFloat

from gymnasium import RewardWrapper


class AffineRewWrapper(RewardWrapper):
    """Applies an affine linear transformation to the reward signal."""

    def __init__(
        self, env, r_offset: SupportsFloat = 0.0, r_scale: SupportsFloat = 1.0
    ):
        super().__init__(env)
        self._r_offset = float(r_offset)
        self._r_scale = float(r_scale)
        self.reward_range = (0.0, 1.0)

    def reward(self, r: SupportsFloat) -> SupportsFloat:
        return self._r_scale * (r + self._r_offset)
