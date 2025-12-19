from typing import SupportsFloat

from gymnasium import RewardWrapper
import numpy as np


class LogRewWrapper(RewardWrapper):
    """Applies log to the reward."""

    def __init__(self, env, r_offset: float = 30.0, r_scale: float = 1.0 / 50):
        super().__init__(env)
        self.reward_range = (0.0, 1.0)
        self._r_offset = r_offset
        self._r_scale = r_scale

    def reward(self, r: SupportsFloat) -> SupportsFloat:
        return self._r_scale * (np.log(r) + self._r_offset)
