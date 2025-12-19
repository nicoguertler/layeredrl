from typing import Dict, Tuple, Optional

import torch

from .policy import Policy


class UniformPolicy(Policy):
    """A policy that randomly samples actions from the action space."""

    def reset(self) -> None:
        pass

    def _get_raw_action(
        self,
        mapped_env_obs: torch.Tensor,
        level_input: Optional[torch.Tensor],
        level_state: Optional[Dict],
        deterministic: bool,
    ) -> Tuple[torch.Tensor, Dict]:
        """Get a raw, untransformed action for the given observation.

        The components of the raw action lie in the range [-1, 1] for
        continuous action spaces.

        Args:
            mapped_env_obs: The observation from the environment after the
                env_obs_map has been applied.
            level_input: The input to this level, i.e., the action from the level above.
            level_state: The state of the level.
            deterministic: Whether to return a deterministic action (as opposed
                to a stochastic one).
        Returns:
            The unscaled action, and a Dict with info about the action (logits etc.)
        """
        return torch.tensor(self.action_space.sample(), device=self.device), {}
