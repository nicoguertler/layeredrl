from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional

from gymnasium.spaces import Box
import torch


class Policy(ABC, torch.nn.Module):
    """Policy mapping env obs, level input, and level state to actions."""

    def __init__(self, action_space: Box, device=torch.device("cpu")):
        """Initialize the policy.

        Args:
            action_space: The action space the policy output has to
                lie in.
            device: The device to use.
        """
        super().__init__()
        self.action_space = action_space
        self.continuous = isinstance(action_space, Box)
        if self.continuous:
            self._low = torch.tensor(action_space.low, device=device)
            self._diff = torch.tensor(action_space.high, device=device) - self._low
        self.device = device

    @abstractmethod
    def reset(self) -> None:
        """Reset the policy, e.g. at the beginning of the episode."""
        pass

    @abstractmethod
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
        pass

    def transform_action(self, action: torch.Tensor) -> torch.Tensor:
        """Transform the raw action to the action space of the environment.

        Args:
            action: The raw action.
        Returns:
            The transformed action (now in the action space of the environment).
        """
        if self.continuous:
            return self._low + 0.5 * (action + 1) * self._diff
        else:
            return action

    def untransform_action(self, action: torch.Tensor) -> torch.Tensor:
        """Undo the transformation of the action.

        Args:
            action: The action in the action space of the environment.
        Returns:
            The action in the raw action space ([-1, 1]^n for continuous
            action spaces).
        """
        if self.continuous:
            return 2 * (action - self._low) / self._diff - 1
        else:
            return action

    def get_action(
        self,
        mapped_env_obs: torch.Tensor,
        level_input: Optional[torch.Tensor],
        level_state: Optional[Dict],
        deterministic: bool,
    ) -> torch.Tensor:
        """Get an action for the given observation.

        Args:
            mapped_env_obs: The observation from the environment after the
                env_obs_map has been applied.
            level_input: The input to this level, i.e., the action from the level above.
            level_state: The state of the level.
            deterministic: Whether to return a deterministic action (as opposed
                to a stochastic one).
        Returns:
            The action, and a Batch with info about the action (logits etc.)
        """
        raw_action, action_info = self._get_raw_action(
            mapped_env_obs, level_input, level_state, deterministic
        )
        return self.transform_action(raw_action), action_info

    def get_log_prob(
        self,
        mapped_env_obs: torch.Tensor,
        level_input: Optional[torch.Tensor],
        level_state: Optional[Dict],
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get the log probability of the given action under the policy.

        Args:
            mapped_env_obs: The observation from the environment after the
                env_obs_map has been applied.
            level_input: The input to this level, i.e., the action from the level above.
            level_state: The state of the level.
            action: The action.
        Returns:
            The log probability of the action under the policy.
        """
        raise NotImplementedError
