from typing import Dict, Optional

import torch

from .level import Level

from gymnasium.spaces import Space


class ConstantLevel(Level):
    """A level that outputs a constant action."""

    def __init__(self, action: torch.Tensor):
        """Initialize the level.

        Args:
            action: The action to output.
        """
        super().__init__()
        self.action = action

    def get_input_space(self) -> Space:
        """Get the input space of level.

        Input space denotes everything the level needs except for the
        environment observation, e.g. a skill vector.

        Returns:
            The input space (or None if no input is required).
        """
        return None

    def get_action(
        self,
        mapped_env_obs: torch.Tensor,
        level_input: Optional[torch.Tensor],
        level_input_info: Optional[Dict],
        active_instances: torch.Tensor = torch.tensor([True], dtype=torch.bool),
    ) -> torch.Tensor:
        """Get an action for the given observation (in this case a constant).

        Note that only the action for the active instances is returned.

        Call this at the beginning of the implementation of get_action in derived classes.

        Args:
            mapped_env_obs: The environment observation after the self.env_obs_map has been
                applied. Note that the observation has a batch dimension (for multiple
                environment instances).
            level_input: The input to this level for the active instances, i.e., the
                action from the level above.
            level_input_info: Additional information about the level input.
            active_instances: In which of the environment instances the level is active.
                env_obs and level_input correspond to these instances.
        Returns:
            The action (also with a batch dimension).
            An info dict containing additional information about the action.
        """
        super().get_action(mapped_env_obs, level_input, active_instances)
        batch_size = mapped_env_obs.shape[0]
        return self.action.repeat(batch_size, 1), dict()
