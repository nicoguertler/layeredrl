from typing import Dict, Optional, Tuple

from .level import Level

from gymnasium.spaces import Space
import torch


class RandomLevel(Level):
    """A level that samples random actions."""

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
    ) -> Tuple[torch.Tensor, Dict]:
        """Get a random action.

        Note that only the action for the active instances is returned.

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
        super().get_action(
            mapped_env_obs, level_input, level_input_info, active_instances
        )
        batch_size = mapped_env_obs.shape[0]
        return self.sample_from_action_space(batch_size), {}
