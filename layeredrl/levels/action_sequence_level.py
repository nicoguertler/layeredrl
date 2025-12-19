from typing import Dict, Optional

import torch

from .level import Level

from gymnasium.spaces import Space


class ActionSequenceLevel(Level):
    """A level that outputs a pre-defined sequence of actions."""

    def __init__(self, action_sequence: torch.Tensor, repeat: int = 1, **kwargs):
        """Initialize the level.

        Args:
            action_sequence: The action sequence to output.
            repeat: How often to repeat each action.
        """
        super().__init__(**kwargs)
        self.action_sequence = action_sequence
        self.indices = None
        self.repeat = repeat

    def get_input_space(self) -> Space:
        """Get the input space of level.

        Input space denotes everything the level needs except for the
        environment observation, e.g. a skill vector.

        Returns:
            The input space (or None if no input is required).
        """
        return None

    def reset(self):
        """Reset the level.

        Call once before the first episode, not between episodes."""
        super().reset()
        self.indices = torch.zeros(
            self.n_env_instances, dtype=torch.long, device=self.device
        )

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
        indices = self.indices[active_instances]
        indices = indices // self.repeat
        action = self.action_sequence[indices]
        return action, {}

    def process_transition(
        self,
        mapped_env_obs: torch.Tensor,
        level_input: Optional[torch.Tensor],
        action: torch.Tensor,
        next_mapped_env_obs: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        active_instances: torch.Tensor,
    ) -> bool:
        """Process transition and check whether level would like to return
        control to the level above.

        This usually involves adding the transition to the replay buffer and possibly
        preprocessing it.

        Note that everything has a batch dimension.

        Args:
            mapped_env_obs: The mapped environment observation for the active instances.
            level_input: The input to this level for the active instances, i.e., the
                action from the level above.
            action: The action that was taken by the level.
            next_mapped_env_obs: The next mapped environment observation for the active instances.
            terminated: Whether the episode terminated for the active instances.
            truncated: Whether the episode was truncated for the active instances.
            active_instances: In which of the environment instances the level is active.
                next_obs and terminated correspond to these instances.
        Returns:
            Whether the level is done, i.e. whether it hands control back to
            the level above.
        """
        self.indices[active_instances] += 1
        self.indices[torch.logical_and(active_instances, truncated)] = 0
        return super().process_transition(
            mapped_env_obs,
            level_input,
            action,
            next_mapped_env_obs,
            terminated,
            truncated,
            active_instances,
        )
