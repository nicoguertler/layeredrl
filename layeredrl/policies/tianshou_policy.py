from typing import Dict, Tuple, Optional

from gymnasium.spaces import Space
import numpy as np
from tianshou.data import ReplayBuffer, Batch
from tianshou.policy import BasePolicy
import torch
from torch.distributions import Categorical, Normal

from .policy import Policy
from ..utils.misc import to_torch


class TianshouPolicy(Policy):
    """Wrapper around a tianshou policy.

    Note: Only tested with SACPolicy and DQNPolicy at the moment."""

    _action_info_keys = ["logits", "log_prob", "state"]

    def __init__(
        self, action_space: Space, ts_policy: BasePolicy, device=torch.device("cpu")
    ):
        """Initialize the policy.

        Args:
            action_space: The action space of the environment the policy
                acts in.
            ts_policy: The tianshou policy.
            device: The device to use.
        """
        super().__init__(action_space, device)
        self.ts_policy = ts_policy
        self.state = None
        self._eps = np.finfo(np.float32).eps.item()

    def reset(self) -> None:
        """Reset the policy, e.g. at the beginning of the episode."""
        self.state = None

    def _process_info_items(self, key: str, value: torch.Tensor) -> torch.Tensor:
        if key == "logits" and self.continuous:
            return value.transpose(0, 1)
        else:
            return value

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
        obs = {
            "mapped_env_obs": mapped_env_obs,
            "level_input": level_input,
            "level_state": level_state,
        }
        input_batch = Batch(obs=obs, info=None)
        output_batch = self.ts_policy(input_batch, self.state)
        if not deterministic:
            output_batch.act = self.ts_policy.exploration_noise(
                output_batch.act, Batch(obs=torch.empty(0, device=self.device))
            )
        action_info = {
            k: self._process_info_items(k, v)
            for k, v in output_batch.items()
            if k in self._action_info_keys
        }
        return to_torch(output_batch.act, self.device), action_info

    def update(self, sample_size: int, buffer: ReplayBuffer) -> None:
        """Update the policy.

        Args:
            sample_size: The number of samples in one batch.
            buffer: The replay buffer.
        """
        self.ts_policy.update(sample_size, buffer)

    def get_log_prob(
        self,
        mapped_env_obs: torch.Tensor,
        level_input: Optional[torch.Tensor],
        level_state: Optional[Dict],
        action: torch.Tensor,
        std: Optional[float] = None,
    ) -> torch.Tensor:
        """Get the log probability of the given action under the policy.

        Args:
            mapped_env_obs: The observation from the environment after the
                env_obs_map has been applied.
            level_input: The input to this level, i.e., the action from the level above.
            level_state: The state of the level.
            action: The action.
            std: The standard deviation of the action distribution. Overwrites
                the std from the policy if given.
        Returns:
            The log probability of the action under the policy.
        """
        _, act_info = self._get_raw_action(
            mapped_env_obs, level_input, level_state, deterministic=True
        )
        logits = act_info["logits"]
        if self.continuous:
            scale = (
                std * torch.ones_like(logits[:, 1, ...])
                if std is not None
                else logits[:, 1, ...]
            )
            distribution = Normal(logits[:, 0, ...], scale)
            unsquashed_action = torch.atanh(action)
        else:
            distribution = Categorical(logits=logits)
            unsquashed_action = action
        log_prob = distribution.log_prob(unsquashed_action)
        if self.continuous:
            # take tanh squashing into account (note that action is already squashed)
            log_prob = (log_prob - torch.log(1 - action**2 + self._eps)).sum(dim=-1)
        return log_prob

    def get_value(
        self,
        mapped_env_obs: torch.Tensor,
        level_input: Optional[torch.Tensor],
        level_state: Optional[Dict],
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get the value of the given obs and action as predicted by the critic.

        Args:
            mapped_env_obs: The observation from the environment after the
                env_obs_map has been applied.
            level_input: The input to this level, i.e., the action from the level above.
            level_state: The state of the level.
            action: The action.
        Returns:
            The value of the given obs and action as predicted by the critic.
        """
        obs = {
            "mapped_env_obs": mapped_env_obs,
            "level_input": level_input,
            "level_state": level_state,
        }
        return self.ts_policy.critic1(obs, action), self.ts_policy.critic2(obs, action)
