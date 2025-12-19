from typing import Any, Dict, Tuple

import numpy as np
import torch

from tianshou.utils.net.common import Net


class ConcatNet(Net):
    """A network that concatenates mapped_env_obs, level_input, action, n_remaining_states, and level_state along
    the last dimension.

    For use with Tianshou level."""

    def __init__(
        self,
        mapped_env_obs_shape: int,
        level_input_dim: int,
        level_state_dims: Dict[str, int],
        scale_remaining_steps: float = 1.0,
        *args,
        **kwargs,
    ):
        """Initialize the network."""
        self.mapped_env_obs_shape = mapped_env_obs_shape
        self.mapped_env_obs_rank = len(mapped_env_obs_shape)
        self.scale_remaining_steps = scale_remaining_steps
        state_shape = (
            np.prod(mapped_env_obs_shape)
            + level_input_dim
            + sum(level_state_dims.values()),
        )
        super().__init__(state_shape=state_shape, *args, **kwargs)

    def forward(
        self, obs: Dict, state: Any = None, **kwargs: Any
    ) -> Tuple[torch.Tensor, Any]:
        """Concatenate the inputs along the last dimension."""
        element_lst = [
            obs["mapped_env_obs"].flatten(start_dim=-self.mapped_env_obs_rank)
        ]
        if obs["level_input"] is not None:
            element_lst.append(obs["level_input"])
        if "action" in obs:
            element_lst.append(obs["action"])
        if "n_remaining_steps" in obs["level_state"]:
            # rescale the remaining steps, usually to [0, 1]
            element_lst.append(
                obs["level_state"]["n_remaining_steps"] * self.scale_remaining_steps
            )
        element_lst.extend(
            v for k, v in obs["level_state"].items() if k != "n_remaining_steps"
        )
        total_obs = torch.cat(element_lst, dim=-1)
        return super().forward(total_obs, state, **kwargs)
