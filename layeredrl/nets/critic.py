from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from tianshou.data import Batch
from tianshou.utils.net.continuous import Critic as TianshouCritic


class Critic(TianshouCritic):
    """Critic network for Tianshou level."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set weights of last layer to zero
        self.last.model[-1].weight.data.zero_()

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        act: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> torch.Tensor:
        """Same as Tianshou critic but do not flatten obs and act before passing to preprocess_net."""
        if act is not None:
            obs_dict = {k: v for k, v in obs.items()}
            obs_dict["action"] = act
            obs = Batch(obs_dict)
        logits, hidden = self.preprocess(obs)
        logits = self.last(logits)
        return logits
