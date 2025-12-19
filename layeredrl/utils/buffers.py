from typing import List, Tuple, Union

import numpy as np
from tianshou.data import Batch, VectorReplayBuffer
import torch


class ToDeviceReplayBuffer(VectorReplayBuffer):
    """Replay buffer that moves batch to target device after sampling."""

    def __init__(self, target_device=torch.device("cpu"), *args, **kwargs):
        """Initialize the wrapper.

        Args:
            target_device: The target device to move the batch to.
        """
        super().__init__(*args, **kwargs)
        self.target_device = target_device

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
        """Sample from replay buffer and move batch to target device.

        If batch_size is 0, return all the data in the buffer

        Args:
            batch_size: The batch size.
        Returns:
            Sample data and its corresponding indices inside the buffer.
        """
        batch, indices = super().sample(batch_size)
        batch.to_torch(device=self.target_device)
        return batch, indices

    def __getitem__(self, index: Union[slice, int, List[int], np.ndarray]) -> Batch:
        """Return a data batch: self[index]."""
        batch = super().__getitem__(index)
        batch.to_torch(device=self.target_device)
        return batch
