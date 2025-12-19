from tianshou.data import Batch, ReplayBuffer
import torch

from .predictor import Predictor


class StaticPredictor(Predictor):
    """A static predictor without any training functionality."""

    def loss(self, batch: Batch) -> torch.Tensor:
        """Compute the loss for the given batch.

        Args:
            batch: The batch. The first dimension corresponds to the batch dimension (e.g. environments).
            For example, batch.state.shape = (batch_size, per_env_size, state_dim)
        Returns:
            The loss.
        """
        return None

    def learn(self, buffer: ReplayBuffer, n_steps: int, batch_size: int) -> None:
        """Learn from the given batch.

        Args:
            buffer: The replay buffer. Partial trajectories will be sampled from this buffer.
            n_steps: The number of optimizer steps to take.
            batch_size: The batch size, i.e., the number of partial trajectories to sample.
        Returns:
            The loss after the updates.
        """
        pass
