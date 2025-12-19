from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
from tianshou.data import Batch


class Model(ABC, torch.nn.Module):
    """Abstract base class for dynamics models."""

    def __init__(
        self,
        n_models: int = 1,
        n_particles_per_model: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize the model.

        Args:
            n_particles: The number of particles in the ensemble.
            device: The device to use.
        """
        super().__init__()
        self.n_models = n_models
        self.n_particles_per_model = n_particles_per_model
        self.device = device
        self.register_buffer("n_total_env_steps", torch.tensor(0, dtype=torch.long))

    @abstractmethod
    def get_parameters(self) -> torch.Tensor:
        """Get the parameters of the model.

        Returns:
            An iterator over the parameters of the model.
        """
        pass

    def set_n_total_env_steps(self, n_total_env_steps: int) -> None:
        """Set the total number of environment steps.

        Args:
            n_total_env_steps: The total number of environment steps.
        """
        self.n_total_env_steps[...] = n_total_env_steps

    @abstractmethod
    def loss(self, batch: Batch) -> torch.Tensor:
        """Compute the loss for the given batch.

        Args:
            batch: The batch. The first dimension corresponds to the batch dimension (e.g. environments).
            For example, batch.state.shape = (batch_size, per_env_size, state_dim)
        Returns:
            The loss.
        """
        pass

    def learn(self, batch_lst: List[Batch]) -> None:
        """Learn from the given batch.

        Args:
            batch_lst: A list of training batches. The first dimension corresponds to the transitions.
        Returns:
            The loss after the updates.
        """
        pass

    @abstractmethod
    def rollout(
        self,
        initial_state: torch.Tensor,
        context: torch.Tensor,
        actions: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Rollout with the given actions from the given initial state (open loop).

        Note that everything is assumed to have a 'batch' dimension, useful for parallelizing,
        e.g. for vectorized environments.

        Args:
            initial_state: The initial state. Shape: (batch_size, state_dim)
            context: The context, i.e., information that is constant over the whole rollout.
            actions: The actions. Shape: (batch_size, horizon, action_dim)
            deterministic: Whether to use the mean of the predicted distribution or to sample
                from it.
        Returns:
            Batch containing the resulting states, termination probabilities and aleatoric
            and epistemic uncertanties.
            state shape: (batch_size, n_models, n_particles_per_model, horizon + 1, state_dim)
            state_mean shape: (batch_size, n_models, horizon + 1, state_dim)
            term_prob shape: (batch_size, n_models, n_particles_per_model, horizon)
        """
        pass

    @abstractmethod
    def predict(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        action: torch.Tensor,
        std: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Predict the next state given the current state and action.

        Note that everything is assumed to have a 'batch' dimension, useful for parallelizing,
        e.g. for vectorized environments. This uses the mean of the predicted distribution
        and does not sample.

        Args:
            state: The current state.
            context: The context, i.e., information that is constant over timesteps.
            action: The action.
            std: Overwrites the standard deviation that the model predicts if provided.
        Returns:
            Mean, weights and standard deviations of the modes of the mixture of Gaussians that
            make up the ensemble. Also averaged termination probability.
            Shape for state and std: (batch_size, n_models, n_modes, state_dim)
            Shape for weights: (batch_size, n_models, n_modes)
            Shape for term_prob: (batch_size)
        """
        pass

    @abstractmethod
    def sample(
        self, state: torch.Tensor, context: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Sample the next state given the current state and action.

        Note that everything is assumed to have a 'batch' dimension, useful for parallelizing,
        e.g. for vectorized environments.

        Args:
            state: The current state.
            context: The context, i.e., information that is constant over timesteps.
            action: The action.
        Returns:
            The sampled next state.
        """
        pass

    def get_prob(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> Tuple:
        """Get probability (density) of next state, the termination probability, and an info dict.

        Note that everything is assumed to have a 'batch' dimension, useful for parallelization.

        Args:
            state: The current state.
            context: The context, i.e., information that is constant over timesteps.
            action: The action.
            next_state: The next state.
        Returns:
            A tuple containing:
                - The probability (density) of the next state given the current state and action under the model.
                - The termination probability given the current state and action under the model.
                - An info dict with additional information.
        """
        log_prob, term_prob, info = self.get_log_prob(
            state, context, action, next_state
        )
        # clip from below before applying exp to avoid nan
        log_prob = log_prob.clamp(min=-700.0)
        return torch.exp(log_prob), term_prob, info

    @abstractmethod
    def get_log_prob(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> Tuple:
        """Get log of probability (density) of next state, the termination probability, and an info dict.

        Note that everything is assumed to have a 'batch' dimension, useful for parallelization.

        Args:
            state: The current state.
            context: The context, i.e., information that is constant over timesteps.
            action: The action.
            next_state: The next state.
        Returns:
            A tuple containing:
                - The log probability (density) of the next state given the current state and action under the model.
                - The termination probability given the current state and action under the model.
                - An info dict with additional information.
        """
        pass
