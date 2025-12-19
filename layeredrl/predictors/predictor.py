from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
from tianshou.data import Batch, ReplayBuffer
import torch
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

from ..models import Model
from ..utils.misc import to_torch


class Predictor(ABC, Module):
    """Abstract base class for prediction module.

    This includes the dynamics and reward model and the value function.
    """

    def __init__(
        self,
        model: Model,
        val_func: Optional[Module],
        rew_func: Optional[Module],
        encoder: Module,
        latent_state_dim: int,
        context_dim: int,
        device: torch.device = torch.device("cpu"),
        writer: Optional[SummaryWriter] = None,
        log_interval: int = 10,
    ):
        """Initialize the predictor.

        Args:
            model: The dynamics, and termination model. Takes in latent state, context and action.
            val_func: Predicts the value given the latent state, the context, and the action.
                Thus, this could be a Q-function or a value function (that ignores the action).
                Required for training the predictor.
            rew_func: Predicts the reward given the latent state, the context, and the action.
            encoder: Maps the observation to the latent state and context.
            latent_state_dim: The dimension of the latent state space.
            context_dim: The dimension of the context variable (encoding time invariant information
                that stays constant during an episode).
            device: The device to use.
            log_interval: The interval in which to log to tensorboard (in calls to learn).
        """
        super().__init__()
        self.model = model
        self.val_func = val_func
        self.rew_func = rew_func
        self.encoder = encoder
        self.latent_state_dim = latent_state_dim
        self.context_dim = context_dim
        self.device = device
        self.writer = writer
        self.log_interval = log_interval
        self.register_buffer("_learn_counter", torch.tensor(0, dtype=torch.long))
        self.n_total_env_steps = 0

    @staticmethod
    def sample_partial_trajectories(
        buffer: ReplayBuffer,
        batch_size: int,
        n_steps: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Sample partial trajectories from the replay buffer.

        Args:
            buffer: The replay buffer.
            batch_size: The batch size, i.e., the number of partial trajectories to sample.
            n_steps: The number of steps per partial trajectory.
        Returns:
            A numpy array with the indices (in the buffer) of the sampled partial trajectories.
            Shape: (n_steps, batch_size)
            A numpy array with the length of each partial trajectory (which can be different
            as some trajectories might have terminated). Shape: (batch_size,)
            A numpy array with the validity mask for each partial trajectory. The mask is 1.0
            for valid time steps and 0.0 for invalid time steps beyond the end of an episode.
            Shape: (batch_size, n_steps)
        """
        indices = buffer.sample_indices(batch_size)
        indices_torch = [to_torch(indices, device=device)]
        indices = [indices]
        traj_len = n_steps * torch.ones(batch_size, dtype=int, device=device)
        # mask out invalid time steps beyond the end of a trajectory
        validity_mask = torch.ones((batch_size, n_steps), dtype=float, device=device)
        for i in range(n_steps - 1):
            # find next index for each trajectory
            indices.append(buffer.next(indices[-1]))
            indices_torch.append(to_torch(indices[-1], device=device))
            ends_here = to_torch(indices[-1] == indices[-2], device=device)
            traj_len[ends_here] -= 1
            validity_mask[ends_here, i + 1] = 0.0
        indices = np.stack(indices)
        return indices, traj_len, validity_mask

    @abstractmethod
    def loss(self, batch: Batch) -> Tuple[torch.Tensor, Dict]:
        """Compute the loss for the given batch.

        Args:
            batch: The batch. The first dimension corresponds to the batch dimension (e.g. environments).
                For example, batch.state.shape = (batch_size, per_env_size, state_dim)
        Returns:
            The loss and an info dictionary with the individual loss terms.
        """
        pass

    @abstractmethod
    def learn(
        self,
        buffer: ReplayBuffer,
        n_steps: int,
        batch_size: int,
        model_batch_size: int,
        n_total_env_steps: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """Learn from the given batch.

        Args:
            buffer: The replay buffer. Partial trajectories will be sampled from this buffer.
            n_steps: The number of optimizer steps to take.
            batch_size: The batch size, i.e., the number of partial trajectories to sample.
            model_batch_size: The batch size for the model training.
            n_total_env_steps: The total number of environment steps taken so far. Useful for logging
                and learning rate schedules.
        Returns:
            The loss and an info dictionary with the individual loss terms.
        """
        self._learn_counter += 1
        self.n_total_env_steps = n_total_env_steps

    def encode(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode the observation into the latent state and context.

        Args:
            obs: The observation.
        Returns:
            The latent state and context.
        """
        return self.encoder(obs)

    def predict(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Predict the next latent state, reward etc. given the current observation and action.

        Args:
            obs: The observation.
            act: The action.
        Returns:
            Mean, weights, and std of the predicted mixture of Gaussians for the next latent state,
            expected reward, and termination probability.
            s_mean shape: (batch_size, n_models, n_modes, latent_state_dim)
            weights shape: (batch_size, n_models, n_modes)
            s_std shape: (batch_size, n_models, n_modes, latent_state_dim)
            reward shape: (batch_size, )
            term_prob shape: (batch_size, )
        """
        latent_state, context = self.encode(obs)
        s_mean, weights, s_std, term_prob = self.model.predict(
            latent_state, context, act
        )
        r = self.rew_func(latent_state, context, act)
        return s_mean, weights, s_std, r, term_prob

    def predict_reward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Predict the reward given the current observation and action.

        Args:
            obs: The observation.
            act: The action.
        Returns:
            The predicted reward.
        """
        latent_state, context = self.encode(obs)
        return self.rew_func(latent_state, context, act)

    def rollout(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[Batch, torch.Tensor]:
        """Rollout the model given the current observation and action.

        Args:
            obs: The observation.
            act: The action.
            deterministic: Whether to rollout deterministically.
        Returns:
            The rollout data and the contexts for the trajectories.
        """
        latent_state, context = self.encode(obs)
        trajectory = self.model.rollout(
            latent_state, context, act, deterministic=deterministic
        )
        n_models = self.model.n_models
        n_particles_per_model = self.model.n_particles_per_model
        traj_act = act[:, None, None].expand(
            -1, n_models, n_particles_per_model, -1, -1
        )
        horizon = traj_act.shape[-2]
        traj_cont = context[:, None, None, None].expand(
            -1, n_models, n_particles_per_model, horizon, -1
        )
        if self.rew_func is not None:
            trajectory.rew = self.rew_func(
                trajectory.state[..., :-1, :], traj_cont, traj_act
            )
        return trajectory, context

    def value(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Predict the value of the given state-action pair.

        Args:
            obs: The observation.
            act: The action.
        Returns:
            The value.
        """
        latent_state, context = self.encode(obs)
        return self.val_func(latent_state, context, act)
