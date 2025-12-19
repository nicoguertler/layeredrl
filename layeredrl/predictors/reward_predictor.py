from copy import deepcopy
from typing import Dict, Optional, Tuple, Union

import numpy as np
from tianshou.data import Batch, ReplayBuffer
import torch
from torch import nn
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

from ..models import Model
from .predictor import Predictor
from ..utils.misc import to_torch
from ..utils.schedules import Schedule, ConstantSchedule
from ..utils.normalization import Standardizer


def _to_schedule(lr: Union[Schedule, float]) -> Schedule:
    if isinstance(lr, Schedule):
        return lr
    return ConstantSchedule(lr)


class RewardPredictor(Predictor):
    """Predictor that learns the encoder via the reward function.

    Gradients from fitting the reward function are backpropagated to the encoder."""

    def __init__(
        self,
        model: Model,
        val_func: Optional[Module],
        rew_func: Optional[Module],
        encoder: Module,
        latent_state_dim: int,
        context_dim: int,
        gamma: float = 0.99,
        tau: float = 0.005,
        k_steps: int = 1,
        n_steps: int = 1,
        learn_encoder: bool = True,
        consistency_loss_weight: float = 1.0,
        value_loss_weight: float = 1.0,
        rew_loss_weight: float = 1.0,
        delta_standardizer_lr: float = 1.0e-3,
        encoder_lr: Union[Schedule, float] = 3.0e-4,
        reward_lr: Union[Schedule, float] = 3.0e-4,
        model_lr: Union[Schedule, float] = 3.0e-4,
        value_lr: Union[Schedule, float] = 3.0e-4,
        encoder_warm_up: int = 20,
        delta_standardizer: bool = True,
        clip_grad_max_norm: Optional[float] = 20.0,
        device: torch.device = torch.device("cpu"),
        writer: Optional[SummaryWriter] = None,
        log_interval: int = 10,
    ):
        """Initialize the predictor.

        Args:
            model: The dynamics, reward and termination model. Takes in latent state and action.
            val_func: Predicts the value given the latent state, the context, and the action.
                Thus, this could be a Q-function or a value function (that ignores the action).
            rew_func: Predicts the reward given the latent state, the context, and the action.
            encoder: Maps the observation to the latent state and context.
            latent_state_dim: The dimension of the latent state space.
            context_dim: The dimension of the context variable (encoding time invariant information).
            gamma: The discount factor.
            tau: The soft update rate for the target networks.
            k_steps: The length of sampled trajectory parts.
            n_steps: The number of steps in the n-step return target for the value function.
            learn_encoder: Whether to learn the map to the latent space.
            consistency_loss_weight: The weight of the consistency loss.
            value_loss_weight: The weight of the value loss.
            rew_loss_weight: The weight of the reward loss.
            delta_standardizer_loss_weight: The learning rate for the delta standardizer.
            encoder_lr: The learning rate for the encoder (can also be a schedule in total env steps).
            reward_lr: The learning rate for the reward function (can also be a schedule in total env steps).
            model_lr: The learning rate for the model (can also be a schedule in total env steps).
            encoder_warm_up: The number of steps during which only to learn the map to the latent space.
            clip_grad_max_norm: If not None, the gradients are clipped to the given maximum norm.
            device: The device to use.
            writer: A tensorboard summary writer for logging (optional).
            log_interval: The interval in which to log to tensorboard (in calls to learn).
        """
        self.k_steps = k_steps
        self.n_steps = n_steps
        self.gamma = gamma
        self.tau = tau
        self.learn_encoder = learn_encoder
        self.cons_weight = consistency_loss_weight
        self.val_weight = value_loss_weight
        self.rew_loss_weight = rew_loss_weight
        self.encoder_warm_up = encoder_warm_up
        self.delta_standardizer = delta_standardizer
        self.clip_grad_max_norm = clip_grad_max_norm
        super().__init__(
            model,
            val_func,
            rew_func,
            encoder,
            latent_state_dim,
            context_dim,
            device,
            writer,
            log_interval,
        )
        self.target_val_func = deepcopy(val_func)

        parameter_groups = [
            {"params": self.model.parameters()},
            {"params": self.val_func.parameters() if self.val_func else []},
            {"params": self.rew_func.parameters()},
        ]
        # NOTE: Has to match the order of the parameter groups in the optimizer.
        self.lr_schedules = [
            _to_schedule(model_lr),
            _to_schedule(value_lr),
            _to_schedule(reward_lr),
        ]
        if self.learn_encoder:
            parameter_groups.append({"params": self.encoder.parameters()})
            self.lr_schedules.append(_to_schedule(encoder_lr))
        self.total_optim = torch.optim.Adam(parameter_groups)
        self.optims = {
            "total_optim": self.total_optim,
        }

        self.standardizer_lr_schedule = _to_schedule(delta_standardizer_lr)
        if self.delta_standardizer:
            self.delta_standardizer = Standardizer(
                latent_state_dim=latent_state_dim, device=device
            )
            self.delta_standardizer_optim = torch.optim.Adam(
                self.delta_standardizer.parameters()
            )
        else:
            self.delta_standardizer = None
            self.delta_standardizer_optim = None
        self.delta_stand_inverse = torch.nn.Parameter(
            torch.eye(self.latent_state_dim, device=device), requires_grad=False
        )

    def soft_update(self, target: nn.Module, source: nn.Module, tau: float) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def update_target_nets(self) -> None:
        if self.val_func is not None:
            self.soft_update(self.target_val_func, self.val_func, tau=self.tau)

    def compute_targets(
        self,
        buffer: ReplayBuffer,
        batch_size: int,
        k_steps: int,
        n_steps: int,
        gamma: float = 0.99,
    ) -> Batch:
        """Compute the targets for the given indices.

        Note: This bootstraps at the end of an unfinished episode (an episode which is incomplete in
        the buffer because it has not ended yet). In other words, the end of an unfinished episode is
        treated like a truncated episode.

        All rewards in a k_steps + n_steps trajectory are first added up with discounting and then
        differences of these values are considered. This lowers the complexity from O(k * n) to O(k + n).

        Args:
            buffer: The replay buffer.
            batch_size: The batch size, i.e., the number of partial trajectories to sample.
            k_steps: The length of sampled trajectory parts.
            n_steps: The number of steps in the n-step return target for the value function.
        Returns:
            A batch containing the states, rewards, bootstrap values etc.
        """
        # have to get longer partial trajectories (n + k) because of n-step return
        indices, traj_len, validity_mask = self.sample_partial_trajectories(
            buffer, batch_size, k_steps + n_steps, self.device
        )

        rew_targets = torch.empty((batch_size, k_steps + n_steps), device=self.device)
        states = torch.empty(
            (batch_size, k_steps + n_steps, self.latent_state_dim), device=self.device
        )
        terminated = torch.empty((batch_size, k_steps + n_steps), device=self.device)
        action_dim = buffer.act[0].shape[0]
        actions = torch.empty(
            (batch_size, k_steps + n_steps, action_dim), device=self.device
        )
        contexts = torch.empty(
            (batch_size, k_steps + n_steps, self.context_dim), device=self.device
        )
        bootstrap_values = torch.empty(
            (batch_size, k_steps + n_steps), device=self.device
        )
        planning_mode = torch.empty((batch_size, k_steps + n_steps), device=self.device)
        obs = []

        for i in range(k_steps + n_steps):
            rew_targets[:, i] = to_torch(buffer.rew[indices[i]], device=self.device)
            obs.append(to_torch(buffer.obs[indices[i]], device=self.device))
            actions[:, i, :] = to_torch(buffer.act[indices[i]], device=self.device)
            states[:, i, :], contexts[:, i, :] = self.encode(
                to_torch(buffer.obs[indices[i]], device=self.device)
            )
            terminated[:, i] = to_torch(
                buffer.terminated[indices[i]], device=self.device
            )
            planning_mode[:, i] = (
                ~to_torch(buffer.info.random_mode[indices[i]], device=self.device)
            ).float()
            # can bootstrap from normal states within the episodes, and from truncated states at the end of
            # episodes
            is_unfinished = to_torch(
                np.isin(indices[i], buffer.unfinished_index()), device=self.device
            )
            is_truncated = to_torch(buffer.truncated[indices[i]], device=self.device)
            bootstrap_from_here = torch.logical_and(
                i < traj_len + n_steps,
                (i < traj_len) | is_truncated | is_unfinished,
            )
            if self.val_func is None:
                bootstrap_values[:, i] = 0.0
            else:
                with torch.no_grad():
                    bootstrap_values[:, i] = self.target_val_func(
                        states[:, i, :], contexts[:, i, :], actions[:, i, :]
                    )
            bootstrap_values[:, i] *= bootstrap_from_here
        gammas = torch.vander(
            torch.tensor([gamma], device=self.device),
            increasing=True,
            N=k_steps + n_steps,
        ).squeeze()

        obs = torch.stack(obs, dim=1)

        # add up discounted reward along full k + n steps
        discounted_rew = rew_targets * gammas * validity_mask
        cum_rew = torch.zeros((batch_size, k_steps + n_steps), device=self.device)
        cum_rew[:, 1:] = torch.cumsum(discounted_rew[:, :-1], dim=1)
        ret = (cum_rew[:, n_steps:] - cum_rew[:, :-n_steps]) / gammas[:k_steps]
        # add bootstrap values
        valid_steps_left = torch.clamp(
            traj_len[:, None].expand(-1, k_steps)
            - torch.arange(0, k_steps, device=self.device),
            min=1,
            max=n_steps,
        )
        bootstrap_gamma = gamma**valid_steps_left
        ret += bootstrap_gamma * bootstrap_values[:, n_steps:]

        targets = Batch(
            rew=rew_targets[:, :k_steps],
            indices=indices,  # returning k_steps + n_steps indices for tests
            traj_len=traj_len,
            validity_mask=validity_mask[:, :k_steps],
            value=ret,
            act=actions[:, :k_steps, :],
            state=states[:, : k_steps + 1, :],
            contexts=contexts[:, :k_steps, :],
            terminated=terminated[:, :k_steps],
            obs=obs,
            planning_mode=planning_mode[:, :k_steps],
        )
        return targets

    def sample_model_batch(self, buffer: ReplayBuffer, batch_size: int) -> Batch:
        """Sample a batch for training the model. Includes only state, next state, context,
        next context, and validity mask. The states are sampled randomly and do not belong
        to partial trajectories.

        Args:
            buffer: The replay buffer.
            batch_size: The batch size.
        Returns:
            The batch.
        """
        # Does not recalculate reward because sample isn't used
        indices = buffer.sample_indices(batch_size)
        obs = buffer.obs[indices]
        state, context = self.encode(to_torch(obs, device=self.device))
        next_state, _ = self.encode(
            to_torch(buffer.obs_next[indices], device=self.device)
        )
        batch = Batch(
            state=state.detach(),
            state_next=next_state.detach(),
            act=to_torch(buffer.act[indices], device=self.device),
            terminated=to_torch(buffer.terminated[indices], device=self.device),
            context=context,
        )
        return batch

    def get_standardized_log_prob(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        fixed_std: torch.Tensor,
    ) -> Tuple:
        """Get log probability (density) of next state in the standardized space, the termination probability,
        and an info dict.

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
        return self.model.get_log_prob(
            state=state,
            context=context,
            action=action,
            next_state=next_state,
            std=fixed_std,
            transform=self.delta_standardizer,
        )

    def loss(
        self,
        batch: Batch,
        model_batch: Batch,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute the loss for the given batch.

        Args:
            batch: Batch used for reward and value learning. The first dimension corresponds to the batch dimension
                (e.g. environments), and the second dimension corresponds to n + k steps time steps.
                For example, batch.state.shape = (batch_size, n + k, state_dim)
            model_batch: Batch used for training the model. The first dimension also corresponds to the batch dimension,
                but there is no second dimension for time steps, only individual transitions are sampled.
        Returns:
            The loss and an info dictionary with the individual loss terms.
        """
        # consistency loss for dynamics model
        consistency_loss = self.model.loss(model_batch)

        # use first context in sequence to prevent state from leaking into it
        first_contexts = batch.contexts[:, 0, :].unsqueeze(1).expand_as(batch.contexts)

        if self.val_func is not None:
            # value loss
            values = self.val_func(
                # NOTE: Not detaching here did not improve performance, but it might warrant
                # further investigation.
                batch.state[:, :-1, :].detach(),
                first_contexts.detach(),
                batch.act,
            )
            value_loss = (
                batch.validity_mask
                * batch.planning_mode
                * (batch.value.detach() - values) ** 2
            )
            value_loss = value_loss.mean() / batch.validity_mask.mean()
        else:
            value_loss = torch.tensor(0.0, device=self.device)

        if self.learn_encoder and hasattr(self.encoder, "mask"):
            # mask separating fixed and changing observation dimensions
            self.encoder.mask.data = torch.logical_or(
                (batch.obs[:, -1] - batch.obs[:, 0]).abs().mean(dim=0) > 0.0,
                self.encoder.mask,
            )

        # reward loss
        rewards = self.rew_func(batch.state[:, :-1, :], first_contexts, batch.act)
        rew_loss = torch.nn.functional.mse_loss(rewards, batch.rew.detach())

        # standardizer loss
        standard_delta = self.delta_standardizer(
            model_batch.state_next.detach() - model_batch.state.detach()
        ).squeeze(1)
        cov = torch.cov(standard_delta.T)
        standardizer_loss = torch.linalg.matrix_norm(
            cov - torch.eye(cov.shape[0], device=self.device)
        )

        loss = (
            +self.val_weight * value_loss
            + self.rew_loss_weight * rew_loss
            + self.cons_weight * consistency_loss
        )

        info = {
            "value_loss": self.val_weight * value_loss,
            "rew_loss": self.rew_loss_weight * rew_loss,
            "consistency_loss": self.cons_weight * consistency_loss,
            "standardizer_loss": standardizer_loss,
        }
        return loss, info

    def _update_standardizer_inverse(self):
        """Update the inverse of the standardizer matrix in the random dynamics."""

        if not hasattr(self.model, "random_dynamics"):
            return

        symm_start = self.model.symmetry_breaking_start
        symm_start = (
            symm_start and self.n_total_env_steps < self.model.sb_start_duration
        )

        # update delta standardizer transformation in random dynamics
        if self.delta_standardizer and symm_start:
            with torch.no_grad():
                stand_mat = self.delta_standardizer.get_symm_mat()
                self.delta_stand_inverse.data = torch.inverse(stand_mat)
                matrix = self.delta_stand_inverse
                if self.model.normalize_targets:
                    matrix = torch.einsum(
                        "i,ij->ij",
                        1.0
                        / (self.model.output_bn.var + self.model.output_bn._eps).sqrt(),
                        matrix,
                    )
                self.model.random_dynamics.transform.data = matrix

    def learn(
        self,
        buffer: ReplayBuffer,
        n_steps: int,
        batch_size: int,
        model_batch_size: int,
        n_total_env_steps: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """Sample batches, calculate losses, and update models.

        Args:
            buffer: The replay buffer. Partial trajectories will be sampled from this buffer.
            n_steps: The number of optimizer steps to take.
            batch_size: The batch size, i.e., the number of partial trajectories to sample (for value and
                reward functions).
            model_batch_size: The batch size for training the model.
            n_total_env_steps: The total number of environment steps taken so far. Useful for logging
                and learning rate schedules.
        Returns:
            The loss and an info dictionary with the individual loss terms.
        """
        super().learn(buffer, n_steps, batch_size, model_batch_size, n_total_env_steps)
        self.model.set_n_total_env_steps(n_total_env_steps)

        # learning rate schedules for parameter groups
        for p_group, lr_sched in zip(self.total_optim.param_groups, self.lr_schedules):
            p_group["lr"] = lr_sched(n_total_env_steps)
        if self.delta_standardizer_optim:
            self.delta_standardizer_optim.param_groups[0]["lr"] = (
                self.standardizer_lr_schedule(n_total_env_steps)
            )

        buffer_empty = len(buffer) == 0
        if self._learn_counter > self.encoder_warm_up and not buffer_empty:
            for _ in range(n_steps):
                batch = self.compute_targets(
                    buffer=buffer,
                    batch_size=batch_size,
                    k_steps=self.k_steps,
                    n_steps=self.n_steps,
                    gamma=self.gamma,
                )
                model_batch = self.sample_model_batch(
                    buffer=buffer,
                    batch_size=model_batch_size,
                )
                loss, loss_info = self.loss(batch, model_batch)
                self.total_optim.zero_grad()
                loss.backward()

                if self.clip_grad_max_norm is not None:
                    for p_group in self.total_optim.param_groups:
                        for param in p_group["params"]:
                            torch.nn.utils.clip_grad_norm_(
                                param, self.clip_grad_max_norm
                            )
                self.total_optim.step()

                # standardizer optimization
                self.delta_standardizer_optim.zero_grad()
                loss_info["standardizer_loss"].backward()
                self.delta_standardizer_optim.step()

                # update target nets
                self.update_target_nets()

                # update inverse of standardizer matrix in random dynamics
                self._update_standardizer_inverse()
        else:
            loss = torch.tensor(0.0, device=self.device)
            loss_info = {}

        return loss, loss_info
