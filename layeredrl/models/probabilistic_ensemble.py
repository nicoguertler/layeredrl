from typing import Callable, Generator, List, Optional, Tuple, Union

from gymnasium.spaces import Box, Discrete
from tianshou.data import Batch
import torch
from torch.nn import Module

from .model import Model
from ..nets import RandomDynamics
from ..utils.distributions import get_normal_log_prob
from ..utils.normalization import RunningBatchNorm


class ProbabilisticEnsemble(Model):
    """Implementation of a Probabilistic Ensemble of dynamics models.

    See https://arxiv.org/abs/1805.12114.
    """

    def __init__(
        self,
        state_space: Box,
        context_space: Box,
        action_space: Union[Box, Discrete],
        partial_net: Callable[[Box, Box, Union[Box, Discrete]], Module],
        n_models: int = 1,
        n_modes: int = 1,
        n_particles_per_model: int = 1,
        learning_rate: float = 1e-3,
        create_optimizer: bool = False,
        device: torch.device = torch.device("cpu"),
        predict_delta: bool = True,
        normalize_targets: bool = True,
        target_bn_momentum: float = 0.01,
        weighted_loss: bool = False,
        symmetry_breaking_start: bool = False,
        sb_start_duration: int = 100000,
        sb_start_factor: float = 1.5,
        partition_batch: bool = False,
        **kwargs
    ):
        """Initialize the model.

        Args:
            state_space: The state space.
            context_space: The context space (containing static information).
            action_space: The action space.
            partial_net: A function that takes in the state, context and action space and the number
                of modes and returns a randomly initialized neural network. This neural network
                should take in a state and an action and return: (state_mean, state_std),
                termination probability
            n_models: The number of models in the ensemble.
            n_modes: The number of modes to use for the Gaussian mixture model.
            n_particles_per_model: The number of particles per model.
            learning_rate: The learning rate for the supervised learning of the model.
            create_optimizer: Whether to create an optimizer for the model.
            device: The device to use.
            predict_delta: Whether to internally predict the change in state instead of the
                next state.
            normalize_targets: Whether to normalize the targets of the model during
                learning. If predict_delta is True, the delta is normalized.
            target_bn_momentum: The momentum for the batch normalization of the targets.
            weighted_loss: Whether to weight the contribution of each transition to the loss.
                Requires the batch to have a 'weight' key.
            symmetry_breaking_start: Whether to start with a fixed but random model for specified
                number of calls to the learn method. This is useful to break the symmetry between
                skills when using the model with a skill learning method like SPlaTES or DADS.
            sb_start_duration: The number of total env steps during which to use the fixed
                but random model.
            sb_start_factor: The factor by which to multiply the deltas predicted by the random
                model during the symmetry breaking start.
            partition_batch: Whether to partition the batch such that each network is trained on a
                different batch.
            **kwargs: Additional keyword arguments for Model class.
        """
        super().__init__(n_models, n_particles_per_model, device, **kwargs)
        self.n_modes = n_modes
        self.state_space = state_space
        self.context_space = context_space
        self.action_space = action_space
        self.net_factory = partial_net
        self.device = device
        self.gaussian_loss = torch.nn.GaussianNLLLoss(reduction="none")
        self.predict_delta = predict_delta
        self.normalize_targets = normalize_targets
        self.target_bn_momentum = target_bn_momentum
        self.weighted_loss = weighted_loss
        self.symmetry_breaking_start = symmetry_breaking_start
        self.sb_start_duration = sb_start_duration
        self.sb_start_factor = sb_start_factor
        self.partition_batch = partition_batch

        self._log_n_models = torch.log(
            torch.tensor(n_models, dtype=torch.float32, device=device)
        )

        # Initialize the ensemble of networks.
        self.nets = torch.nn.ModuleList(
            [
                self.net_factory(state_space, context_space, action_space, n_modes).to(
                    self.device
                )
                for _ in range(self.n_models)
            ]
        )

        self.register_buffer("_learn_counter", torch.tensor(0))
        if self.symmetry_breaking_start:
            self.random_dynamics = RandomDynamics(
                self.state_space,
                self.action_space,
                device=self.device,
                std=0.3,
                n_modes=self.n_modes,
            )

        if self.normalize_targets:
            self.output_bn = RunningBatchNorm(
                self.state_space.shape[0],
                momentum=target_bn_momentum,
                device=self.device,
                track_mean=False,
            )
        else:
            self.output_bn = None

        if create_optimizer:
            self.optimizer = torch.optim.Adam(self.get_parameters(), lr=learning_rate)
        else:
            self.optimizer = None

    def _partition_batch(self, batch: Batch):
        """Partition the batch by adding a model and particle dimension if partition_batch is True.

        Else simply expand the batch along the model dimension.

        Args:
            batch: The batch to partition."""

        batch_size = batch.state.shape[0]
        new_first_dim = batch_size // self.n_models
        new_dict = {}
        for k, v in batch.items():
            if (
                v is not None
                and isinstance(v, torch.Tensor)
                and v.shape[0] == batch_size
            ):
                if self.partition_batch:
                    new_dict[k] = v.view((new_first_dim, self.n_models) + v.shape[1:])
                else:
                    new_dict[k] = v.unsqueeze(1).expand(
                        (batch_size, self.n_models) + v.shape[1:]
                    )
        return Batch(**new_dict)

    def get_parameters(self) -> Generator[torch.Tensor, None, None]:
        """Get the parameters of the model.

        Returns:
            An iterator over the parameters of the model.
        """
        for net in self.nets:
            for param in net.parameters():
                yield param

    def _run_net(
        self,
        net: Module,
        state: torch.Tensor,
        context: torch.Tensor,
        action: torch.Tensor,
        loss_mode: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Run the given network (one member of the ensemble).

        Args:
            net: The network.
            state: The state.
            context: The context, i.e., information that does not change over timesteps.
            action: The action.
        Returns:
            A tuple containing:
                - The mean and standard deviation of the predicted state or delta.
                - The weights of the modes of the mixture of Gaussians.
                - The termination probability."""
        if (
            self.symmetry_breaking_start
            and self.n_total_env_steps < self.sb_start_duration
            and not loss_mode
        ):
            # Use the symmetry breaking model during the symmetry breaking start.
            # After half of the duration, the symmetry breaking model is
            # linearly interpolated with the learned model.
            sb_logits, sb_weights, sb_term_prob = self.random_dynamics(state, action)
            sb_logits = (self.sb_start_factor * sb_logits[0], sb_logits[1])
            logits, weights, term_prob = net(state, context, action)
            c = max(0, -0.5 * self.sb_start_duration + self.n_total_env_steps) / (
                0.5 * self.sb_start_duration
            )
            logits = (
                (1 - c) * sb_logits[0] + c * logits[0],
                (1 - c) * sb_logits[1] + c * logits[1],
            )
            weights = (1 - c) * sb_weights + c * weights
            term_prob = (1 - c) * sb_term_prob + c * term_prob
        else:
            logits, weights, term_prob = net(state, context, action)

        return logits, weights, term_prob

    def _get_member_pred(
        self,
        id: int,
        state: torch.Tensor,
        context: torch.Tensor,
        action: torch.Tensor,
        std: Optional[torch.Tensor] = None,
        loss_mode: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Get the prediction of a member of the ensemble.

        Args:
            id: The id of the member.
            state: The state.
            context: The context, i.e., information that does not change over timesteps.
            action: The action.
            std: Overwrites the standard deviation that the model predicts if provided.
            loss_mode: Whether to always use the true model instead of the symmetry breaking model.
        Returns:
            A tuple containing:
                - The mean and standard deviation of the predicted state or delta (of all modes).
                - The weights of the modes.
                - The termination probability."""
        logits, weights, term_prob = self._run_net(
            self.nets[id], state, context, action, loss_mode
        )
        if self.normalize_targets:
            # multiply with running variance
            logits = tuple(
                logit * torch.sqrt(self.output_bn.var + self.output_bn._eps)
                for logit in logits
            )
        mean = logits[0]
        if std is None:
            scale = logits[1]
        else:
            scale = std.expand_as(logits[1])
        if self.predict_delta:
            mean += state[..., None, :]
        logits = (mean, scale)

        return logits, weights, term_prob

    def loss(self, batch: Batch) -> torch.Tensor:
        """Compute the loss for the given batch.

        Args:
            batch: The batch. The first dimension corresponds to the batch dimension (e.g. environments).
            For example, batch.state.shape = (batch_size, per_env_size, state_dim)
        Returns:
            The loss.
        """
        self._learn_counter += 1

        # keep track of variance of state (deltas)
        if self.predict_delta:
            x = batch.state_next - batch.state
        else:
            x = batch.state
        if self.normalize_targets:
            x_view = x.view(-1, x.shape[-1])
            # keep track of variance of state (deltas)
            self.output_bn(x_view).view(x.shape)

        # partition the batch such that each network is trained on a different batch
        # (if self.partition_batch is True, otherwise just add a model dimension)
        batch = self._partition_batch(batch)

        loss = 0
        for i, net in enumerate(self.nets):
            state_log_prob, term_prob, _ = self.get_member_log_prob(
                i,
                batch.state[:, i, ...],
                batch.context[:, i, ...],
                batch.act[:, i, ...],
                batch.state_next[:, i, ...],
                # To use learned model in loss and not symmetry breaking model
                loss_mode=True,
            )
            loss += -state_log_prob
            loss += torch.nn.functional.binary_cross_entropy(
                term_prob, batch.terminated[:, i, ...].float(), reduction="none"
            )
            if self.weighted_loss:
                loss *= batch.weight.squeeze(-1)
            if "validity_mask" in batch:
                loss *= batch.validity_mask[:, i, ...]
        return loss.mean()

    def learn(self, batch_lst: List[Batch]) -> None:
        """Learn from the given batch.

        Args:
            batch_lst: A list of training batches. The first dimension corresponds to the transitions.
        Returns:
            The loss after the updates.
        """
        for batch in batch_lst:
            loss = self.loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss

    def rollout(
        self,
        initial_state: torch.Tensor,
        context: torch.Tensor,
        actions: torch.Tensor,
        deterministic: bool = False,
        loss_mode: bool = False,
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
            loss_mode: Whether to always use the true model instead of the symmetry breaking model.
        Returns:
            Batch containing the resulting states, termination probabilities and aleatoric
            and epistemic uncertanties.
            state shape: (batch_size, n_models, n_particles_per_model, horizon + 1, state_dim)
            state_mean shape: (batch_size, n_models, horizon + 1, state_dim)
            term_prob shape: (batch_size, n_models, n_particles_per_model, horizon)
        """
        H = actions.shape[1]  # rollout horizon
        batch_size = initial_state.shape[0]
        s = torch.zeros(
            (batch_size, self.n_models, self.n_particles_per_model, H + 1)
            + self.state_space.shape,
            device=self.device,
        )
        s[..., 0, :] = initial_state[:, None, None, :]
        # expand actions along the particle dimension
        actions = actions[:, None, ...].expand(
            (batch_size, self.n_particles_per_model) + actions.shape[1:]
        )
        term_prob = torch.zeros(
            (batch_size, self.n_models, self.n_particles_per_model, H),
            device=self.device,
        )
        expanded_context = context[:, None, :].expand(
            (-1, self.n_particles_per_model, -1)
        )
        for k in range(H):
            for i in range(self.n_models):
                logits, weights, term_prob[:, i, :, k] = self._get_member_pred(
                    i,
                    s[:, i, :, k, ...],
                    expanded_context,
                    actions[..., k, :],
                    loss_mode=loss_mode,
                )
                # clamp to avoid exploding values
                logits = (logits[0].clamp(-10e6, 10e6), logits[1].clamp(-10e6, 10e6))
                if deterministic:
                    s[:, i, :, k + 1, ...] = (logits[0] * weights[..., None]).sum(
                        dim=-2
                    )
                else:
                    # sample a mode according to the weights
                    mode = torch.multinomial(weights.view(-1, self.n_modes), 1)
                    mode = mode.view(weights.shape[:-1])
                    modes_expanded = mode[..., None, None].expand_as(logits[0])
                    mean = logits[0].gather(-2, modes_expanded)[..., 0, :]
                    std = logits[1].gather(-2, modes_expanded)[..., 0, :]
                    s[:, i, :, k + 1, ...] = mean + std * torch.randn(
                        size=mean.shape, device=logits[0].device
                    )

        # compute epistemic and aleatoric uncertainty
        s_mean = s.mean(dim=2)
        s_var = s.var(dim=2, correction=0)
        s_epistemic = s_mean.var(dim=1, correction=0)
        s_aleatoric = s_var.mean(dim=1)

        batch = Batch(
            state=s,
            state_mean=s_mean,
            state_var=s_var,
            state_epistemic_var=s_epistemic,
            state_aleatoric_var=s_aleatoric,
            term_prob=term_prob,
        )
        return batch

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
        batch_size = state.shape[0]
        # third dimension is for current and predicted state
        s = torch.zeros(
            (batch_size, self.n_models, 2, self.n_modes) + self.state_space.shape,
            device=self.device,
        )
        s_std = torch.zeros(
            (batch_size, self.n_models, self.n_modes) + self.state_space.shape,
            device=self.device,
        )
        weights = torch.zeros(
            (batch_size, self.n_models, self.n_modes), device=self.device
        )
        term_prob = torch.zeros((batch_size, self.n_models), device=self.device)
        for i in range(self.n_models):
            logits, ws, term_prob[:, i] = self._get_member_pred(
                i, state, context, action, std=std
            )
            if logits[1].isnan().any():
                torch.nan_to_num(logits[1], out=logits[1], nan=1e-4)
            s[:, i, 1, ...] = logits[0]
            s_std[:, i, ...] = logits[1]
            weights[:, i, :] = ws

        return s[:, :, 1], weights, s_std, term_prob.mean(dim=1)

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
        raise NotImplementedError("This method is not implemented yet.")

    def get_member_log_prob(
        self,
        id: int,
        state: torch.Tensor,
        context: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        std: Optional[torch.Tensor] = None,
        loss_mode: bool = False,
        transform: Optional[Module] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Get probability (density) of next state and the termination probability for ensemble member.

        If batch normalization is used, it matters whether the Module is in eval or train mode. Note that
        everything is assumed to have a 'batch' dimension, useful for parallelization.

        Args:
            state: The current state.
            context: The context, i.e., information that is constant over timesteps.
            action: The action.
            next_state: The next state.
            std: Overwrites the standard deviation that the model predicts if provided.
            loss_mode: Whether to always use the true model instead of the symmetry breaking model.
            transform: A transformation to apply to the state before computing the log probability. Note that this
                only makes sense with a given standard deviation, not with the learned one.

        Returns:
            A tuple containing:
                - The probability (density) of the next state given the current state and action under the model.
                - The termination probability given the current state and action under the model.
                - The mean of the predicted distribution.
        """
        if self.predict_delta:
            x = next_state - state
        else:
            x = next_state

        logits, weights, term_prob = self._run_net(
            self.nets[id], state, context, action, loss_mode=loss_mode
        )
        if self.normalize_targets:
            # multiply with running variance
            logits = tuple(
                logit * torch.sqrt(self.output_bn.var + self.output_bn._eps)
                for logit in logits
            )
        mean = logits[0]
        if std is None:
            scale = logits[1]
        else:
            scale = std.expand_as(logits[1])
        if transform is not None:
            x = transform(x)
            mean = transform(mean)
        log_prob = get_normal_log_prob(
            x=x[..., None, :],
            mean=mean,
            std=scale,
        ).sum(dim=-1)
        # max log prob for (weighted) logsumexp trick
        max_log_prob = log_prob.max(dim=-1).values
        # sum over weighted probabilities of the modes
        log_prob = (weights * (log_prob - max_log_prob[..., None]).double().exp()).sum(
            dim=-1
        ).log() + max_log_prob
        mean = (weights[..., None] * mean).sum(dim=-2)
        return log_prob, term_prob, mean

    def get_log_prob(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        std: Optional[torch.Tensor] = None,
        loss_mode: bool = False,
        transform: Optional[Module] = None,
    ) -> Tuple:
        """Get log probability (density) of next state and the termination probability for full ensemble.

        Note that everything is assumed to have a 'batch' dimension, useful for parallelization.

        Args:
            state: The current state.
            context: The context, i.e., information that is constant over timesteps.
            action: The action.
            next_state: The next state.
            std: Overwrites the standard deviation that the model predicts if provided.
            loss_mode: Whether to always use the true model instead of the symmetry breaking model.
            transform: A transformation to apply to the state before computing the log probability. Note that this
                only makes sense with a given standard deviation, not with the learned one.
        Returns:
            A tuple containing:
                - The log probability (density) of the next state given the current state and action under the model.
                - The termination probability given the current state and action under the model.
                - An info dict with additional information.
        """
        total_term_prob = torch.zeros(state.shape[0], device=self.device)
        total_log_prob = torch.zeros(
            state.shape[0], device=self.device, dtype=torch.float64
        )
        means = torch.zeros(
            state.shape[:-1] + (self.n_models, state.shape[-1]), device=self.device
        )
        log_probs = torch.zeros(state.shape[:-1] + (self.n_models,), device=self.device)
        for i in range(self.n_models):
            log_prob, term_prob, mean = self.get_member_log_prob(
                i,
                state,
                context,
                action,
                next_state,
                std,
                loss_mode=loss_mode,
                transform=transform,
            )
            total_term_prob += term_prob
            means[..., i, :] = mean
            log_probs[..., i] = log_prob
        # logsumexp for numerical stability
        total_log_prob = torch.logsumexp(log_probs, dim=-1) - self._log_n_models
        return (
            total_log_prob,
            total_term_prob / self.n_models,
            {"means": means},
        )
