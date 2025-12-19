from typing import Callable, Optional

from tianshou.data import Batch
import torch

from .planner import Planner
from ..optimizers import Optimizer


class BlackboxPlanner(Planner):
    """Plans by optimizing a sequence of actions using a given optimizer."""

    def __init__(
        self,
        optimizer: Optimizer,
        rollout_callback: Callable[[Batch], None] = None,
        deterministic: bool = False,
        use_value_func: bool = False,
        use_term_prob: bool = True,
        use_ensemble_disagreement: bool = False,
        **kwargs
    ):
        """Initialize the planner.

        Note that the class only supports predictors with value functions that do
        not depend on the action (no Q-functions). The reason for this is that the
        value function is used to calculate the value of the final state of the
        trajectory and the action is not known at that point.

        The planner does not use the policy.

        Args:
            optimizer: An instance of Optimizer that does the optimization of
                the action sequence.
            rollout_callback: A function that is called after rolling out a new
                set of trajectories. Expects a batch with the trajectory data, the
                context, the const and the actions as input.
            deterministic: Whether to use the deterministic version of the model.
            use_value_func: Whether to use the value function to calculate the terminal cost.
            use_term_prob: Whether to use the termination probability to calculate the cost.
            use_ensemble_disagreement: Whether to use the ensemble disagreement instead of the task reward.
            **kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__(**kwargs)
        self.optimizer = optimizer
        self.rollout_callback = rollout_callback
        self.deterministic = deterministic
        self.use_value_func = use_value_func
        self.use_term_prob = use_term_prob
        self.use_ensemble_disagreement = use_ensemble_disagreement
        self.initial_guess = torch.zeros(
            (self.n_env_instances, self.horizon, self.action_space.shape[0]),
            device=self.device,
        )

    def _reintroduce_sample_dimension(
        self, trajectory: Batch, batch_size: int, n_samples: int
    ):
        """Reintroduce the sample dimension to the trajectory.

        This is the inverse of collapsing the batch and sample dimensions into one dimension.
        Args:
            trajectory: The trajectory to reintroduce the sample dimension to.
            batch_size: The batch size of the trajectory.
            n_samples: The number of samples to reintroduce.
        """
        for k, v in trajectory.items():
            if v is not None:
                trajectory[k] = v.view((batch_size, n_samples) + v.shape[1:])

    def cost(
        self, initial_obs: torch.Tensor, x: torch.Tensor, horizon: int, batch_size: int
    ) -> torch.Tensor:
        """Compute the cost of the given actions.

        Makes use of the predictor (containing the dynamics model, reward and value functions).

        Args:
            initial_obs: The initial observation of the environment(s) and samples.
                Shape: (batch_size * n_samples, state_dim)
                The batch dimension is collapsed with the sample dimension by using repeat_interleave.
            x: The actions to compute the cost of but in shape for blackbox optimizer:
                (batch_size, n_samples, horizon * action_dim)
            horizon: The horizon of the plan.
            batch_size: The batch size of the trajectory.

        Returns:
            The cost of the actions (for each environment instance). Shape: (batch_size,)
        """
        # Split up x into actions and get rid of sample dimension
        actions = x.view((-1, horizon, self.action_space.shape[0]))
        # Perform rollout
        trajectory, context = self.predictor.rollout(
            initial_obs, actions, self.deterministic
        )
        # Reintroduce the sample dimension
        self._reintroduce_sample_dimension(trajectory, batch_size, x.shape[1])
        context = context.view((batch_size, x.shape[1]) + context.shape[1:])
        # sum over time and take termination probability into account
        if self.use_term_prob:
            cont_prob = 1.0 - trajectory.term_prob
        else:
            cont_prob = torch.ones_like(trajectory.term_prob)
        if self.use_ensemble_disagreement:
            rewards = trajectory.state_epistemic_var[..., 1:, :].mean(dim=-1)
            rewards = rewards[..., None, None, :].expand_as(trajectory.rew)
        else:
            rewards = trajectory.rew + self.get_aux_rewards(trajectory)
        alive_prob = torch.cumprod(cont_prob, dim=-1)
        cumulative_reward = rewards[..., 0] + (
            rewards[..., 1:] * alive_prob[..., :-1]
        ).sum(dim=-1)
        # Average over models and particles
        cumulative_reward = cumulative_reward.mean(dim=[2, 3])
        cost = -cumulative_reward
        # Calculate the value for the final states of all particles and then average
        if self.predictor.val_func is None:
            final_state_value = 0.0
        else:
            # NOTE: The actions are only passed to the value function to comply with the interface of the predictor.
            # The value function should not depend on the action as it does not correspond to the last state.
            state_shape = trajectory.state.shape
            expanded_context = context[:, :, None, None].expand(
                -1, -1, state_shape[2], state_shape[3], -1
            )
            final_state_value = self.predictor.val_func(
                trajectory.state[..., -1, :], expanded_context, actions[:, -1, :]
            ).mean(dim=[2, 3])
        if self.use_value_func:
            cost -= alive_prob[..., -1].mean(dim=[-2, -1]) * final_state_value
        if self.rollout_callback is not None:
            self.rollout_callback(
                trajectory,
                context,
                cost,
                x.view((batch_size, -1, horizon, self.action_space.shape[0])),
            )
        return cost

    def plan(
        self,
        initial_obs: torch.Tensor,
        active_instances: torch.Tensor,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Plan a trajectory from the given observation and return it.

        Note that observation has a batch dimension (for multiple environment instances).

        Args:
            initial_obs: The initial observation of the environment(s).
            active_instances: A boolean tensor indicating which instances are active.
            verbose: Whether to print out the cost during optimization.
        Returns:
            The actions corresponding to the planned trajectory (a sequence of actions
            for each environment instance), and an info dictionary with additional
            information about the optimization.
        """
        batch_size = initial_obs.shape[0]
        # Repeat initial observation to accomodate for number of samples
        initial_obs = initial_obs.repeat_interleave(
            repeats=self.optimizer.n_samples, dim=0
        )
        self.optimizer.reset(
            self.initial_guess[active_instances].reshape(
                (active_instances.sum().item(), -1)
            )
        )
        x_opt, x_info = self.optimizer.optimize(
            lambda x: self.cost(initial_obs, x, self.horizon, batch_size),
            verbose=verbose,
        )
        for k, v in x_info.items():
            if isinstance(v, torch.Tensor):
                if v.shape[0] == batch_size:
                    x_info[k] = v.view(batch_size, self.horizon, -1)
        return x_opt.view(x_opt.shape[0], self.horizon, -1), x_info

    def reset(
        self,
        initial_guess: torch.Tensor,
        reset_instances: Optional[torch.Tensor] = None,
    ) -> None:
        """Reset the planner.

        When doing MPC, this should be called at the beginning of each episode
        to reset the planner's internal state.

        Args:
            initial_guess: The initial guess for the optimal actions. Shape: (batch_size, horizon, action_dim)
            reset_instances: A boolean tensor indicating which instances to reset.
        """
        if reset_instances is None:
            self.initial_guess = initial_guess
        else:
            self.initial_guess[reset_instances] = initial_guess
