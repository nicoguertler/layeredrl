from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from gymnasium.spaces import Box
from tianshou.data import Batch
import torch

from ..predictors import Predictor


class Planner(ABC):
    def __init__(
        self,
        predictor: Predictor,
        action_space: Box,
        n_env_instances: int,
        horizon: int,
        policy: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        aux_rewards: Optional[List[Callable[[Batch], torch.Tensor]]] = None,
        aux_reward_weights: Optional[List[float]] = None,
        device=torch.device("cpu"),
    ):
        """Initialize the planner.

        Args:
            predictor: The predictor to use for planning. This includes the dynamics and reward model
                and the value function.
            action_space: The action space.
            n_env_instances: The number of environment instances. Relevant for vectorization.
            horizon: The horizon of the plan.
            policy: A function that takes in a state and returns an action.
            aux_rewards: A list of functions that take in a batch and return a tensor with
                cumulative auxiliary rewards for each environment instance. These auxiliary rewards
                can also make use of the epistemic and aleatoric uncertainties of trajectories.
            aux_reward_weights: A list of weights for the auxiliary rewards.
            device: The device to use.
        """

        self.predictor = predictor
        self.action_space = action_space
        self.n_env_instances = n_env_instances
        self.horizon = horizon
        self.policy = policy
        self.aux_rewards = [] if aux_rewards is None else aux_rewards
        self.aux_reward_weights = (
            [] if aux_reward_weights is None else aux_reward_weights
        )
        self.device = device

    def set_predictor(self, predictor: Predictor):
        """Set the predictor for the planner.

        Args:
            predictor: The new predictor to use for planning.
        """
        self.predictor = predictor

    def get_aux_rewards(self, trajectory: Batch) -> torch.Tensor:
        """Get the auxiliary rewards for the given trajectory.

        Args:
            trajectory: The trajectory to get the auxiliary rewards for.
        Returns:
            The auxiliary rewards for each environment instance.
        """
        aux_rewards = 0
        for aux_reward, weight in zip(self.aux_rewards, self.aux_reward_weights):
            aux_rewards += weight * aux_reward(trajectory)
        return aux_rewards

    @abstractmethod
    def plan(self, initial_obs: torch.Tensor) -> torch.Tensor:
        """Plan a trajectory from the given observation and return it.

        Note that observation has a batch dimension (for multiple environment instances).

        Args:
            initial_obs: The initial observation of the environment(s).
        Returns:
            The actions corresponding to the planned trajectory (a sequence of actions
            for each environment instance), and an info dictionary with additional
            information about the optimization.
        """
        pass

    @abstractmethod
    def shift_initialization(
        self,
        n_shift_steps: int,
        initial_guess: torch.Tensor,
        active_instances: torch.Tensor,
    ):
        """Shift the initial action sequence by n_shift_steps and pad with initial_guess.

        Args:
            n_shift_steps: The number of steps to shift the initial action sequence by.
            initial_guess: The initial guess for the last n_shift_steps of the new action sequence.
                Shape: (n_envs, n_shift_steps, action_dim)
            active_instances: A boolean tensor indicating which instances to shift.
        """
        pass

    @abstractmethod
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
        pass
