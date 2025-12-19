from typing import Union

import torch

from .blackbox_planner import BlackboxPlanner
from ..optimizers import CEM, ICEM
from ..utils.misc import to_torch


class CEMPlanner(BlackboxPlanner):
    def __init__(
        self,
        initial_sigma: Union[torch.Tensor, float],
        use_icem: bool = True,
        cem_params: dict = None,
        device: torch.device = torch.device("cpu"),
        *args,
        **kwargs
    ):
        """Initialize the planner.

        Args:
            initial_sigma: The initial standard deviation of the samples. Shape: (action_dim, )
            use_icem: Whether to use ICEM instead of CEM.
            cem_params: Parameters for the CEM optimizer.
            device: The device to use.
            *args: Arguments for the BlackboxPlanner.
            **kwargs: Keyword arguments for the BlackboxPlanner and optimizer."""
        self.initial_sigma = initial_sigma
        if not isinstance(initial_sigma, torch.Tensor):
            assert isinstance(
                initial_sigma, float
            ), "initial_sigma must be a float or a torch.Tensor"
            # add batch dimension
            self.initial_sigma = initial_sigma * torch.ones(
                (1, kwargs["horizon"] * kwargs["action_space"].shape[0]), device=device
            )
        else:
            self.initial_sigma.to(device)
        lower_bound = kwargs.pop("lower_bound", None)
        upper_bound = kwargs.pop("upper_bound", None)
        optimizer_class = ICEM if use_icem else CEM
        if cem_params is None:
            cem_params = {}
        if use_icem:
            cem_params["action_dim"] = kwargs["action_space"].shape[0]
        optimizer = optimizer_class(
            # placeholders until horizon is known
            initial_sigma=self.initial_sigma,
            lower_bound=None,
            upper_bound=None,
            device=device,
            **cem_params,
        )
        super().__init__(optimizer=optimizer, device=device, **kwargs)
        if lower_bound is not None:
            self.lower_bound = torch.tensor(lower_bound, device=self.device)
        else:
            self.lower_bound = to_torch(self.action_space.low, device=self.device)
        if upper_bound is not None:
            self.upper_bound = torch.tensor(upper_bound, device=self.device)
        else:
            self.upper_bound = to_torch(self.action_space.high, device=self.device)
        optimizer.truncated = lower_bound is not None and upper_bound is not None

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
        # set horizon-dependent parameters of optimizer
        self.optimizer.initial_sigma = self.initial_sigma.clone().detach()
        if self.lower_bound is not None:
            self.optimizer.lower_bound = self.lower_bound.repeat(1, self.horizon).to(
                self.device
            )
        if self.upper_bound is not None:
            self.optimizer.upper_bound = self.upper_bound.repeat(1, self.horizon).to(
                self.device
            )
        self.optimizer.truncated = (
            self.lower_bound is not None and self.upper_bound is not None
        )
        actions, info = super().plan(initial_obs, active_instances, verbose)
        # update initial guess to include the planned actions
        self.initial_guess[active_instances] = self.optimizer.mu.view(
            self.optimizer.mu.shape[0], -1, self.action_space.shape[0]
        )

        return actions, info

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
        self.initial_guess[active_instances] = torch.cat(
            [self.initial_guess[active_instances, n_shift_steps:], initial_guess], dim=1
        )
