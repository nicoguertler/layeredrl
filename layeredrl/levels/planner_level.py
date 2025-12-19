from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from gymnasium.spaces import Space
from tianshou.data import Batch
import torch

from .level import Level
from ..predictors import Predictor
from ..planners import Planner
from ..utils.buffers import ToDeviceReplayBuffer
from ..utils.misc import temp_eval_mode


class PlannerLevel(Level):
    """Level that does MPC with a provided predictor (for dynamics, reward, and value) and planner."""

    def __init__(
        self,
        partial_planner: Callable[..., Planner],
        predictor_factory: Callable[..., Predictor],
        initial_guess: torch.Tensor,
        horizon: int = 10,
        shift_initialization: bool = True,
        verbose: bool = False,
        alternate_with_noise: bool = False,
        switch_random_prob: float = 0.1,
        switch_planner_prob: float = 0.1,
        only_noise: bool = False,
        resample_on_end: bool = False,
        resample_random_action_prob: float = 1.0,
        buffer_size: int = 10000,
        batch_size: int = 256,
        model_batch_size: int = 256,
        warm_up_steps: int = 1000,
        no_planning_steps: int = 100000,
        update_interval: int = 1,
        n_updates: int = 1,
        param_reset_freeze: int = 2000,
        log_interval: int = 1000,
        use_ensemble_disagreement: bool = False,
        **kwargs,
    ):
        """Initialize the level.

        Args:
            partial_planner: A partial planner that expects predictor, action space etc. The Planner is
                responsible for the optimization of the action sequence.
            predictor_factory: A factory that creates a new instance of Predictor. The Predictor is
                responsible for the dynamics model, reward and value functions (and potentially
                for mapping the observation to a latent space).
            initial_guess: The initial guess for the optimal action. Shape: (action_dim, )
            horizon: The horizon for planning.
            shift_initialization: Whether to shift the initialization of the action sequence
                by the number of steps that have been executed in the environment.
            verbose: Whether to print additional information during planning.
            alternate_with_noise: Whether to alternate between planning and taking random actions.
            switch_random_prob: The probability of switching from planning to random actions (per
                call to get_action).
            switch_planner_prob: The probability of switching from planning to random actions (per
                call to get_action).
            only_noise: Whether to use only noise instead of planning (and noise).
            resample_on_end: Whether to resample the action only after the end of an episode.
            resample_random_action_prob: The probability of resampling the action when in random mode.
                By default, the action is resampled every time.
            buffer_size: The size of the replay buffer being filled with the transitions of this level.
            batch_size: The batch size for learning the predictor (reward and value).
            model_batch_size: The batch size for learning the dynamics model.
            warm_up_steps: The number of steps without learning (only filling the replay buffer).
            no_planning_steps: The number of steps without planning (only random actions). When the lower
                level has not learned anything yet, planning would be wasteful.
            update_interval: The interval at which the predictor is updated (in terms of new transitions
                on this level and not in the environment).
            n_updates: The number of updates per interval.
            param_reset_freeze: The number of steps after a parameter reset during which no learning
                is performed. This gives the lower level(s) some time to recover from the parameter
                reset before the predictor is updated again.
            log_interval: The interval in which to log statistics.
            use_ensemble_disagreement: Whether to use ensemble disagreement as reward when planning.
        """
        super().__init__(**kwargs)
        self.partial_planner = partial_planner
        self.predictor_factory = predictor_factory
        self.initial_guess = initial_guess.to(self.device)
        self.horizon = horizon
        self.shift_initialization = shift_initialization
        self.verbose = verbose
        self.alternate_with_noise = alternate_with_noise
        self.switch_random_prob = switch_random_prob
        self.switch_planner_prob = switch_planner_prob
        self.only_noise = only_noise
        self.resample_on_end = resample_on_end
        self.resample_random_action_prob = resample_random_action_prob
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.model_batch_size = model_batch_size
        self.warm_up_steps = warm_up_steps
        self.no_planning_steps = no_planning_steps
        self.update_interval = update_interval
        self.predictor_updates = n_updates
        self.param_reset_freeze = param_reset_freeze
        self.log_interval = log_interval
        self.use_ensemble_disagreement = use_ensemble_disagreement
        self._warm_up = True
        self._random_mode = False
        self._last_log_step = 0
        self._rew_error_sum = 0.0
        self._last_param_reset = None
        self._done = None

    def initialize(
        self,
        env_obs_space: Space,
        action_space: Space,
        n_env_instances: int,
        parent_predictor: Predictor,
        env_obs_map: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        mapped_env_obs_shape: Optional[Tuple] = None,
        keep_params: bool = False,
    ) -> None:
        """Construct the level."""
        super().initialize(
            env_obs_space,
            action_space,
            n_env_instances,
            parent_predictor,
            env_obs_map,
            mapped_env_obs_shape,
            keep_params,
        )
        self.action_space = action_space
        if not keep_params:
            # initialize the networks in the predictor
            self.predictor = self.predictor_factory(
                self.mapped_env_obs_shape,
                action_space,
                device=self.device,
                writer=self.writer,
            )
        self.planner = self.partial_planner(
            predictor=self.predictor,
            action_space=action_space,
            n_env_instances=n_env_instances,
            horizon=self.horizon,
            device=self.device,
        )
        # buffer for transitions
        self.buffer = ToDeviceReplayBuffer(
            total_size=self.buffer_size,
            buffer_num=self.n_env_instances,
            target_device=self.device,
        )
        self._done = torch.zeros(
            self.n_env_instances, dtype=torch.bool, device=self.device
        )
        self._last_random_action = torch.zeros(
            self.n_env_instances, self.action_space.shape[0], device=self.device
        )
        self._random_action_prob = 1.0 / (2 ** self.action_space.shape[0])

    def set_n_env_instances(self, n_env_instances: int) -> None:
        """Set the number of environment instances.

        Args:
            n_env_instances: The number of environment instances.
        """
        super().set_n_env_instances(n_env_instances)
        self._done = torch.zeros(
            self.n_env_instances, dtype=torch.bool, device=self.device
        )
        self._last_random_action = torch.zeros(
            self.n_env_instances, self.action_space.shape[0], device=self.device
        )

    def get_input_space(self) -> Space:
        """Get the input space of level.

        Input space denotes everything the level needs except for the
        environment observation, e.g. a skill vector.

        Returns:
            The input space (or None if no input is required).
        """
        return None

    def reset(self):
        """Reset the level.

        Call once before the first episode, not between episodes."""
        super().reset()
        initial_guess = self.initial_guess.expand(
            self.n_env_instances, self.horizon, self.action_space.shape[0]
        )
        self.planner.reset(initial_guess=initial_guess.detach().clone())
        self._random_mode = False
        self._last_log_step = self.n_total_steps
        self._rew_error_sum = 0.0
        self._last_param_reset = None
        self._done = torch.zeros(
            self.n_env_instances, dtype=torch.bool, device=self.device
        )
        self._last_random_action = torch.zeros(
            self.n_env_instances, self.action_space.shape[0], device=self.device
        )

    def soft_reset(self):
        """Soft reset the hierarchy.

        This is called when manually resetting the (vector) environment. This will not
        affect things like warm up periods etc."""
        super().soft_reset()
        initial_guess = self.initial_guess.expand(
            self.n_env_instances, self.horizon, self.action_space.shape[0]
        )
        self.planner.reset(initial_guess=initial_guess.detach().clone())
        self._random_mode = False
        self._last_log_step = self.n_total_steps
        self._rew_error_sum = 0.0

    def get_copy(self) -> "Level":
        """Get a copy of the level.

        No parameters are copied, only the state of the level.

        Returns:
            A copy of the level.
        """
        level = super().get_copy()
        return level

    def get_action(
        self,
        mapped_env_obs: torch.Tensor,
        level_input: Optional[torch.Tensor],
        level_input_info: Optional[Dict],
        active_instances: torch.Tensor = torch.tensor([True], dtype=torch.bool),
    ) -> torch.Tensor:
        """Get an action for the given observation.

        Note that only the action for the active instances is returned.

        Call this at the beginning of the implementation of get_action in derived classes.

        Args:
            mapped_env_obs: The environment observation after the self.env_obs_map has been
                applied. Note that the observation has a batch dimension (for multiple
                environment instances).
            level_input: The input to this level for the active instances, i.e., the
                action from the level above.
            level_input_info: Additional information about the level input.
            active_instances: In which of the environment instances the level is active.
                env_obs and level_input correspond to these instances.
        Returns:
            The action (also with a batch dimension).
            An info dict containing additional information about the action.
        """
        super().get_action(mapped_env_obs, level_input, active_instances)
        self.predictor.model.set_n_total_env_steps(self.n_total_env_steps)
        self.predictor.n_total_env_steps = self.n_total_env_steps

        n_active_instances = active_instances.sum().item()

        if not self.training:
            self._random_mode = False
            self.planner.use_ensemble_disagreement = False
        else:
            # handle stochastic switching between planning and random actions
            if self.only_noise or self.n_total_env_steps < self.warm_up_steps:
                self._random_mode = True
            elif self.alternate_with_noise:
                if self._random_mode:
                    if (
                        self.n_total_env_steps > self.no_planning_steps
                        and torch.rand(1) < self.switch_planner_prob
                    ):
                        # switch to planning
                        if self.use_ensemble_disagreement:
                            self.planner.use_ensemble_disagreement = (
                                not self.planner.use_ensemble_disagreement
                            )
                        else:
                            self.planner.use_ensemble_disagreement = False
                        if self.verbose:
                            print("Switch to planning.")
                            print(
                                "Use ensemble disagreement: ",
                                self.planner.use_ensemble_disagreement,
                            )
                        self._random_mode = False
                else:
                    if torch.rand(1) < self.switch_random_prob:
                        # switch to random mode
                        if self.verbose:
                            print("Switch to random mode.")
                        self._random_mode = True
            else:
                self._random_mode = False

        if self._random_mode:
            # use random action
            if self.verbose:
                print("Use random action.")
            if self.resample_on_end:
                if self._done.any():
                    batch_size = self._done.sum().item()
                    self._last_random_action[self._done] = (
                        self.sample_from_action_space(batch_size)
                    )
                action = self._last_random_action[active_instances]
            else:
                if (
                    self.resample_random_action_prob == 1.0
                    or torch.rand(1) < self.resample_random_action_prob
                ):
                    if self.verbose:
                        print("Resample random action")
                    self._last_random_action[active_instances] = (
                        self.sample_from_action_space(n_active_instances)
                    )
                action = self._last_random_action[active_instances]
            mean = torch.full_like(action, torch.nan, device=self.device)
            std = torch.full_like(action, torch.nan, device=self.device)
            return action, {"mean": mean, "std": std}
        else:
            # plan
            if self.verbose:
                print("Plan")
            if self.shift_initialization:
                # shift initialization of action sequence
                self.planner.shift_initialization(
                    n_shift_steps=1,
                    initial_guess=self.initial_guess.expand(
                        n_active_instances,
                        1,
                        self.action_space.shape[0],
                    )
                    .detach()
                    .clone(),
                    active_instances=active_instances,
                )
            actions, action_info = self.planner.plan(
                mapped_env_obs, active_instances, verbose=self.verbose
            )
            return actions[:, 0, :], {k: v[:, 0, :] for k, v in action_info.items()}

    def get_reward(
        self,
        mapped_env_obs: torch.Tensor,
        level_input: Optional[torch.Tensor],
        level_state: Dict,
        action: torch.Tensor,
        next_mapped_env_obs: torch.Tensor,
        next_level_state: Dict,
        terminated: bool,
        cum_env_reward: torch.Tensor,
        elapsed_env_steps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Get the reward for the given batch of transitions.

        Note that everything has a batch dimension. Also note that there is no next_level_input
        because the level input doeos not change while the level is in control.

        Args:
            mapped_env_obs: The mapped environment observation.
            level_input: The input to this level.
            level_state: The state of the level.
            action: The action this level took.
            next_mapped_env_obs: The next mapped environment observation.
            next_level_state: The next state of the level.
            terminated: Whether the episode terminated.
            cum_env_reward: The sum of the environment rewards that were received
                during the transition.
            elapsed_env_steps: The number of environment steps that elapsed during
                the transition.
        Returns:
            The reward and the reward info dict.
        """
        # return scaled environment reward that has accumulated
        return cum_env_reward / elapsed_env_steps, {}

    def process_transition(
        self,
        mapped_env_obs: torch.Tensor,
        level_input: Optional[torch.Tensor],
        action: torch.Tensor,
        next_mapped_env_obs: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        active_instances: torch.Tensor,
    ) -> bool:
        """Process transition and check whether level would like to return
        control to the level above.

        This usually involves adding the transition to the replay buffer and possibly
        preprocessing it.

        Note that everything has a batch dimension.

        Args:
            mapped_env_obs: The mapped environment observation for the active instances.
            level_input: The input to this level for the active instances, i.e., the
                action from the level above.
            action: The action that was taken by the level.
            next_mapped_env_obs: The next mapped environment observation for the active instances.
            terminated: Whether the episode terminated for the active instances.
            truncated: Whether the episode was truncated for the active instances.
            active_instances: In which of the environment instances the level is active.
                next_obs and terminated correspond to these instances.
        Returns:
            Whether the level is done, i.e. whether it hands control back to
            the level above.
        """
        cum_env_reward = self.cum_reward[active_instances]
        elapsed_env_steps = self.elapsed_env_steps[active_instances]
        level_state = self.get_level_state(
            mapped_env_obs, level_input, active_instances
        )

        # Important: This increments the number of steps in control, resets self.cum_reward etc.,
        # so changes state of the level to after transition
        super().process_transition(
            mapped_env_obs,
            level_input,
            action,
            next_mapped_env_obs,
            terminated,
            truncated,
            active_instances,
        )

        # keep track of when an episode ended in case self._resample_on_end is True
        self._done[active_instances] = torch.logical_or(terminated, truncated)

        reset_pause = (
            self._last_param_reset is not None
            and self.n_total_steps < self._last_param_reset + self.param_reset_freeze
        )

        # reset the planner for active instances that terminated or truncated
        # (otherwise shift initialization would still have an effect)
        reset_instances = active_instances.clone()
        term_or_trunc = torch.logical_or(terminated, truncated)
        reset_instances[active_instances] = torch.logical_and(
            term_or_trunc, reset_instances[active_instances]
        )
        self.planner.reset(
            initial_guess=self.initial_guess.expand(
                reset_instances.sum().item(), self.horizon, self.action_space.shape[0]
            )
            .detach()
            .clone(),
            reset_instances=reset_instances,
        )

        if self.training:
            next_level_state = self.get_level_state(
                next_mapped_env_obs, level_input, active_instances
            )
            reward, reward_info = self.get_reward(
                mapped_env_obs,
                level_input,
                level_state,
                action,
                next_mapped_env_obs,
                next_level_state,
                terminated,
                cum_env_reward,
                elapsed_env_steps,
            )
            info = {
                "env_rew": cum_env_reward,
                "random_mode": torch.full_like(
                    cum_env_reward, self._random_mode, dtype=torch.bool
                ),
            }
            transition = Batch(
                obs=mapped_env_obs,
                act=action,
                rew=reward,
                terminated=terminated,
                truncated=truncated,
                obs_next=next_mapped_env_obs,
                info=info,
            )
            transition.to_numpy()
            self._newest_transition = transition
            self._reward_info = reward_info
            if not reset_pause:
                buffer_ids = active_instances.nonzero().squeeze(dim=1).cpu().numpy()
                self._newest_obs_buffer_index = self.buffer.add(
                    batch=transition,
                    buffer_ids=buffer_ids,
                )[0]

            # logging
            with temp_eval_mode(self.predictor), torch.no_grad():
                pred_reward = self.predictor.predict_reward(mapped_env_obs, action)
            self._rew_error_sum += torch.sum((reward - pred_reward) ** 2).item()
            steps_elapsed_since_log = self.n_total_steps - self._last_log_step
            if self.writer is not None and steps_elapsed_since_log > self.log_interval:
                # logging
                self._last_log_step = self.n_total_steps
                # log statistics of new transition
                self.writer.add_scalar(
                    "planner-level/reward-error",
                    self._rew_error_sum / steps_elapsed_since_log,
                    self.n_total_env_steps,
                )
                self._rew_error_sum = 0.0

        # do not return control to level above by default
        return torch.zeros(active_instances.shape, dtype=torch.bool, device=self.device)

    def learn(self) -> None:
        """Learn from the collected (semi-MDP) transitions."""
        super().learn()
        if (
            self._last_param_reset is not None
            and self.n_total_steps < self._last_param_reset + self.param_reset_freeze
        ):
            self._n_new_transitions = 0
            return
        if self.n_total_env_steps > self.warm_up_steps:
            if self._warm_up:
                self._n_new_transitions = 0
                self._warm_up = False
            # learn skill dynamics model
            if self._n_new_transitions >= self.update_interval:
                n_updates = (
                    self._n_new_transitions * self.predictor_updates
                ) // self.update_interval
                loss, loss_info = self.predictor.learn(
                    self.buffer,
                    n_updates,
                    self.batch_size,
                    self.model_batch_size,
                    self.n_total_env_steps,
                )
                if self.writer is not None:
                    self.writer.add_scalar(
                        "planner-level/model-loss", loss.item(), self.n_total_env_steps
                    )
                    for k, v in loss_info.items():
                        self.writer.add_scalar(
                            f"planner-level/model-{k}", v.item(), self.n_total_env_steps
                        )
                self._n_new_transitions = 0

    def eval(self) -> None:
        """Set the level to evaluation mode."""
        super().eval()
        self.predictor.eval()

    def train(self) -> None:
        """Set the level to training mode."""
        super().train()
        self.predictor.train()

    def save(self, path: Path) -> None:
        """Save the level to the given path."""
        super().save(path)
        torch.save(self.predictor.state_dict(), path / "predictor.pth")

    def load(self, path: Path) -> None:
        """Load the level from the given path."""
        super().load(path)
        self.predictor.load_state_dict(torch.load(path / "predictor.pth"))

    def save_buffers(self, path: Path) -> None:
        """Save the replay buffer to the given path."""
        super().save_buffers(path)
        self.buffer.save_hdf5(path / "planner_replay_buffer.hdf5")
        print(f"Saved buffer of length {len(self.buffer)} in planner level.")

    def load_buffers(self, path: Path) -> None:
        """Load the replay buffer from the given path."""
        super().load_buffers(path)
        self.buffer = self.buffer.load_hdf5(path / "planner_replay_buffer.hdf5")
        self._warm_up = False
        print(f"Loaded buffer of length {len(self.buffer)} in planner level.")
