from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from gymnasium.spaces import Box, Space
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict
import torch

from .level import Level
from ..predictors import Predictor
from ..tdmpc2.tdmpc2 import TDMPC2
from ..tdmpc2.goal_based_tdmpc2 import GoalBasedTDMPC2
from ..tdmpc2.common.buffer import Buffer
from ..tdmpc2 import get_default_tdmpc2_config


class ProcessObsBuffer(Buffer):
    """TD-MPC2 buffer that processes env obs, level input and level state to obtain observations when sampling."""

    def __init__(
        self,
        cfg: Dict,
        get_obs: Callable,
        goal_based: bool = False,
        use_her: bool = False,
        n_her_goals: int = 2,
        achieved_goal_range: Optional[list] = None,
        desired_goal_range: Optional[list] = None,
        reward_func: Optional[Callable] = None,
        reward_offset: float = 0.0,
    ):
        """Initialize the buffer.

        Args:
            cfg: The config of the buffer.
            get_obs: A function that concatenates the observations.
            goal_based: Whether the buffer is goal-based. Requires desired_goal_range.
            use_her: Whether to use hindsight experience replay.
            achieved_goal_slice: The slice of the observation that contains the achieved goal.
            desired_goal_slice: The slice of the observation that contains the desired goal.
            reward_func: A function that computes the reward from the achieved and desired goal.
            reward_offset: An offset that is added to the reward.
        """
        super().__init__(cfg)
        assert not use_her or goal_based, "HER requires goal-based buffer."
        assert (
            not goal_based or desired_goal_range is not None
        ), "Need desired goal slice for goal-based buffer."
        assert (
            not use_her or achieved_goal_range is not None
        ), "Need achieved goal slice for HER."
        assert (
            not use_her or desired_goal_range is not None
        ), "Need desired goal slice for HER."
        self._get_obs = get_obs
        self.goal_based = goal_based
        self.use_her = use_her
        self.n_her_goals = n_her_goals
        self.achieved_goal_slice = (
            None if achieved_goal_range is None else slice(*achieved_goal_range)
        )
        self.desired_goal_slice = (
            None if desired_goal_range is None else slice(*desired_goal_range)
        )
        self.reward_func = reward_func
        self.reward_offset = reward_offset

    def sample(self):
        """Sample a batch of subsequences from the buffer."""
        td, info = self._buffer.sample(return_info=True)

        td = td.view(-1, self.cfg.horizon + 1).permute(1, 0)

        if self.goal_based:
            desired_goal = td["mapped_env_obs"][..., self.desired_goal_slice]

        # This removes the desired goal from the observation if goal-based
        obs = self._get_obs(td["mapped_env_obs"], td["level_input"], td["level_state"])
        action = td["action"][1:]
        reward = td["reward"][1:].unsqueeze(-1)
        task = td["task"][0] if "task" in td.keys() else None
        terminated = td["terminated"][:-1]

        if self.use_her:
            index = info["index"][0]
            ep_end = self._buffer["ep_end"][index]
            desired_goal_lst = [desired_goal]
            reward_lst = [reward]
            for _ in range(self.n_her_goals):
                # sample a random goal between the current time step and the end of the episode
                # (future strategy)
                rand_nums = torch.rand(index.shape, device=index.device)
                goal_index = index + torch.floor((ep_end - index) * rand_nums).to(
                    dtype=torch.long
                )
                _achieved_goal = self._buffer["mapped_env_obs"][index][
                    :, self.achieved_goal_slice
                ]
                _desired_goal = self._buffer["mapped_env_obs"][goal_index][
                    :, self.achieved_goal_slice
                ]
                _achieved_goal = _achieved_goal.view(
                    -1, self.cfg.horizon + 1, _achieved_goal.shape[-1]
                ).permute(1, 0, 2)
                _desired_goal = _desired_goal.view(
                    -1, self.cfg.horizon + 1, _desired_goal.shape[-1]
                ).permute(1, 0, 2)
                with torch.no_grad():
                    _reward = self.reward_func(
                        achieved_goal=_achieved_goal[1:],
                        desired_goal=_desired_goal[:-1],
                    ).unsqueeze(-1)
                desired_goal_lst.append(_desired_goal)
                reward_lst.append(_reward)
            desired_goal = torch.cat(desired_goal_lst, dim=1)
            reward = torch.cat(reward_lst, dim=1)
            obs = obs.repeat(1, self.n_her_goals + 1, 1)
            action = action.repeat(1, self.n_her_goals + 1, 1)
            terminated = terminated.repeat(1, self.n_her_goals + 1)
            if task is not None:
                task = torch.cat([task] * (self.n_her_goals + 1), dim=1)

        reward += self.reward_offset

        if self.goal_based:
            return self._to_device(obs, desired_goal, action, reward, terminated, task)
        else:
            return self._to_device(obs, action, reward, terminated, task)

    def save(self, path: Path):
        """Save the buffer to the given path.

        Args:
            path: The path to save the buffer to."""
        assert hasattr(self, "_buffer"), "No buffer to save."
        self._buffer.dumps(path)

    def load(self, path: Path, td: TensorDict):
        """Load the buffer from the given path.

        Args:
            path: The path to load the buffer from.
            td: A TensorDict with the same keys as the buffer
                to infer the shape of the buffer from.
        """
        if self._num_eps == 0:
            self._buffer = self._init(td)
        self._buffer.loads(path)
        unique_ep_ids = self._buffer.storage["episode"].unique()
        self._num_eps = unique_ep_ids.shape[0]

    def add(self, td: TensorDict):
        """Add an episode to the buffer.
        Args:
            td: A TensorDict containing the transitions to add.
        Returns:
            The number of episodes and the indices of the added transitions.
        """

        td["ep_end"] = torch.zeros_like(td["reward"], dtype=torch.long)
        num_eps, indices = super().add(td)
        self._buffer["ep_end"][indices] = indices[-1]
        return num_eps, indices


class TDMPC2Level(Level):
    def __init__(
        self,
        tdmpc2_config: Optional[Dict[str, Any]] = None,
        update_interval: int = 1,
        n_updates: int = 1,
        reward_calc_interval: int = 1,
        warm_up_steps: int = 2000,
        goal_based: bool = False,
        goal_dim: int = 0,
        use_her: bool = False,
        achieved_goal_range: Optional[list] = None,
        desired_goal_range: Optional[list] = None,
        reward_func: Optional[Callable] = None,
        reward_offset: float = 0.0,
        **kwargs,
    ):
        """Level that uses TD-MPC2 to plan and learn a policy.

        For an explanation of TD-MPC2 see: https://arxiv.org/abs/2310.16828

        Args:
            tdmpc2_config: The configuration for the TD-MPC2 algorithm.
            update_interval: The interval in which the level/the policy is updated.
            n_updates: The number of updates per interval.
            reward_calc_inteval: The interval in which the reward is calculated.
                Setting this to something larger than 1 can be useful for performance
                reasons (if the reward is computed later during sampling).
            warm_up_steps: The number of steps to collect with random actoins before learning starts.
            goal_based: Whether TDMPC2 should treat the goal separately. Requires desired_goal_range.
            goal_dim: The dimension of the goal.
            use_her: Whether to use hindsight experience replay. Requires goal_based and achieved_goal_range.
            achieved_goal_range: The slice of the observation that contains the achieved goal.
            desired_goal_range: The slice of the observation that contains the desired goal.
            reward_func: A function that computes the reward from the achieved and desired goal.
            reward_offset: An offset that is added to the reward.
        """
        super().__init__(**kwargs)
        if tdmpc2_config is None:
            tdmpc2_config = get_default_tdmpc2_config()
        self.tdmpc2_config = OmegaConf.create(tdmpc2_config)
        self.tdmpc2_config.task_dim = 0
        self.tdmpc2_config.bin_size = (
            self.tdmpc2_config.vmax - self.tdmpc2_config.vmin
        ) / (self.tdmpc2_config.num_bins - 1)
        self.update_interval = update_interval
        self.n_updates = n_updates
        self.reward_calc_interval = reward_calc_interval
        self.warm_up_steps = warm_up_steps
        self.goal_based = goal_based
        self.goal_dim = goal_dim
        self.use_her = use_her
        self.achieved_goal_range = achieved_goal_range
        self.desired_goal_range = desired_goal_range
        self.reward_func = reward_func
        self.reward_offset = reward_offset
        # will store a Batch with the last transitions that have been added to the buffer
        self._newest_transition = None
        self._tdmpc2 = None
        self._warm_up = True
        self._last_tdmpc2_log_step = 0
        # need to keep track of last action and reward because of the way the TDMPC2
        # buffer is organized
        self._last_action = None
        self._last_reward = None
        assert not goal_based or not (
            desired_goal_range is None and goal_dim == 0
        ), "Need goal_dim or desired goal range for goal-based TDMP2."
        self._goal_dim = (
            desired_goal_range[1] - desired_goal_range[0]
            if desired_goal_range is not None
            else goal_dim
        )

    def _create_buffer(self) -> None:
        """Create the replay buffer."""
        self.buffer = ProcessObsBuffer(
            cfg=self.tdmpc2_config,
            get_obs=self._get_tdmpc2_obs,
            goal_based=self.goal_based,
            use_her=self.use_her,
            achieved_goal_range=self.achieved_goal_range,
            desired_goal_range=self.desired_goal_range,
            reward_func=self.reward_func,
            reward_offset=self.reward_offset,
        )

    def get_tdmpc2_obs_shape(
        self, mapped_env_obs_shape: Tuple[int, ...], env_obs_space: Box
    ) -> int:
        """Get the shape of the observation for the TDMP2 algorithm."""
        actual_mapped_env_obs_dim = (
            mapped_env_obs_shape[0]
            if mapped_env_obs_shape is not None
            else env_obs_space.shape[0]
        )
        input_space_dim = (
            self.get_input_space().shape[0] if self.get_input_space() is not None else 0
        )
        level_state_dim = sum(v for v in self.get_level_state_dims().values())
        if self.goal_based:
            goal_dim = self._goal_dim
        else:
            goal_dim = 0
        return (
            actual_mapped_env_obs_dim + input_space_dim + level_state_dim - goal_dim,
        )

    def initialize(
        self,
        env_obs_space: Space,
        action_space: Space,
        n_env_instances: int,
        parent_predictor: Predictor,
        env_obs_map: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        mapped_env_obs_shape: Optional[int] = None,
        keep_params: bool = False,
    ) -> None:
        """Construct the level.

        Args:
            env_obs_space: The observation space of the environment.
            action_space: The action space of the level. If this is the lowest level,
                this is the action space of the environment. Otherwise, it is the
                input space of the level below this one.
            n_env_instances: The number of environment instances.
            parent_predictor: The predictor of the parent level (or None if there is no parent
                or the parent does not have a predictor).
            env_obs_map: A map that is applied to the environment observation.
                This can be used to implement information hiding and for moving a trained
                level from one environment to another with a different observation space.
                If None, the identity map is used.
            mapped_env_obs_shape: Shape of the output of env_obs_map. If None, the
                shape of the environment observation space is used.
            keep_params: Whether to keep the parameters of the level (e.g. the policy)
                when initializing. If False, the parameters are reset.
        """
        super().initialize(
            env_obs_space,
            action_space,
            n_env_instances,
            parent_predictor,
            env_obs_map,
            mapped_env_obs_shape,
            keep_params,
        )
        assert isinstance(
            env_obs_space, Box
        ), "Only Box observation spaces are supported in TD-MPC2 level."
        self._action_low = torch.tensor(action_space.low, device=self.device)
        self._action_high = torch.tensor(action_space.high, device=self.device)
        # observation and action space for the TDMP2 algorithm
        tdmpc2_obs_dim = self.get_tdmpc2_obs_shape(mapped_env_obs_shape, env_obs_space)
        self.tdmpc2_config.obs_shape = {self.tdmpc2_config["obs"]: tdmpc2_obs_dim}
        if self.goal_based:
            self.tdmpc2_config.goal_shape = {"state": (self._goal_dim,)}
        self.tdmpc2_config.action_dim = action_space.shape[0]
        self.tdmpc2_config.goal_dim = self._goal_dim
        self._create_buffer()
        if not keep_params:
            if self.goal_based:
                self._tdmpc2 = GoalBasedTDMPC2(self.tdmpc2_config, device=self.device)
            else:
                self._tdmpc2 = TDMPC2(self.tdmpc2_config, device=self.device)
        self._last_tdmpc2_log_step = self.n_total_steps

    def reset(self):
        """Reset the level.

        Call once before the first episode, not between episodes."""
        super().reset()
        self._tds = [[] for _ in range(self.n_env_instances)]
        self._warm_up = True
        self._last_tdmpc2_log_step = self.n_total_steps
        self._last_action = torch.zeros(
            (self.n_env_instances, self.action_dim), device=self.device
        )
        self._last_reward = torch.zeros((self.n_env_instances), device=self.device)

    def soft_reset(self):
        """Soft reset the hierarchy.

        This is called when manually resetting the (vector) environment. This will not
        affect things like warm up periods etc."""
        super().soft_reset()
        self._tds = [[] for _ in range(self.n_env_instances)]

    def set_n_env_instances(self, n_env_instances: int) -> None:
        """Set the number of environment instances.

        Note that this deletes and recreates the replay buffer.

        Args:
            n_env_instances: The number of environment instances.
        """
        super().set_n_env_instances(n_env_instances)
        self._create_buffer()
        self._tds = [[] for _ in range(self.n_env_instances)]
        self._last_action = torch.zeros(
            (n_env_instances, self.action_dim), device=self.device
        )
        self._last_reward = torch.zeros((n_env_instances), device=self.device)

    def get_input_space(self) -> Space:
        """Get the input space of level.

        Input space denotes everything the level needs except for the
        environment observation, e.g. a skill vector.

        Returns:
            The input space (or None if no input is required).
        """
        return None

    def _get_tdmpc2_obs(
        self,
        mapped_env_obs: torch.Tensor,
        level_input: torch.Tensor,
        level_state: torch.Tensor,
    ):
        """Get the observation for the TDMPC2 algorithm.

        Args:
            mapped_env_obs: The environment observation after the self.env_obs_map has been
                applied. Note that the observation has a batch dimension (for multiple
                environment instances). Only for active instances.
            level_input: The input to this level for the active instances, i.e., the
                action from the level above.
            level_state: The state of the level."""
        # concatenate environment observation, level input and level state
        if self.goal_based:
            # remove the goal from the observation in goal-based mode
            mask = torch.ones(
                mapped_env_obs.shape[-1], dtype=torch.bool, device=self.device
            )
            mask[self.desired_goal_range[0] : self.desired_goal_range[1]] = False
            obs_lst = [mapped_env_obs[..., mask]]
        else:
            obs_lst = [mapped_env_obs]
        if len(level_state.keys()) > 0:
            level_state_tensor = torch.cat([v for v in level_state.values()], dim=-1)
            obs_lst.append(level_state_tensor)
        if level_input is not None:
            obs_lst.append(level_input)
        return torch.cat(obs_lst, dim=-1)

    def get_action(
        self,
        mapped_env_obs: torch.Tensor,
        level_input: Optional[torch.Tensor],
        level_input_info: Optional[Dict],
        active_instances: torch.Tensor = torch.tensor([True], dtype=torch.bool),
    ) -> torch.Tensor:
        """Get an action for the given observation using TD-MPC2 (so MPC with MPPI + policy).

        Note that only the action for the active instances is returned.

        Call this at the beginning of the implementation of get_action in derived classes.

        Args:
            mapped_env_obs: The environment observation after the self.env_obs_map has been
                applied. Note that the observation has a batch dimension (for multiple
                environment instances).
            level_input: The input to this level for the active instances, i.e., the
                action from the level above.
            active_instances: In which of the environment instances the level is active.
                env_obs and level_input correspond to these instances.
        Returns:
            The action (also with a batch dimension).
            An info dict containing additional information about the action.
        """
        if self.n_total_steps > self.warm_up_steps:
            level_state = self.get_level_state(
                mapped_env_obs, level_input, active_instances
            )
            # get observation from environment observation, level input and level state
            tdmpc2_obs = self._get_tdmpc2_obs(
                mapped_env_obs[active_instances], level_input, level_state
            )
            if self.goal_based:
                goal = mapped_env_obs[
                    active_instances,
                    self.desired_goal_range[0] : self.desired_goal_range[1],
                ]
                action = self._tdmpc2.act(
                    tdmpc2_obs,
                    goal,
                    t0=self.n_steps_in_control[active_instances] == 0,
                    eval_mode=not self.training,
                )
            else:
                action = self._tdmpc2.act(
                    tdmpc2_obs,
                    t0=self.n_steps_in_control[active_instances] == 0,
                    eval_mode=not self.training,
                )
            action = self._action_low + 0.5 * (action + 1) * (
                self._action_high - self._action_low
            )
        else:
            action = self.sample_from_action_space(active_instances.sum())
        return action, dict()

    def _to_td(
        self,
        mapped_env_obs,
        level_state,
        level_input=None,
        action=None,
        reward=None,
        cum_env_reward=None,
        terminated=None,
        truncated=None,
    ):
        """Creates a TensorDict for a new episode (for tdmpc2)."""
        level_state = TensorDict(level_state, batch_size=(), device="cpu")
        if action is None:
            action = torch.full(
                mapped_env_obs.shape[:-1] + (self.action_dim,), float("nan")
            )
        if reward is None:
            reward = torch.tensor(float("nan"))
        if cum_env_reward is None:
            cum_env_reward = torch.tensor(float("nan"))
        if level_input is None:
            level_input = torch.full(
                mapped_env_obs.shape[:-1] + (self.input_dim,), float("nan")
            )
        if terminated is None:
            terminated = torch.zeros(mapped_env_obs.shape[:-1], dtype=torch.bool)
        if truncated is None:
            truncated = torch.zeros(mapped_env_obs.shape[:-1], dtype=torch.bool)
        td = TensorDict(
            dict(
                action=action.unsqueeze(0).cpu(),  # noqa
                reward=reward.unsqueeze(0).cpu(),
                level_input=level_input.unsqueeze(0).cpu(),
                mapped_env_obs=mapped_env_obs.unsqueeze(0).cpu(),
                level_state={k: v.unsqueeze(0).cpu() for k, v in level_state.items()},
                terminated=terminated.unsqueeze(0).cpu(),
                truncated=truncated.unsqueeze(0).cpu(),
            ),
            batch_size=(1,),
        )
        return td

    def _transform_from_action_space(self, action: torch.Tensor) -> torch.Tensor:
        """Tranform the action from the action space to [-1, 1]^dim.

        Args:
            action: The action in the action space.
        Returns:
            The action in the interval [-1, 1]^dim.
        """
        return (action - self._action_low) / (
            self._action_high - self._action_low
        ) * 2 - 1

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
        level_input_tensor = (
            torch.zeros((active_instances.shape[0], 0))
            if level_input is None
            else level_input
        )
        level_state = self.get_level_state(
            mapped_env_obs, level_input, active_instances
        )
        cum_env_reward = self.cum_reward[active_instances]
        elapsed_env_steps = self.elapsed_env_steps[active_instances]

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

        if self.training:
            next_level_state = self.get_level_state(
                next_mapped_env_obs, level_input, active_instances
            )
            if self.n_total_steps % self.reward_calc_interval == 0:
                # calculate reward for logging
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
            else:
                reward = torch.zeros(
                    active_instances.shape, dtype=torch.float, device=self.device
                )
                reward_info = {}

            active_tds = [
                tds for active, tds in zip(active_instances, self._tds) if active
            ]
            first_transitions = [len(tds) == 0 for tds in active_tds]
            for i, tds in enumerate(active_tds):
                if first_transitions[i]:
                    # at beginning of skill execution, add only obs to buffer
                    tds.append(
                        self._to_td(
                            mapped_env_obs=mapped_env_obs[i],
                            level_input=level_input_tensor[i],
                            level_state={k: v[i] for k, v in level_state.items()},
                        )
                    )
                else:
                    tds.append(
                        self._to_td(
                            mapped_env_obs=mapped_env_obs[i],
                            level_input=level_input_tensor[i],
                            level_state={k: v[i] for k, v in level_state.items()},
                            action=self._transform_from_action_space(
                                self._last_action[active_instances][i]
                            ).float(),
                            reward=self._last_reward[active_instances][i],
                            cum_env_reward=cum_env_reward[i],
                            terminated=terminated[i],
                            truncated=truncated[i],
                        )
                    )

            self._last_action[active_instances] = action
            self._last_reward[active_instances] = reward

            # if the terminated or truncated is true (not necessarily stemming from environment),
            # then add the transitions saved in the tds list to the buffer
            done = terminated | truncated
            for i, tds in enumerate(active_tds):
                if done[i]:
                    # add last state, simply repeat last action and reward
                    tds.append(
                        self._to_td(
                            mapped_env_obs=next_mapped_env_obs[i],
                            level_input=level_input_tensor[i],
                            level_state={k: v[i] for k, v in next_level_state.items()},
                            action=self._transform_from_action_space(
                                self._last_action[active_instances][i]
                            ).float(),
                            reward=self._last_reward[active_instances][i],
                            cum_env_reward=cum_env_reward[i],
                            terminated=terminated[i],
                            truncated=truncated[i],
                        )
                    )
                    self.buffer.add(torch.cat(tds))
                    self._tds[i] = []

            transitions = {
                "mapped_env_obs": mapped_env_obs,
                "level_input": level_input_tensor,
                "level_state": level_state,
                "action": action,
                "reward": reward,
                "next_mapped_env_obs": next_mapped_env_obs,
                "next_level_state": next_level_state,
                "next_level_input": level_input_tensor,
                "terminated": terminated,
                "truncated": truncated,
            }
            self._newest_transition = transitions
            self._reward_info = reward_info

        # do not return control to level above by default
        return torch.zeros(active_instances.shape, dtype=torch.bool, device=self.device)

    def learn(self) -> None:
        """Learn from the collected (semi-MDP) transitions."""
        if self.n_total_steps > self.warm_up_steps:
            if self._warm_up:
                self._n_new_transitions = 0
                self._warm_up = False
            if self._n_new_transitions >= self.update_interval:
                n_updates = (
                    self._n_new_transitions // self.update_interval * self.n_updates
                )
                if not hasattr(self.buffer, "_buffer"):
                    print(
                        "Warning: Buffer not initialized. No updates will be performed."
                    )
                    return
                for _ in range(n_updates):
                    _train_metrics = self._tdmpc2.update(self.buffer)
                self._n_new_transitions = 0

                steps_elapsed = self.n_total_steps - self._last_tdmpc2_log_step
                if self.writer is not None and steps_elapsed > self.log_interval:
                    for k, v in _train_metrics.items():
                        self.writer.add_scalar(f"tdmpc2/{k}", v, self.n_total_env_steps)
                    self._last_tdmpc2_log_step = self.n_total_steps

    def save(self, path: Path) -> None:
        """Save the level to the given path."""
        super().save(path)
        if self._tdmpc2 is not None:
            self._tdmpc2.save(path / "tdmpc2.pth")

    def load(self, path: Path) -> None:
        """Load the level from the given path."""
        super().load(path)
        if self._tdmpc2 is not None:
            self._tdmpc2.load(path / "tdmpc2.pth")

    def save_buffers(self, path: Path) -> None:
        """Save the replay buffer to the given path."""
        self.buffer.save(path / "replay_buffer")

    def load_buffers(self, path: Path) -> None:
        """Load the replay buffer from the given path."""
        input_space = self.get_input_space()
        if input_space is None:
            input_dim = 0
        else:
            input_dim = self.get_input_space().shape[0]
        # create dummy data to initialize the buffer
        dummy_mapped_env_obs = torch.zeros(
            (self.n_env_instances, self.mapped_env_obs_shape[0]), device=self.device
        )
        dummy_level_input = torch.zeros(
            (self.n_env_instances, input_dim), device=self.device
        )
        dummy_level_state = self.get_level_state(
            dummy_mapped_env_obs,
            dummy_level_input,
            torch.tensor(
                [True] * self.n_env_instances, dtype=torch.bool, device=self.device
            ),
        )
        dummy_action = torch.zeros(
            (self.n_env_instances, self.action_space.shape[0]), device=self.device
        )
        dummy_reward = torch.ones((self.n_env_instances, 1), device=self.device)
        dummy_td = self._to_td(
            mapped_env_obs=dummy_mapped_env_obs,
            level_input=dummy_level_input,
            level_state=dummy_level_state,
            action=dummy_action,
            reward=dummy_reward,
        )
        self.buffer.load(path / "replay_buffer", dummy_td)
