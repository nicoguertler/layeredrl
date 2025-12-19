from copy import copy
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml

from ..levels import Level
from ..utils.misc import to_torch


def _filter_dict_active_instances(active_instances: torch.Tensor, info: dict) -> dict:
    """Filter the info dictionary for the given active instances.

    Args:
        active_instances: A boolean tensor indicating which instances to keep.
        info: The info dictionary to filter.
    Returns:
        The filtered info dictionary.
    """
    return {key: value[active_instances] for key, value in info.items()}


class Hierarchy:
    def __init__(
        self,
        levels: List[Level],
        env: gym.Env,
        env_obs_maps: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
        mapped_env_obs_shapes: Optional[List[Tuple]] = None,
        keep_params: bool = False,
        device: torch.device = torch.device("cpu"),
        writer: Optional[SummaryWriter] = None,
    ):
        """Initialize the hierarchy. Makes sure the action a level emits fits the input
        expected by the level below it.

        Args:
            levels: The levels of the hierarchy.
            env: The environment.
            obs_input_maps: A list of functions that map the environment observation to a vector
                that is provided to the corresponding level. This can be used to implement
                information hiding and for moving a trained level from one environment to another
                with a different observation space. If None, the identity map is used.
            mapped_env_obs_shapes: The shapes of the output of the env_obs_map of each level. If
                None, the dimension of the environment observation space is used. If negative, the
                negative dimension is added to the dimension of the environment observation space.
                This is useful if the map from the environment observation to the level
                observation is dropping some components.
            keep_params: Whether to keep the parameters of the levels instead of resetting them.
                Setting this to True is only valid if the levels were already initialized before.
            device: The device to use.
            writer: The TensorBoard writer to use for logging. If None, no logging is done.
        """
        self.levels = levels
        self.env = env
        self.active_level = None
        self.n_levels = len(levels)
        assert (
            mapped_env_obs_shapes is None or len(mapped_env_obs_shapes) == self.n_levels
        ), "mapped_env_obs_shapes must be None or have the same length as levels"
        assert (
            env_obs_maps is None or len(env_obs_maps) == self.n_levels
        ), "env_obs_maps must be None or have the same length as levels"
        self.mapped_env_obs_shapes = (
            [None] * self.n_levels
            if mapped_env_obs_shapes is None
            else mapped_env_obs_shapes
        )
        # Total number of environment steps the hierarchy has seen during training (summed over
        # environment instances).
        self.n_total_env_steps = 0
        # if mapped env obs dimension is negative, add to dimension of the environment observation
        # space
        for i, shape in enumerate(self.mapped_env_obs_shapes):
            if shape is not None and len(shape) == 1 and shape[0] < 0:
                self.mapped_env_obs_shapes[i] = (
                    self.env.observation_space.shape[-1] + shape[0],
                )
        self.env_obs_maps = (
            [None] * self.n_levels if env_obs_maps is None else env_obs_maps
        )
        self.device = device
        self.writer = writer

        if isinstance(env, gym.vector.VectorEnv):
            self.action_space = env.single_action_space
            self.observation_space = env.single_observation_space
            self.n_env_instances = env.num_envs
        else:
            self.action_space = env.action_space
            self.observation_space = env.observation_space
            self.n_env_instances = 1

        # initialize levels such that their action space is the input space of the level below
        # (or the environment action space for the lowest level)
        for i in range(self.n_levels):
            if i == self.n_levels - 1:
                action_space = self.action_space
            else:
                action_space = levels[i + 1].get_input_space()
            if i == 0:
                parent_predictor = None
            elif hasattr(levels[i - 1], "predictor"):
                parent_predictor = levels[i - 1].predictor
            else:
                parent_predictor = None
            levels[i].initialize(
                env_obs_space=self.observation_space,
                action_space=action_space,
                n_env_instances=self.n_env_instances,
                parent_predictor=parent_predictor,
                env_obs_map=self.env_obs_maps[i],
                mapped_env_obs_shape=self.mapped_env_obs_shapes[i],
                keep_params=keep_params,
            )
        # It is not necessary to set the number of environment instances for the
        # levels again because this already happens when calling Level.initialize(...).
        self.set_n_env_instances(self.n_env_instances, propagate_to_levels=False)
        self.training = True

    def reset(self) -> None:
        """Reset the hierarchy.

        Call this at the beginning of the session. The highest level
        is active at the beginning for all environment instances.

        Do not call at the end of episodes.
        """
        self.active_level = torch.zeros(
            self.n_env_instances, dtype=torch.int, device=self.device
        )
        for level in self.levels:
            level.reset()

    def soft_reset(self) -> None:
        """Soft reset the hierarchy.

        Call this when manually resetting the (vector) environment. This will not
        affect things like warm up periods etc. and can therefore be called
        without influencing the learning process."""
        self.active_level = torch.zeros(
            self.n_env_instances, dtype=torch.int, device=self.device
        )
        for level in self.levels:
            level.soft_reset()

    def get_copy(self) -> "Hierarchy":
        """Return a copy of the hierarchy.

        No models are copied, only the structure and state of the hierarchy.

        The copy of the hierarchy can be used for testing rollouts without influencing
        the state of the original hierarchy, for example. Learning with the copy
        will influence the original hierarchy, however, and is not recommended.
        """
        hierarchy = copy(self)
        hierarchy.levels = [level.get_copy() for level in self.levels]
        hierarchy.set_n_env_instances(self.n_env_instances, propagate_to_levels=False)
        hierarchy.reset()
        return hierarchy

    def set_n_env_instances(
        self, n_env_instances: int, propagate_to_levels: bool = True
    ) -> None:
        """Set the number of environment instances.

        Args:
            n_env_instances: The number of environment instances.
        """
        self.n_env_instances = n_env_instances
        self.mapped_env_obs = []
        self.level_action = []
        self.level_action_info = []
        for level in self.levels:
            if propagate_to_levels:
                level.set_n_env_instances(n_env_instances)
            self.mapped_env_obs.append(
                torch.zeros(
                    size=(self.n_env_instances,) + level.mapped_env_obs_shape,
                    dtype=torch.float,
                    device=self.device,
                )
            )
            action_shape, action_dtype = level.get_action_shape_and_type()
            level_action = torch.zeros(
                size=(self.n_env_instances,) + action_shape,
                dtype=action_dtype,
                device=self.device,
            )
            self.level_action.append(level_action)
            self.level_action_info.append(dict())

    def _update_level_action_info(
        self, level: int, info: dict, active_instances: torch.Tensor
    ) -> None:
        """Update the level action info dictionary for the given level.

        Args:
            level: The level for which to update the action info.
            info: The info dictionary to update the level action info with.
            active_instances: A boolean tensor indicating which instances to update.
        """
        for key, value in info.items():
            if key not in self.level_action_info[level]:
                self.level_action_info[level][key] = torch.zeros(
                    (self.n_env_instances,) + value.shape[1:],
                    dtype=value.dtype,
                    device=self.device,
                )
            self.level_action_info[level][key][active_instances] = value

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get an action for the given observation.

        Note that obs and the returned action have a batch dimension
        corresponding to environment instances.

        The method descends the hierarchy from top to bottom, starting
        with the active level. From thereon, an action is obtained for
        each level which is then passed to the level below. The action
        of the lowest level is returned (to be executed in the environment).

        Args:
            obs: The environment observation.
        Returns:
            The action for the environment.
        """
        # Make sure obs is tensor
        obs = to_torch(obs, self.device)

        # downward pass for obtaining actions
        for i, level in enumerate(self.levels):
            # only get action for those instances for which the level is active
            active_instances = self.active_level == i
            if active_instances.any():
                level_obs = obs[active_instances].float()
                higher_level_action = (
                    self.level_action[i - 1][active_instances] if i > 0 else None
                )
                higher_level_action_info = (
                    _filter_dict_active_instances(
                        active_instances, self.level_action_info[i - 1]
                    )
                    if i > 0
                    else None
                )
                mapped_level_obs = level.env_obs_map(level_obs)
                self.mapped_env_obs[i][active_instances] = mapped_level_obs
                level_action, level_info = level.get_action(
                    mapped_level_obs,
                    higher_level_action,
                    higher_level_action_info,
                    active_instances,
                )
                self.level_action[i][active_instances] = level_action.type_as(
                    self.level_action[i]
                )
                self._update_level_action_info(i, level_info, active_instances)
                if i < self.n_levels - 1:
                    # after obtaining the action, the level below becomes active
                    # except for the last level which stays active
                    self.active_level[active_instances] = i + 1

        # return the action of the lowest level
        return self.level_action[-1]

    def process_transition(
        self,
        obs_next: torch.Tensor,
        rew: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Process the environment transition and return control to the higher levels where appropriate.

        Starting from the lowest level, the active level can pass control back to
        the level above. This can continue until a level stays active or the highest
        level is reached.

        While ascending the hierarchy, register the (semi-MDP) transitions with the
        levels.

        Args:
            obs_next: The next environment observation.
            rew: The reward of the environment transition.
            terminated: Whether the episode terminated. Tensor with one entry per environment instance.
            truncated: Whether the episode was truncated. Tensor with one entry per environment instance.
        """
        obs_next = to_torch(obs_next, self.device).float()
        rew = to_torch(rew, self.device)
        terminated = to_torch(terminated, self.device)
        truncated = to_torch(truncated, self.device)

        self.n_total_env_steps += self.n_env_instances
        # register environment reward
        for i in range(self.n_levels):
            self.levels[i].register_env_reward(rew)
            self.levels[i].n_total_env_steps = self.n_total_env_steps
        # upward pass for returning control to the higher levels
        for i in reversed(range(self.n_levels)):
            # only check those instances for which the level is active
            active_instances = self.active_level == i
            if active_instances.any():
                next_level_obs = obs_next[active_instances].float()
                next_mapped_level_obs = self.levels[i].env_obs_map(next_level_obs)
                higher_level_action = (
                    self.level_action[i - 1][active_instances] if i > 0 else None
                )
                level_action = self.level_action[i][active_instances]
                return_control = self.levels[i].process_transition(
                    self.mapped_env_obs[i][active_instances],
                    higher_level_action,
                    level_action,
                    next_mapped_level_obs,
                    terminated[active_instances],
                    truncated[active_instances],
                    active_instances,
                )
                if i > 0:
                    # return control to the level above if the level is done,
                    # if the episode was terminated, or if the episode was truncated
                    return_control_or_terminated = torch.logical_or(
                        return_control, terminated[active_instances]
                    )
                    # This generates somewhat incomplete transitions on the higher level
                    # but it's still better than not having a transition that truncates
                    # or terminates on the higher level
                    rc_or_term_or_trunc = torch.logical_or(
                        return_control_or_terminated, truncated[active_instances]
                    )
                    self.active_level[rc_or_term_or_trunc] = i - 1
                    # reset the number of steps the level has been in control in
                    # these cases
                    self.levels[i].n_steps_in_control[return_control_or_terminated] = 0

        # give control to the highest level if the episode truncated
        self.active_level[truncated] = 0

    def learn(self) -> None:
        """Learn from the collected transitions."""
        for level in self.levels:
            level.learn()

    def eval(self) -> None:
        """Set all levels of the hierarchy to evaluation mode."""
        self.training = False
        for level in self.levels:
            level.eval()

    def train(self) -> None:
        """Set all levels of the hierarchy to training mode."""
        self.training = True
        for level in self.levels:
            level.train()

    def save(self, path: Path) -> None:
        """Save the hierarchy to the given path."""
        path.mkdir(parents=True, exist_ok=True)
        hierarchy_state = {"self.n_total_env_steps": self.n_total_env_steps}
        with open(path / "hierarchy_state.yaml", "w") as f:
            yaml.dump(hierarchy_state, f)
        for i, level in enumerate(self.levels):
            level_path = path / f"level_{i}"
            level_path.mkdir(parents=True, exist_ok=True)
            level.save(level_path)

    def load(self, path: Path) -> None:
        """Load the hierarchy from the given path."""
        with open(path / "hierarchy_state.yaml", "r") as f:
            hierarchy_state = yaml.safe_load(f)
        self.n_total_env_steps = hierarchy_state["self.n_total_env_steps"]
        for i, level in enumerate(self.levels):
            level.load(path / f"level_{i}")
            level.n_total_env_steps = self.n_total_env_steps

    def save_buffers(self, path: Path) -> None:
        """Save the replay buffers of all levels to the given path."""
        for i, level in enumerate(self.levels):
            level_path = path / f"level_{i}"
            level_path.mkdir(parents=True, exist_ok=True)
            level.save_buffers(level_path)

    def load_buffers(self, path: Path) -> None:
        """Load the replay buffers of all levels from the given path."""
        for i, level in enumerate(self.levels):
            level.load_buffers(path / f"level_{i}")
