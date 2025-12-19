from abc import ABC, abstractmethod
from copy import copy, deepcopy
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from gymnasium.spaces import Space, Discrete, Box
from gymnasium.spaces.utils import flatdim
from tianshou.data import Batch
import torch
from torch.utils.tensorboard import SummaryWriter

from ..predictors import Predictor


class Level(ABC):
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        writer: Optional[SummaryWriter] = None,
        log_interval: int = 1000,
    ):
        """Create the level of the hierarchy.

        Args:
            device: The device to use.
            writer: The TensorBoard writer to use for logging. If None, no logging is done.
            log_interval: The interval in which to log statistics in total (summed) environment steps.
        """
        self.action_space = None
        self.device = device
        self.writer = writer
        self.log_interval = log_interval
        self.optimizers = []

        # predictor of parent level (set automatically during initialization of the hierarchy)
        self.parent_predictor = None
        # environment reward accumulated over last transition (which could span multiple
        # environment time steps)
        self.cum_reward = None
        # number of environment steps elapsed during last transition
        self.elapsed_env_steps = None
        # last action info of level
        self.action_info = None
        self._n_new_transitions = 0
        # number of steps for which the level has been in control (for each env instance)
        # Note: steps refers to the number of steps in the SMDP corresponding to this level
        # Summed over environment instances.
        self.n_steps_in_control = None
        # Total number of environment steps seen by the whole hierarchy during training
        self.n_total_env_steps = 0
        # total number of steps
        # Note: steps refers to the number of steps in the SMDP corresponding to this level
        # and not in the environment
        self.n_total_steps = 0
        self.training = True

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
        self.env_obs_space = env_obs_space
        self.action_space = action_space
        self.n_env_instances = n_env_instances
        self.action_dim = flatdim(action_space)
        self.input_space = self.get_input_space()
        self.parent_predictor = parent_predictor
        # If no map from observation to level input is given, make sure the environment
        # observation is discrete or in a box space so that it can be handled by the
        # default map.
        assert env_obs_map is not None or isinstance(env_obs_space, (Discrete, Box))
        assert isinstance(self.input_space, (Discrete, Box)) or self.input_space is None

        self.input_dim = 0 if self.input_space is None else flatdim(self.input_space)
        self.mapped_env_obs_shape = (
            env_obs_space.shape
            if mapped_env_obs_shape is None
            else mapped_env_obs_shape
        )

        if env_obs_map is None:
            self.env_obs_map = lambda x: x
        else:
            self.env_obs_map = env_obs_map

        self.set_n_env_instances(n_env_instances)
        self._n_new_transitions = 0

    @abstractmethod
    def get_input_space(self) -> Space:
        """Get the input space of level.

        Input space denotes everything the level needs except for the
        environment observation, e.g. a skill vector.

        Returns:
            The input space (or None if no input is required).
        """
        pass

    def get_action_shape_and_type(self) -> Tuple[torch.Size, torch.dtype]:
        """Get the shape and type of the action.

        Returns:
            The shape and type of the action.
        """
        if isinstance(self.action_space, Discrete):
            return torch.Size([]), torch.int64
        elif isinstance(self.action_space, Box):
            return (
                self.action_space.shape,
                torch.float32,
            )
        else:
            raise NotImplementedError

    def sample_from_action_space(self, batch_size: int) -> torch.Tensor:
        """Sample from the action space.

        Args:
            batch_size: The batch size.
        Returns:
            The sampled actions.
        """
        if isinstance(self.action_space, Discrete):
            return torch.randint(
                low=0, high=self.action_space.n, size=(batch_size,), device=self.device
            )
        elif isinstance(self.action_space, Box):
            u = torch.rand((batch_size,) + self.action_space.shape, device=self.device)
            return torch.as_tensor(
                self.action_space.low, device=self.device
            ) + u * torch.as_tensor(
                self.action_space.high - self.action_space.low, device=self.device
            )
        else:
            raise NotImplementedError

    def reset(self):
        """Reset the level.

        Call once before the first episode, not between episodes."""
        self.soft_reset()

    def soft_reset(self):
        """Soft reset the hierarchy.

        This is called when manually resetting the (vector) environment. This will not
        affect things like warm up periods etc."""
        self.cum_reward = torch.zeros(self.n_env_instances, device=self.device)
        self.elapsed_env_steps = torch.zeros(
            self.n_env_instances, dtype=torch.int64, device=self.device
        )
        self._n_new_transitions = 0
        self.n_steps_in_control = torch.zeros(
            self.n_env_instances, dtype=torch.int64, device=self.device
        )
        self.action_info = Batch()

    def set_n_env_instances(self, n_env_instances: int) -> None:
        """Set the number of environment instances.

        Args:
            n_env_instances: The number of environment instances.
        """
        self.n_env_instances = n_env_instances
        self.cum_reward = torch.zeros(self.n_env_instances, device=self.device)
        self.elapsed_env_steps = torch.zeros(
            self.n_env_instances, dtype=torch.int64, device=self.device
        )
        self.action_info = Batch()
        self.n_steps_in_control = torch.zeros(
            self.n_env_instances, dtype=torch.int64, device=self.device
        )

    def get_copy(self) -> "Level":
        """Get a copy of the level.

        Resets the copied level.

        No parameters are copied, only the state of the level. The copy of the level
        can be used for testing rollouts without influencing the state of the original
        level, for example.

        Returns:
            A copy of the level.
        """
        level = copy(self)
        level.action_space = deepcopy(self.action_space)
        level.reset()
        return level

    def get_level_state(
        self,
        mapped_env_obs: torch.Tensor,
        level_input: torch.Tensor,
        active_instances: torch.Tensor,
    ) -> Dict:
        """Compute the state of the level.

        This contains all information relevant for determining the level action except for
        the environment observation and the level input, which are handled separately.

        Args:
            mapped_env_obs: The mapped environment observation of the active instances.
            level_input: The input to this level for the active instances, i.e., the
                action from the level above.
            active_instances: The instances for which to return the level state.
        Returns:
            The level state.
        """
        return {}

    def get_level_state_dims(self) -> Dict[str, int]:
        """Get the dimensions of the level state.

        Implement this in derived classes.

        Returns:
            A dictionary with the dimensions of each item in the level state.
        """
        return {}

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
        # return environment reward that has accumulated by default
        return cum_env_reward, {}

    def _update_action_info(self, active_instances: torch.Tensor, action_info: Dict):
        """Update the action info for the given instances.

        Args:
            active_instances: The instances to update.
            action_info: The action info to update with.
        """
        for k, v in action_info.items():
            if v is not None:
                if k not in self.action_info:
                    self.action_info[k] = torch.empty(
                        (self.n_env_instances,) + v.shape[1:],
                        dtype=v.dtype,
                        device=self.device,
                    )
                self.action_info[k][active_instances] = v

    @abstractmethod
    def get_action(
        self,
        mapped_env_obs: torch.Tensor,
        level_input: Optional[torch.Tensor],
        level_input_info: Optional[Dict],
        active_instances: torch.Tensor = torch.tensor([True], dtype=torch.bool),
    ) -> Tuple[torch.Tensor, Dict]:
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
        # To be implemented by subclasses.
        pass

    def register_env_reward(self, env_rew: torch.Tensor) -> None:
        """Register the environment reward with the level.

        Args:
            env_rew: The environment reward.
        """
        self.cum_reward += env_rew
        self.elapsed_env_steps += 1

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
        self.cum_reward[active_instances] = 0.0
        self.elapsed_env_steps[active_instances] = 0
        self._n_new_transitions += active_instances.sum().item()
        self.n_steps_in_control[active_instances] += 1
        self.n_total_steps += active_instances.sum().item()
        # Do not return control to the level above by default
        return torch.zeros(active_instances.shape, dtype=torch.bool, device=self.device)

    def add_transitions(self, transitions: Batch) -> None:
        """Add a transition to the level.

        Args:
            transitions: The batch of transitions to add. The first dimension
                corresponds to the batch dimension (e.g. environments). The following
                keys must be included:
                obs: The observation.
                act: The action.
                rew: The reward.
                terminated: Whether the episode terminated.
                truncated: Whether the episode was truncated.
                obs_next: The next observation.
        """
        raise NotImplementedError

    def learn(self) -> None:
        """Learn from the collected (semi-MDP) transitions."""
        pass

    def eval(self) -> None:
        """Set the level to evaluation mode."""
        self.training = False

    def train(self) -> None:
        """Set the level to training mode."""
        self.training = True

    def save(self, path: Path) -> None:
        """Save the level to the given path."""
        torch.save(
            [opt.state_dict() for opt in self.optimizers], path / "optimizers.pth"
        )
        torch.save({"n_total_steps": self.n_total_steps}, path / "level_state.pth")

    def load(self, path: Path) -> None:
        """Load the level from the given path."""
        for opt, state in zip(self.optimizers, torch.load(path / "optimizers.pth")):
            opt.load_state_dict(state)
        level_state = torch.load(path / "level_state.pth")
        self.n_total_steps = level_state["n_total_steps"]

    def save_buffers(self, path: Path) -> None:
        """Save the replay buffer to the given path."""
        pass

    def load_buffers(self, path: Path) -> None:
        """Load the replay buffer from the given path."""
        pass
