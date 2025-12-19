from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatdim
from hydra.utils import instantiate
from tianshou.data import Batch
import torch

from .level import Level
from ..policies import TianshouPolicy
from ..utils.buffers import ToDeviceReplayBuffer
from ..predictors import Predictor


def _replace_value(
    original_value: str, new_value: Any, config: Dict[str, Any]
) -> Dict[str, Any]:
    for key, value in config.items():
        if isinstance(value, dict):
            config[key] = _replace_value(original_value, new_value, value)
        elif value == original_value:
            config[key] = new_value
    return config


class TianshouLevel(Level):
    def __init__(
        self,
        tianshou_config: Dict[str, Any],
        buffer_size: Optional[int] = None,
        batch_size: int = 64,
        update_interval: int = 1,
        n_updates: int = 1,
        reward_calc_interval: int = 1,
        **kwargs,
    ):
        """A level which uses a Tianshou algorithm to learn a policy.

        Args:
            tianshou_config: The config containing all parameters of the Tianshou
                objects. Has to contain the following keys:

                - n_critics: The number of critic neural networks.
                - nets: The configs of the neural networks.

                    - actor_net: The config of the actor neural network.
                    - critic_net: The config of the critic neural network.

                - optims: The config of the optimizers.

                    - actor_optim: The config of the actor optimizer.
                    - critic_optim: The config of the critic optimizer.

                - actor: The config of the actor.
                - critic: The config of the critic.
                - policy: The config of the policy. In Tianshou policy refers to
                  the whole algorithm, not only to the policy itself.

                Note that Hydra's instantiate function is used to create the objects
                from the configs. Hence, the configs should specify the full class path
                of the objects to instantiate. For example, to create an Adam optimizer,
                the actor_optim item in the config should contain:

                .. code-block:: yaml

                    _target_: torch.optim.Adam
                    lr: 0.001

            buffer_size: The size of the (replay) buffer. If None, no buffer is
                created.
            batch_size: The number of samples to draw from the buffer for one training
                update.
            update_interval: The interval in which the level/the policy is updated.
            n_updates: The number of updates per interval.
            reward_calc_inteval: The interval in which the reward is calculated.
                Setting this to something larger than 1 can be useful for performance
                reasons (if the reward is computed later during sampling).
        """
        super().__init__(**kwargs)
        self.config = tianshou_config
        self.buffer_size = buffer_size
        self.sample_size = batch_size
        self.update_interval = update_interval
        self.n_updates = n_updates
        self.reward_calc_interval = reward_calc_interval
        # will store a Batch with the last transitions that have been added to the buffer
        self._newest_transition = None

    def _create_buffer(self) -> None:
        """Create the replay buffer."""
        if self.buffer_size is not None:
            self.buffer = ToDeviceReplayBuffer(
                total_size=self.buffer_size,
                buffer_num=self.n_env_instances,
                target_device=self.device,
            )
        else:
            self.buffer = None

    def prepare_config(self) -> Dict[str, Any]:
        """Prepare config for instantiation of the policy."""
        # fill in dimensions of spaces and device in config
        level_input_dim = 0 if self.input_space is None else flatdim(self.input_space)
        level_state_dims = self.get_level_state_dims()
        final_config = deepcopy(self.config)
        _replace_value(
            "__mapped_env_obs_shape__", self.mapped_env_obs_shape, final_config
        )
        _replace_value("__level_input_dim__", level_input_dim, final_config)
        _replace_value("__level_state_dims__", level_state_dims, final_config)
        _replace_value("__action_dim__", self.action_dim, final_config)
        _replace_value("__action_space__", self.action_space, final_config)
        _replace_value("__device__", self.device, final_config)
        _replace_value("__writer__", self.writer, final_config)

        self.n_critics = final_config["n_critics"]

        # actor
        if "actor" in final_config:
            self.actor = instantiate(final_config["actor"]).to(self.device)
            _replace_value("__actor__", self.actor, final_config)

        # critics
        if "critic" in final_config:
            self.critics = [
                instantiate(final_config["critic"]).to(self.device)
                for _ in range(self.n_critics)
            ]
            for i, critic in enumerate(self.critics):
                _replace_value(f"__critic_{i}__", critic, final_config)

        # optimizers
        if "actor_optim" in final_config["optims"]:
            self.actor_optim = instantiate(
                final_config["optims"]["actor_optim"],
                params=self.actor.parameters(),
            )
            self.optimizers.append(self.actor_optim)
            _replace_value("__actor_optim__", self.actor_optim, final_config)
        self.critic_optims = [
            instantiate(
                final_config["optims"]["critic_optim"], params=critic.parameters()
            )
            for critic in self.critics
        ]
        for i, critic_optim in enumerate(self.critic_optims):
            self.optimizers.append(critic_optim)
            _replace_value(f"__critic_optim_{i}__", critic_optim, final_config)

        if "alpha" in final_config:
            alpha_config = final_config["alpha"]
            # tune alpha for target entropy (has to be a parameter to be saved)
            log_alpha = torch.nn.Parameter(
                torch.tensor(
                    [alpha_config["initial_log_alpha"]],
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=True,
                )
            )
            alpha_optim = instantiate(alpha_config["optim"], params=[log_alpha])
            self.optimizers.append(alpha_optim)
            alpha = (alpha_config["target_entropy"], log_alpha, alpha_optim)
            _replace_value("__alpha__", alpha, final_config)

        return final_config

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
        self._create_buffer()

        if (
            "learning_rate_schedule" in self.config
            and self.config["learning_rate_schedule"] is not None
        ):
            self.learning_rate_schedule = instantiate(
                self.config["learning_rate_schedule"]
            )
        else:
            self.learning_rate_schedule = None

        if not keep_params:
            final_config = self.prepare_config()

            # policy
            self.ts_policy = instantiate(
                config=final_config["policy"], _convert_="all"
            )(**final_config["policy_dynamic_args"])
            self.policy = TianshouPolicy(
                action_space=self.action_space,
                ts_policy=self.ts_policy,
                device=self.device,
            )
            if "eps" in final_config:
                self.policy.ts_policy.set_eps(final_config["eps"])

    def set_n_env_instances(self, n_env_instances: int) -> None:
        """Set the number of environment instances.

        Note that this deletes and recreates the replay buffer.

        Args:
            n_env_instances: The number of environment instances.
        """
        super().set_n_env_instances(n_env_instances)
        self._create_buffer()

    def get_input_space(self) -> Space:
        """Get the input space of level.

        Input space denotes everything the level needs except for the
        environment observation, e.g. a skill vector.

        Returns:
            The input space (or None if no input is required).
        """
        return None

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
            active_instances: In which of the environment instances the level is active.
                env_obs and level_input correspond to these instances.
        Returns:
            The action (also with a batch dimension).
        """
        level_state = self.get_level_state(
            mapped_env_obs, level_input, active_instances
        )
        action, action_info = self.policy.get_action(
            mapped_env_obs[active_instances],
            level_input,
            level_state,
            deterministic=False,
        )
        self._update_action_info(active_instances, action_info)
        return action, dict()

    def _preprocess_batch(self, batch: Batch) -> Batch:
        """Preprocess a batch before it is added to the replay buffer.

        Args:
            batch: The batch to preprocess.
        Returns:
            The preprocessed batch.
        """
        return batch

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
        raw_action = self.policy.untransform_action(action)
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
            transition = Batch(
                obs={
                    "mapped_env_obs": mapped_env_obs,
                    "level_input": level_input_tensor,
                    "level_state": level_state,
                },
                act=raw_action,
                rew=reward,
                terminated=terminated,
                truncated=truncated,
                obs_next={
                    "mapped_env_obs": next_mapped_env_obs,
                    "level_input": level_input_tensor,
                    "level_state": next_level_state,
                },
                policy=self.action_info[active_instances],
                info={"env_rew": cum_env_reward},
            )
            transition.to_numpy()
            transition = self._preprocess_batch(transition)
            self._newest_transition = transition
            self._reward_info = reward_info
            self._newest_obs_buffer_index = self.buffer.add(transition)[0]

        # do not return control to level above by default
        return torch.zeros(active_instances.shape, dtype=torch.bool, device=self.device)

    def learn(self) -> None:
        """Learn from the collected (semi-MDP) transitions."""
        if self._n_new_transitions >= self.update_interval:
            n_updates = self._n_new_transitions // self.update_interval * self.n_updates
            if self.learning_rate_schedule is not None:
                for optimizer in self.optimizers:
                    optimizer.param_groups[0]["lr"] = self.learning_rate_schedule(
                        self.n_total_steps, optimizer.param_groups[0]["lr"]
                    )
            for _ in range(n_updates):
                self.policy.update(self.sample_size, self.buffer)
            self._n_new_transitions = 0

    def eval(self) -> None:
        """Set the level to evaluation mode."""
        super().eval()
        if self.policy is not None:
            self.policy.eval()

    def train(self) -> None:
        """Set the level to training mode."""
        super().train()
        if self.policy is not None:
            self.policy.train()

    def save(self, path: Path) -> None:
        """Save the level to the given path."""
        super().save(path)
        if self.policy is not None:
            torch.save(self.policy.state_dict(), path / "policy.pth")

    def load(self, path: Path) -> None:
        """Load the level from the given path."""
        super().load(path)
        if self.policy is not None:
            self.policy.load_state_dict(torch.load(path / "policy.pth"))

    def save_buffers(self, path: Path) -> None:
        """Save the replay buffer to the given path."""
        self.buffer.save_hdf5(path / "replay_buffer.hdf5")

    def load_buffers(self, path: Path) -> None:
        """Load the replay buffer from the given path."""
        self.buffer = self.buffer.load_hdf5(path / "replay_buffer.hdf5")
